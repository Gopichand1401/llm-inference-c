/*
 * xlsx_reader.c - general-purpose .xlsx reader (see include/xlsx_reader.h).
 *
 * Layout of an .xlsx file: a ZIP archive of XML files.
 *   xl/workbook.xml            tab names and their relationship IDs
 *   xl/_rels/workbook.xml.rels relationship ID -> worksheet file
 *   xl/sharedStrings.xml       every distinct text value, referenced by index
 *   xl/styles.xml              number formats (needed to recognise dates)
 *   xl/worksheets/sheetN.xml   the cells of each tab
 *
 * Sections of this file:
 *   1. small helpers (errors, growable buffer, string arena)
 *   2. ZIP directory + extraction (the only use of zlib is zlib_inflate_raw)
 *   3. minimal XML scanner and text clean-up
 *   4. number / date formatting
 *   5. parsers: workbook, relationships, shared strings, styles, worksheet
 *   6. public API
 */
#include "xlsx_reader.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define MAX_FILE_BYTES ((size_t)1024 * 1024 * 1024)
#define MAX_ENTRY_BYTES ((size_t)512 * 1024 * 1024)
#define MAX_COLUMNS 16384          /* Excel's own limit (XFD) */
#define ARENA_BLOCK ((size_t)256 * 1024)

static const char EMPTY[] = "";

/* ================================================================== */
/* 1. Helpers                                                          */
/* ================================================================== */

static void set_error(char *error, size_t capacity, const char *fmt, ...)
{
    va_list ap;

    if (error == NULL || capacity == 0) {
        return;
    }
    va_start(ap, fmt);
    vsnprintf(error, capacity, fmt, ap);
    va_end(ap);
}

/* Growable byte buffer, always NUL-terminated once it has data. */
typedef struct {
    char *data;
    size_t len;
    size_t cap;
} Buf;

static int buf_reserve(Buf *b, size_t extra)
{
    size_t need = b->len + extra + 1;

    if (need <= b->cap) {
        return 0;
    }
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) {
        cap *= 2;
    }
    char *p = realloc(b->data, cap);
    if (p == NULL) {
        return -1;
    }
    b->data = p;
    b->cap = cap;
    return 0;
}

static void buf_clear(Buf *b)
{
    b->len = 0;
    if (b->data != NULL) {
        b->data[0] = '\0';
    }
}

static const char *buf_str(const Buf *b)
{
    return b->data != NULL ? b->data : EMPTY;
}

static void buf_free(Buf *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static int buf_append(Buf *b, const char *s, size_t n)
{
    if (buf_reserve(b, n) != 0) {
        return -1;
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 0;
}

/* String arena: every cell text lives here and is freed in one go. */
typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used;
    size_t cap;
} ArenaBlock;

static const char *arena_dup(ArenaBlock **head, const char *s, size_t n)
{
    ArenaBlock *blk = *head;
    size_t need = n + 1;

    if (n == 0) {
        return EMPTY;
    }
    if (blk == NULL || blk->cap - blk->used < need) {
        size_t cap = need > ARENA_BLOCK ? need : ARENA_BLOCK;
        blk = malloc(sizeof(*blk) + cap);
        if (blk == NULL) {
            return NULL;
        }
        blk->next = *head;
        blk->used = 0;
        blk->cap = cap;
        *head = blk;
    }
    char *p = (char *)(blk + 1) + blk->used;
    memcpy(p, s, n);
    p[n] = '\0';
    blk->used += need;
    return p;
}

/* ================================================================== */
/* 2. ZIP container                                                    */
/* ================================================================== */

typedef struct {
    const unsigned char *name;
    size_t name_len;
    unsigned method;           /* 0 stored, 8 deflate */
    uint32_t csize;
    uint32_t usize;
    uint32_t offset;           /* of the local header */
} ZipEntry;

typedef struct {
    const unsigned char *data;
    size_t size;
    ZipEntry *entries;
    size_t count;
} Zip;

static unsigned le16(const unsigned char *p)
{
    return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static uint32_t le32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* The single place zlib is used: raw DEFLATE (no zlib header) -> bytes. */
static int zlib_inflate_raw(const unsigned char *src, size_t src_len,
                            unsigned char *dst, size_t dst_len)
{
    z_stream zs;
    int rc;

    memset(&zs, 0, sizeof(zs));
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
        return -1;
    }
    zs.next_in = (Bytef *)(uintptr_t)src;
    zs.avail_in = (uInt)src_len;
    zs.next_out = dst;
    zs.avail_out = (uInt)dst_len;
    rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END || zs.total_out != dst_len) {
        return -1;
    }
    return 0;
}

static int zip_index(Zip *z, const unsigned char *data, size_t size,
                     char *error, size_t cap)
{
    size_t eocd = 0;
    int found = 0;

    memset(z, 0, sizeof(*z));
    z->data = data;
    z->size = size;

    if (size < 22) {
        set_error(error, cap, "file is too small to be an .xlsx workbook");
        return -1;
    }
    if (size >= 8 && memcmp(data, "\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", 8) == 0) {
        set_error(error, cap,
                  "this is an old binary .xls file or a password-protected "
                  "workbook; save it as an unprotected .xlsx first");
        return -1;
    }
    if (memcmp(data, "PK", 2) != 0) {
        set_error(error, cap, "not a ZIP/.xlsx file (missing 'PK' signature)");
        return -1;
    }

    /* The end-of-central-directory record is in the last 64 KB + 22 bytes. */
    size_t lowest = size > 65557 ? size - 65557 : 0;
    for (size_t i = size - 22 + 1; i-- > lowest;) {
        if (le32(data + i) == 0x06054B50u) {
            eocd = i;
            found = 1;
            break;
        }
    }
    if (!found) {
        set_error(error, cap, "ZIP directory not found (corrupt file?)");
        return -1;
    }

    size_t total = le16(data + eocd + 10);
    uint32_t cd_size = le32(data + eocd + 12);
    uint32_t cd_off = le32(data + eocd + 16);

    if (total == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_off == 0xFFFFFFFFu) {
        set_error(error, cap, "ZIP64 archives are not supported");
        return -1;
    }
    if ((size_t)cd_off > size || (size_t)cd_size > size - (size_t)cd_off) {
        set_error(error, cap, "ZIP directory lies outside the file");
        return -1;
    }

    z->entries = calloc(total ? total : 1, sizeof(ZipEntry));
    if (z->entries == NULL) {
        set_error(error, cap, "out of memory");
        return -1;
    }

    const unsigned char *p = data + cd_off;
    const unsigned char *end = data + cd_off + cd_size;
    for (size_t i = 0; i < total; i++) {
        if ((size_t)(end - p) < 46 || le32(p) != 0x02014B50u) {
            set_error(error, cap, "corrupt ZIP directory entry %zu", i);
            return -1;
        }
        size_t nlen = le16(p + 28);
        size_t xlen = le16(p + 30);
        size_t clen = le16(p + 32);
        if ((size_t)(end - p) < 46 + nlen + xlen + clen) {
            set_error(error, cap, "corrupt ZIP directory entry %zu", i);
            return -1;
        }
        if (le16(p + 8) & 1u) {
            set_error(error, cap, "encrypted workbooks are not supported");
            return -1;
        }
        ZipEntry *e = &z->entries[z->count++];
        e->method = le16(p + 10);
        e->csize = le32(p + 20);
        e->usize = le32(p + 24);
        e->offset = le32(p + 42);
        e->name = p + 46;
        e->name_len = nlen;
        p += 46 + nlen + xlen + clen;
    }
    return 0;
}

static const ZipEntry *zip_find(const Zip *z, const char *name)
{
    size_t n = strlen(name);

    for (size_t i = 0; i < z->count; i++) {
        if (z->entries[i].name_len == n &&
            memcmp(z->entries[i].name, name, n) == 0) {
            return &z->entries[i];
        }
    }
    return NULL;
}

/* Extract one entry into a new NUL-terminated buffer (caller frees). */
static char *zip_extract(const Zip *z, const ZipEntry *e, size_t *out_len,
                         char *error, size_t cap)
{
    size_t off = e->offset;

    if (off > z->size || z->size - off < 30 ||
        le32(z->data + off) != 0x04034B50u) {
        set_error(error, cap, "corrupt ZIP local header");
        return NULL;
    }
    size_t data_off = off + 30 + le16(z->data + off + 26) +
                      le16(z->data + off + 28);
    if (data_off > z->size || (size_t)e->csize > z->size - data_off) {
        set_error(error, cap, "ZIP entry data lies outside the file");
        return NULL;
    }
    if ((size_t)e->usize > MAX_ENTRY_BYTES) {
        set_error(error, cap, "ZIP entry is unreasonably large");
        return NULL;
    }

    unsigned char *buf = malloc((size_t)e->usize + 1);
    if (buf == NULL) {
        set_error(error, cap, "out of memory");
        return NULL;
    }
    if (e->method == 0) {
        if (e->csize != e->usize) {
            free(buf);
            set_error(error, cap, "corrupt stored ZIP entry");
            return NULL;
        }
        memcpy(buf, z->data + data_off, e->usize);
    } else if (e->method == 8) {
        if (zlib_inflate_raw(z->data + data_off, e->csize, buf, e->usize) != 0) {
            free(buf);
            set_error(error, cap, "decompression failed (corrupt entry)");
            return NULL;
        }
    } else {
        free(buf);
        set_error(error, cap, "unsupported ZIP compression method %u",
                  e->method);
        return NULL;
    }
    buf[e->usize] = '\0';
    *out_len = e->usize;
    return (char *)buf;
}

/* Extract by name. *missing is set to 1 when the entry does not exist. */
static char *zip_read(const Zip *z, const char *name, size_t *len,
                      int *missing, char *error, size_t cap)
{
    const ZipEntry *e = zip_find(z, name);

    *missing = (e == NULL);
    if (e == NULL) {
        return NULL;
    }
    return zip_extract(z, e, len, error, cap);
}

/* ================================================================== */
/* 3. XML scanner and text clean-up                                    */
/* ================================================================== */

typedef enum { XML_EOF, XML_START, XML_END, XML_TEXT } XmlKind;

typedef struct {
    XmlKind kind;
    const char *name;      /* START / END */
    size_t name_len;
    const char *attrs;     /* START: raw text after the name */
    size_t attrs_len;
    int self_closing;      /* START: <tag ... /> */
    const char *text;      /* TEXT: raw, entities not yet decoded */
    size_t text_len;
    int cdata;             /* TEXT came from <![CDATA[ ]]> */
} XmlToken;

typedef struct {
    const char *p;
    const char *end;
} XmlScan;

static int is_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static const char *find_str(const char *s, const char *end, const char *pat)
{
    size_t n = strlen(pat);

    while (s < end && (size_t)(end - s) >= n) {
        const char *q = memchr(s, pat[0], (size_t)(end - s) - n + 1);
        if (q == NULL) {
            return NULL;
        }
        if (memcmp(q, pat, n) == 0) {
            return q;
        }
        s = q + 1;
    }
    return NULL;
}

/* Next token. Comments, <? ?> and <! > are skipped. 0 at end of input. */
static int xml_next(XmlScan *x, XmlToken *t)
{
    memset(t, 0, sizeof(*t));

    for (;;) {
        if (x->p >= x->end) {
            t->kind = XML_EOF;
            return 0;
        }
        size_t rem = (size_t)(x->end - x->p);

        if (*x->p != '<') {
            const char *e = memchr(x->p, '<', rem);
            if (e == NULL) {
                e = x->end;
            }
            t->kind = XML_TEXT;
            t->text = x->p;
            t->text_len = (size_t)(e - x->p);
            x->p = e;
            return 1;
        }
        if (rem >= 4 && memcmp(x->p, "<!--", 4) == 0) {
            const char *e = find_str(x->p + 4, x->end, "-->");
            x->p = e != NULL ? e + 3 : x->end;
            continue;
        }
        if (rem >= 9 && memcmp(x->p, "<![CDATA[", 9) == 0) {
            const char *e = find_str(x->p + 9, x->end, "]]>");
            if (e == NULL) {
                x->p = x->end;
                continue;
            }
            t->kind = XML_TEXT;
            t->cdata = 1;
            t->text = x->p + 9;
            t->text_len = (size_t)(e - (x->p + 9));
            x->p = e + 3;
            return 1;
        }
        if (rem >= 2 && x->p[1] == '?') {
            const char *e = find_str(x->p + 2, x->end, "?>");
            x->p = e != NULL ? e + 2 : x->end;
            continue;
        }
        if (rem >= 2 && x->p[1] == '!') {
            const char *e = memchr(x->p, '>', rem);
            x->p = e != NULL ? e + 1 : x->end;
            continue;
        }
        if (rem >= 2 && x->p[1] == '/') {
            const char *s = x->p + 2;
            const char *q = s;
            while (q < x->end && *q != '>' && !is_ws(*q)) {
                q++;
            }
            const char *gt = memchr(q, '>', (size_t)(x->end - q));
            if (gt == NULL) {
                x->p = x->end;
                continue;
            }
            t->kind = XML_END;
            t->name = s;
            t->name_len = (size_t)(q - s);
            x->p = gt + 1;
            return 1;
        }

        /* Start tag. */
        const char *s = x->p + 1;
        const char *q = s;
        while (q < x->end && *q != '>' && *q != '/' && !is_ws(*q)) {
            q++;
        }
        const char *attrs = q;
        char quote = 0;
        while (q < x->end) {
            if (quote) {
                if (*q == quote) {
                    quote = 0;
                }
            } else if (*q == '"' || *q == '\'') {
                quote = *q;
            } else if (*q == '>') {
                break;
            }
            q++;
        }
        if (q >= x->end) {
            x->p = x->end;
            t->kind = XML_EOF;
            return 0;
        }
        t->kind = XML_START;
        t->name = s;
        t->name_len = (size_t)(attrs - s);
        t->self_closing = (q > attrs && q[-1] == '/');
        t->attrs = attrs;
        t->attrs_len = (size_t)(q - attrs) - (t->self_closing ? 1u : 0u);
        x->p = q + 1;
        return 1;
    }
}

static int name_is(const XmlToken *t, const char *name)
{
    size_t n = strlen(name);

    return t->name_len == n && memcmp(t->name, name, n) == 0;
}

static size_t put_utf8(char *out, unsigned long cp)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

/*
 * Decode XML entities (&amp; &#65; &#x41;) and Excel's _xHHHH_ escapes
 * from s[0..n) into out. Decoded text is never longer than the input, so
 * out needs room for n + 1 bytes. Returns the decoded length.
 */
static size_t xml_decode(const char *s, size_t n, char *out)
{
    size_t w = 0;

    for (size_t i = 0; i < n;) {
        if (s[i] == '&') {
            size_t j = i + 1;
            while (j < n && j - i <= 10 && s[j] != ';') {
                j++;
            }
            if (j < n && s[j] == ';') {
                size_t len = j - i - 1;
                const char *e = s + i + 1;
                if (len == 3 && memcmp(e, "amp", 3) == 0) {
                    out[w++] = '&';
                } else if (len == 2 && memcmp(e, "lt", 2) == 0) {
                    out[w++] = '<';
                } else if (len == 2 && memcmp(e, "gt", 2) == 0) {
                    out[w++] = '>';
                } else if (len == 4 && memcmp(e, "quot", 4) == 0) {
                    out[w++] = '"';
                } else if (len == 4 && memcmp(e, "apos", 4) == 0) {
                    out[w++] = '\'';
                } else if (len >= 2 && e[0] == '#') {
                    unsigned long cp = 0;
                    int ok = 1;
                    if (e[1] == 'x' || e[1] == 'X') {
                        for (size_t k = 2; k < len; k++) {
                            int h = hex_value(e[k]);
                            if (h < 0) {
                                ok = 0;
                                break;
                            }
                            cp = cp * 16 + (unsigned long)h;
                        }
                        if (len == 2) {
                            ok = 0;
                        }
                    } else {
                        for (size_t k = 1; k < len; k++) {
                            if (e[k] < '0' || e[k] > '9') {
                                ok = 0;
                                break;
                            }
                            cp = cp * 10 + (unsigned long)(e[k] - '0');
                        }
                    }
                    if (ok && cp > 0 && cp <= 0x10FFFF) {
                        w += put_utf8(out + w, cp);
                    } else if (!ok) {
                        out[w++] = '&';
                        i++;
                        continue;
                    }
                } else {
                    out[w++] = '&';
                    i++;
                    continue;
                }
                i = j + 1;
                continue;
            }
        } else if (s[i] == '_' && i + 6 < n && s[i + 1] == 'x' &&
                   s[i + 6] == '_') {
            int h0 = hex_value(s[i + 2]), h1 = hex_value(s[i + 3]);
            int h2 = hex_value(s[i + 4]), h3 = hex_value(s[i + 5]);
            if (h0 >= 0 && h1 >= 0 && h2 >= 0 && h3 >= 0) {
                unsigned long cp = (unsigned long)((h0 << 12) | (h1 << 8) |
                                                   (h2 << 4) | h3);
                if (cp >= 0xD800 && cp <= 0xDFFF) {
                    out[w++] = '?';
                } else if (cp != 0) {
                    w += put_utf8(out + w, cp);
                }
                i += 7;
                continue;
            }
        }
        out[w++] = s[i++];
    }
    out[w] = '\0';
    return w;
}

/* Append s[0..n) to a buffer, decoding entities unless it is CDATA. */
static int buf_append_xml(Buf *b, const char *s, size_t n, int raw)
{
    if (raw) {
        return buf_append(b, s, n);
    }
    if (buf_reserve(b, n) != 0) {
        return -1;
    }
    b->len += xml_decode(s, n, b->data + b->len);
    return 0;
}

/*
 * Value of attribute 'name' of a start tag, decoded, into out. A name that
 * starts with ':' matches by suffix (":id" matches r:id, ns1:id ...).
 */
static int xml_attr(const XmlToken *t, const char *name, char *out, size_t cap)
{
    const char *p = t->attrs;
    const char *end = t->attrs + t->attrs_len;
    const int suffix = (name[0] == ':');
    const size_t nl = strlen(name);

    while (p < end) {
        while (p < end && is_ws(*p)) {
            p++;
        }
        const char *ns = p;
        while (p < end && *p != '=' && !is_ws(*p)) {
            p++;
        }
        size_t nlen = (size_t)(p - ns);
        while (p < end && is_ws(*p)) {
            p++;
        }
        if (p >= end || *p != '=') {
            break;
        }
        p++;
        while (p < end && is_ws(*p)) {
            p++;
        }
        if (p >= end || (*p != '"' && *p != '\'')) {
            break;
        }
        char q = *p++;
        const char *vs = p;
        while (p < end && *p != q) {
            p++;
        }
        size_t vlen = (size_t)(p - vs);
        if (p < end) {
            p++;
        }
        int match = suffix ? (nlen >= nl && memcmp(ns + nlen - nl, name, nl) == 0)
                           : (nlen == nl && memcmp(ns, name, nl) == 0);
        if (match) {
            if (vlen + 1 > cap) {
                vlen = cap - 1;
            }
            xml_decode(vs, vlen, out);
            return 1;
        }
    }
    return 0;
}

/* Byte length of the whitespace character at r (0 if it is not one). */
static size_t space_len(const unsigned char *r)
{
    if (r[0] < 0x20 || r[0] == ' ') {
        return 1;
    }
    if (r[0] == 0xC2 && (r[1] == 0xA0 || r[1] == 0x85)) {        /* NBSP, NEL */
        return 2;
    }
    if (r[0] == 0xE1 && r[1] == 0x9A && r[2] == 0x80) {          /* U+1680 */
        return 3;
    }
    if (r[0] == 0xE2 && r[1] == 0x80 &&
        ((r[2] >= 0x80 && r[2] <= 0x8A) || r[2] == 0xA8 || r[2] == 0xA9 ||
         r[2] == 0xAF)) {                         /* U+2000-200A, 2028/9, 202F */
        return 3;
    }
    if (r[0] == 0xE2 && r[1] == 0x81 && r[2] == 0x9F) {          /* U+205F */
        return 3;
    }
    if (r[0] == 0xE3 && r[1] == 0x80 && r[2] == 0x80) {          /* U+3000 */
        return 3;
    }
    return 0;
}

/*
 * Make cell text one clean line, in place: control characters, tabs, line
 * breaks and Unicode spaces become single spaces; zero-width characters are
 * dropped; leading and trailing space is removed.
 */
static void clean_text(char *s)
{
    char *w = s;
    int pending = 0;
    int started = 0;
    const unsigned char *r = (const unsigned char *)s;

    while (*r) {
        size_t sp = space_len(r);

        if (sp > 0) {
            pending = 1;
            r += sp;
        } else if (r[0] == 0xE2 && r[1] == 0x80 && r[2] == 0x8B) {   /* U+200B */
            r += 3;
        } else if (r[0] == 0xEF && r[1] == 0xBB && r[2] == 0xBF) {   /* BOM */
            r += 3;
        } else {
            if (pending && started) {
                *w++ = ' ';
            }
            pending = 0;
            started = 1;
            *w++ = (char)*r++;
        }
    }
    *w = '\0';
}

/* ================================================================== */
/* 4. Numbers and dates                                                */
/* ================================================================== */

/* Excel built-in number format IDs that are dates or times. */
static int builtin_is_date(int id)
{
    return (id >= 14 && id <= 22) || (id >= 27 && id <= 36) ||
           (id >= 45 && id <= 47) || (id >= 50 && id <= 58);
}

/* Does a custom format code (e.g. "dd-mmm-yy hh:mm") show a date or time? */
static int format_is_date(const char *f)
{
    for (; *f; f++) {
        if (*f == '"') {
            f++;
            while (*f && *f != '"') {
                f++;
            }
            if (*f == '\0') {
                break;
            }
            continue;
        }
        if (*f == '\\' || *f == '_' || *f == '*') {
            if (f[1]) {
                f++;
            }
            continue;
        }
        if (*f == '[') {
            const char *e = strchr(f, ']');
            if (e == NULL) {
                break;
            }
            /* [h] [mm] [ss] are elapsed time; [Red] [$-409] are not. */
            int elapsed = (e > f + 1);
            for (const char *c = f + 1; c < e; c++) {
                if (!strchr("hmsHMS", *c)) {
                    elapsed = 0;
                    break;
                }
            }
            if (elapsed) {
                return 1;
            }
            f = e;
            continue;
        }
        if (strncmp(f, "General", 7) == 0 || strncmp(f, "general", 7) == 0) {
            f += 6;
            continue;
        }
        if (strchr("dDmMyYhHsS", *f)) {
            return 1;
        }
    }
    return 0;
}

/* Days since 1970-01-01 -> calendar date (proleptic Gregorian). */
static void civil_from_days(long z, long *y, int *m, int *d)
{
    z += 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    long doe = z - era * 146097;
    long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long yy = yoe + era * 400;
    long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long mp = (5 * doy + 2) / 153;

    *d = (int)(doy - (153 * mp + 2) / 5 + 1);
    *m = (int)(mp < 10 ? mp + 3 : mp - 9);
    *y = yy + (*m <= 2);
}

/* Excel serial number -> "YYYY-MM-DD", "YYYY-MM-DD HH:MM[:SS]" or "HH:MM". */
static void format_date(double serial, int date1904, char *out, size_t cap)
{
    long days = (long)serial;
    long secs = (long)((serial - (double)days) * 86400.0 + 0.5);
    int hh, mm, ss;
    long y;
    int mo, d;

    if (secs >= 86400) {
        days++;
        secs -= 86400;
    }
    hh = (int)(secs / 3600);
    mm = (int)((secs % 3600) / 60);
    ss = (int)(secs % 60);

    if (days == 0 && secs > 0) {                      /* time of day only */
        if (ss == 0) {
            snprintf(out, cap, "%02d:%02d", hh, mm);
        } else {
            snprintf(out, cap, "%02d:%02d:%02d", hh, mm, ss);
        }
        return;
    }
    if (!date1904 && days < 60) {
        days++;      /* Excel's 1900 system counts a 29 Feb 1900 that never existed */
    }
    civil_from_days(days - (date1904 ? 24107L : 25569L), &y, &mo, &d);

    if (secs == 0) {
        snprintf(out, cap, "%04ld-%02d-%02d", y, mo, d);
    } else if (ss == 0) {
        snprintf(out, cap, "%04ld-%02d-%02d %02d:%02d", y, mo, d, hh, mm);
    } else {
        snprintf(out, cap, "%04ld-%02d-%02d %02d:%02d:%02d", y, mo, d, hh, mm, ss);
    }
}

/* Plain numbers: up to 15 significant digits, like Excel's General format. */
static void format_number(const char *v, char *out, size_t cap)
{
    char *end = NULL;
    double d = strtod(v, &end);

    if (end == v) {
        snprintf(out, cap, "%s", v);
        return;
    }
    snprintf(out, cap, "%.15g", d);
}

/* ================================================================== */
/* 5. Workbook model and parsers                                       */
/* ================================================================== */

typedef struct {
    const char *name;
    int hidden;
    XlsxRow *rows;
    int nrows;
    int cap;
} Sheet;

struct XlsxBook {
    ArenaBlock *arena;
    Sheet *sheets;
    int nsheets;
    const char **shared;
    size_t nshared;
    unsigned char *xf_date;     /* per cell style: 1 if its format is a date */
    size_t nxf;
    int date1904;
};

typedef struct {
    int r1, c1, r2, c2;         /* 0-based columns, 1-based rows */
} MergeRange;

static const char *book_dup(XlsxBook *b, const char *s, size_t n)
{
    return arena_dup(&b->arena, s, n);
}

/* "AB12" -> col 27 (0-based), row 12. Returns 0 on success. */
static int parse_ref(const char *ref, int *col, int *row)
{
    long c = 0;
    const char *p = ref;

    if (!((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z'))) {
        return -1;
    }
    while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
        c = c * 26 + ((*p & 0x1F));
        if (c > MAX_COLUMNS + 1) {
            return -1;
        }
        p++;
    }
    if (*p < '0' || *p > '9') {
        return -1;
    }
    *col = (int)c - 1;
    *row = atoi(p);
    return 0;
}

/* ---- relationships: id -> target ---------------------------------- */

typedef struct {
    char id[64];
    char type[192];
    char target[512];
} Rel;

static int parse_rels(const char *xml, size_t n, Rel **out, size_t *count)
{
    XmlScan x = {xml, xml + n};
    XmlToken t;
    Rel *rels = NULL;
    size_t cnt = 0, cap = 0;

    while (xml_next(&x, &t)) {
        if (t.kind == XML_START && name_is(&t, "Relationship")) {
            if (cnt == cap) {
                size_t ncap = cap ? cap * 2 : 16;
                Rel *p = realloc(rels, ncap * sizeof(*rels));
                if (p == NULL) {
                    free(rels);
                    return -1;
                }
                rels = p;
                cap = ncap;
            }
            Rel *r = &rels[cnt++];
            r->id[0] = r->type[0] = r->target[0] = '\0';
            xml_attr(&t, "Id", r->id, sizeof(r->id));
            xml_attr(&t, "Type", r->type, sizeof(r->type));
            xml_attr(&t, "Target", r->target, sizeof(r->target));
        }
    }
    *out = rels;
    *count = cnt;
    return 0;
}

/* ---- shared strings ----------------------------------------------- */

static int parse_shared(XlsxBook *b, const char *xml, size_t n)
{
    XmlScan x = {xml, xml + n};
    XmlToken t;
    Buf buf = {0};
    int in_si = 0, in_t = 0, in_rph = 0;
    size_t cap = 0;
    int rc = -1;

    while (xml_next(&x, &t)) {
        const char *value = NULL;

        if (t.kind == XML_START) {
            if (name_is(&t, "si")) {
                buf_clear(&buf);
                in_t = in_rph = 0;
                in_si = !t.self_closing;
                if (t.self_closing) {
                    value = EMPTY;
                }
            } else if (in_si && name_is(&t, "rPh")) {      /* phonetic guide */
                in_rph = !t.self_closing;
            } else if (in_si && !in_rph && name_is(&t, "t")) {
                in_t = !t.self_closing;
            }
        } else if (t.kind == XML_END) {
            if (name_is(&t, "t")) {
                in_t = 0;
            } else if (name_is(&t, "rPh")) {
                in_rph = 0;
            } else if (name_is(&t, "si") && in_si) {
                in_si = 0;
                if (buf.data != NULL) {
                    clean_text(buf.data);
                }
                value = book_dup(b, buf_str(&buf), strlen(buf_str(&buf)));
                if (value == NULL) {
                    goto done;
                }
            }
        } else if (t.kind == XML_TEXT && in_t) {
            if (buf_append_xml(&buf, t.text, t.text_len, t.cdata) != 0) {
                goto done;
            }
        }

        if (value != NULL) {
            if (b->nshared == cap) {
                size_t ncap = cap ? cap * 2 : 1024;
                const char **p = realloc((void *)b->shared, ncap * sizeof(*p));
                if (p == NULL) {
                    goto done;
                }
                b->shared = p;
                cap = ncap;
            }
            b->shared[b->nshared++] = value;
        }
    }
    rc = 0;
done:
    buf_free(&buf);
    return rc;
}

/* ---- styles: which cell formats are dates ------------------------- */

typedef struct {
    int id;
    int is_date;
} CustomFmt;

static int custom_is_date(const CustomFmt *f, size_t n, int id)
{
    for (size_t i = 0; i < n; i++) {
        if (f[i].id == id) {
            return f[i].is_date;
        }
    }
    return 0;
}

static int parse_styles(XlsxBook *b, const char *xml, size_t n)
{
    XmlScan x = {xml, xml + n};
    XmlToken t;
    CustomFmt *fmts = NULL;
    size_t nfmt = 0, fcap = 0, xcap = 0;
    int in_xfs = 0;
    char attr[512];
    int rc = -1;

    while (xml_next(&x, &t)) {
        if (t.kind == XML_START && name_is(&t, "numFmt")) {
            if (nfmt == fcap) {
                size_t ncap = fcap ? fcap * 2 : 16;
                CustomFmt *p = realloc(fmts, ncap * sizeof(*p));
                if (p == NULL) {
                    goto done;
                }
                fmts = p;
                fcap = ncap;
            }
            fmts[nfmt].id = xml_attr(&t, "numFmtId", attr, sizeof(attr))
                                ? atoi(attr) : -1;
            fmts[nfmt].is_date = xml_attr(&t, "formatCode", attr, sizeof(attr))
                                     ? format_is_date(attr) : 0;
            nfmt++;
        } else if (t.kind == XML_START && name_is(&t, "cellXfs")) {
            in_xfs = !t.self_closing;
        } else if (t.kind == XML_END && name_is(&t, "cellXfs")) {
            in_xfs = 0;
        } else if (t.kind == XML_START && in_xfs && name_is(&t, "xf")) {
            int id = xml_attr(&t, "numFmtId", attr, sizeof(attr)) ? atoi(attr) : 0;
            int is_date = builtin_is_date(id) || custom_is_date(fmts, nfmt, id);

            if (b->nxf == xcap) {
                size_t ncap = xcap ? xcap * 2 : 64;
                unsigned char *p = realloc(b->xf_date, ncap);
                if (p == NULL) {
                    goto done;
                }
                b->xf_date = p;
                xcap = ncap;
            }
            b->xf_date[b->nxf++] = (unsigned char)is_date;
        }
    }
    rc = 0;
done:
    free(fmts);
    return rc;
}

/* ---- worksheet ----------------------------------------------------- */

static int sheet_push_row(Sheet *sh, int row, const char **cells, int ncols)
{
    if (sh->nrows == sh->cap) {
        int ncap = sh->cap ? sh->cap * 2 : 64;
        XlsxRow *p = realloc(sh->rows, (size_t)ncap * sizeof(*p));
        if (p == NULL) {
            return -1;
        }
        sh->rows = p;
        sh->cap = ncap;
    }
    const char **copy = malloc((size_t)ncols * sizeof(*copy));
    if (copy == NULL) {
        return -1;
    }
    memcpy((void *)copy, (const void *)cells, (size_t)ncols * sizeof(*copy));
    sh->rows[sh->nrows].row = row;
    sh->rows[sh->nrows].ncols = ncols;
    sh->rows[sh->nrows].cells = copy;
    sh->nrows++;
    return 0;
}

static int cmp_rows(const void *a, const void *b)
{
    const XlsxRow *ra = a, *rb = b;

    return (ra->row > rb->row) - (ra->row < rb->row);
}

/* Set row->cells[col], widening the row with empty cells if needed. */
static int row_set(XlsxRow *row, int col, const char *value)
{
    if (col >= row->ncols) {
        const char **p = realloc((void *)row->cells,
                                 (size_t)(col + 1) * sizeof(*p));
        if (p == NULL) {
            return -1;
        }
        for (int i = row->ncols; i < col; i++) {
            p[i] = EMPTY;
        }
        row->cells = p;
        row->ncols = col + 1;
    }
    ((const char **)row->cells)[col] = value;
    return 0;
}

/* Merged ranges spanning several rows: repeat the top-left value downwards. */
static int apply_merges(Sheet *sh, const MergeRange *m, size_t nm)
{
    for (size_t k = 0; k < nm; k++) {
        if (m[k].r2 <= m[k].r1) {
            continue;                      /* horizontal-only: leave alone */
        }
        int lo = 0, hi = sh->nrows;        /* first row with number >= r1 */
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sh->rows[mid].row < m[k].r1) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= sh->nrows || sh->rows[lo].row != m[k].r1) {
            continue;
        }
        const char *value = xlsx_cell(&sh->rows[lo], m[k].c1);
        if (value[0] == '\0') {
            continue;
        }
        for (int i = lo + 1; i < sh->nrows && sh->rows[i].row <= m[k].r2; i++) {
            if (xlsx_cell(&sh->rows[i], m[k].c1)[0] == '\0') {
                if (row_set(&sh->rows[i], m[k].c1, value) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int parse_sheet(XlsxBook *b, Sheet *sh, const char *xml, size_t n)
{
    XmlScan x = {xml, xml + n};
    XmlToken t;
    Buf vbuf = {0}, tbuf = {0};
    const char **cv = NULL;
    int cn = 0, ccap = 0;
    MergeRange *merges = NULL;
    size_t nmerge = 0, mcap = 0;
    int cur_row = 0, cur_col = -1, col = 0;
    int in_row = 0, in_c = 0, in_v = 0, in_is = 0, in_t = 0, in_rph = 0;
    char ctype[16] = "";
    int cstyle = -1;
    char attr[512];
    int rc = -1;

    while (xml_next(&x, &t)) {
        if (t.kind == XML_START) {
            if (name_is(&t, "row")) {
                cur_row = xml_attr(&t, "r", attr, sizeof(attr)) ? atoi(attr)
                                                                : cur_row + 1;
                cn = 0;
                cur_col = -1;
                in_row = !t.self_closing;
            } else if (in_row && name_is(&t, "c")) {
                int rr;
                if (xml_attr(&t, "r", attr, sizeof(attr)) &&
                    parse_ref(attr, &col, &rr) == 0) {
                    /* col set */
                } else {
                    col = cur_col + 1;
                }
                cur_col = col;
                if (!xml_attr(&t, "t", ctype, sizeof(ctype))) {
                    ctype[0] = '\0';
                }
                cstyle = xml_attr(&t, "s", attr, sizeof(attr)) ? atoi(attr) : -1;
                buf_clear(&vbuf);
                buf_clear(&tbuf);
                in_c = !t.self_closing;
                in_v = in_is = in_t = in_rph = 0;
            } else if (in_c && name_is(&t, "v")) {
                in_v = !t.self_closing;
            } else if (in_c && name_is(&t, "is")) {
                in_is = !t.self_closing;
            } else if (in_c && in_is && name_is(&t, "rPh")) {
                in_rph = !t.self_closing;
            } else if (in_c && in_is && !in_rph && name_is(&t, "t")) {
                in_t = !t.self_closing;
            } else if (name_is(&t, "mergeCell")) {
                int r1, c1, r2, c2;
                char *colon;
                if (xml_attr(&t, "ref", attr, sizeof(attr)) &&
                    (colon = strchr(attr, ':')) != NULL) {
                    *colon = '\0';
                    if (parse_ref(attr, &c1, &r1) == 0 &&
                        parse_ref(colon + 1, &c2, &r2) == 0) {
                        if (nmerge == mcap) {
                            size_t ncap = mcap ? mcap * 2 : 16;
                            MergeRange *p = realloc(merges, ncap * sizeof(*p));
                            if (p == NULL) {
                                goto done;
                            }
                            merges = p;
                            mcap = ncap;
                        }
                        merges[nmerge].r1 = r1;
                        merges[nmerge].c1 = c1;
                        merges[nmerge].r2 = r2;
                        merges[nmerge].c2 = c2;
                        nmerge++;
                    }
                }
            }
        } else if (t.kind == XML_TEXT) {
            if (in_v) {
                if (buf_append(&vbuf, t.text, t.text_len) != 0) {
                    goto done;
                }
            } else if (in_t) {
                if (buf_append_xml(&tbuf, t.text, t.text_len, t.cdata) != 0) {
                    goto done;
                }
            }
        } else if (t.kind == XML_END) {
            if (name_is(&t, "v")) {
                in_v = 0;
            } else if (name_is(&t, "t")) {
                in_t = 0;
            } else if (name_is(&t, "rPh")) {
                in_rph = 0;
            } else if (name_is(&t, "is")) {
                in_is = 0;
            } else if (name_is(&t, "c") && in_c) {
                const char *val = EMPTY;
                char tmp[64];

                in_c = 0;
                if (strcmp(ctype, "s") == 0) {
                    long idx = vbuf.len ? strtol(buf_str(&vbuf), NULL, 10) : -1;
                    if (idx >= 0 && (size_t)idx < b->nshared) {
                        val = b->shared[idx];
                    }
                } else if (strcmp(ctype, "inlineStr") == 0 ||
                           strcmp(ctype, "str") == 0 ||
                           strcmp(ctype, "e") == 0 ||
                           strcmp(ctype, "d") == 0) {
                    if (strcmp(ctype, "inlineStr") != 0) {
                        buf_clear(&tbuf);
                        if (buf_append_xml(&tbuf, buf_str(&vbuf), vbuf.len, 0) != 0) {
                            goto done;
                        }
                    }
                    if (tbuf.data != NULL) {
                        clean_text(tbuf.data);
                        val = book_dup(b, tbuf.data, strlen(tbuf.data));
                        if (val == NULL) {
                            goto done;
                        }
                    }
                } else if (strcmp(ctype, "b") == 0) {
                    val = (vbuf.len > 0 && vbuf.data[0] == '1') ? "TRUE" : "FALSE";
                } else if (vbuf.len > 0) {
                    char *end = NULL;
                    double d = strtod(buf_str(&vbuf), &end);
                    if (end != buf_str(&vbuf) && cstyle >= 0 &&
                        (size_t)cstyle < b->nxf && b->xf_date[cstyle] &&
                        d >= 0.0 && d < 2958466.0) {
                        format_date(d, b->date1904, tmp, sizeof(tmp));
                    } else {
                        format_number(buf_str(&vbuf), tmp, sizeof(tmp));
                    }
                    val = book_dup(b, tmp, strlen(tmp));
                    if (val == NULL) {
                        goto done;
                    }
                }

                if (val[0] != '\0' && col >= 0 && col < MAX_COLUMNS) {
                    if (col >= ccap) {
                        int ncap = ccap ? ccap * 2 : 32;
                        while (ncap <= col) {
                            ncap *= 2;
                        }
                        const char **p = realloc((void *)cv,
                                                 (size_t)ncap * sizeof(*p));
                        if (p == NULL) {
                            goto done;
                        }
                        cv = p;
                        ccap = ncap;
                    }
                    for (int i = cn; i < col; i++) {
                        cv[i] = EMPTY;
                    }
                    cv[col] = val;
                    if (col >= cn) {
                        cn = col + 1;
                    }
                }
            } else if (name_is(&t, "row") && in_row) {
                in_row = 0;
                if (cn > 0) {
                    if (sheet_push_row(sh, cur_row, cv, cn) != 0) {
                        goto done;
                    }
                }
            }
        }
    }

    /* Rows are normally ascending; make sure, then apply merged cells. */
    for (int i = 1; i < sh->nrows; i++) {
        if (sh->rows[i].row < sh->rows[i - 1].row) {
            qsort(sh->rows, (size_t)sh->nrows, sizeof(*sh->rows), cmp_rows);
            break;
        }
    }
    if (apply_merges(sh, merges, nmerge) != 0) {
        goto done;
    }
    rc = 0;
done:
    buf_free(&vbuf);
    buf_free(&tbuf);
    free((void *)cv);
    free(merges);
    return rc;
}

/* ---- workbook ------------------------------------------------------- */

typedef struct {
    const char *name;
    int hidden;
    char rid[64];
} SheetRef;

static int ends_with(const char *s, const char *suffix)
{
    size_t a = strlen(s), b = strlen(suffix);

    return a >= b && strcmp(s + a - b, suffix) == 0;
}

static void read_file_into(const char *path, unsigned char **data, size_t *size,
                           char *error, size_t cap)
{
    FILE *f = fopen(path, "rb");
    long len;

    *data = NULL;
    if (f == NULL) {
        set_error(error, cap, "cannot open the file");
        return;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        set_error(error, cap, "cannot determine the file size");
        return;
    }
    if ((size_t)len > MAX_FILE_BYTES) {
        fclose(f);
        set_error(error, cap, "file is too large");
        return;
    }
    *data = malloc((size_t)len + 1);
    if (*data == NULL) {
        fclose(f);
        set_error(error, cap, "out of memory");
        return;
    }
    if (fread(*data, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(*data);
        *data = NULL;
        set_error(error, cap, "cannot read the file");
        return;
    }
    fclose(f);
    *size = (size_t)len;
}

XlsxBook *xlsx_open(const char *path, char *error, size_t error_capacity)
{
    unsigned char *file = NULL;
    size_t file_size = 0;
    Zip zip;
    XlsxBook *b = NULL;
    SheetRef *refs = NULL;
    int nrefs = 0, refs_cap = 0;
    Rel *rels = NULL;
    size_t nrels = 0;
    char *part = NULL;
    size_t part_len = 0;
    int missing = 0;
    char attr[512];
    char msg[256];

    memset(&zip, 0, sizeof(zip));
    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }
    if (path == NULL || path[0] == '\0') {
        set_error(error, error_capacity, "No workbook path given.");
        return NULL;
    }

    read_file_into(path, &file, &file_size, error, error_capacity);
    if (file == NULL) {
        return NULL;
    }
    if (zip_index(&zip, file, file_size, error, error_capacity) != 0) {
        goto fail;
    }

    b = calloc(1, sizeof(*b));
    if (b == NULL) {
        set_error(error, error_capacity, "out of memory");
        goto fail;
    }

    /* workbook.xml: tab names, hidden flag, date system */
    msg[0] = '\0';
    part = zip_read(&zip, "xl/workbook.xml", &part_len, &missing, msg, sizeof(msg));
    if (part == NULL) {
        set_error(error, error_capacity, "not an Excel workbook: %s",
                  missing ? "xl/workbook.xml is missing" : msg);
        goto fail;
    }
    {
        XmlScan x = {part, part + part_len};
        XmlToken t;
        while (xml_next(&x, &t)) {
            if (t.kind != XML_START) {
                continue;
            }
            if (name_is(&t, "workbookPr")) {
                if (xml_attr(&t, "date1904", attr, sizeof(attr)) &&
                    (strcmp(attr, "1") == 0 || strcmp(attr, "true") == 0)) {
                    b->date1904 = 1;
                }
            } else if (name_is(&t, "sheet")) {
                if (nrefs == refs_cap) {
                    int ncap = refs_cap ? refs_cap * 2 : 8;
                    SheetRef *p = realloc(refs, (size_t)ncap * sizeof(*p));
                    if (p == NULL) {
                        set_error(error, error_capacity, "out of memory");
                        goto fail;
                    }
                    refs = p;
                    refs_cap = ncap;
                }
                SheetRef *r = &refs[nrefs];
                if (!xml_attr(&t, "name", attr, sizeof(attr))) {
                    attr[0] = '\0';
                }
                clean_text(attr);
                r->name = book_dup(b, attr, strlen(attr));
                if (r->name == NULL) {
                    set_error(error, error_capacity, "out of memory");
                    goto fail;
                }
                r->hidden = xml_attr(&t, "state", attr, sizeof(attr)) &&
                            (strcmp(attr, "hidden") == 0 ||
                             strcmp(attr, "veryHidden") == 0);
                if (!xml_attr(&t, ":id", r->rid, sizeof(r->rid))) {
                    r->rid[0] = '\0';
                }
                nrefs++;
            }
        }
    }
    free(part);
    part = NULL;

    /* relationships: rId -> worksheet file */
    part = zip_read(&zip, "xl/_rels/workbook.xml.rels", &part_len, &missing,
                    msg, sizeof(msg));
    if (part == NULL) {
        set_error(error, error_capacity, "workbook relationships unreadable: %s",
                  missing ? "xl/_rels/workbook.xml.rels is missing" : msg);
        goto fail;
    }
    if (parse_rels(part, part_len, &rels, &nrels) != 0) {
        set_error(error, error_capacity, "out of memory");
        goto fail;
    }
    free(part);
    part = NULL;

    /* shared strings (optional: a workbook with no text has none) */
    part = zip_read(&zip, "xl/sharedStrings.xml", &part_len, &missing,
                    msg, sizeof(msg));
    if (part == NULL && !missing) {
        set_error(error, error_capacity, "sharedStrings.xml: %s", msg);
        goto fail;
    }
    if (part != NULL) {
        if (parse_shared(b, part, part_len) != 0) {
            set_error(error, error_capacity, "out of memory reading strings");
            goto fail;
        }
        free(part);
        part = NULL;
    }

    /* styles (optional): which formats are dates */
    part = zip_read(&zip, "xl/styles.xml", &part_len, &missing, msg, sizeof(msg));
    if (part == NULL && !missing) {
        set_error(error, error_capacity, "styles.xml: %s", msg);
        goto fail;
    }
    if (part != NULL) {
        if (parse_styles(b, part, part_len) != 0) {
            set_error(error, error_capacity, "out of memory reading styles");
            goto fail;
        }
        free(part);
        part = NULL;
    }

    /* worksheets, in workbook order */
    b->sheets = calloc((size_t)(nrefs ? nrefs : 1), sizeof(*b->sheets));
    if (b->sheets == NULL) {
        set_error(error, error_capacity, "out of memory");
        goto fail;
    }
    for (int i = 0; i < nrefs; i++) {
        const Rel *rel = NULL;
        char name[640];

        for (size_t k = 0; k < nrels; k++) {
            if (strcmp(rels[k].id, refs[i].rid) == 0) {
                rel = &rels[k];
                break;
            }
        }
        if (rel == NULL) {
            continue;
        }
        if (!ends_with(rel->type, "/worksheet")) {
            continue;                       /* chart sheets, dialogs, ... */
        }
        if (rel->target[0] == '/') {
            snprintf(name, sizeof(name), "%s", rel->target + 1);
        } else {
            snprintf(name, sizeof(name), "xl/%s", rel->target);
        }

        part = zip_read(&zip, name, &part_len, &missing, msg, sizeof(msg));
        if (part == NULL) {
            set_error(error, error_capacity, "tab '%s': %s", refs[i].name,
                      missing ? "worksheet file is missing" : msg);
            goto fail;
        }
        Sheet *sh = &b->sheets[b->nsheets++];
        sh->name = refs[i].name;
        sh->hidden = refs[i].hidden;
        if (parse_sheet(b, sh, part, part_len) != 0) {
            set_error(error, error_capacity, "tab '%s': out of memory",
                      refs[i].name);
            goto fail;
        }
        free(part);
        part = NULL;
    }

    free(rels);
    free(refs);
    free(zip.entries);
    free(file);
    return b;

fail:
    free(part);
    free(rels);
    free(refs);
    free(zip.entries);
    free(file);
    xlsx_close(b);
    return NULL;
}

/* ================================================================== */
/* 6. Public API                                                       */
/* ================================================================== */

void xlsx_close(XlsxBook *book)
{
    if (book == NULL) {
        return;
    }
    for (int s = 0; s < book->nsheets; s++) {
        for (int r = 0; r < book->sheets[s].nrows; r++) {
            free((void *)book->sheets[s].rows[r].cells);
        }
        free(book->sheets[s].rows);
    }
    free(book->sheets);
    free((void *)book->shared);
    free(book->xf_date);
    while (book->arena != NULL) {
        ArenaBlock *next = book->arena->next;
        free(book->arena);
        book->arena = next;
    }
    free(book);
}

/* 1 if sheet is a valid tab index of book. */
static int valid_sheet(const XlsxBook *book, int sheet)
{
    return book != NULL && sheet >= 0 && sheet < book->nsheets;
}

int xlsx_sheet_count(const XlsxBook *book)
{
    return book == NULL ? 0 : book->nsheets;
}

const char *xlsx_sheet_name(const XlsxBook *book, int sheet)
{
    return valid_sheet(book, sheet) ? book->sheets[sheet].name : NULL;
}

int xlsx_sheet_hidden(const XlsxBook *book, int sheet)
{
    return valid_sheet(book, sheet) ? book->sheets[sheet].hidden : 0;
}

int xlsx_row_count(const XlsxBook *book, int sheet)
{
    return valid_sheet(book, sheet) ? book->sheets[sheet].nrows : 0;
}

const XlsxRow *xlsx_row(const XlsxBook *book, int sheet, int index)
{
    if (!valid_sheet(book, sheet) || index < 0 ||
        index >= book->sheets[sheet].nrows) {
        return NULL;
    }
    return &book->sheets[sheet].rows[index];
}

const char *xlsx_cell(const XlsxRow *row, int col)
{
    if (row == NULL || col < 0 || col >= row->ncols) {
        return EMPTY;
    }
    return row->cells[col];
}
