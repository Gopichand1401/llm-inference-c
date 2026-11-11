/*
 * llm_rag.c - one chat program, three ways to answer:
 *
 *   llm_rag --retrieve-only [options] ["question"]
 *       Answers come straight from the .xlsx / .csv files (no model is
 *       loaded): the cell that was asked for, its source, and an offer of
 *       one more cell. See include/rag_answer.h.
 *
 *   llm_rag <model.gguf> --rag [options] ["question"]
 *       Retrieval-augmented generation: the best rows are found (BM25),
 *       put into the prompt, and the chat model writes the answer.
 *       The sources are printed under it. Needs a chat model (TinyLlama).
 *
 *   llm_rag <model file> [options] ["prompt"]
 *       The model alone, no files: multi-turn chat for GGUF chat models,
 *       plain text completion for llama2.c models (stories42M ...).
 *
 * All three stay open for more input until /exit. In the two modes that
 * use files, the --docs folders are checked before every question: a new,
 * edited or deleted file is picked up at once.
 *
 * Options:
 *   --docs PATH        folder or .xlsx/.csv file; repeatable  (default: data)
 *   --header-row N     header row for every table             (default: detect)
 *   --include-hidden   also read hidden tabs
 *   -n N               maximum new tokens      (default 160 with --rag, else 256)
 *   --sampler S        greedy | temperature | top-k
 *                                              (default greedy with --rag, else top-k)
 *   --temperature T    (default 0.7)           --top-k K   (default 40)
 *   --seed N           random seed (default: from the clock)
 *   --system TEXT      system message (replaces the RAG instruction with --rag)
 *   --ctx N            context to allocate (default: the model's)
 *   --tokenizer F      tokenizer.bin for llama2.c models
 *   -k N               --rag: rows given to the model          (default 3)
 *   --min-score S      --rag: best score needed to answer      (default 1.0)
 *   --keep R           --rag: keep rows scoring >= R x best    (default 0.35)
 *   --compact          --rag: give the model only the key columns and the
 *                      columns the question asks about (shorter prompt,
 *                      faster answers; all columns if none is named)
 *   --show-context     --rag: print the prompt given to the model
 *   --show-rows        --retrieve-only: print the rows --rag would use
 *   -v                 status lines, warnings, scores, timings
 *
 * Commands while it runs (any case): /exit  /help  /files  /reload
 *   /reset (forget the conversation)  /context (--rag: show the prompt)
 *
 *   --chat-format F    zephyr | llama2 | mistral | none
 *                                              (default zephyr for GGUF models)
 *
 * This program only reads the command line and prints: the work is done by
 * the library functions of include/pocket.h, the same ones the Python
 * package (python/pocketllm) calls.
 */
#define _POSIX_C_SOURCE 200809L

#include "pocket.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_INPUT 4096
#define MAX_DOCS 32
#define MAX_K 16

typedef enum { MODE_MODEL, MODE_RETRIEVE, MODE_RAG } Mode;

typedef struct {
    Mode mode;
    const char *docs[MAX_DOCS];
    int n_docs;
    PocketRag *rag;
    PocketModel *model;
    PocketGenOptions gen;
    PocketRagOptions ragopt;
    int show_context;
    int show_rows;
    int verbose;
} App;

static void usage(const char *p)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --retrieve-only [\"question\"]\n"
            "        answers straight from the .xlsx/.csv files in data/ (no model)\n"
            "  %s <model.gguf> --rag [\"question\"]\n"
            "        the rows are found, then the chat model writes the answer\n"
            "  %s <model file> [\"prompt\"]\n"
            "        the model alone (chat for GGUF, completion for llama2.c)\n"
            "Options: [--docs PATH]... [--header-row N] [--include-hidden] [-n N]\n"
            "         [--sampler greedy|temperature|top-k] [--temperature T] [--top-k K]\n"
            "         [--seed N] [--system TEXT] [--ctx N] [--tokenizer F]\n"
            "         [--chat-format zephyr|llama2|mistral|none] [-k N] [--min-score S]\n"
            "         [--keep R] [--compact] [--show-context] [--show-rows] [-v]\n"
            "All modes keep running; type /exit to leave, /help for commands.\n",
            p, p, p);
}

static void print_piece(const char *piece, void *user)
{
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Documents                                                           */
/* ------------------------------------------------------------------ */

static void print_load_summary(const App *a, double seconds)
{
    int ok = 0, tables = 0;
    for (int i = 0; i < pocket_rag_file_count(a->rag); i++) {
        const int readable = pocket_rag_file_readable(a->rag, i);
        ok += readable;
        tables += pocket_rag_file_tables(a->rag, i);
        if (!readable) {    /* an unreadable file is always worth a line */
            fprintf(stderr, "[%s could not be read and is skipped]\n",
                    pocket_rag_file_name(a->rag, i));
        }
    }
    if (a->verbose) {
        for (int i = 0; i < pocket_rag_warning_count(a->rag); i++) {
            fprintf(stderr, "[warning: %s]\n", pocket_rag_warning(a->rag, i));
        }
        fprintf(stderr, "[documents: %d file(s), %d table(s), %d rows/notes, "
                        "%d search terms", ok, tables, pocket_rag_row_count(a->rag),
                pocket_rag_term_count(a->rag));
        if (seconds >= 0.0) {
            fprintf(stderr, " | loaded in %.2f s", seconds);
        }
        fprintf(stderr, "]\n");
    }
    if (ok == 0) {
        fprintf(stderr, "[no readable .xlsx or .csv file yet: copy files into "
                        "the documents folder, they are picked up before the "
                        "next question]\n");
    }
}

/* Reload when a file was added, edited or removed since the last load. */
static void refresh_documents(App *a)
{
    int changed = pocket_rag_refresh(a->rag);

    if (changed < 0) {
        fprintf(stderr, "[%s; using what was loaded]\n", pocket_rag_error(a->rag));
    } else if (changed > 0) {
        fprintf(stderr, "[documents changed: %s; reloading]\n",
                pocket_rag_change(a->rag));
        print_load_summary(a, -1.0);
    }
}

static void list_files(const App *a)
{
    for (int i = 0; i < pocket_rag_file_count(a->rag); i++) {
        if (pocket_rag_file_readable(a->rag, i)) {
            fprintf(stderr, "  %s: %d table(s), %d rows/notes\n",
                    pocket_rag_file_name(a->rag, i), pocket_rag_file_tables(a->rag, i),
                    pocket_rag_file_rows(a->rag, i));
        } else {
            fprintf(stderr, "  %s: not readable\n", pocket_rag_file_name(a->rag, i));
        }
    }
    for (int i = 0; i < pocket_rag_warning_count(a->rag); i++) {
        fprintf(stderr, "  note: %s\n", pocket_rag_warning(a->rag, i));
    }
}

/* ------------------------------------------------------------------ */
/* One line of input, whatever the mode                                */
/* ------------------------------------------------------------------ */

static void print_stats(const App *a)
{
    PocketStats st;

    if (!a->verbose) {
        return;
    }
    pocket_model_stats(a->model, &st);
    fprintf(stderr,
            "[prompt %d tokens %.2f s (%.1f tok/s) | reply %d tokens %.2f s "
            "(%.1f tok/s)%s | context %d/%d]\n",
            st.prompt_tokens, st.prompt_seconds,
            st.prompt_seconds > 0 ? st.prompt_tokens / st.prompt_seconds : 0.0,
            st.generated_tokens, st.generate_seconds,
            st.generate_seconds > 0 ? st.generated_tokens / st.generate_seconds : 0.0,
            st.stopped_by_eos ? "" : " | stopped at token limit", st.position,
            st.context);
}

/* --retrieve-only --show-rows: the rows --rag would use, without a model. */
static void show_rows(App *a, const char *question)
{
    int n = pocket_rag_search(a->rag, question, 4 * MAX_K, 1, &a->ragopt);

    if (n <= 0) {
        printf("I could not find this in the files.\n");
        return;
    }
    printf("Best matching rows:\n");
    for (int i = 0; i < n; i++) {
        printf("  [%d] %s", i + 1, pocket_rag_hit_source(a->rag, i));
        printf("  score %.2f\n", pocket_rag_hit_score(a->rag, i));
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%s\n", pocket_rag_hit_text(a->rag, i));
    }
}

static int rag_turn(App *a, const char *question)
{
    const char *reply = pocket_rag_ask_model(a->rag, a->model, question, &a->gen,
                                             &a->ragopt, print_piece, NULL);
    if (reply == NULL) {
        fprintf(stderr, "\n[%s]\n", pocket_rag_error(a->rag));
        return 0;
    }
    printf("\n");
    const char *sources = pocket_rag_last_source(a->rag);
    if (sources[0] != '\0') {
        int rows = 1;
        for (const char *p = sources; *p; p++) {
            rows += *p == '\n';
        }
        if (a->show_context) {
            fprintf(stderr, "----- prompt given to the model -----%s\n"
                            "-------------------------------------\n",
                    pocket_rag_last_prompt(a->rag));
        }
        printf("\nSources:\n");
        for (int i = 0; i < rows; i++) {
            printf("  [%d] %s", i + 1, pocket_rag_hit_source(a->rag, i));
            if (a->verbose) {
                printf("  score %.2f", pocket_rag_hit_score(a->rag, i));
            }
            printf("\n");
        }
        fflush(stdout);
        if (a->verbose) {
            PocketStats st;
            pocket_model_stats(a->model, &st);
            fprintf(stderr,
                    "[%d row(s) in prompt | prompt %d tokens %.2f s (%.1f tok/s) | "
                    "reply %d tokens %.2f s (%.1f tok/s)%s | context %d/%d]\n",
                    rows, st.prompt_tokens, st.prompt_seconds,
                    st.prompt_seconds > 0 ? st.prompt_tokens / st.prompt_seconds : 0.0,
                    st.generated_tokens, st.generate_seconds,
                    st.generate_seconds > 0 ? st.generated_tokens / st.generate_seconds : 0.0,
                    st.stopped_by_eos ? "" : " | stopped at token limit",
                    st.position, st.context);
        }
    }
    return 1;
}

static int answer(App *a, const char *line)
{
    int ok = 1;

    if (a->mode != MODE_MODEL) {
        refresh_documents(a);
    }
    switch (a->mode) {
    case MODE_RETRIEVE:
        if (a->show_rows) {
            show_rows(a, line);
        } else {
            const char *reply = pocket_rag_ask(a->rag, line);
            if (reply == NULL) {
                fprintf(stderr, "[%s]\n", pocket_rag_error(a->rag));
                ok = 0;
            } else {
                fputs(reply, stdout);
            }
        }
        break;
    case MODE_RAG:
        ok = rag_turn(a, line);
        break;
    default:
        if (pocket_model_is_chat(a->model)) {
            ok = pocket_model_chat(a->model, line, &a->gen, print_piece, NULL) != NULL;
        } else {
            /* llama2.c models: print the prompt, then continue it */
            fputs(line, stdout);
            fflush(stdout);
            ok = pocket_model_generate(a->model, line, &a->gen, print_piece, NULL) != NULL;
        }
        printf("\n");
        if (!ok) {
            fprintf(stderr, "[%s]\n", pocket_model_error(a->model));
        } else {
            print_stats(a);
        }
        break;
    }
    fflush(stdout);
    return ok;
}

static int read_line(char *buffer, size_t capacity)
{
    if (fgets(buffer, (int)capacity, stdin) == NULL) {
        return 0;
    }
    buffer[strcspn(buffer, "\r\n")] = '\0';
    return 1;
}

static void print_help(const App *a)
{
    fprintf(stderr, "  /exit     leave\n"
                    "  /reset    forget the conversation so far\n");
    if (a->mode != MODE_MODEL) {
        fprintf(stderr, "  /files    list the loaded files\n"
                        "  /reload   read every file again\n");
    }
    if (a->mode == MODE_RAG) {
        fprintf(stderr, "  /context  show or hide the prompt given to the model\n");
    }
    if (a->mode == MODE_RETRIEVE) {
        fprintf(stderr, "  Ask about anything in the files. After an answer you can "
                        "reply yes or no to\n  the offer, name another column "
                        "(\"room\", \"mail id\"), or ask something new.\n");
    }
}

/* Returns 1 to keep going, 0 to leave. */
static int run_command(App *a, const char *typed)
{
    char cmd[32];
    size_t n = 0;

    for (; typed[n] != '\0' && typed[n] != ' ' && n < sizeof(cmd) - 1; n++) {
        cmd[n] = (char)tolower((unsigned char)typed[n]);
    }
    cmd[n] = '\0';

    if (strcmp(cmd, "/exit") == 0 || strcmp(cmd, "/quit") == 0) {
        return 0;
    }
    if (strcmp(cmd, "/help") == 0) {
        print_help(a);
    } else if (strcmp(cmd, "/reset") == 0) {
        if (a->model != NULL) {
            pocket_model_reset(a->model);
        }
        if (a->rag != NULL) {
            pocket_rag_reset(a->rag);
        }
        fprintf(stderr, "[conversation reset]\n");
    } else if (strcmp(cmd, "/files") == 0 && a->mode != MODE_MODEL) {
        refresh_documents(a);
        list_files(a);
    } else if (strcmp(cmd, "/reload") == 0 && a->mode != MODE_MODEL) {
        if (pocket_rag_reload(a->rag) != 0) {
            fprintf(stderr, "[%s]\n", pocket_rag_error(a->rag));
        } else {
            print_load_summary(a, -1.0);
            fprintf(stderr, "[files read again]\n");
        }
    } else if (strcmp(cmd, "/context") == 0 && a->mode == MODE_RAG) {
        a->show_context = !a->show_context;
        fprintf(stderr, "[showing the prompt: %s]\n", a->show_context ? "on" : "off");
    } else {
        fprintf(stderr, "[unknown command; /help lists the commands]\n");
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    App a;
    const char *question = NULL, *model_path = NULL, *tokenizer_path = NULL;
    const char *system = NULL, *chat_format = NULL;
    unsigned long seed = 0;
    int seed_given = 0, status = 1, retrieve = 0, rag = 0, max_context = 0;
    int header_row = 0, include_hidden = 0;
    char error[1024];

    memset(&a, 0, sizeof(a));

    int first = 1;
    if (argc >= 2 && argv[1][0] != '-') {
        model_path = argv[1];
        first = 2;
    }
    for (int i = first; i < argc; i++) {
        const char *s = argv[i];
        const int has_value = (i + 1 < argc);

        if (strcmp(s, "--retrieve-only") == 0) {
            retrieve = 1;
        } else if (strcmp(s, "--rag") == 0) {
            rag = 1;
        } else if (strcmp(s, "--docs") == 0 && has_value) {
            if (a.n_docs == MAX_DOCS) {
                fprintf(stderr, "At most %d --docs.\n", MAX_DOCS);
                return 1;
            }
            a.docs[a.n_docs++] = argv[++i];
        } else if (strcmp(s, "-k") == 0 && has_value) {
            a.ragopt.k = atoi(argv[++i]);
            if (a.ragopt.k < 1 || a.ragopt.k > MAX_K) {
                fprintf(stderr, "-k must be between 1 and %d.\n", MAX_K);
                return 1;
            }
        } else if (strcmp(s, "-n") == 0 && has_value) {
            a.gen.max_tokens = atoi(argv[++i]);
            if (a.gen.max_tokens <= 0) {
                fprintf(stderr, "-n must be positive.\n");
                return 1;
            }
        } else if (strcmp(s, "--min-score") == 0 && has_value) {
            a.ragopt.min_score = atof(argv[++i]);
        } else if (strcmp(s, "--keep") == 0 && has_value) {
            a.ragopt.keep = atof(argv[++i]);
            if (a.ragopt.keep < 0.0 || a.ragopt.keep > 1.0) {
                fprintf(stderr, "--keep must be between 0 and 1.\n");
                return 1;
            }
        } else if (strcmp(s, "--header-row") == 0 && has_value) {
            header_row = atoi(argv[++i]);
        } else if (strcmp(s, "--include-hidden") == 0) {
            include_hidden = 1;
        } else if (strcmp(s, "--show-context") == 0) {
            a.show_context = 1;
        } else if (strcmp(s, "--compact") == 0) {
            a.ragopt.compact = 1;
        } else if (strcmp(s, "--show-rows") == 0) {
            a.show_rows = 1;
        } else if (strcmp(s, "--sampler") == 0 && has_value) {
            a.gen.sampler = argv[++i];
            if (strcmp(a.gen.sampler, "greedy") != 0 &&
                strcmp(a.gen.sampler, "temperature") != 0 &&
                strcmp(a.gen.sampler, "top-k") != 0) {
                fprintf(stderr, "Unknown sampler '%s'.\n", a.gen.sampler);
                return 1;
            }
        } else if (strcmp(s, "--temperature") == 0 && has_value) {
            a.gen.temperature = (float)atof(argv[++i]);
            if (a.gen.temperature <= 0.0f) {
                fprintf(stderr, "--temperature must be positive.\n");
                return 1;
            }
        } else if (strcmp(s, "--top-k") == 0 && has_value) {
            a.gen.top_k = atoi(argv[++i]);
            if (a.gen.top_k <= 0) {
                fprintf(stderr, "--top-k must be positive.\n");
                return 1;
            }
        } else if (strcmp(s, "--seed") == 0 && has_value) {
            errno = 0;
            seed = strtoul(argv[++i], NULL, 10);
            seed_given = (errno == 0);
        } else if (strcmp(s, "--system") == 0 && has_value) {
            system = argv[++i];
        } else if (strcmp(s, "--chat-format") == 0 && has_value) {
            chat_format = argv[++i];
        } else if (strcmp(s, "--tokenizer") == 0 && has_value) {
            tokenizer_path = argv[++i];
        } else if (strcmp(s, "--ctx") == 0 && has_value) {
            max_context = atoi(argv[++i]);
            if (max_context <= 0) {
                fprintf(stderr, "--ctx must be positive.\n");
                return 1;
            }
        } else if (strcmp(s, "-v") == 0) {
            a.verbose = 1;
        } else if (s[0] == '-' && s[1] != '\0') {
            usage(argv[0]);
            return 1;
        } else {
            question = s;
        }
    }

    /* ---- Which mode? ---- */
    if (retrieve && rag) {
        fprintf(stderr, "Use either --retrieve-only (no model) or --rag (rows + "
                        "model), not both.\n");
        return 1;
    }
    if (rag && model_path == NULL) {
        fprintf(stderr, "--rag needs a chat model:\n"
                        "  %s models/gguf/tinyllama.gguf --rag\n", argv[0]);
        return 1;
    }
    if (!retrieve && model_path == NULL) {
        usage(argv[0]);
        return 1;
    }
    a.mode = retrieve ? MODE_RETRIEVE : rag ? MODE_RAG : MODE_MODEL;
    if (retrieve && model_path != NULL && a.verbose) {
        fprintf(stderr, "[--retrieve-only answers from the files; %s is not "
                        "loaded]\n", model_path);
    }
    if (a.n_docs == 0) {
        a.docs[a.n_docs++] = "data";
    }
    a.ragopt.system = a.mode == MODE_RAG ? system : NULL;
    if (!seed_given) {
        seed = (unsigned long)time(NULL);
    }

    /* ---- Documents ---- */
    if (a.mode != MODE_MODEL) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        a.rag = pocket_rag_open(a.docs, a.n_docs, header_row, include_hidden,
                                error, sizeof(error));
        if (a.rag == NULL) {
            fprintf(stderr, "[cannot read documents: %s]\n", error);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        pocket_rag_set_auto_refresh(a.rag, 0);   /* done here, to report it */
        print_load_summary(&a, (double)(t1.tv_sec - t0.tv_sec) +
                                   (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9);
    }

    /* ---- Model ---- */
    if (a.mode != MODE_RETRIEVE) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        a.model = pocket_model_open(model_path, tokenizer_path, max_context, error,
                                    sizeof(error));
        if (a.model == NULL) {
            fprintf(stderr, "%s\n", error);
            goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        pocket_model_seed(a.model, seed);
        if (a.mode == MODE_RAG && !pocket_model_is_chat(a.model)) {
            fprintf(stderr, "--rag needs a chat model that can follow instructions "
                            "(TinyLlama GGUF).\n%s is a llama2.c story model: it "
                            "can only continue text.\nUse --retrieve-only for "
                            "answers from the files, or run this model without "
                            "--rag.\n", model_path);
            goto cleanup;
        }
        if (chat_format != NULL &&
            pocket_model_set_chat_format(a.model, chat_format) != 0) {
            fprintf(stderr, "%s\n", pocket_model_error(a.model));
            goto cleanup;
        }
        if (a.mode == MODE_MODEL && system != NULL) {
            pocket_model_set_system(a.model, system);
        }
        if (a.verbose) {
            fprintf(stderr, "[model: %s | context %d | loaded in %.2f s | seed %lu]\n",
                    pocket_model_description(a.model), pocket_model_context(a.model),
                    (double)(t1.tv_sec - t0.tv_sec) +
                        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9, seed);
        }
    }

    /* ---- A question on the command line ---- */
    if (question != NULL) {
        status = answer(&a, question) ? 0 : 1;
        if (!isatty(STDIN_FILENO)) {
            goto cleanup;        /* scripted: answer and leave */
        }
    }

    /* ---- Chat ---- */
    {
        char line[MAX_INPUT];

        if (a.mode == MODE_RETRIEVE) {
            fprintf(stderr, "Ask about the files (answers come straight from "
                            "them). /help for commands, /exit to leave.\n");
        } else if (a.mode == MODE_RAG) {
            fprintf(stderr, "Ask about the files (the model answers from the rows "
                            "found). /help for commands, /exit to leave.\n");
        } else if (pocket_model_is_chat(a.model)) {
            fprintf(stderr, "Chat with the model (no files are used). /reset "
                            "starts over, /exit leaves.\n");
        } else {
            fprintf(stderr, "Type the start of a text and the model continues it "
                            "(no files are used). /exit leaves.\n");
        }
        for (;;) {
            fprintf(stderr, "\n> ");
            fflush(stderr);
            if (!read_line(line, sizeof(line))) {
                break;
            }
            const char *text = line;
            while (*text == ' ' || *text == '\t') {
                text++;
            }
            if (*text == '\0') {
                continue;
            }
            if (*text == '/') {
                if (!run_command(&a, text)) {
                    break;
                }
                continue;
            }
            if (!answer(&a, text)) {
                fprintf(stderr, "[that did not work; /reset starts a new "
                                "conversation]\n");
            }
        }
        status = 0;
    }

cleanup:
    pocket_rag_close(a.rag);
    pocket_model_close(a.model);
    return status;
}
