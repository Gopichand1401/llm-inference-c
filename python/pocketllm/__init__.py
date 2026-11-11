"""pocketllm - a LLaMA-style inference engine written in C, with RAG over
.xlsx / .csv files, usable from Python.

    import pocketllm

    model = pocketllm.Model("models/gguf/tinyllama.gguf")
    print(model.generate("My name is", sampler="top-k", temperature=0.7))
    print(model.chat("What is machine learning?", sampler="greedy"))

    rag = pocketllm.Rag("data")
    answer = rag.ask("Who is the instructor for ML?")
    print(answer.text, "|", answer.source)
    print(rag.ask("Who teaches DS3010?", model=model).text)

All the work is done by the C library (include/pocket.h); this module only
calls it through ctypes.
"""
import codecs
import ctypes
import glob
import os
import sys

__all__ = ["Model", "Rag", "Answer", "Row", "PocketError", "version"]

SAMPLERS = ("greedy", "temperature", "top-k")
CHAT_FORMATS = ("zephyr", "llama2", "mistral", "none")


class PocketError(RuntimeError):
    """An error reported by the C library."""


# ---------------------------------------------------------------- loading

def _find_library():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.environ.get("POCKETLLM_LIB"):
        candidates.append(os.environ["POCKETLLM_LIB"])
    candidates += sorted(glob.glob(os.path.join(here, "_pocket*.so")))
    candidates += sorted(glob.glob(os.path.join(here, "_pocket*.dylib")))
    # running from the project folder without installing: make lib
    candidates.append(os.path.join(here, "..", "..", "build", "libpocketllm.so"))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise ImportError(
        "pocketllm: the compiled library was not found. Install the package "
        "with 'pip install .' in the project folder, or run 'make lib' there "
        "and set POCKETLLM_LIB=build/libpocketllm.so")


class _GenOptions(ctypes.Structure):
    _fields_ = [("sampler", ctypes.c_char_p), ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int), ("max_tokens", ctypes.c_int)]


class _Stats(ctypes.Structure):
    _fields_ = [("prompt_tokens", ctypes.c_int), ("generated_tokens", ctypes.c_int),
                ("stopped_by_eos", ctypes.c_int), ("prompt_seconds", ctypes.c_double),
                ("generate_seconds", ctypes.c_double), ("position", ctypes.c_int),
                ("context", ctypes.c_int)]


class _RagOptions(ctypes.Structure):
    _fields_ = [("k", ctypes.c_int), ("min_score", ctypes.c_double),
                ("keep", ctypes.c_double), ("system", ctypes.c_char_p),
                ("compact", ctypes.c_int)]


_TokenFn = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def _load():
    lib = ctypes.CDLL(_find_library())
    P, S, I, D = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double
    sig = {
        "pocket_version": (S, []),
        "pocket_model_open": (P, [S, S, I, S, ctypes.c_size_t]),
        "pocket_model_close": (None, [P]),
        "pocket_model_description": (S, [P]),
        "pocket_model_context": (I, [P]),
        "pocket_model_vocab_size": (I, [P]),
        "pocket_model_is_chat": (I, [P]),
        "pocket_model_seed": (None, [P, ctypes.c_ulong]),
        "pocket_model_set_chat_format": (I, [P, S]),
        "pocket_model_chat_format": (S, [P]),
        "pocket_model_set_system": (I, [P, S]),
        "pocket_model_reset": (None, [P]),
        "pocket_model_generate": (S, [P, S, ctypes.POINTER(_GenOptions), _TokenFn, P]),
        "pocket_model_chat": (S, [P, S, ctypes.POINTER(_GenOptions), _TokenFn, P]),
        "pocket_model_stats": (None, [P, ctypes.POINTER(_Stats)]),
        "pocket_model_error": (S, [P]),
        "pocket_rag_open": (P, [ctypes.POINTER(S), I, I, I, S, ctypes.c_size_t]),
        "pocket_rag_close": (None, [P]),
        "pocket_rag_refresh": (I, [P]),
        "pocket_rag_reload": (I, [P]),
        "pocket_rag_set_auto_refresh": (None, [P, I]),
        "pocket_rag_change": (S, [P]),
        "pocket_rag_ask": (S, [P, S]),
        "pocket_rag_last_answer": (S, [P]),
        "pocket_rag_last_source": (S, [P]),
        "pocket_rag_last_offer": (S, [P]),
        "pocket_rag_last_found": (I, [P]),
        "pocket_rag_reset": (None, [P]),
        "pocket_rag_ask_model": (S, [P, P, S, ctypes.POINTER(_GenOptions),
                                     ctypes.POINTER(_RagOptions), _TokenFn, P]),
        "pocket_rag_last_prompt": (S, [P]),
        "pocket_rag_search": (I, [P, S, I, I, ctypes.POINTER(_RagOptions)]),
        "pocket_rag_hit_text": (S, [P, I]),
        "pocket_rag_hit_source": (S, [P, I]),
        "pocket_rag_hit_score": (D, [P, I]),
        "pocket_rag_row_count": (I, [P]),
        "pocket_rag_term_count": (I, [P]),
        "pocket_rag_file_count": (I, [P]),
        "pocket_rag_file_name": (S, [P, I]),
        "pocket_rag_file_readable": (I, [P, I]),
        "pocket_rag_file_rows": (I, [P, I]),
        "pocket_rag_file_tables": (I, [P, I]),
        "pocket_rag_warning_count": (I, [P]),
        "pocket_rag_warning": (S, [P, I]),
        "pocket_rag_error": (S, [P]),
    }
    for name, (restype, argtypes) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = restype, argtypes
    return lib


_lib = _load()


def _text(raw):
    return raw.decode("utf-8", "replace") if raw is not None else ""


def _bytes(text):
    return os.fspath(text).encode("utf-8") if text is not None else None


def version():
    """Version of the C library, e.g. '0.3.0'."""
    return _text(_lib.pocket_version())


def _gen_options(sampler, temperature, top_k, max_tokens):
    if sampler is not None and sampler not in SAMPLERS:
        raise ValueError("sampler must be one of %s, not %r" % (", ".join(SAMPLERS), sampler))
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return _GenOptions(_bytes(sampler), temperature or 0.0, top_k or 0, max_tokens or 0)


class _Stream:
    """Turns the C callback's byte pieces into text for a Python callable."""

    def __init__(self, stream, on_token):
        self.sinks = []
        if stream:
            self.sinks.append(lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        if on_token is not None:
            self.sinks.append(on_token)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.error = None
        self.callback = _TokenFn(self._piece) if self.sinks else _TokenFn()

    def _piece(self, piece, _user):
        try:
            text = self.decoder.decode(piece)
            if text:
                for sink in self.sinks:
                    sink(text)
        except BaseException as exc:      # never unwind through C
            self.error = self.error or exc

    def finish(self):
        if self.sinks:
            tail = self.decoder.decode(b"", final=True)
            if tail and self.error is None:
                for sink in self.sinks:
                    sink(tail)
        if self.error is not None:
            raise self.error


# ---------------------------------------------------------------- Model

class Model:
    """A model file loaded into the C engine.

    path         a llama2.c .bin (stories15M / 42M / 110M) or a GGUF file of
                 the "llama" architecture (TinyLlama ...)
    tokenizer    tokenizer.bin for .bin models (default: next to the model,
                 then the current folder)
    context      positions to allocate (default: the model's, at most 4096)
    chat_format  "zephyr" (TinyLlama, default for GGUF), "llama2", "mistral"
                 or "none"
    system       system message for chat()
    seed         random seed (default: from the clock)
    """

    def __init__(self, path, tokenizer=None, context=0, chat_format=None,
                 system=None, seed=None):
        self._h = None
        error = ctypes.create_string_buffer(1024)
        handle = _lib.pocket_model_open(_bytes(path), _bytes(tokenizer), int(context or 0),
                                        error, len(error))
        if not handle:
            raise PocketError(_text(error.value))
        self._h = handle
        self.path = os.fspath(path)
        if chat_format is not None:
            self.chat_format = chat_format
        if system is not None:
            self.system = system
        if seed is not None:
            self.seed(seed)

    # -- housekeeping
    def close(self):
        """Free the model. The object cannot be used afterwards."""
        if self._h:
            _lib.pocket_model_close(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "<pocketllm.Model %s (%s)>" % (os.path.basename(self.path), self.description)

    def _handle(self):
        if not self._h:
            raise PocketError("the model is closed")
        return self._h

    # -- information
    @property
    def description(self):
        return _text(_lib.pocket_model_description(self._handle()))

    @property
    def context(self):
        """Positions allocated for the KV cache."""
        return _lib.pocket_model_context(self._handle())

    @property
    def vocab_size(self):
        return _lib.pocket_model_vocab_size(self._handle())

    @property
    def is_chat(self):
        """True for GGUF chat models, False for llama2.c story models."""
        return bool(_lib.pocket_model_is_chat(self._handle()))

    @property
    def stats(self):
        """Token counts and timings of the last call, as a dict."""
        st = _Stats()
        _lib.pocket_model_stats(self._handle(), ctypes.byref(st))
        out = {name: getattr(st, name) for name, _ in _Stats._fields_}
        out["stopped_by_eos"] = bool(out["stopped_by_eos"])
        g, s = out["generated_tokens"], out["generate_seconds"]
        out["tokens_per_second"] = g / s if s > 0 else 0.0
        return out

    # -- settings
    @property
    def chat_format(self):
        return _text(_lib.pocket_model_chat_format(self._handle()))

    @chat_format.setter
    def chat_format(self, name):
        if _lib.pocket_model_set_chat_format(self._handle(), _bytes(name)) != 0:
            raise ValueError(_text(_lib.pocket_model_error(self._h)))

    @property
    def system(self):
        return getattr(self, "_system", None)

    @system.setter
    def system(self, text):
        if _lib.pocket_model_set_system(self._handle(), _bytes(text or "")) != 0:
            raise PocketError(_text(_lib.pocket_model_error(self._h)))
        self._system = text or None

    def seed(self, value):
        """Set the random seed (the same seed gives the same text)."""
        _lib.pocket_model_seed(self._handle(), int(value) & 0xFFFFFFFF)

    def reset(self):
        """Forget the chat so far."""
        _lib.pocket_model_reset(self._handle())

    # -- generation
    def _run(self, fn, text, sampler, temperature, top_k, max_tokens, seed, stream,
             on_token):
        options = _gen_options(sampler, temperature, top_k, max_tokens)
        if seed is not None:
            self.seed(seed)
        sink = _Stream(stream, on_token)
        raw = fn(self._handle(), _bytes(text), ctypes.byref(options), sink.callback, None)
        sink.finish()
        if raw is None:
            raise PocketError(_text(_lib.pocket_model_error(self._h)))
        return _text(raw)

    def generate(self, prompt, sampler="top-k", temperature=0.7, top_k=40,
                 max_tokens=256, seed=None, stream=False, on_token=None):
        """Continue a text. Returns the continuation (without the prompt).

        sampler      "greedy", "temperature" or "top-k"
        stream       True: also print the text while it is generated
        on_token     a function called with each new piece of text
        """
        return self._run(_lib.pocket_model_generate, prompt, sampler, temperature,
                         top_k, max_tokens, seed, stream, on_token)

    def chat(self, message=None, sampler="top-k", temperature=0.7, top_k=40,
             max_tokens=256, seed=None, stream=False, on_token=None):
        """One chat turn; earlier turns are remembered until reset().

        Without a message it starts an interactive chat in the terminal or
        notebook (/reset starts over, /exit leaves).
        """
        if message is None:
            return self._interactive(sampler, temperature, top_k, max_tokens)
        return self._run(_lib.pocket_model_chat, message, sampler, temperature, top_k,
                         max_tokens, seed, stream, on_token)

    def _interactive(self, sampler, temperature, top_k, max_tokens):
        print("Chat with the model. /reset starts over, /exit leaves."
              if self.is_chat else
              "Type the start of a text and the model continues it. /exit leaves.")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("/exit", "/quit"):
                break
            if line.lower() == "/reset":
                self.reset()
                print("[conversation reset]")
                continue
            try:
                if not self.is_chat:
                    sys.stdout.write(line)
                self.chat(line, sampler, temperature, top_k, max_tokens, stream=True)
                print()
            except PocketError as exc:
                print("\n[%s]" % exc)


# ---------------------------------------------------------------- RAG

class Answer:
    """The reply to Rag.ask().

    text     the answer itself ("Swapnil Hingmire")
    source   where it is from ("04_UG_DS.xlsx, tab Core, row 10 (...)"), or ""
    offer    the follow-up question ("Do you want the slot?"), or ""
    found    True if rows of the files were used
    full     everything, as the command-line program prints it
    """

    def __init__(self, text, source, offer, found, full):
        self.text, self.source, self.offer = text, source, offer
        self.found, self.full = found, full

    def __str__(self):
        return self.full

    def __repr__(self):
        return "Answer(text=%r, source=%r, offer=%r, found=%r)" % (
            self.text, self.source, self.offer, self.found)

    def __bool__(self):
        return self.found


class Row:
    """One search hit of Rag.search(): text, source and BM25 score."""

    def __init__(self, text, source, score):
        self.text, self.source, self.score = text, source, score

    def __repr__(self):
        return "Row(source=%r, score=%.2f)" % (self.source, self.score)


class Rag:
    """Question answering over the .xlsx and .csv files of one or more folders.

    rag = pocketllm.Rag("data")                # or Rag("a.xlsx", "more/")
    rag.ask("Who is the instructor for ML?")   # straight from the rows
    rag.ask("yes")                             # replies to the offer work
    rag.ask("Who teaches DS3010?", model=m)    # the model writes the answer

    New, edited and deleted files are picked up before every question
    (auto_refresh=False turns that off; reload() reads everything again).
    """

    def __init__(self, *paths, header_row=0, include_hidden=False, auto_refresh=True):
        self._h = None
        if len(paths) == 1 and isinstance(paths[0], (list, tuple)):
            paths = tuple(paths[0])
        if not paths:
            paths = ("data",)
        self.paths = [os.fspath(p) for p in paths]
        array = (ctypes.c_char_p * len(self.paths))(*[_bytes(p) for p in self.paths])
        error = ctypes.create_string_buffer(1024)
        handle = _lib.pocket_rag_open(array, len(self.paths), int(header_row or 0),
                                      int(bool(include_hidden)), error, len(error))
        if not handle:
            raise PocketError(_text(error.value))
        self._h = handle
        if not auto_refresh:
            _lib.pocket_rag_set_auto_refresh(self._h, 0)

    def close(self):
        if self._h:
            _lib.pocket_rag_close(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return "<pocketllm.Rag %d file(s), %d rows>" % (len(self.files), self.row_count)

    def _handle(self):
        if not self._h:
            raise PocketError("the Rag object is closed")
        return self._h

    def _fail(self):
        raise PocketError(_text(_lib.pocket_rag_error(self._h)))

    # -- what is loaded
    @property
    def files(self):
        """[{'name', 'readable', 'rows', 'tables'}] for every file found."""
        h = self._handle()
        return [{"name": _text(_lib.pocket_rag_file_name(h, i)),
                 "readable": bool(_lib.pocket_rag_file_readable(h, i)),
                 "rows": _lib.pocket_rag_file_rows(h, i),
                 "tables": _lib.pocket_rag_file_tables(h, i)}
                for i in range(_lib.pocket_rag_file_count(h))]

    @property
    def row_count(self):
        return _lib.pocket_rag_row_count(self._handle())

    @property
    def warnings(self):
        h = self._handle()
        return [_text(_lib.pocket_rag_warning(h, i))
                for i in range(_lib.pocket_rag_warning_count(h))]

    def refresh(self):
        """Read the files again if any changed. Returns what changed, or ''."""
        r = _lib.pocket_rag_refresh(self._handle())
        if r < 0:
            self._fail()
        return _text(_lib.pocket_rag_change(self._h)) if r else ""

    def reload(self):
        """Read every file again."""
        if _lib.pocket_rag_reload(self._handle()) != 0:
            self._fail()

    def reset(self):
        """Forget the conversation (the follow-up memory of ask())."""
        _lib.pocket_rag_reset(self._handle())

    # -- answers
    def ask(self, question, model=None, sampler="greedy", temperature=0.7, top_k=40,
            max_tokens=160, seed=None, k=3, system=None, stream=False, on_token=None,
            compact=False):
        """Answer a question from the files. Returns an Answer.

        Without a model the answer is copied straight from the rows, and
        "yes" / "no" / "and its room?" / a number work as follow-ups.
        With model=... the best k rows are given to the chat model, which
        writes the answer; sampler, temperature, top_k, max_tokens, seed
        and system then apply, and stream=True prints while it writes.
        compact=True gives the model only the key columns and the columns
        the question asks about: a shorter prompt and a faster answer.
        """
        h = self._handle()
        if model is None:
            raw = _lib.pocket_rag_ask(h, _bytes(question))
            if raw is None:
                self._fail()
            return Answer(_text(_lib.pocket_rag_last_answer(h)),
                          _text(_lib.pocket_rag_last_source(h)),
                          _text(_lib.pocket_rag_last_offer(h)),
                          bool(_lib.pocket_rag_last_found(h)), _text(raw))
        if not isinstance(model, Model):
            raise TypeError("model must be a pocketllm.Model")
        if not 1 <= int(k) <= 16:
            raise ValueError("k must be between 1 and 16")
        options = _gen_options(sampler, temperature, top_k, max_tokens)
        rag_options = _RagOptions(int(k), 0.0, 0.0, _bytes(system), int(bool(compact)))
        if seed is not None:
            model.seed(seed)
        sink = _Stream(stream, on_token)
        raw = _lib.pocket_rag_ask_model(h, model._handle(), _bytes(question),
                                        ctypes.byref(options), ctypes.byref(rag_options),
                                        sink.callback, None)
        sink.finish()
        if raw is None:
            self._fail()
        text = _text(raw)
        source = _text(_lib.pocket_rag_last_source(h))
        full = text + ("\n\nSources:\n" + source if source else "")
        return Answer(text, source, "", bool(source), full)

    def search(self, question, k=5):
        """The k best rows for a question (BM25), best first, as Row objects."""
        h = self._handle()
        n = _lib.pocket_rag_search(h, _bytes(question), int(k), 0, None)
        if n < 0:
            self._fail()
        return [Row(_text(_lib.pocket_rag_hit_text(h, i)),
                    _text(_lib.pocket_rag_hit_source(h, i)),
                    _lib.pocket_rag_hit_score(h, i)) for i in range(n)]

    def chat(self, model=None, **options):
        """Interactive question answering (/files, /reset, /exit)."""
        print("Ask about the files%s. /files, /reset, /exit."
              % (" (the model writes the answers)" if model is not None else ""))
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            low = line.lower()
            if low in ("/exit", "/quit"):
                break
            if low == "/reset":
                self.reset()
                if model is not None:
                    model.reset()
                print("[conversation reset]")
                continue
            if low == "/files":
                for f in self.files:
                    print("  %s: %s" % (f["name"], "%d rows" % f["rows"]
                                        if f["readable"] else "not readable"))
                continue
            try:
                if model is None:
                    sys.stdout.write(self.ask(line).full)
                else:
                    answer = self.ask(line, model=model, stream=True, **options)
                    print()
                    if answer.source:
                        print("\nSources:\n" + answer.source)
            except PocketError as exc:
                print("[%s]" % exc)
