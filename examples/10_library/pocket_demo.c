/*
 * pocket_demo.c - a C program that uses the project as a library.
 * Build:  make lib
 *         gcc -Iinclude pocket_demo.c -Lbuild -lpocketllm -o pocket_demo
 * Run:    LD_LIBRARY_PATH=build ./pocket_demo models/gguf/tinyllama.gguf
 */
#include "pocket.h"

#include <stdio.h>

/* Called with every new piece of text while the model is writing. */
static void show(const char *piece, void *user)
{
    (void)user;
    fputs(piece, stdout);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    char error[1024];

    /* ---- 1. Answers from the Excel / CSV files (no model) ---- */
    const char *docs[] = {"data"};
    PocketRag *rag = pocket_rag_open(docs, 1, 0, 0, error, sizeof(error));
    if (rag == NULL) {
        printf("Cannot read the files: %s\n", error);
        return 1;
    }
    printf("Loaded %d files, %d rows.\n\n", pocket_rag_file_count(rag),
           pocket_rag_row_count(rag));

    pocket_rag_ask(rag, "Who is the instructor for ML?");
    printf("Q: Who is the instructor for ML?\n");
    printf("A: %s\n", pocket_rag_last_answer(rag));
    printf("   from %s\n\n", pocket_rag_last_source(rag));

    pocket_rag_ask(rag, "yes");                 /* accepts the offered follow-up */
    printf("Q: yes\nA: %s\n\n", pocket_rag_last_answer(rag));

    if (argc < 2) {
        printf("(give a model file as argument to see generation)\n");
        pocket_rag_close(rag);
        return 0;
    }

    /* ---- 2. A model, with a choice of sampler ---- */
    PocketModel *model = pocket_model_open(argv[1], NULL, 0, error, sizeof(error));
    if (model == NULL) {
        printf("%s\n", error);
        pocket_rag_close(rag);
        return 1;
    }
    printf("Model: %s\n\n", pocket_model_description(model));

    PocketGenOptions greedy = {"greedy", 0.7f, 40, 40};
    PocketGenOptions topk = {"top-k", 0.8f, 40, 40};
    pocket_model_seed(model, 42);

    printf("[greedy] My name is");
    if (pocket_model_generate(model, "My name is", &greedy, show, NULL) == NULL) {
        printf("\nerror: %s", pocket_model_error(model));
    }
    printf("\n[top-k]  My name is");
    if (pocket_model_generate(model, "My name is", &topk, show, NULL) == NULL) {
        printf("\nerror: %s", pocket_model_error(model));
    }
    printf("\n\n");

    /* ---- 3. Files + model: the model writes the answer from the rows ---- */
    if (pocket_model_is_chat(model)) {
        printf("Q: Who teaches DS3010?\nA: ");
        if (pocket_rag_ask_model(rag, model, "Who teaches DS3010?", NULL, NULL,
                                 show, NULL) == NULL) {
            printf("error: %s", pocket_rag_error(rag));
        }
        printf("\n   rows used:\n%s\n", pocket_rag_last_source(rag));
    }

    PocketStats st;
    pocket_model_stats(model, &st);
    printf("\n%d tokens in %.2f s (%.1f tokens/s)\n", st.generated_tokens,
           st.generate_seconds,
           st.generate_seconds > 0 ? st.generated_tokens / st.generate_seconds : 0.0);

    pocket_model_close(model);
    pocket_rag_close(rag);
    return 0;
}
