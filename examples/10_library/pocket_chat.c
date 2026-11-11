
#include "pocket.h"
#include <stdio.h>
#include <string.h>

#define MAX_INPUT 4096

// Print generated tokens as they arrive
static void print_token(const char *piece, void *user)
{
    (void)user;
    printf("%s", piece);
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    char error[1024];
    char line[MAX_INPUT];

    // Get model path from terminal
    if (argc != 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    // Load LLM
    printf("Loading model...\n");

    PocketModel *model = pocket_model_open(
        argv[1], NULL, 0, error, sizeof(error)
    );

    if (model == NULL) {
        printf("Model loading failed: %s\n", error);
        return 1;
    }

    printf("Model loaded: %s\n",
           pocket_model_description(model));

    if (!pocket_model_is_chat(model)) {
        printf("Error: Model does not support chat.\n");
        pocket_model_close(model);
        return 1;
    }

    // Load documents
    printf("\nLoading documents...\n");

    const char *docs[] = {"data"};

    PocketRag *rag = pocket_rag_open(
        docs, 1, 0, 0, error, sizeof(error)
    );

    if (rag == NULL) {
        printf("Document loading failed: %s\n", error);
        pocket_model_close(model);
        return 1;
    }

    printf("Loaded %d files, %d rows.\n",
           pocket_rag_file_count(rag),
           pocket_rag_row_count(rag));

    // LLM generation settings
    PocketGenOptions options = {
    "greedy",
    0.0f,
    1,
    50
};

    // RAG retrieval settings
    
    PocketRagOptions rag_options = {
    3,
    1.0,
    0.35,
    "You are a university course information assistant.\n"
    "Answer questions using ONLY the spreadsheet context.\n"
    "Treat every course row as an independent record.\n"
    "Never combine information from different course rows.\n"
    "Never assume or invent information.\n"
    "Preserve course codes, names, credits, departments, "
    "availability, instructors, emails, slots and rooms exactly.\n"
    "Do not infer the institution or department from outside knowledge.\n"
    "Answer the exact question in one or two natural sentences.\n"
    "If the requested information is present, answer directly.\n"
    "If it is missing, say that the information is unavailable.\n",
    1   /* compact: only the key and asked-for columns (faster answers) */
    };


    printf("\n========================================\n");
    printf("          PocketLLM RAG + LLM Chat\n");
    printf("========================================\n");
    printf("Type /exit to quit.\n");

    // Interactive chat
    while (1) {

        printf("\nYou: ");
        fflush(stdout);

        if (fgets(line, sizeof(line), stdin) == NULL)
            break;

        line[strcspn(line, "\r\n")] = '\0';

        if (strcmp(line, "/exit") == 0)
            break;

        if (line[0] == '\0')
            continue;

        // Debug: retrieve relevant rows
        int hits =pocket_rag_search(rag, line, 1, 1, &rag_options);

        printf("\n========== RETRIEVED CONTEXT ==========\n");

        if (hits <= 0) {
            printf("No relevant rows found.\n");
        }

        for (int i = 0; i < hits; i++) {

            printf("\nHit %d | Score: %.3f\n",
                   i + 1,
                   pocket_rag_hit_score(rag, i));

            printf("%s\n",
                   pocket_rag_hit_text(rag, i));

            printf("Source: %s\n",
                   pocket_rag_hit_source(rag, i));
        }

        // Generate answer using LLM
        printf("\n========== GENERATED ANSWER ==========\n");
        printf("\nAssistant: ");
        fflush(stdout);

        const char *reply = pocket_rag_ask_model(
            rag,
            model,
            line,
            &options,
            &rag_options,
            print_token,
            NULL
        );

        if (reply == NULL) {
            printf("\nError: %s\n",
                   pocket_rag_error(rag));
            continue;
        }

        printf("\n");

        // Display actual prompt sent to LLM
        const char *prompt = pocket_rag_last_prompt(rag);

        if (prompt != NULL) {
            printf("\n========== LLM PROMPT ==========\n");
            printf("%s\n", prompt);
        }

        // Display sources
        const char *source = pocket_rag_last_source(rag);

        if (source != NULL && source[0] != '\0') {
            printf("\n========== SOURCES ==========\n");
            printf("%s\n", source);
        }

        // Display performance statistics
        PocketStats stats;
        pocket_model_stats(model, &stats);

        printf("\n========== PERFORMANCE ==========\n");
        printf("Prompt tokens: %d\n", stats.prompt_tokens);
        printf("Generated tokens: %d\n", stats.generated_tokens);
        printf("Prompt time: %.2f seconds\n", stats.prompt_seconds);
        printf("Generation time: %.2f seconds\n",
               stats.generate_seconds);

        if (stats.generate_seconds > 0) {
            printf("Generation speed: %.2f tokens/sec\n",
                   stats.generated_tokens / stats.generate_seconds);
        }
    }

    // Cleanup
    pocket_rag_close(rag);
    pocket_model_close(model);

    printf("\nChat session ended.\n");

    return 0;
}
