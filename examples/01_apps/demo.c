
#include "llm.h"

#include <stdio.h>

static void verify_weights(const Transformer *model)
{
    const TransformerWeights *w = &model->weights;
    const RunState *s = &model->state;

    printf("\n=== Weight Pointer Verification ===\n");

    printf("Token embeddings: %p\n",
           (void *)w->token_embedding_table);
    printf("Attention RMSNorm: %p\n", (void *)w->rms_att_weight);
    printf("WQ: %p\n", (void *)w->wq);
    printf("WK: %p\n", (void *)w->wk);
    printf("WV: %p\n", (void *)w->wv);
    printf("WO: %p\n", (void *)w->wo);
    printf("FFN RMSNorm: %p\n", (void *)w->rms_ffn_weight);
    printf("W1: %p\n", (void *)w->w1);
    printf("W2: %p\n", (void *)w->w2);
    printf("W3: %p\n", (void *)w->w3);
    printf("Final RMSNorm: %p\n", (void *)w->rms_final_weight);
    printf("Classifier: %p\n", (void *)w->wcls);

    printf("\n=== Runtime Buffer Verification ===\n");

    printf("x: %p\n", (void *)s->x);
    printf("xb: %p\n", (void *)s->xb);
    printf("xb2: %p\n", (void *)s->xb2);
    printf("q: %p\n", (void *)s->q);
    printf("k: %p\n", (void *)s->k);
    printf("v: %p\n", (void *)s->v);
    printf("attention: %p\n", (void *)s->att);
    printf("logits: %p\n", (void *)s->logits);
    printf("key cache: %p\n", (void *)s->key_cache);
    printf("value cache: %p\n", (void *)s->value_cache);

    printf("\n=== Verification Result ===\n");

    if (w->token_embedding_table &&
        w->rms_att_weight && w->rms_ffn_weight &&
        w->wq && w->wk && w->wv && w->wo &&
        w->w1 && w->w2 && w->w3 &&
        w->rms_final_weight && w->wcls &&
        s->x && s->xb && s->xb2 &&
        s->hb && s->hb2 && s->q && s->k && s->v &&
        s->att && s->logits &&
        s->key_cache && s->value_cache) {
        printf("PASS: All expected pointers are non-NULL.\n");
    } else {
        printf("FAIL: At least one pointer is NULL.\n");
    }
}

static void verify_tensors(const Transformer *model)
{
    printf("\n=== Generic Tensor Verification ===\n");

    printf("Number of tensors: %zu\n", model->num_tensors);

    for (size_t i = 0; i < model->num_tensors; i++) {
        const Tensor *t = &model->tensors[i];

        printf("\nTensor: %s\n", t->name);
        printf("Address: %p\n", t->data);

        printf("Shape: [");

        for (size_t j = 0; j < t->ndim; j++) {
            printf("%zu", t->shape[j]);

            if (j + 1 < t->ndim) {
                printf(", ");
            }
        }

        printf("]\n");

        printf("Data type: %s\n",
               t->dtype == TENSOR_F32 ? "FP32" : "Other");

        printf("Storage: %s\n",
               t->storage == TENSOR_STORAGE_MAPPED
                   ? "Memory mapped" : "Other");

        printf("Size: %zu bytes\n", t->byte_size);

        
    }

    /* Verify tensor lookup for every registered tensor. */
    printf("\n=== Tensor Lookup Verification ===\n");

    int lookup_passed = 1;

    for (size_t i = 0; i < model->num_tensors; i++) {

        Tensor *found = tensor_find(
            model->tensors,
            model->num_tensors,
            model->tensors[i].name
        );

        if (found == NULL || found != &model->tensors[i]) {
            printf("FAIL: %s\n", model->tensors[i].name);
            lookup_passed = 0;
        } else {
            printf("PASS: %s\n", found->name);
        }
    }

    /* Verify invalid tensor lookup. */
    Tensor *missing = tensor_find(
        model->tensors,
        model->num_tensors,
        "non_existing_tensor"
    );

    if (missing == NULL) {
        printf("PASS: Invalid tensor name correctly returns NULL\n");
    } else {
        printf("FAIL: Invalid tensor lookup\n");
        lookup_passed = 0;
    }

    printf("\nOverall tensor lookup: %s\n",
        lookup_passed ? "PASS" : "FAIL");   

    printf("\nEmbedding and classifier share memory: %s\n",
           model->tensors[0].data == model->tensors[11].data
               ? "YES" : "NO");
    
}

int main(int argc, char **argv)
{
    char error[256];
    Transformer *model;
    const Config *config;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    model = llm_load(argv[1], error, sizeof(error));

    if (model == NULL) {
        fprintf(stderr, "Error: %s\n", error);
        return 1;
    }

    config = llm_model_config(model);

    printf("Model loaded successfully!\n");
    printf("File size: %zu bytes\n", llm_model_file_size(model));
    printf("Embedding dimension: %d\n", config->dim);
    printf("Hidden dimension: %d\n", config->hidden_dim);
    printf("Transformer layers: %d\n", config->n_layers);
    printf("Attention heads: %d\n", config->n_heads);
    printf("Key/value heads: %d\n", config->n_kv_heads);
    printf("Vocabulary size: %d\n", config->vocab_size);
    printf("Maximum sequence length: %d\n", config->seq_len);

    verify_weights(model);
    verify_tensors(model);
    llm_free_model(model);
    return 0;
}
