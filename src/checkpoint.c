#include "checkpoint.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static CheckpointFormat detect_checkpoint_format(
    const char *checkpoint
)
{
    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        return CHECKPOINT_FORMAT_AUTO;
    }

    unsigned char header[9];

    size_t bytes_read = fread(header, 1, sizeof(header), file);

    fclose(file);

    /* GGUF magic signature */
    if (bytes_read >= 4 &&
        memcmp(header, "GGUF", 4) == 0) {
        return CHECKPOINT_FORMAT_GGUF;
    }

    /* SafeTensors: 8-byte header length + JSON object */
    if (bytes_read == 9 && header[8] == '{') {

        uint64_t header_length = 0;

        for (int i = 0; i < 8; i++) {
            header_length |=
                (uint64_t)header[i] << (i * 8);
        }

        if (header_length > 0) {
            return CHECKPOINT_FORMAT_SAFETENSORS;
        }
    }

    /* Existing Stories checkpoint format */
    return CHECKPOINT_FORMAT_STORIES;
}

/* Generic checkpoint format dispatcher. */
int checkpoint_load(
    const char *checkpoint,
    CheckpointFormat format,
    Config *config,
    TransformerWeights *weights,
    int *fd,
    float **data,
    ssize_t *file_size,
    char *error,
    size_t error_capacity
)
{
    switch (format) {

        case CHECKPOINT_FORMAT_STORIES:

            if (!stories_checkpoint_load(
                    checkpoint,
                    config,
                    weights,
                    fd,
                    data,
                    file_size,
                    error,
                    error_capacity
                )) {
                return 0;
            }

        if (!config_validate(config, error, error_capacity)) {
            return 0;
        }

        return 1;

        case CHECKPOINT_FORMAT_GGUF:
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "GGUF format is not supported yet.");
            }
            return 0;

        case CHECKPOINT_FORMAT_SAFETENSORS:
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "SafeTensors format is not supported yet.");
            }
            return 0;

        case CHECKPOINT_FORMAT_AUTO:

            format = detect_checkpoint_format(checkpoint);

            if (format == CHECKPOINT_FORMAT_AUTO) {
                if (error != NULL && error_capacity > 0) {
                    snprintf(error, error_capacity,
                            "Unable to detect checkpoint format.");
                }
                return 0;
            }

            return checkpoint_load(
                checkpoint,
                format,
                config,
                weights,
                fd,
                data,
                file_size,
                error,
                error_capacity
            );
        default:
            if (error != NULL && error_capacity > 0) {
                snprintf(error, error_capacity,
                         "Unknown checkpoint format.");
            }
            return 0;
    }
}