#ifndef ARIA_H
#define ARIA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct AriaModel AriaModel;

const char *aria_last_error(void);

AriaModel *aria_model_init(const char *bundle_path);
void aria_model_destroy(AriaModel *model);

int aria_complete(
    AriaModel *model,
    const char *messages_json,
    const char *options_json,
    const char *tools_json,
    char *out,
    size_t out_len
);

int aria_complete_stream(
    AriaModel *model,
    const char *messages_json,
    const char *options_json,
    const char *tools_json,
    char *out,
    size_t out_len,
    void (*callback)(const char *chunk, void *user_data),
    void *user_data
);

int aria_embed(
    AriaModel *model,
    const char *input_json,
    char *out,
    size_t out_len
);

int aria_transcribe(
    AriaModel *model,
    const unsigned char *pcm,
    size_t pcm_len,
    const char *options_json,
    char *out,
    size_t out_len
);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_H */
