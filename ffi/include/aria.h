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

/* Model-name → cache-directory helpers (SDK auto-download support). */

/* Returns "~/.ariacompute/models/{model}" as a `const char *`. Valid until the
 * next call to a returning FFI function on this thread. NULL on error. */
const char *aria_model_cache_dir(const char *model);

/* Returns 1 if `ref_` looks like a local bundle path (contains a path
 * separator or already exists on disk), 0 if it is a model name, -1 on error. */
int aria_is_local_path(const char *ref_);

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
