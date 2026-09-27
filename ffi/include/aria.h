#ifndef ARIA_H
#define ARIA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct AriaModel AriaModel;

const char *aria_last_error(void);

/* track: "encoder" | "decoder" */
AriaModel *aria_model_init(const char *checkpoint_path, const char *track);
void aria_model_destroy(AriaModel *model);

const char *aria_model_cache_dir(const char *model);
int aria_is_local_path(const char *ref_);

/* System One JSON in → JSON out (null-terminated). Returns 0 on success. */
int aria_systemone(
    AriaModel *model,
    const char *request_json,
    char *out,
    size_t out_len
);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_H */
