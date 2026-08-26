package com.ariacompute.engine

import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import org.json.JSONObject

/** Kotlin/JVM + Android JNI wrapper over libaria_ffi. */
class AriaEngine(bundlePath: String) : AutoCloseable {
    private var handle: Long = nativeInit(bundlePath)

    init {
        if (handle == 0L) throw IllegalStateException(nativeLastError() ?: "init failed")
    }

    fun complete(messagesJson: String, optionsJson: String = """{"max_tokens":16}""", toolsJson: String = "[]"): String {
        return nativeComplete(handle, messagesJson, optionsJson, toolsJson)
            ?: throw IllegalStateException(nativeLastError() ?: "complete failed")
    }

    fun embed(inputJson: String): String =
        nativeEmbed(handle, inputJson) ?: throw IllegalStateException(nativeLastError() ?: "embed failed")

    fun transcribe(pcm: ByteArray): String =
        nativeTranscribe(handle, pcm) ?: throw IllegalStateException(nativeLastError() ?: "transcribe failed")

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeInit(path: String): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeComplete(handle: Long, messages: String, options: String, tools: String): String?
    private external fun nativeEmbed(handle: Long, input: String): String?
    private external fun nativeTranscribe(handle: Long, pcm: ByteArray): String?
    private external fun nativeLastError(): String?

    companion object {
        init {
            try {
                System.loadLibrary("aria_ffi")
            } catch (_: UnsatisfiedLinkError) {
                // Host tests may System.load(ARIA_FFI_LIB) before constructing.
            }
        }

        private const val DEFAULT_SITE = "https://ariacompute.com"

        private fun ariaHome(): String {
            val override = System.getenv("ARIA_COMPUTE_HOME")
            if (!override.isNullOrEmpty()) return override
            return System.getProperty("user.home") + File.separator + ".ariacompute"
        }

        private fun cacheDir(model: String): String =
            ariaHome() + File.separator + "models" + File.separator + model

        fun parseBundleName(model: String): Pair<String, String> {
            require(model.isNotEmpty() && !model.contains('/') && !model.contains('\\')) {
                "invalid model name: $model"
            }
            val idx = model.lastIndexOf("_q")
            if (idx != -1) {
                val slug = model.substring(0, idx)
                var suffix = model.substring(idx + 2)
                if (suffix.endsWith("_channel") || suffix.endsWith("_group")) {
                    suffix = suffix.substring(0, suffix.lastIndexOf('_'))
                }
                val quant = when (suffix) {
                    "4" -> "int4"
                    "8" -> "int8"
                    "326", "3.26" -> "int326"
                    else -> throw IllegalArgumentException("unknown quant suffix _q$suffix")
                }
                require(slug.isNotEmpty()) { "invalid model name: $model" }
                return slug to quant
            }
            return model to "int4"
        }

        private fun isValidBundle(dir: String): Boolean {
            val weight = File(dir, "weight.bin")
            val config = File(dir, "config.json")
            if (!weight.isFile || !config.isFile) return false
            return try {
                val meta = JSONObject(config.readText())
                meta.optString("format") == "aria-quant-bundle"
            } catch (_: Exception) {
                false
            }
        }

        private fun isLocalRef(ref: String): Boolean =
            ref.contains('/') || ref.contains('\\') || File(ref).exists()

        private const val DEFAULT_SDK = "v1.0"
        private val HUB_REQUIRED = listOf("config.json", "weight.bin")
        private val HUB_OPTIONAL = listOf(
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        )

        private fun preferredPublicHub(site: String): String =
            if (site.lowercase().contains("ariacompute.cn")) "modelscope" else "huggingface"

        private fun hubBearer(token: String): String? {
            val t = token.trim()
            if (t.isEmpty()) return null
            val low = t.lowercase()
            if (low.startsWith("sk-") || low.startsWith("bfvk-")) return null
            return t
        }

        private fun unquoteYaml(v: String): String {
            val t = v.trim()
            if (t.length >= 2 &&
                ((t.startsWith("\"") && t.endsWith("\"")) || (t.startsWith("'") && t.endsWith("'")))
            ) {
                return t.substring(1, t.length - 1)
            }
            return t
        }

        private fun configYmlScalar(key: String): String? {
            val path = File(ariaHome(), "config.yml")
            if (!path.isFile) return null
            return try {
                path.readLines().firstNotNullOfOrNull { line ->
                    if (line.startsWith(" ") || line.startsWith("\t")) return@firstNotNullOfOrNull null
                    val s = line.trim()
                    if (s.isEmpty() || s.startsWith("#") || ':' !in s) return@firstNotNullOfOrNull null
                    val idx = s.indexOf(':')
                    if (s.substring(0, idx).trim() != key) return@firstNotNullOfOrNull null
                    unquoteYaml(s.substring(idx + 1)).ifEmpty { null }
                }
            } catch (_: Exception) {
                null
            }
        }

        private fun resolveHubToken(
            source: String,
            token: String = "",
            hfToken: String = "",
            modelscopeApiToken: String = "",
        ): String? {
            val named = if (source == "modelscope") modelscopeApiToken else hfToken
            val field = if (source == "modelscope") "modelscope_api_token" else "hf_token"
            for (cand in listOf(named, token, configYmlScalar(field) ?: "")) {
                val b = hubBearer(cand)
                if (b != null) return b
            }
            return null
        }

        private fun hubPathNames(model: String): List<String> {
            val names = mutableListOf(model)
            var lower = model.lowercase()
            var core = model
            for (suf in listOf("_channel", "_group")) {
                if (lower.endsWith(suf)) {
                    core = model.dropLast(suf.length)
                    lower = core.lowercase()
                    break
                }
            }
            val stems = mutableListOf(core)
            if (lower.endsWith("_q326")) {
                stems.add(core.dropLast(5) + "q3.26")
            } else if (lower.endsWith("_q3.26")) {
                stems.add(core.dropLast(6) + "q326")
            }
            for (stem in stems) {
                for (share in listOf("", "_channel", "_group")) {
                    val cand = stem + share
                    if (cand !in names) names.add(cand)
                }
            }
            return names
        }

        private fun hubFileUrls(source: String, model: String, file: String): List<String> {
            val urls = mutableListOf<String>()
            for (name in hubPathNames(model)) {
                if (source == "modelscope") {
                    for (repo in listOf("AriaCompute/$name", "AriaCompute/model")) {
                        urls.add("https://www.modelscope.cn/models/$repo/resolve/master/$DEFAULT_SDK/$name/$file")
                        urls.add("https://modelscope.cn/models/$repo/resolve/master/$DEFAULT_SDK/$name/$file")
                    }
                } else {
                    for (repo in listOf("ariacompute/$name", "ariacompute/model")) {
                        urls.add("https://huggingface.co/$repo/resolve/main/$DEFAULT_SDK/$name/$file")
                    }
                }
            }
            return urls
        }

        private fun fetchUrlToFile(url: String, dest: File, token: String?, client: HttpClient) {
            val builder = HttpRequest.newBuilder(URI.create(url)).GET()
            if (!token.isNullOrEmpty()) {
                builder.header("Authorization", "Bearer $token")
            }
            dest.parentFile?.mkdirs()
            val resp = client.send(builder.build(), HttpResponse.BodyHandlers.ofFile(dest.toPath()))
            val code = resp.statusCode()
            if (code == 401 || code == 403) {
                throw RuntimeException("HTTP $code")
            }
            if (code != 200) {
                dest.delete()
                throw RuntimeException("HTTP $code")
            }
        }

        private fun fetchHubFile(
            source: String,
            model: String,
            file: String,
            dest: File,
            token: String?,
            required: Boolean,
            client: HttpClient,
        ) {
            var last: Exception? = null
            for (url in hubFileUrls(source, model, file)) {
                try {
                    fetchUrlToFile(url, dest, token, client)
                    return
                } catch (e: Exception) {
                    last = e
                    val msg = e.message ?: ""
                    if (msg.contains("HTTP 401") || msg.contains("HTTP 403")) {
                        val field = if (source == "modelscope") "modelscope_api_token" else "hf_token"
                        val code = if (msg.contains("401")) 401 else 403
                        throw RuntimeException(
                            "auth failed HTTP $code; set $field via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)"
                        )
                    }
                }
            }
            if (required) {
                throw RuntimeException("$source: missing $file${last?.let { ": $it" } ?: ""}")
            }
        }

        /** Download `model` from the regional public hub into
         * `~/.ariacompute/models/{model}`; skips when a valid bundle is cached.
         * Dashboard is not used. Token is optional; Dashboard sk-/bfvk- keys are ignored. */
        @JvmOverloads
        @JvmStatic
        fun downloadModel(
            model: String,
            token: String = "",
            site: String = DEFAULT_SITE,
            hfToken: String = "",
            modelscopeApiToken: String = "",
        ): String {
            parseBundleName(model)
            val source = preferredPublicHub(site)
            val hubToken = resolveHubToken(source, token, hfToken, modelscopeApiToken)
            val cache = cacheDir(model)
            if (File(cache).exists() && isValidBundle(cache)) return cache

            val client = HttpClient.newHttpClient()
            val staging = cacheDir(".$model.partial")
            val stagingDir = File(staging)
            stagingDir.deleteRecursively()
            stagingDir.mkdirs()
            try {
                for (file in HUB_REQUIRED) {
                    fetchHubFile(source, model, file, File(stagingDir, file), hubToken, true, client)
                }
                for (extra in HUB_OPTIONAL) {
                    try {
                        fetchHubFile(source, model, extra, File(stagingDir, extra), hubToken, false, client)
                    } catch (_: Exception) {
                    }
                }
                if (!isValidBundle(staging)) {
                    throw RuntimeException(
                        "$source fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)"
                    )
                }
                val cacheFile = File(cache)
                cacheFile.deleteRecursively()
                Files.move(stagingDir.toPath(), cacheFile.toPath(), StandardCopyOption.ATOMIC_MOVE)
                return cache
            } catch (e: Exception) {
                stagingDir.deleteRecursively()
                throw e
            }
        }

        /** Open a model by reference. A value containing a separator or already
         * on disk is a local path; otherwise it is a model name downloaded from
         * the regional public hub then loaded. */
        @JvmOverloads
        @JvmStatic
        fun open(
            modelRef: String,
            token: String = "",
            site: String = DEFAULT_SITE,
            hfToken: String = "",
            modelscopeApiToken: String = "",
        ): AriaEngine {
            if (isLocalRef(modelRef)) return AriaEngine(modelRef)
            val bundle = downloadModel(modelRef, token, site, hfToken, modelscopeApiToken)
            return AriaEngine(bundle)
        }
    }
}
