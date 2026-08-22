package com.ariacompute.engine

import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.ZipInputStream
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
                val suffix = model.substring(idx + 2)
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

        /** Download `model` from the Dashboard private source into
         * `~/.ariacompute/models/{model}`; skips when a valid bundle is cached. */
        @JvmOverloads
        @JvmStatic
        fun downloadModel(model: String, token: String, site: String = DEFAULT_SITE): String {
            require(token.isNotEmpty()) { "dashboard token is required to download a model" }
            val (slug, quant) = parseBundleName(model)
            val cache = cacheDir(model)
            if (File(cache).exists() && isValidBundle(cache)) return cache

            val client = HttpClient.newHttpClient()
            val metaUrl = "${site.trimEnd('/')}/api/dashboard/models/" +
                java.net.URLEncoder.encode(slug, "UTF-8") +
                "/download?quant=${java.net.URLEncoder.encode(quant, "UTF-8")}&sdk=v1.0&format=json"
            val metaReq = HttpRequest.newBuilder(URI.create(metaUrl))
                .header("Authorization", "Bearer $token")
                .GET()
                .build()
            val metaResp = client.send(metaReq, HttpResponse.BodyHandlers.ofString())
            if (metaResp.statusCode() != 200) {
                throw RuntimeException("dashboard request failed: ${metaResp.statusCode()}")
            }
            val meta = JSONObject(metaResp.body())
            val url = meta.optString("url")
            if (url.isEmpty()) throw RuntimeException("dashboard meta returned empty url")

            val zipReq = HttpRequest.newBuilder(URI.create(url))
                .header("Authorization", "Bearer $token")
                .GET()
                .build()
            val zipResp = client.send(zipReq, HttpResponse.BodyHandlers.ofByteArray())
            if (zipResp.statusCode() != 200) {
                throw RuntimeException("download stream failed: ${zipResp.statusCode()}")
            }
            val data = zipResp.body()

            val staging = cacheDir(".$model.partial")
            File(staging).deleteRecursively()
            val stagingDir = File(staging)
            stagingDir.mkdirs()
            extractZip(data.inputStream().buffered(), stagingDir)
            if (!isValidBundle(staging)) {
                stagingDir.deleteRecursively()
                throw RuntimeException("downloaded archive did not contain a valid aria-quant-bundle")
            }
            val cacheFile = File(cache)
            cacheFile.deleteRecursively()
            Files.move(stagingDir.toPath(), cacheFile.toPath(), StandardCopyOption.ATOMIC_MOVE)
            return cache
        }

        private fun extractZip(stream: BufferedInputStream, dest: File) {
            ZipInputStream(stream).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    val out = File(dest, entry.name)
                    if (entry.isDirectory) {
                        out.mkdirs()
                    } else {
                        out.parentFile?.mkdirs()
                        FileOutputStream(out).use { fos -> zis.copyTo(fos) }
                    }
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
            // flatten a single top-level subdir
            val entries = dest.listFiles()?.filter { !it.name.startsWith(".") } ?: emptyList()
            if (entries.size == 1 && entries[0].isDirectory) {
                val inner = entries[0]
                if (File(inner, "config.json").isFile) {
                    inner.listFiles()?.forEach { f ->
                        Files.move(f.toPath(), File(dest, f.name).toPath(), StandardCopyOption.REPLACE_EXISTING)
                    }
                    inner.deleteRecursively()
                }
            }
        }

        /** Open a model by reference. A value containing a separator or already
         * on disk is a local path; otherwise it is a model name that is
         * downloaded (requires `token`) then loaded. */
        @JvmOverloads
        @JvmStatic
        fun open(modelRef: String, token: String = "", site: String = DEFAULT_SITE): AriaEngine {
            if (isLocalRef(modelRef)) return AriaEngine(modelRef)
            if (token.isEmpty()) {
                throw IllegalArgumentException("model name '$modelRef' requires a dashboard token to download")
            }
            val bundle = downloadModel(modelRef, token, site)
            return AriaEngine(bundle)
        }
    }
}
