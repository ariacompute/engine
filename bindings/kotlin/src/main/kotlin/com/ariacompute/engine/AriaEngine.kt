package com.ariacompute.engine

import java.io.File
import java.io.FileInputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.GZIPInputStream
import org.json.JSONArray
import org.json.JSONObject

/** Kotlin/JVM + Android JNI wrapper over libaria_ffi. */
class AriaEngine : AutoCloseable {
    private var handle: Long = 0L
    private var cfg: SetupConfig = defaultSetupConfig()
    private var genericToken: String = ""

    constructor()

    constructor(bundlePath: String) {
        loadNative()
        initBundle(bundlePath)
    }

    private fun initBundle(bundlePath: String) {
        handle = nativeInit(bundlePath)
        if (handle == 0L) throw IllegalStateException(nativeLastError() ?: "init failed")
    }

    fun setup(
        router: String? = null,
        routerApiKey: String? = null,
        siteUrl: String? = null,
        upgradeUrl: String? = null,
        compute: String? = null,
        hfToken: String? = null,
        modelscopeApiToken: String? = null,
    ): AriaEngine {
        cfg = applySetupFields(
            cfg,
            router = router,
            routerApiKey = routerApiKey,
            siteUrl = siteUrl,
            upgradeUrl = upgradeUrl,
            compute = compute,
            hfToken = hfToken,
            modelscopeApiToken = modelscopeApiToken,
        )
        return this
    }

    fun setupStatus(): SetupConfig = cfg.copy()

    fun setupClear(): AriaEngine {
        cfg = defaultSetupConfig()
        return this
    }

    /** Download (if needed) and load a model using instance setup. */
    @JvmName("openNamed")
    fun open(modelRef: String): AriaEngine {
        val site = cfg.siteUrl.ifEmpty { DEFAULT_SITE }
        loadNative(site)
        val bundle = if (isLocalRef(modelRef)) modelRef else downloadModel(
            modelRef,
            genericToken,
            site,
            cfg.hfToken,
            cfg.modelscopeApiToken,
        )
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
        initBundle(bundle)
        return this
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
        @Volatile
        private var nativeLoaded = false

        @Synchronized
        internal fun loadNative(site: String = DEFAULT_SITE) {
            if (nativeLoaded) return
            val env = System.getenv("ARIA_FFI_LIB")
            if (!env.isNullOrEmpty() && File(env).isFile) {
                System.load(File(env).absolutePath)
                nativeLoaded = true
                return
            }
            try {
                System.loadLibrary("aria_ffi")
                nativeLoaded = true
                return
            } catch (_: UnsatisfiedLinkError) {
                // Fall through to ~/.ariacompute/lib or Releases download.
            }
            System.load(File(ensureFfiLib(site)).absolutePath)
            nativeLoaded = true
        }

        private const val DEFAULT_SITE = "https://ariacompute.com"
        private const val SDK_UA = "aria-engine-sdk/0.1.0"

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

        internal fun ffiLibName(osName: String = System.getProperty("os.name") ?: ""): String {
            val n = osName.lowercase()
            return when {
                n.contains("win") -> "aria_ffi.dll"
                n.contains("mac") || n.contains("darwin") -> "libaria_ffi.dylib"
                else -> "libaria_ffi.so"
            }
        }

        private fun libDir(): String = ariaHome() + File.separator + "lib"

        private fun cachedFfiPath(): String? {
            val p = File(libDir(), ffiLibName())
            return if (p.isFile) p.absolutePath else null
        }

        internal fun ffiAssetOs(
            osName: String = System.getProperty("os.name") ?: "",
            arch: String = System.getProperty("os.arch") ?: "",
        ): String {
            val os = osName.lowercase()
            val a = arch.lowercase()
            if (os.contains("linux") && (a == "amd64" || a == "x86_64")) return "linux_x86_64"
            if (os.contains("linux") && (a == "aarch64" || a == "arm64")) return "linux_arm64"
            if (os.contains("mac") || os.contains("darwin")) return "macos"
            if (os.contains("win") && (a == "amd64" || a == "x86_64")) return "windows_x86_64"
            throw IllegalStateException("unsupported platform $osName/$arch for libaria_ffi")
        }

        private fun stripV(tag: String): String {
            val t = tag.trim()
            return if (t.startsWith("v") || t.startsWith("V")) t.substring(1) else t
        }

        private fun parseSemver(tag: String): Triple<Int, Int, Int>? {
            val core = stripV(tag).split("-", limit = 2)[0].split("+", limit = 2)[0]
            val parts = core.split(".")
            if (parts.isEmpty() || parts[0].toIntOrNull() == null) return null
            return Triple(
                parts[0].toInt(),
                parts.getOrNull(1)?.toIntOrNull() ?: 0,
                parts.getOrNull(2)?.toIntOrNull() ?: 0,
            )
        }

        internal fun selectLatestStable(releases: JSONArray): String {
            var bestTag: String? = null
            var best = Triple(-1, -1, -1)
            for (i in 0 until releases.length()) {
                val rel = releases.getJSONObject(i)
                if (rel.optBoolean("draft") || rel.optBoolean("prerelease")) continue
                val tag = rel.optString("tag_name").ifEmpty { rel.optString("tag") }
                val parsed = parseSemver(tag) ?: continue
                if (parsed.first > best.first ||
                    (parsed.first == best.first && parsed.second > best.second) ||
                    (parsed.first == best.first && parsed.second == best.second && parsed.third > best.third)
                ) {
                    best = parsed
                    bestTag = tag
                }
            }
            if (bestTag == null) throw IllegalStateException("no stable release found for libaria_ffi")
            return stripV(bestTag)
        }

        private fun upgradeOrg(site: String): String {
            configYmlScalar("upgrade_url")?.let { return it.trimEnd('/') }
            val hint = (site.ifEmpty { configYmlScalar("site_url") ?: DEFAULT_SITE }).lowercase()
            return if (hint.contains("ariacompute.cn") || hint.contains("gitee.com")) {
                "https://gitee.com/ariacompute"
            } else {
                "https://github.com/ariacompute"
            }
        }

        private fun releasesApiUrl(org: String): String {
            val owner = org.trimEnd('/').substringAfterLast('/')
            return if (org.lowercase().contains("gitee.com")) {
                "https://gitee.com/api/v5/repos/$owner/engine/releases?per_page=30"
            } else {
                "https://api.github.com/repos/$owner/engine/releases?per_page=30"
            }
        }

        private fun httpGetBytes(url: String, dest: File? = null): ByteArray {
            val client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
            val req = HttpRequest.newBuilder(URI.create(url))
                .header("User-Agent", SDK_UA)
                .GET()
                .build()
            if (dest != null) {
                dest.parentFile?.mkdirs()
                val resp = client.send(req, HttpResponse.BodyHandlers.ofFile(dest.toPath()))
                if (resp.statusCode() !in 200..299) {
                    dest.delete()
                    throw RuntimeException("HTTP ${resp.statusCode()} $url")
                }
                return ByteArray(0)
            }
            val resp = client.send(req, HttpResponse.BodyHandlers.ofByteArray())
            if (resp.statusCode() !in 200..299) {
                throw RuntimeException("HTTP ${resp.statusCode()} $url")
            }
            return resp.body()
        }

        internal fun extractFfiArchive(archive: File, destDir: File, want: String = ffiLibName()): String {
            GZIPInputStream(FileInputStream(archive)).use { gzip ->
                val tar = gzip.readBytes()
                var offset = 0
                while (offset + 512 <= tar.size) {
                    val allZero = (0 until 512).all { tar[offset + it].toInt() == 0 }
                    if (allZero) break
                    val nameBytes = tar.copyOfRange(offset, offset + 100)
                    val entryName = String(nameBytes, Charsets.UTF_8).substringBefore('\u0000')
                    val sizeStr = String(tar.copyOfRange(offset + 124, offset + 136), Charsets.US_ASCII)
                        .substringBefore('\u0000').trim()
                    val size = sizeStr.toLongOrNull(8) ?: 0L
                    val typeFlag = tar[offset + 156].toInt()
                    offset += 512
                    val isFile = typeFlag == 0 || typeFlag == '0'.code
                    val base = File(entryName).name
                    if (isFile && base == want) {
                        destDir.mkdirs()
                        val dest = File(destDir, want)
                        dest.writeBytes(tar.copyOfRange(offset, offset + size.toInt()))
                        dest.setExecutable(true)
                        return dest.absolutePath
                    }
                    val padded = ((size + 511) / 512) * 512
                    offset += padded.toInt()
                }
            }
            throw IllegalStateException("$want not found in ${archive.path}")
        }

        @JvmStatic
        @JvmOverloads
        fun ensureFfiLib(site: String = DEFAULT_SITE): String {
            val env = System.getenv("ARIA_FFI_LIB")
            if (!env.isNullOrEmpty() && File(env).isFile) return env
            cachedFfiPath()?.let { return it }

            val org = upgradeOrg(site)
            val raw = httpGetBytes(releasesApiUrl(org))
            val releases = JSONArray(String(raw, Charsets.UTF_8))
            val ver = selectLatestStable(releases)
            val assetName = "libaria_ffi_${ver}_${ffiAssetOs()}.tar.gz"
            var url: String? = null
            for (i in 0 until releases.length()) {
                val rel = releases.getJSONObject(i)
                val tag = rel.optString("tag_name").ifEmpty { rel.optString("tag") }
                if (stripV(tag) != ver) continue
                val assets = rel.optJSONArray("assets") ?: continue
                for (j in 0 until assets.length()) {
                    val asset = assets.getJSONObject(j)
                    if (asset.optString("name") == assetName) {
                        url = asset.optString("browser_download_url").ifEmpty {
                            asset.optString("direct_asset_url")
                        }
                        break
                    }
                }
                if (!url.isNullOrEmpty()) break
            }
            if (url.isNullOrEmpty()) throw IllegalStateException("release asset not found: $assetName")

            val staging = File(ariaHome(), "tmp${File.separator}ffi-$ver")
            staging.deleteRecursively()
            staging.mkdirs()
            return try {
                val archive = File(staging, assetName)
                httpGetBytes(url, archive)
                extractFfiArchive(archive, File(libDir()), ffiLibName())
            } finally {
                staging.deleteRecursively()
            }
        }

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
            val home = File(ariaHome())
            for (name in listOf("engine.yml", "config.yml")) {
                val path = File(home, name)
                if (!path.isFile) continue
                try {
                    path.readLines().forEach { line ->
                        if (line.startsWith(" ") || line.startsWith("\t")) return@forEach
                        val s = line.trim()
                        if (s.isEmpty() || s.startsWith("#") || ':' !in s) return@forEach
                        val idx = s.indexOf(':')
                        if (s.substring(0, idx).trim() != key) return@forEach
                        val v = unquoteYaml(s.substring(idx + 1))
                        if (v.isNotEmpty()) return v
                    }
                } catch (_: Exception) {
                    continue
                }
            }
            return null
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
                            "auth failed HTTP $code; set $field via aria-engine setup (do not pass a Dashboard sk-/bfvk- key as the hub token)"
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
            loadNative(site)
            if (isLocalRef(modelRef)) return AriaEngine(modelRef)
            val bundle = downloadModel(modelRef, token, site, hfToken, modelscopeApiToken)
            return AriaEngine(bundle)
        }
    }
}
