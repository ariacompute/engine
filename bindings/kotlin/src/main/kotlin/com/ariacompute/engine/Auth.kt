package com.ariacompute.engine

internal const val INTL_CLOUD = "https://gateway.ariacompute.com"
internal const val INTL_SITE = "https://ariacompute.com"
internal const val INTL_UPGRADE = "https://github.com/ariacompute"
const val CN_CLOUD = "https://gateway.ariacompute.cn"
const val CN_SITE = "https://ariacompute.cn"
const val CN_UPGRADE = "https://gitee.com/ariacompute"

data class AuthConfig(
    var cloudApiKey: String = "",
    var cloudUrl: String = "",
    var siteUrl: String = "",
    var upgradeUrl: String = "",
    var hybridMode: String = "balance",
    var hybridExecution: String = "hybrid",
    var hybridSemantic: Boolean = true,
    var hybridSemanticTimeoutMs: Int = 800,
    var hybridSemanticCacheSize: Int = 512,
    var compute: String = "auto",
    var hfToken: String = "",
    var modelscopeApiToken: String = "",
)

internal fun defaultAuthConfig() = AuthConfig()

internal fun gatewayRegion(url: String): String? {
    val lower = url.lowercase()
    if (lower.contains("ariacompute.cn") || lower.contains("gitee.com/ariacompute")) return "cn"
    if (lower.contains("ariacompute.com") || lower.contains("github.com/ariacompute")) return "intl"
    return null
}

internal fun pairUrls(region: String): Triple<String, String, String> =
    if (region == "cn") Triple(CN_CLOUD, CN_SITE, CN_UPGRADE)
    else Triple(INTL_CLOUD, INTL_SITE, INTL_UPGRADE)

fun fillAuthUrls(cfg: AuthConfig): AuthConfig {
    val region = gatewayRegion(cfg.siteUrl) ?: gatewayRegion(cfg.cloudUrl) ?: gatewayRegion(cfg.upgradeUrl)
        ?: return cfg
    val (cloud, site, upgrade) = pairUrls(region)
    if (cfg.cloudUrl.isEmpty()) cfg.cloudUrl = cloud
    if (cfg.siteUrl.isEmpty()) cfg.siteUrl = site
    if (cfg.upgradeUrl.isEmpty()) cfg.upgradeUrl = upgrade
    return cfg
}

internal fun localePrefersCn(): Boolean {
    val lang = ((System.getenv("LANG") ?: "") + (System.getenv("LC_ALL") ?: "")).lowercase()
    return lang.contains("zh") || lang.contains(".cn") || lang.startsWith("cn")
}

internal var probeDashboard: (String, String) -> Boolean = { site, key ->
    defaultProbeDashboard(site, key)
}

internal fun defaultProbeDashboard(siteUrl: String, apiKey: String): Boolean {
    return try {
        val url = java.net.URI.create(siteUrl.trimEnd('/') + "/api/dashboard/models").toURL()
        val conn = url.openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = 10_000
        conn.readTimeout = 10_000
        conn.setRequestProperty("User-Agent", "aria-engine-sdk/0.1.0")
        conn.setRequestProperty("Authorization", "Bearer $apiKey")
        conn.requestMethod = "GET"
        conn.responseCode in 200..299
    } catch (_: Exception) {
        false
    }
}

fun detectGatewayPair(apiKey: String): Triple<String, String, String> {
    val key = apiKey.trim()
    val first = if (localePrefersCn()) "cn" else "intl"
    val second = if (first == "cn") "intl" else "cn"
    for (region in listOf(first, second)) {
        val (cloud, site, upgrade) = pairUrls(region)
        if (key.isNotEmpty() && probeDashboard(site, key)) return Triple(cloud, site, upgrade)
    }
    return pairUrls(first)
}

fun applyAuthFields(
    existing: AuthConfig,
    cloudApiKey: String? = null,
    cloudUrl: String? = null,
    siteUrl: String? = null,
    upgradeUrl: String? = null,
    hybridMode: String? = null,
    hybridExecution: String? = null,
    hybridSemantic: Boolean? = null,
    hybridSemanticTimeoutMs: Int? = null,
    hybridSemanticCacheSize: Int? = null,
    compute: String? = null,
    hfToken: String? = null,
    modelscopeApiToken: String? = null,
): AuthConfig {
    val out = existing.copy()
    if (cloudApiKey != null) out.cloudApiKey = cloudApiKey
    if (cloudUrl != null) out.cloudUrl = cloudUrl
    if (siteUrl != null) out.siteUrl = siteUrl
    if (upgradeUrl != null) out.upgradeUrl = upgradeUrl
    if (hybridMode != null) out.hybridMode = hybridMode
    if (hybridExecution != null) out.hybridExecution = hybridExecution
    if (hybridSemantic != null) out.hybridSemantic = hybridSemantic
    if (hybridSemanticTimeoutMs != null) out.hybridSemanticTimeoutMs = hybridSemanticTimeoutMs
    if (hybridSemanticCacheSize != null) out.hybridSemanticCacheSize = hybridSemanticCacheSize
    if (compute != null) out.compute = compute
    if (hfToken != null) out.hfToken = hfToken
    if (modelscopeApiToken != null) out.modelscopeApiToken = modelscopeApiToken
    require(out.hybridMode in setOf("cost", "balance", "intelligence")) { "invalid hybrid_mode: ${out.hybridMode}" }
    require(out.hybridExecution in setOf("hybrid", "device", "cloud")) { "invalid hybrid_execution: ${out.hybridExecution}" }
    require(out.compute in setOf("auto", "cpu", "cuda")) { "invalid compute: ${out.compute}" }
    require(out.hybridSemanticTimeoutMs > 0 && out.hybridSemanticCacheSize > 0) {
        "hybrid_semantic_timeout_ms / cache_size must be positive integers"
    }
    fillAuthUrls(out)
    if (out.cloudApiKey.isNotEmpty() && (out.cloudUrl.isEmpty() || out.siteUrl.isEmpty() || out.upgradeUrl.isEmpty())) {
        val (cloud, site, upgrade) = detectGatewayPair(out.cloudApiKey)
        if (out.cloudUrl.isEmpty()) out.cloudUrl = cloud
        if (out.siteUrl.isEmpty()) out.siteUrl = site
        if (out.upgradeUrl.isEmpty()) out.upgradeUrl = upgrade
    }
    return out
}
