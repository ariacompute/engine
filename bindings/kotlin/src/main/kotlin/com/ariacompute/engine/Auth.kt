package com.ariacompute.engine

const val INTL_SITE = "https://ariacompute.com"
const val INTL_UPGRADE = "https://github.com/ariacompute"
const val CN_SITE = "https://ariacompute.cn"
const val CN_UPGRADE = "https://gitee.com/ariacompute"

data class AuthConfig(
    var router: String = "",
    var siteUrl: String = "",
    var upgradeUrl: String = "",
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

internal fun pairUrls(region: String): Pair<String, String> =
    if (region == "cn") Pair(CN_SITE, CN_UPGRADE) else Pair(INTL_SITE, INTL_UPGRADE)

fun fillAuthUrls(cfg: AuthConfig): AuthConfig {
    val region = gatewayRegion(cfg.siteUrl) ?: gatewayRegion(cfg.upgradeUrl) ?: return cfg
    val (site, upgrade) = pairUrls(region)
    if (cfg.siteUrl.isEmpty()) cfg.siteUrl = site
    if (cfg.upgradeUrl.isEmpty()) cfg.upgradeUrl = upgrade
    return cfg
}

fun applyAuthFields(
    existing: AuthConfig,
    router: String? = null,
    siteUrl: String? = null,
    upgradeUrl: String? = null,
    compute: String? = null,
    hfToken: String? = null,
    modelscopeApiToken: String? = null,
): AuthConfig {
    val out = existing.copy()
    if (router != null) out.router = router
    if (siteUrl != null) out.siteUrl = siteUrl
    if (upgradeUrl != null) out.upgradeUrl = upgradeUrl
    if (compute != null) out.compute = compute
    if (hfToken != null) out.hfToken = hfToken
    if (modelscopeApiToken != null) out.modelscopeApiToken = modelscopeApiToken
    require(out.compute in setOf("auto", "cpu", "cuda")) { "invalid compute: ${out.compute}" }
    fillAuthUrls(out)
    return out
}
