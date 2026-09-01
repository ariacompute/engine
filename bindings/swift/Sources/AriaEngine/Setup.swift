import Foundation

public let intlSite = "https://ariacompute.com"
public let intlUpgrade = "https://github.com/ariacompute"
public let cnSite = "https://ariacompute.cn"
public let cnUpgrade = "https://gitee.com/ariacompute"

public struct SetupConfig: Equatable {
    public var router: String = ""
    public var routerApiKey: String = ""
    public var siteUrl: String = ""
    public var upgradeUrl: String = ""
    public var compute: String = "auto"
    public var hfToken: String = ""
    public var modelscopeApiToken: String = ""

    public init() {}
}

public struct SetupUpdates {
    public var router: String?
    public var routerApiKey: String?
    public var siteUrl: String?
    public var upgradeUrl: String?
    public var compute: String?
    public var hfToken: String?
    public var modelscopeApiToken: String?

    public init(
        router: String? = nil,
        routerApiKey: String? = nil,
        siteUrl: String? = nil,
        upgradeUrl: String? = nil,
        compute: String? = nil,
        hfToken: String? = nil,
        modelscopeApiToken: String? = nil
    ) {
        self.router = router
        self.routerApiKey = routerApiKey
        self.siteUrl = siteUrl
        self.upgradeUrl = upgradeUrl
        self.compute = compute
        self.hfToken = hfToken
        self.modelscopeApiToken = modelscopeApiToken
    }
}

public enum AriaSetupError: Error, Equatable {
    case invalidCompute(String)
}

func gatewayRegion(_ url: String) -> String? {
    let lower = url.lowercased()
    if lower.contains("ariacompute.cn") || lower.contains("gitee.com/ariacompute") { return "cn" }
    if lower.contains("ariacompute.com") || lower.contains("github.com/ariacompute") { return "intl" }
    return nil
}

func pairUrls(_ region: String) -> (String, String) {
    region == "cn" ? (cnSite, cnUpgrade) : (intlSite, intlUpgrade)
}

func fillSetupUrls(_ cfg: SetupConfig) -> SetupConfig {
    var out = cfg
    guard let region = gatewayRegion(out.siteUrl) ?? gatewayRegion(out.upgradeUrl) else { return out }
    let (site, upgrade) = pairUrls(region)
    if out.siteUrl.isEmpty { out.siteUrl = site }
    if out.upgradeUrl.isEmpty { out.upgradeUrl = upgrade }
    return out
}

public func applySetup(_ existing: SetupConfig, _ updates: SetupUpdates) throws -> SetupConfig {
    var out = existing
    if let v = updates.router { out.router = v }
    if let v = updates.routerApiKey { out.routerApiKey = v }
    if let v = updates.siteUrl { out.siteUrl = v }
    if let v = updates.upgradeUrl { out.upgradeUrl = v }
    if let v = updates.compute { out.compute = v }
    if let v = updates.hfToken { out.hfToken = v }
    if let v = updates.modelscopeApiToken { out.modelscopeApiToken = v }
    switch out.compute {
    case "auto", "cpu", "cuda":
        break
    default:
        throw AriaSetupError.invalidCompute(out.compute)
    }
    return fillSetupUrls(out)
}
