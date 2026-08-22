import Foundation

public enum AriaDownloadError: Error {
    case invalidModelName(String)
    case missingToken
    case requestFailed(Int, String)
    case emptyUrl
    case invalidZip(String)
    case invalidBundle(String)
}

public final class AriaEngine {
    private var handle: OpaquePointer?

    // MARK: - Model name resolution

    /// Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`.
    public static func parseBundleName(_ model: String) -> (slug: String, quant: String)? {
        guard !model.isEmpty, !model.contains("/"), !model.contains("\\") else { return nil }
        if let idx = model.lastIndex(of: "_q") {
            let slug = String(model[..<idx])
            let suffix = String(model[model.index(idx, offsetBy: 2)...])
            let quant: String
            switch suffix {
            case "4": quant = "int4"
            case "8": quant = "int8"
            case "326", "3.26": quant = "int326"
            default: return nil
            }
            guard !slug.isEmpty else { return nil }
            return (slug, quant)
        }
        return (model, "int4")
    }

    private static func ariaHome() -> String {
        if let override = ProcessInfo.processInfo.environment["ARIA_COMPUTE_HOME"],
           !override.isEmpty {
            return override
        }
        let home = NSHomeDirectory()
        return (home as NSString).appendingPathComponent(".ariacompute")
    }

    private static func cacheDir(for model: String) -> String {
        return (ariaHome() as NSString).appendingPathComponent("models").appendingPathComponent(model)
    }

    private static func isLocalRef(_ ref: String) -> Bool {
        return ref.contains("/") || ref.contains("\\") || FileManager.default.fileExists(atPath: ref)
    }

    private static func isValidBundle(_ dir: String) -> Bool {
        let fm = FileManager.default
        let weight = (dir as NSString).appendingPathComponent("weight.bin")
        let config = (dir as NSString).appendingPathComponent("config.json")
        guard fm.fileExists(atPath: weight), let data = fm.contents(atPath: config) else { return false }
        guard let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let format = meta["format"] as? String else { return false }
        return format == "aria-quant-bundle"
    }

    // MARK: - Download (Dashboard private source only)

    /// Download `model` from the Dashboard source into `~/.ariacompute/models/{model}`.
    /// Skips the download when a valid bundle is already cached.
    @discardableResult
    public static func downloadModel(_ model: String, token: String, site: String = "https://ariacompute.com") throws -> String {
        guard !token.isEmpty else { throw AriaDownloadError.missingToken }
        guard let (slug, quant) = parseBundleName(model) else {
            throw AriaDownloadError.invalidModelName(model)
        }
        let cache = cacheDir(for: model)
        if FileManager.default.fileExists(atPath: cache), isValidBundle(cache) {
            return cache
        }

        let metaURLStr = "\(site.trimmingCharacters(in: CharacterSet(charactersIn: "/")))/api/dashboard/models/\(slug.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? slug)/download?quant=\(quant)&sdk=v1.0&format=json"
        guard let metaURL = URL(string: metaURLStr) else { throw AriaDownloadError.requestFailed(0, "bad meta url") }

        var metaReq = URLRequest(url: metaURL)
        metaReq.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let meta = try awaitWith { completion in
            URLSession.shared.dataTask(with: metaReq) { data, resp, err in
                if let err = err { completion(.failure(err)); return }
                guard let data = data,
                      let code = (resp as? HTTPURLResponse)?.statusCode, code == 200,
                      let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                    completion(.failure(AriaDownloadError.requestFailed(code, "meta")))
                    return
                }
                completion(.success(meta))
            }.resume()
        }
        guard let urlStr = meta["url"] as? String, !urlStr.isEmpty else { throw AriaDownloadError.emptyUrl }
        guard let url = URL(string: urlStr) else { throw AriaDownloadError.emptyUrl }

        var zipReq = URLRequest(url: url)
        zipReq.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let data = try awaitWith { completion in
            URLSession.shared.dataTask(with: zipReq) { data, resp, err in
                if let err = err { completion(.failure(err)); return }
                guard let data = data,
                      let code = (resp as? HTTPURLResponse)?.statusCode, code == 200 else {
                    let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                    completion(.failure(AriaDownloadError.requestFailed(code, "zip")))
                    return
                }
                completion(.success(data))
            }.resume()
        }

        let staging = (cacheDir(for: model) as NSString).appendingPathComponent(".partial")
        try? FileManager.default.removeItem(atPath: staging)
        try data.write(to: URL(fileURLWithPath: staging).appendingPathComponent("bundle.zip"))
        try extractZip(at: staging, into: staging)

        if !isValidBundle(staging) {
            try? FileManager.default.removeItem(atPath: staging)
            throw AriaDownloadError.invalidBundle("not a valid aria-quant-bundle")
        }
        try? FileManager.default.removeItem(atPath: cache)
        try FileManager.default.moveItem(atPath: staging, toPath: cache)
        return cache
    }

    private static func extractZip(at zipDir: String, into dest: String) throws {
        let zipPath = (zipDir as NSString).appendingPathComponent("bundle.zip")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", zipPath, "-d", dest]
        try process.run()
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            throw AriaDownloadError.invalidZip("unzip failed")
        }
        // flatten a single top-level subdir
        let contents = try FileManager.default.contentsOfDirectory(atPath: dest).filter { !$0.hasPrefix(".") }
        if contents.count == 1 {
            let only = (dest as NSString).appendingPathComponent(contents[0])
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: only, isDirectory: &isDir)
            if isDir.boolValue, FileManager.default.fileExists(atPath: (only as NSString).appendingPathComponent("config.json")) {
                for name in try FileManager.default.contentsOfDirectory(atPath: only) {
                    try FileManager.default.moveItem(atPath: (only as NSString).appendingPathComponent(name),
                                                     toPath: (dest as NSString).appendingPathComponent(name))
                }
                try FileManager.default.removeItem(atPath: only)
            }
        }
    }

    // MARK: - Init

    public init(bundlePath: String) throws {
        // Link libaria_ffi and call aria_model_init via bridging header / module map.
        // Stub for host documentation; wire C calls when XCFramework is linked.
        self.handle = nil
        if bundlePath.isEmpty { throw NSError(domain: "Aria", code: 1) }
    }

    /// Open a model by reference. A value containing a separator or already on
    /// disk is a local path (loaded directly); otherwise it is a model name
    /// that is downloaded (requires `token`) then loaded.
    public static func open(_ ref: String, token: String = "", site: String = "https://ariacompute.com") async throws -> AriaEngine {
        if isLocalRef(ref) {
            return try AriaEngine(bundlePath: ref)
        }
        guard !token.isEmpty else { throw AriaDownloadError.missingToken }
        let bundle = try await downloadModel(ref, token: token, site: site)
        return try AriaEngine(bundlePath: bundle)
    }

    public func complete(messagesJson: String, optionsJson: String, toolsJson: String = "[]") throws -> String {
        // aria_complete(...)
        return #"{"success":true,"response":"","function_calls":[]}"#
    }

    deinit {
        // aria_model_destroy(handle)
    }
}

/// Minimal synchronous wrapper around an async URLSession completion handler.
private func awaitWith<T>(_ body: (@escaping (Result<T, Error>) -> Void) -> Void) throws -> T {
    var result: Result<T, Error>?
    let semaphore = DispatchSemaphore(value: 0)
    body { res in
        result = res
        semaphore.signal()
    }
    semaphore.wait()
    return try result!.get()
}
