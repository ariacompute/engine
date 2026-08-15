import Foundation

public final class AriaEngine {
    private var handle: OpaquePointer?

    public init(bundlePath: String) throws {
        // Link libaria_ffi and call aria_model_init via bridging header / module map.
        // Stub for host documentation; wire C calls when XCFramework is linked.
        self.handle = nil
        if bundlePath.isEmpty { throw NSError(domain: "Aria", code: 1) }
    }

    public func complete(messagesJson: String, optionsJson: String, toolsJson: String = "[]") throws -> String {
        // aria_complete(...)
        return #"{"success":true,"response":"","function_calls":[]}"#
    }

    deinit {
        // aria_model_destroy(handle)
    }
}
