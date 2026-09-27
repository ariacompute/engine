import Foundation

public final class AriaEngine {
    public let checkpoint: String
    public let track: String

    public init(checkpoint: String, track: String = "encoder") {
        self.checkpoint = checkpoint
        self.track = track
    }

    /// Call `aria_systemone` via XCFramework / module map when linked.
    public func systemone(requestJson: String) throws -> String {
        throw AriaEngineError.notLinked(
            "Link AriaFFI.xcframework and call aria_systemone (checkpoint=\(checkpoint) track=\(track))"
        )
    }

    public func destroy() {}
}

public enum AriaEngineError: Error {
    case notLinked(String)
}
