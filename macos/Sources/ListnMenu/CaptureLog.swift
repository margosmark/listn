import Foundation

/// Append-only capture diagnostics. Safe to call from any thread; the audio
/// callback should hop off the realtime thread before calling it.
enum CaptureLog {
    private static let lock = NSLock()
    private static let formatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
    /// When set, lines are also printed and mirrored to selftest.log.
    static var echo = false
    private static var truncatedMirror = false

    static var directory: URL { PythonSupervisor.dataDirectory }

    static func beginSelfTest() {
        lock.lock()
        echo = true
        truncatedMirror = false
        lock.unlock()
    }

    static func write(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        let line = formatter.string(from: Date()) + " " + message
        let doEcho = echo
        let truncate = doEcho && !truncatedMirror
        if truncate { truncatedMirror = true }
        if doEcho {
            print(line)
            fflush(stdout)
        }
        append("capture.log", line, truncate: false)
        if doEcho {
            append("selftest.log", line, truncate: truncate)
        }
    }

    private static func append(_ name: String, _ line: String, truncate: Bool) {
        let directory = self.directory
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            let url = directory.appendingPathComponent(name)
            if truncate || !FileManager.default.fileExists(atPath: url.path) {
                let created = FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
                if truncate && !created {
                    try Data().write(to: url)
                }
            }
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            if truncate {
                try handle.truncate(atOffset: 0)
            } else {
                _ = try handle.seekToEnd()
            }
            if let data = (line + "\n").data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } catch {
            // Logging must never take down capture.
        }
    }
}
