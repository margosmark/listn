import Darwin
import Foundation

struct ServerInfo: Decodable {
    var pid: Int32
    var port: Int?
    var sock: String?
    var version: String?
}

enum EngineError: LocalizedError {
    case missing
    case exited(String)
    case timedOut
    case noSocket

    var errorDescription: String? {
        switch self {
        case .missing:
            return "Could not find the listn engine. From the repo run `uv sync --extra mlx`, then rebuild the app, or set LISTN_BIN to the listn executable."
        case .exited(let detail):
            return detail
        case .timedOut:
            return "The transcription engine did not come up. See engine.log in the listn data folder."
        case .noSocket:
            return "The engine is running but it has no capture socket. Quit any older listn process and try again."
        }
    }
}

/// Starts `listn ui --no-browser` if it is not already serving, and waits until
/// server.json points at a live loopback server and a capture socket.
final class PythonSupervisor {
    static var dataDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["LISTN_DATA_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/listn", isDirectory: true)
    }

    private(set) var spawned = false
    private var process: Process?
    let port = 8765

    func ensureRunning(_ completion: @escaping (Result<ServerInfo, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Result { try self.ensureRunningSync() }
            DispatchQueue.main.async { completion(result) }
        }
    }

    func stop() {
        guard spawned, let process, process.isRunning else { return }
        process.terminate()
        spawned = false
    }

    private func ensureRunningSync() throws -> ServerInfo {
        if let existing = readInfo(),
           pidAlive(existing.pid),
           let port = existing.port,
           let sock = existing.sock,
           FileManager.default.fileExists(atPath: sock),
           httpReady(port) {
            spawned = false
            return existing
        }
        let binary = try findEngine()
        let spec = launchSpec(binary)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: spec.executable)
        process.arguments = spec.arguments + ["ui", "--no-browser", "--port", String(port)]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["DO_NOT_TRACK"] = "1"
        process.environment = environment
        let log = try openLog()
        process.standardOutput = log
        process.standardError = log
        try process.run()
        self.process = process
        spawned = true

        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            if !process.isRunning {
                throw EngineError.exited(tail(of: Self.dataDirectory.appendingPathComponent("engine.log"))
                    ?? "The transcription engine exited before it started listening.")
            }
            if let info = readInfo(),
               info.pid == process.processIdentifier,
               let readyPort = info.port,
               let sock = info.sock,
               FileManager.default.fileExists(atPath: sock),
               httpReady(readyPort) {
                return info
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
        throw EngineError.timedOut
    }

    private func findEngine() throws -> String {
        let environment = ProcessInfo.processInfo.environment["LISTN_BIN"]
        var candidates: [String] = []
        if let environment, !environment.isEmpty { candidates.append(environment) }
        if let bundled = Bundle.main.url(forResource: "engine", withExtension: "path"),
           let text = try? String(contentsOf: bundled, encoding: .utf8) {
            let path = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !path.isEmpty { candidates.append(path) }
        }
        if let executable = Bundle.main.executableURL {
            var directory = executable.deletingLastPathComponent()
            for _ in 0..<8 {
                candidates.append(directory.appendingPathComponent(".venv/bin/listn").path)
                let parent = directory.deletingLastPathComponent()
                if parent.path == directory.path { break }
                directory = parent
            }
        }
        candidates.append(contentsOf: ["/opt/homebrew/bin/listn", "/usr/local/bin/listn"])
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        throw EngineError.missing
    }

    /// `Process` cannot exec a shebang script. listn's console entry is one, so
    /// run the interpreter named on its first line.
    private func launchSpec(_ path: String) -> (executable: String, arguments: [String]) {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8), text.hasPrefix("#!") else {
            return (path, [])
        }
        let line = text.split(separator: "\n", maxSplits: 1)[0].dropFirst(2)
        let interpreter = line.trimmingCharacters(in: .whitespaces)
        if interpreter.hasPrefix("/usr/bin/env ") {
            let program = String(interpreter.dropFirst("/usr/bin/env ".count)).split(separator: " ").first.map(String.init) ?? "python3"
            return ("/usr/bin/env", [program, path])
        }
        let parts = interpreter.split(separator: " ").map(String.init)
        if parts.count > 1 {
            return (parts[0], Array(parts.dropFirst()) + [path])
        }
        return (interpreter, [path])
    }

    private func readInfo() -> ServerInfo? {
        let url = Self.dataDirectory.appendingPathComponent("server.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ServerInfo.self, from: data)
    }

    private func httpReady(_ port: Int) -> Bool {
        guard let url = URL(string: "http://127.0.0.1:\(port)/api/status") else { return false }
        var request = URLRequest(url: url, timeoutInterval: 0.6)
        request.httpMethod = "GET"
        let semaphore = DispatchSemaphore(value: 0)
        let ok = Locked(false)
        URLSession.shared.dataTask(with: request) { data, response, _ in
            let status = (response as? HTTPURLResponse)?.statusCode == 200
            let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
            ok.value = status && body.contains("\"capture_sock\"")
            semaphore.signal()
        }.resume()
        _ = semaphore.wait(timeout: .now() + 0.8)
        return ok.value
    }

    private func openLog() throws -> FileHandle {
        let directory = Self.dataDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        let url = directory.appendingPathComponent("engine.log")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
        }
        let handle = try FileHandle(forWritingTo: url)
        _ = try handle.seekToEnd()
        return handle
    }

    private func tail(of url: URL) -> String? {
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return nil }
        let text = String(data: data.suffix(1200), encoding: .utf8) ?? ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

private func pidAlive(_ pid: Int32) -> Bool {
    if pid <= 0 { return false }
    if kill(pid, 0) == 0 { return true }
    return errno == EPERM
}
