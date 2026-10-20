import Darwin
import Foundation

// Frame layout shared with listn/ipc.py. Little-endian, which is native on Apple Silicon.
// uint32 length | uint8 type | payload. AUDIO payload is uint8 track, uint32 rate,
// uint16 channels, then interleaved float32 PCM.

private let frameHello: UInt8 = 1
private let frameWelcome: UInt8 = 2
private let frameAudio: UInt8 = 3
private let frameStart: UInt8 = 4
private let frameStop: UInt8 = 5
private let frameEvent: UInt8 = 6
private let frameError: UInt8 = 9
private let maxFrame = 8 * 1024 * 1024

enum IPCError: LocalizedError {
    case pathTooLong
    case connectFailed(Int32)
    case notConnected

    var errorDescription: String? {
        switch self {
        case .pathTooLong:
            return "The capture socket path is too long for a Unix socket. Move LISTN_DATA_DIR somewhere shorter."
        case .connectFailed(let code):
            return "Could not connect to the transcription engine (\(code))."
        case .notConnected:
            return "The transcription engine is not connected."
        }
    }
}

/// One client of the engine's capture socket. Writes are serialized. Audio
/// callbacks only enqueue; if the engine falls behind, extra blocks are dropped
/// instead of blocking Core Audio.
final class IPCClient {
    var onEvent: (([String: Any]) -> Void)?
    var onError: ((String) -> Void)?
    var onDisconnect: (() -> Void)?

    private var fd: Int32 = -1
    private let writeQueue = DispatchQueue(label: "local.listn.ipc.write")
    private let readQueue = DispatchQueue(label: "local.listn.ipc.read")
    private let lock = NSLock()
    private var outstanding = 0
    private var generation = 0

    func connect(_ path: String) throws {
        close()
        let socketFD = socket(AF_UNIX, SOCK_STREAM, 0)
        if socketFD < 0 {
            throw IPCError.connectFailed(errno)
        }
        var one: Int32 = 1
        setsockopt(socketFD, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout.size(ofValue: one)))
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        let capacity = MemoryLayout.size(ofValue: addr.sun_path)
        guard bytes.count < capacity else {
            Darwin.close(socketFD)
            throw IPCError.pathTooLong
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            for (index, byte) in bytes.enumerated() {
                raw[index] = byte
            }
        }
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
                Darwin.connect(socketFD, pointer, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if rc != 0 {
            let code = errno
            Darwin.close(socketFD)
            throw IPCError.connectFailed(code)
        }
        lock.lock()
        generation += 1
        let token = generation
        fd = socketFD
        outstanding = 0
        lock.unlock()
        readQueue.async { [weak self] in
            self?.readLoop(token)
        }
    }

    func sendHelloAndStart(source: String, appName: String?, appPID: Int32?, mic: Bool) {
        var body: [String: Any] = ["source": source, "mic": mic]
        if let appName { body["app_name"] = appName }
        if let appPID { body["app_pid"] = appPID }
        writeQueue.async { [weak self] in
            guard let self else { return }
            self.writeAll(self.jsonFrame(frameHello, ["client": "listn-menu", "version": 1]))
            self.writeAll(self.jsonFrame(frameStart, body))
        }
    }

    func sendStop() {
        writeQueue.async { [weak self] in
            guard let self else { return }
            self.writeAll(self.frame(frameStop, Data()))
        }
    }

    func sendAudio(track: UInt8, rate: Double, channels: Int, samples: [Float]) {
        guard channels > 0, rate > 0, !samples.isEmpty else { return }
        lock.lock()
        if outstanding >= 24 || fd < 0 {
            lock.unlock()
            return
        }
        outstanding += 1
        lock.unlock()
        writeQueue.async { [weak self] in
            guard let self else { return }
            var payload = Data()
            payload.append(track)
            appendLE(UInt32(rate.rounded()), to: &payload)
            appendLE(UInt16(channels), to: &payload)
            samples.withUnsafeBufferPointer { buffer in
                payload.append(contentsOf: UnsafeRawBufferPointer(buffer))
            }
            self.writeAll(self.frame(frameAudio, payload))
            self.lock.lock()
            self.outstanding -= 1
            self.lock.unlock()
        }
    }

    func close() {
        lock.lock()
        generation += 1
        let socketFD = fd
        fd = -1
        lock.unlock()
        if socketFD >= 0 {
            Darwin.close(socketFD)
        }
    }

    private func readLoop(_ token: Int) {
        var buffer = Data()
        let capacity = 65536
        var chunk = [UInt8](repeating: 0, count: capacity)
        while true {
            lock.lock()
            let current = generation
            let socketFD = fd
            lock.unlock()
            if current != token || socketFD < 0 { break }
            let count = Darwin.read(socketFD, &chunk, capacity)
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                break
            }
            buffer.append(contentsOf: chunk[0..<count])
            if !drain(&buffer) { break }
        }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let current = self.generation == token
            self.lock.unlock()
            if current { self.onDisconnect?() }
        }
    }

    private func drain(_ buffer: inout Data) -> Bool {
        while buffer.count >= 5 {
            let length = readU32(buffer, 0)
            if length < 1 || length > maxFrame {
                DispatchQueue.main.async { [weak self] in
                    self?.onError?("The transcription engine sent a frame listn does not understand.")
                }
                return false
            }
            let total = 4 + Int(length)
            if buffer.count < total { break }
            let kind = buffer[buffer.startIndex + 4]
            let payload = buffer.subdata(in: (buffer.startIndex + 5)..<(buffer.startIndex + total))
            buffer.removeFirst(total)
            deliver(kind, payload)
        }
        return true
    }

    private func deliver(_ kind: UInt8, _ payload: Data) {
        switch kind {
        case frameEvent:
            guard let object = jsonObject(payload) else { return }
            DispatchQueue.main.async { [weak self] in self?.onEvent?(object) }
        case frameError:
            let message = (jsonObject(payload)?["message"] as? String) ?? "The transcription engine reported an error."
            DispatchQueue.main.async { [weak self] in self?.onError?(message) }
        case frameWelcome:
            break
        default:
            break
        }
    }

    private func writeAll(_ data: Data) {
        lock.lock()
        let socketFD = fd
        lock.unlock()
        guard socketFD >= 0, !data.isEmpty else { return }
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var sent = 0
            while sent < raw.count {
                let wrote = Darwin.write(socketFD, base.advanced(by: sent), raw.count - sent)
                if wrote < 0 {
                    if errno == EINTR { continue }
                    return
                }
                if wrote == 0 { return }
                sent += wrote
            }
        }
    }

    private func jsonFrame(_ kind: UInt8, _ object: [String: Any]) -> Data {
        let payload = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
        return frame(kind, payload)
    }

    private func frame(_ kind: UInt8, _ payload: Data) -> Data {
        var data = Data()
        appendLE(UInt32(payload.count + 1), to: &data)
        data.append(kind)
        data.append(payload)
        return data
    }
}

private func jsonObject(_ data: Data) -> [String: Any]? {
    (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

private func readU32(_ data: Data, _ offset: Int) -> UInt32 {
    let index = data.startIndex + offset
    return UInt32(data[index])
        | (UInt32(data[index + 1]) << 8)
        | (UInt32(data[index + 2]) << 16)
        | (UInt32(data[index + 3]) << 24)
}

private func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var little = value.littleEndian
    withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
}
