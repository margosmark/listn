import AppKit
import AVFoundation
import CoreAudio
import Darwin

// System audio is captured with a Core Audio process tap (macOS 14.2+), not
// ScreenCaptureKit. A tap needs only the system-audio recording permission
// (NSAudioCaptureUsageDescription). It can take every process or one app, and
// muteBehavior .unmuted leaves speakers and headphones playing normally.
// ScreenCaptureKit would also work, but it asks for screen recording and is
// built around video frames, which this app never wants.
//
// The microphone is a separate AVAudioEngine input. The two streams stay
// separate tracks: track 0 is other people, track 1 is you.

// The SDK imports kAudioObjectSystemObject as Int32 and kAudioObjectUnknown
// differs between SDK versions, so both are normalized to AudioObjectID once.
private let systemObject = AudioObjectID(kAudioObjectSystemObject)
private let noObject = AudioObjectID(kAudioObjectUnknown)

/// A thread-safe slot for values written from a @Sendable completion handler.
final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value

    init(_ value: Value) { stored = value }

    var value: Value {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}

struct AudioApp {
    var objectID: AudioObjectID
    var pid: pid_t
    var name: String
    var bundleID: String
}

func listAudioApps() -> [AudioApp] {
    let ids = audioObjectIDs(systemObject, kAudioHardwarePropertyProcessObjectList)
    let selfPID = ProcessInfo.processInfo.processIdentifier
    var apps: [AudioApp] = []
    for objectID in ids {
        guard let pid = readUInt32(objectID, kAudioProcessPropertyPID).map({ pid_t($0) }), pid != selfPID else { continue }
        let application = NSRunningApplication(processIdentifier: pid)
        let bundleID = application?.bundleIdentifier ?? ""
        let name = application?.localizedName ?? (bundleID.isEmpty ? "Process \(pid)" : bundleID)
        apps.append(AudioApp(objectID: objectID, pid: pid, name: name, bundleID: bundleID))
    }
    return apps
}

func audioObjectID(for pid: pid_t) -> AudioObjectID? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var pidValue = pid
    var object = noObject
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        systemObject, &address,
        UInt32(MemoryLayout<pid_t>.size), &pidValue,
        &size, &object)
    guard status == noErr, object != noObject else { return nil }
    return object
}

enum CaptureError: LocalizedError {
    case tap(String)
    case aggregate(String)
    case io(String)
    case format
    case noProcess(String)

    var errorDescription: String? {
        switch self {
        case .tap(let detail):
            return "Could not capture system audio (\(detail)). Allow listn under System Settings → Privacy & Security → System Audio Recording, then try again."
        case .aggregate(let detail):
            return "Could not open the audio tap (\(detail))."
        case .io(let detail):
            return "Could not start the audio tap (\(detail))."
        case .format:
            return "The audio tap did not report a usable format."
        case .noProcess(let name):
            return "\(name) is not playing audio yet. Start the meeting, then pick it again."
        }
    }
}

final class CaptureEngine {
    var onPCM: ((UInt8, Double, Int, [Float]) -> Void)?
    var onLevels: ((Float, Float) -> Void)?
    var onWarning: ((String) -> Void)?

    private var tapID = noObject
    private var aggregateID = noObject
    private var procID: AudioDeviceIOProcID?
    private var format = AudioStreamBasicDescription()
    private var micEngine: AVAudioEngine?
    private var running = false
    private var systemDB: Float = -120
    private var micDB: Float = -120
    private var lastLevel = Date.distantPast
    private let levelLock = NSLock()

    func start(appPID: pid_t?, appName: String?, mic: Bool) throws {
        stop()
        let description: CATapDescription
        if let appPID {
            guard let objectID = audioObjectID(for: appPID) else {
                throw CaptureError.noProcess(appName ?? "That app")
            }
            description = CATapDescription(stereoMixdownOfProcesses: [objectID])
        } else {
            var excluded: [AudioObjectID] = []
            if let own = audioObjectID(for: ProcessInfo.processInfo.processIdentifier) {
                excluded = [own]
            }
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: excluded)
        }
        description.name = "listn"
        description.isPrivate = true
        description.isExclusive = false
        // .unmuted keeps the meeting audible. .mutedWhenTapped would silence it.
        description.muteBehavior = .unmuted

        var newTap = noObject
        let tapStatus = AudioHardwareCreateProcessTap(description, &newTap)
        guard tapStatus == noErr, newTap != noObject else {
            throw CaptureError.tap(statusText(tapStatus))
        }
        tapID = newTap

        do {
            try createAggregate(tapUUID: description.uuid.uuidString)
            try installProc()
            running = true
            let startStatus = AudioDeviceStart(aggregateID, procID)
            guard startStatus == noErr else {
                throw CaptureError.io("AudioDeviceStart \(statusText(startStatus))")
            }
            if mic {
                do {
                    try startMicrophone()
                } catch {
                    onWarning?("Microphone not captured: \(error.localizedDescription). System audio is still recording.")
                }
            }
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        running = false
        if let micEngine {
            micEngine.inputNode.removeTap(onBus: 0)
            micEngine.stop()
            self.micEngine = nil
        }
        if let procID {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
            self.procID = nil
        }
        if aggregateID != noObject {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = noObject
        }
        if tapID != noObject {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = noObject
        }
        systemDB = -120
        micDB = -120
    }

    deinit { stop() }

    private func createAggregate(tapUUID: String) throws {
        // Tap auto-start is off. When it is on, AudioDeviceStart blocks until
        // some tapped app produces sound, which looks like a hung menu.
        let description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "listn",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: NSNumber(value: 1),
            kAudioAggregateDeviceIsStackedKey: NSNumber(value: 0),
            kAudioAggregateDeviceTapAutoStartKey: NSNumber(value: 0),
            kAudioAggregateDeviceSubDeviceListKey: [] as [[String: Any]],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUUID,
                    kAudioSubTapDriftCompensationKey: NSNumber(value: 1),
                ],
            ],
        ]
        var aggregate = noObject
        let status = AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregate)
        guard status == noErr, aggregate != noObject else {
            throw CaptureError.aggregate(statusText(status))
        }
        aggregateID = aggregate
        format = try tapFormat(tapID)
        guard format.mSampleRate > 0, format.mChannelsPerFrame > 0 else {
            throw CaptureError.format
        }
    }

    private func installProc() throws {
        var proc: AudioDeviceIOProcID?
        let status = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregateID, nil) { [weak self] _, inputData, _, _, _ in
            self?.consumeSystem(inputData)
        }
        guard status == noErr, let proc else {
            throw CaptureError.io("AudioDeviceCreateIOProcID \(statusText(status))")
        }
        procID = proc
    }

    private func startMicrophone() throws {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        if status == .notDetermined {
            let semaphore = DispatchSemaphore(value: 0)
            let granted = Locked(false)
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                granted.value = ok
                semaphore.signal()
            }
            semaphore.wait()
            if !granted.value {
                throw CaptureError.io("microphone permission denied")
            }
        } else if status == .denied || status == .restricted {
            throw CaptureError.io("microphone permission denied. Allow listn under System Settings → Privacy & Security → Microphone")
        }

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError.format
        }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            self?.consumeMicrophone(buffer)
        }
        engine.prepare()
        try engine.start()
        micEngine = engine
    }

    private func consumeSystem(_ list: UnsafePointer<AudioBufferList>) {
        guard running else { return }
        let channels = Int(format.mChannelsPerFrame)
        guard channels > 0 else { return }
        let flags = format.mFormatFlags
        guard flags & AudioFormatFlags(kAudioFormatFlagIsFloat) != 0 else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        guard !buffers.isEmpty else { return }
        let nonInterleaved = flags & AudioFormatFlags(kAudioFormatFlagIsNonInterleaved) != 0
        let samples: [Float]
        if nonInterleaved || buffers.count >= channels {
            let frames = Int(buffers[0].mDataByteSize) / MemoryLayout<Float>.size
            guard frames > 0 else { return }
            var interleaved = [Float](repeating: 0, count: frames * channels)
            for channel in 0..<min(channels, buffers.count) {
                guard let data = buffers[channel].mData else { continue }
                let source = data.assumingMemoryBound(to: Float.self)
                let count = min(frames, Int(buffers[channel].mDataByteSize) / MemoryLayout<Float>.size)
                for index in 0..<count {
                    interleaved[index * channels + channel] = source[index]
                }
            }
            samples = interleaved
        } else {
            guard let data = buffers[0].mData else { return }
            let count = Int(buffers[0].mDataByteSize) / MemoryLayout<Float>.size
            guard count > 0 else { return }
            samples = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: count))
        }
        noteLevel(track: 0, samples)
        onPCM?(0, format.mSampleRate, channels, samples)
    }

    private func consumeMicrophone(_ buffer: AVAudioPCMBuffer) {
        guard running, let channels = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return }
        var interleaved = [Float](repeating: 0, count: frameCount * channelCount)
        for channel in 0..<channelCount {
            let source = channels[channel]
            for index in 0..<frameCount {
                interleaved[index * channelCount + channel] = source[index]
            }
        }
        noteLevel(track: 1, interleaved)
        onPCM?(1, buffer.format.sampleRate, channelCount, interleaved)
    }

    private func noteLevel(track: Int, _ samples: [Float]) {
        let db = rmsDB(samples)
        levelLock.lock()
        if track == 0 { systemDB = db } else { micDB = db }
        let system = systemDB
        let mic = micDB
        let due = Date().timeIntervalSince(lastLevel) >= 0.1
        if due { lastLevel = Date() }
        levelLock.unlock()
        if due { onLevels?(system, mic) }
    }
}

private func rmsDB(_ samples: [Float]) -> Float {
    if samples.isEmpty { return -120 }
    var sum: Float = 0
    for sample in samples { sum += sample * sample }
    let mean = sum / Float(samples.count)
    if mean < 1e-12 { return -120 }
    return 20 * log10(sqrt(mean))
}

private func statusText(_ status: OSStatus) -> String {
    let code = UInt32(bitPattern: status)
    let bytes = [
        UInt8((code >> 24) & 0xff),
        UInt8((code >> 16) & 0xff),
        UInt8((code >> 8) & 0xff),
        UInt8(code & 0xff),
    ]
    if bytes.allSatisfy({ $0 >= 32 && $0 < 127 }) {
        let text = String(bytes: bytes, encoding: .ascii) ?? ""
        return "'\(text)' (\(status))"
    }
    return "OSStatus \(status)"
}

private func tapFormat(_ tap: AudioObjectID) throws -> AudioStreamBasicDescription {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var format = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    let status = AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format)
    guard status == noErr else { throw CaptureError.format }
    return format
}

private func audioObjectIDs(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr, size > 0 else { return [] }
    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    var ids = [AudioObjectID](repeating: 0, count: count)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

private func readUInt32(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
    return value
}
