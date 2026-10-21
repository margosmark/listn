import AppKit
import CoreAudio
import Foundation

// System audio is a Core Audio process tap (macOS 14.2+). The tap is placed
// in a private aggregate device whose clock is the default output, including
// an HDMI or DisplayPort monitor that has no volume control. Without that
// clock the IO proc runs and every sample is zero.
//
// The microphone is its own AudioDeviceIOProc on the default input device.
// It does not use AVAudioEngine and it does not share the tap aggregate, so
// a silent tap cannot zero the mic (or the other way around).

private let systemObject = AudioObjectID(kAudioObjectSystemObject)
private let noObject = AudioObjectID(kAudioObjectUnknown)

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
            return "Could not start audio capture (\(detail))."
        case .format:
            return "The audio device did not report a usable format."
        case .noProcess(let name):
            return "\(name) is not playing audio yet. Start the meeting, then pick it again."
        }
    }
}

struct StreamMeter {
    var peak: Float = 0
    var sum: Float = 0
    var count: Int = 0
    var callbacks: Int = 0

    var rmsDB: Float {
        guard count > 0 else { return -120 }
        let mean = sum / Float(count)
        if mean < 1e-12 { return -120 }
        return 20 * log10(sqrt(mean))
    }

    mutating func add(_ samples: [Float]) {
        callbacks += 1
        count += samples.count
        for sample in samples {
            let value = abs(sample)
            if value > peak { peak = value }
            sum += sample * sample
        }
    }
}

final class CaptureEngine {
    var onPCM: ((UInt8, Double, Int, [Float]) -> Void)?
    /// Microphone level is nil when this recording is not capturing the mic,
    /// so a system-only session is not reported as a silent microphone.
    var onLevels: ((Float, Float?) -> Void)?
    var onWarning: ((String) -> Void)?
    var onDeviceChange: (() -> Void)?
    /// Fired once a track has delivered only zeros. The menu also watches
    /// levels, so a missed callback still surfaces.
    var onSilent: ((PrivacyPane) -> Void)?
    private let silenceWindow: Double = 2.5
    private var heard: [Int: Double] = [:]
    private var peaks: [Int: Float] = [:]
    private var reported: Set<Int> = []

    private var tapID = noObject
    private var aggregateID = noObject
    private var procID: AudioDeviceIOProcID?
    private var format = AudioStreamBasicDescription()
    private var dropChannels = 0
    private var tapChannels = 2
    private var loggedSystemLayout = false

    private var micDevice = noObject
    private var micProc: AudioDeviceIOProcID?
    private var microphoneOpen = false
    private var micFormat = AudioStreamBasicDescription()

    private(set) var openedOutputUID: String?
    private(set) var openedInputUID: String?
    private var running = false
    private var watchingDevices = false
    private var systemDB: Float = -120
    private var micDB: Float = -120
    private var systemMeter = StreamMeter()
    private var micMeter = StreamMeter()
    private var lastLevel = Date.distantPast
    private var loggedFirst: Set<Int> = []
    private var firstPeak: [Int: Float] = [:]
    private var firstSeconds: [Int: Double] = [:]
    private let levelLock = NSLock()

    /// Starts whichever side it can. A failure of the tap does not stop the
    /// microphone, and a failure of the microphone does not stop the tap.
    func start(appPID: pid_t?, appName: String?, mic: Bool) throws {
        stop()
        running = true
        openedOutputUID = AudioHAL.defaultOutput().flatMap(AudioHAL.uid)
        openedInputUID = AudioHAL.defaultInput().flatMap(AudioHAL.uid)
        if let output = AudioHAL.defaultOutput(), let input = AudioHAL.defaultInput(), output == input {
            CaptureLog.write("default input and output are the same device; mic IOProc and the tap clock share it")
        }
        if let output = AudioHAL.defaultOutput() {
            CaptureLog.write("default output \(AudioHAL.describe(output)) display=\(AudioHAL.isDisplayOutput(output))")
        } else {
            CaptureLog.write("default output missing")
        }
        if let input = AudioHAL.defaultInput() {
            CaptureLog.write("default input \(AudioHAL.describe(input))")
        } else {
            CaptureLog.write("default input missing")
        }

        var problems: [String] = []
        if mic {
            do {
                try startMicrophone()
            } catch {
                problems.append("Microphone: \(error.localizedDescription)")
                CaptureLog.write("mic failed: \(error.localizedDescription)")
            }
        }
        do {
            try startSystem(appPID: appPID, appName: appName)
        } catch {
            problems.append("System audio: \(error.localizedDescription)")
            CaptureLog.write("system tap failed: \(error.localizedDescription)")
            teardownSystem()
        }
        if micProc == nil && procID == nil {
            running = false
            throw CaptureError.io(problems.joined(separator: " "))
        }
        if !problems.isEmpty {
            let message = problems.joined(separator: " ")
            DispatchQueue.main.async { [weak self] in self?.onWarning?(message) }
        }
        CaptureLog.write("capture running mic=\(micProc != nil) system=\(procID != nil)")
    }

    func stop() {
        running = false
        levelLock.lock()
        heard = [:]
        peaks = [:]
        reported = []
        loggedFirst = []
        firstPeak = [:]
        firstSeconds = [:]
        systemMeter = StreamMeter()
        micMeter = StreamMeter()
        levelLock.unlock()
        teardownMicrophone()
        teardownSystem()
        systemDB = -120
        micDB = -120
        loggedSystemLayout = false
    }

    /// Restart when the default input or output actually changes. Creating the
    /// private aggregate must not count: headphones unplugging changes the
    /// default output UID, and that does.
    func watchDevices() {
        guard !watchingDevices else { return }
        watchingDevices = true
        observe(kAudioHardwarePropertyDefaultOutputDevice)
        observe(kAudioHardwarePropertyDefaultInputDevice)
    }

    func takeMeters() -> (system: StreamMeter, mic: StreamMeter) {
        levelLock.lock()
        let snapshot = (systemMeter, micMeter)
        systemMeter = StreamMeter()
        micMeter = StreamMeter()
        levelLock.unlock()
        return snapshot
    }

    deinit { stop() }

    private func observe(_ selector: AudioObjectPropertySelector) {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        AudioObjectAddPropertyListenerBlock(systemObject, &address, DispatchQueue.main) { [weak self] _, _ in
            guard let self, self.running else { return }
            let output = AudioHAL.defaultOutput().flatMap(AudioHAL.uid)
            let input = AudioHAL.defaultInput().flatMap(AudioHAL.uid)
            let outputChanged = self.openedOutputUID != nil && output != self.openedOutputUID
            let inputChanged = self.openedInputUID != nil && input != self.openedInputUID
            guard outputChanged || inputChanged else { return }
            CaptureLog.write("default device changed output \(self.openedOutputUID ?? "-") -> \(output ?? "-") input \(self.openedInputUID ?? "-") -> \(input ?? "-")")
            self.onDeviceChange?()
        }
    }

    private func startSystem(appPID: pid_t?, appName: String?) throws {
        if let appPID, audioObjectID(for: appPID) == nil {
            throw CaptureError.noProcess(appName ?? "That app")
        }
        let output = AudioHAL.defaultOutput()
        let outputUID = output.flatMap(AudioHAL.uid)
        let display = output.map(AudioHAL.isDisplayOutput) ?? false
        let (description, tap) = try createTap(appPID: appPID, appName: appName, outputUID: outputUID, display: display)
        tapID = tap
        let tapASBD = try tapFormat(tap)
        CaptureLog.write("tap format \(AudioHAL.describe(tapASBD))")
        guard tapASBD.mSampleRate > 0, tapASBD.mChannelsPerFrame > 0 else { throw CaptureError.format }
        tapChannels = Int(tapASBD.mChannelsPerFrame)
        format = tapASBD

        try createAggregate(tapUUID: description.uuid.uuidString, clockUID: outputUID)
        if let aggregateFormat = AudioHAL.streamFormat(aggregateID, scope: kAudioDevicePropertyScopeInput) {
            CaptureLog.write("aggregate input format \(AudioHAL.describe(aggregateFormat))")
            format = aggregateFormat
            let aggregateChannels = Int(aggregateFormat.mChannelsPerFrame)
            dropChannels = aggregateChannels > tapChannels ? aggregateChannels - tapChannels : 0
        }
        CaptureLog.write("reading tap channels=\(tapChannels) dropping leading channels=\(dropChannels)")
        try installProc()
        let startStatus = AudioDeviceStart(aggregateID, procID)
        guard startStatus == noErr else {
            throw CaptureError.io("AudioDeviceStart \(statusText(startStatus))")
        }
        CaptureLog.write("system tap started \(statusText(startStatus))")
        if let after = AudioHAL.streamFormat(aggregateID, scope: kAudioDevicePropertyScopeInput) {
            format = after
            let aggregateChannels = Int(after.mChannelsPerFrame)
            dropChannels = aggregateChannels > tapChannels ? aggregateChannels - tapChannels : 0
            CaptureLog.write("aggregate input format after start \(AudioHAL.describe(after)) drop=\(dropChannels)")
        }
    }

    private func createTap(appPID: pid_t?, appName: String?, outputUID: String?, display: Bool) throws -> (CATapDescription, AudioObjectID) {
        // An app tap stays a process mixdown. A display output with no chosen
        // app is tapped on that device, because a global stereo tap returns
        // zeros for some HDMI and DisplayPort monitors.
        if appPID == nil, display, let outputUID {
            let description = CATapDescription(processes: [AudioObjectID](), deviceUID: outputUID, stream: 0)
            description.name = "listn"
            description.isPrivate = true
            description.isExclusive = true
            description.muteBehavior = .unmuted
            var tap = noObject
            let status = AudioHardwareCreateProcessTap(description, &tap)
            CaptureLog.write("display-output device tap \(statusText(status)) uid=\(outputUID)")
            if status == noErr, tap != noObject {
                return (description, tap)
            }
            CaptureLog.write("device tap failed; falling back to a global stereo tap")
        }
        let description = globalTap(appPID: appPID, appName: appName)
        var tap = noObject
        let status = AudioHardwareCreateProcessTap(description, &tap)
        CaptureLog.write("global tap \(statusText(status))")
        guard status == noErr, tap != noObject else {
            throw CaptureError.tap(statusText(status))
        }
        return (description, tap)
    }

    private func globalTap(appPID: pid_t?, appName: String?) -> CATapDescription {
        _ = appName
        let description: CATapDescription
        if let appPID, let objectID = audioObjectID(for: appPID) {
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
        description.muteBehavior = .unmuted
        return description
    }

    private func createAggregate(tapUUID: String, clockUID: String?) throws {
        if let clockUID {
            let status = installAggregate(tapUUID: tapUUID, clockUID: clockUID)
            if status == noErr {
                CaptureLog.write("aggregate clocked to \(clockUID) \(statusText(status)) id=\(aggregateID)")
                return
            }
            CaptureLog.write("clocked aggregate failed \(statusText(status)); retrying tap-only")
            if aggregateID != noObject {
                AudioHardwareDestroyAggregateDevice(aggregateID)
                aggregateID = noObject
            }
        }
        let status = installAggregate(tapUUID: tapUUID, clockUID: nil)
        guard status == noErr, aggregateID != noObject else {
            throw CaptureError.aggregate(statusText(status))
        }
        CaptureLog.write("aggregate tap-only \(statusText(status)) id=\(aggregateID)")
    }

    private func installAggregate(tapUUID: String, clockUID: String?) -> OSStatus {
        var description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "listn",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: NSNumber(value: 1),
            kAudioAggregateDeviceIsStackedKey: NSNumber(value: 0),
            // Off: when it is on, AudioDeviceStart waits until some app plays,
            // which looks like a hung menu. The clock device keeps the IO proc running.
            kAudioAggregateDeviceTapAutoStartKey: NSNumber(value: 0),
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUUID,
                    kAudioSubTapDriftCompensationKey: NSNumber(value: 1),
                ],
            ],
        ]
        if let clockUID {
            description[kAudioAggregateDeviceMainSubDeviceKey] = clockUID
            description[kAudioAggregateDeviceClockDeviceKey] = clockUID
            description[kAudioAggregateDeviceSubDeviceListKey] = [
                [
                    kAudioSubDeviceUIDKey: clockUID,
                    // The clock itself is not drift-compensated.
                    kAudioSubDeviceDriftCompensationKey: NSNumber(value: 0),
                ],
            ]
        } else {
            description[kAudioAggregateDeviceSubDeviceListKey] = [] as [[String: Any]]
        }
        var aggregate = noObject
        let status = AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregate)
        if status == noErr, aggregate != noObject {
            aggregateID = aggregate
        }
        return status
    }

    private func installProc() throws {
        let capturedFormat = format
        var proc: AudioDeviceIOProcID?
        let status = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregateID, nil) { [weak self] _, inputData, _, _, _ in
            self?.consumeSystem(inputData, format: capturedFormat)
        }
        guard status == noErr, let proc else {
            throw CaptureError.io("AudioDeviceCreateIOProcID \(statusText(status))")
        }
        procID = proc
    }

    private func startMicrophone() throws {
        guard let device = AudioHAL.defaultInput() else {
            throw CaptureError.io("no default input device")
        }
        guard let deviceFormat = AudioHAL.streamFormat(device, scope: kAudioDevicePropertyScopeInput) else {
            throw CaptureError.format
        }
        guard deviceFormat.mSampleRate > 0, deviceFormat.mChannelsPerFrame > 0 else {
            throw CaptureError.format
        }
        micDevice = device
        micFormat = deviceFormat
        CaptureLog.write("mic IOProc \(AudioHAL.describe(device)) format \(AudioHAL.describe(deviceFormat))")
        let captured = deviceFormat
        var proc: AudioDeviceIOProcID?
        let created = AudioDeviceCreateIOProcIDWithBlock(&proc, device, nil) { [weak self] _, inputData, _, _, _ in
            self?.consumeMicrophone(inputData, format: captured)
        }
        guard created == noErr, let proc else {
            micDevice = noObject
            throw CaptureError.io("mic IOProc \(statusText(created))")
        }
        micProc = proc
        let started = AudioDeviceStart(device, proc)
        guard started == noErr else {
            AudioDeviceDestroyIOProcID(device, proc)
            micProc = nil
            micDevice = noObject
            throw CaptureError.io("mic AudioDeviceStart \(statusText(started))")
        }
        microphoneOpen = true
        CaptureLog.write("mic started \(statusText(started))")
    }

    private func consumeSystem(_ list: UnsafePointer<AudioBufferList>, format captured: AudioStreamBasicDescription) {
        guard running else { return }
        let active = format.mSampleRate > 0 ? format : captured
        // The tap is appended after the clock device's input buffers. On an
        // HDMI output those leading buffers are often absent; when they exist
        // they are silence and must not be what we record.
        var drop = dropChannels
        let listed = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        let nonInterleaved = active.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        if nonInterleaved {
            drop = listed.count > tapChannels ? listed.count - tapChannels : 0
        }
        if !loggedSystemLayout {
            loggedSystemLayout = true
            let report = AudioHAL.bufferReport(list)
            let take = tapChannels
            let dropped = drop
            DispatchQueue.global(qos: .utility).async {
                CaptureLog.write("system first buffers drop=\(dropped) take=\(take) \(report)")
            }
        }
        let samples = AudioHAL.readFloat(list, format: active, dropChannels: drop, takeChannels: tapChannels)
        deliver(track: 0, samples: samples, rate: active.mSampleRate, channels: max(1, tapChannels))
    }

    private func consumeMicrophone(_ list: UnsafePointer<AudioBufferList>, format: AudioStreamBasicDescription) {
        guard running else { return }
        let channels = max(1, Int(format.mChannelsPerFrame))
        let samples = AudioHAL.readFloat(list, format: format, dropChannels: 0, takeChannels: channels)
        deliver(track: 1, samples: samples, rate: format.mSampleRate, channels: channels)
    }

    private func deliver(track: Int, samples: [Float], rate: Double, channels: Int) {
        guard !samples.isEmpty, rate > 0 else { return }
        noteLevel(track: track, samples)
        let seconds = Double(samples.count / max(1, channels)) / rate
        watchSilence(track: track, samples, seconds: seconds)
        noteFirstSecond(track: track, samples: samples, seconds: seconds)
        onPCM?(UInt8(track), rate, channels, samples)
    }

    private func watchSilence(track: Int, _ samples: [Float], seconds: Double) {
        var peak: Float = 0
        for sample in samples {
            let value = abs(sample)
            if value > peak { peak = value }
        }
        levelLock.lock()
        if reported.contains(track) {
            levelLock.unlock()
            return
        }
        heard[track, default: 0] += seconds
        peaks[track] = max(peaks[track] ?? 0, peak)
        let due = heard[track, default: 0] >= silenceWindow && (peaks[track] ?? 0) <= 1e-5
        if due { reported.insert(track) }
        let heardSeconds = heard[track] ?? 0
        let peakNow = peaks[track] ?? 0
        levelLock.unlock()
        guard due else { return }
        let name = track == 0 ? "system" : "mic"
        DispatchQueue.global(qos: .utility).async {
            CaptureLog.write(String(format: "watchdog %@ peak=%.6g heard=%.2fs", name, peakNow, heardSeconds))
        }
        onSilent?(track == 0 ? .audioCapture : .microphone)
    }

    private func noteFirstSecond(track: Int, samples: [Float], seconds: Double) {
        var peak: Float = 0
        for sample in samples {
            peak = max(peak, abs(sample))
        }
        levelLock.lock()
        firstPeak[track] = max(firstPeak[track] ?? 0, peak)
        firstSeconds[track, default: 0] += seconds
        let ready = (firstSeconds[track] ?? 0) >= 1 && !loggedFirst.contains(track)
        if ready { loggedFirst.insert(track) }
        let peakNow = firstPeak[track] ?? 0
        let secs = firstSeconds[track] ?? 0
        levelLock.unlock()
        guard ready else { return }
        let name = track == 0 ? "system" : "mic"
        DispatchQueue.global(qos: .utility).async {
            CaptureLog.write(String(format: "first second %@ peak=%.6g after %.2fs", name, peakNow, secs))
        }
    }

    private func noteLevel(track: Int, _ samples: [Float]) {
        let db = rmsDB(samples)
        levelLock.lock()
        if track == 0 {
            systemDB = db
            systemMeter.add(samples)
        } else {
            micDB = db
            micMeter.add(samples)
        }
        let system = systemDB
        let mic = micDB
        let due = Date().timeIntervalSince(lastLevel) >= 0.1
        if due { lastLevel = Date() }
        let micLevel: Float? = microphoneOpen ? mic : nil
        levelLock.unlock()
        if due { onLevels?(system, micLevel) }
    }

    private func teardownMicrophone() {
        if let micProc {
            AudioDeviceStop(micDevice, micProc)
            AudioDeviceDestroyIOProcID(micDevice, micProc)
            self.micProc = nil
        }
        micDevice = noObject
        microphoneOpen = false
    }

    private func teardownSystem() {
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
