import CoreAudio
import Foundation

/// What the tap listens to.
enum TapKind {
    /// Every process except listn, stereo mixdown. The production default.
    case globalStereo
    /// Same, mono mixdown.
    case globalMono
    /// Every process, with an empty exclude list.
    case globalNoExclude
    /// The configuration of 0.4.0 and 0.4.1: an exclude-list initializer
    /// with `isExclusive` then forced to false, which turns "everyone but
    /// listn" into "only listn". It always records zeros. Probe only.
    case legacyIncludeOnly
    /// Stereo mixdown of exactly these process objects.
    case processes([AudioObjectID])
    /// Every process playing to this output device (stream 0).
    case device(String)

    var name: String {
        switch self {
        case .globalStereo: return "global-stereo"
        case .globalMono: return "global-mono"
        case .globalNoExclude: return "global-no-exclude"
        case .legacyIncludeOnly: return "include-only-self"
        case .processes: return "processes"
        case .device: return "device"
        }
    }
}

struct TapConfig {
    var kind: TapKind
    /// Output device UID that clocks the aggregate and is its main
    /// sub-device. nil is a tap-only aggregate.
    var clockUID: String?
    /// kAudioAggregateDeviceTapAutoStartKey.
    var autoStart: Bool

    var id: String {
        var text = kind.name + (clockUID == nil ? "/tap-only" : "/clocked")
        if autoStart { text += "/autostart" }
        return text
    }

    /// The order a recording of all system audio tries. A strategy that
    /// starts but records only zeros while some app is playing is replaced
    /// by the next one; see CaptureEngine.
    static func productionCandidates(outputUID: String?, playing: [AudioObjectID]) -> [TapConfig] {
        var list = [TapConfig(kind: .globalStereo, clockUID: outputUID, autoStart: false)]
        if let outputUID {
            list.append(TapConfig(kind: .device(outputUID), clockUID: outputUID, autoStart: false))
        }
        if !playing.isEmpty {
            list.append(TapConfig(kind: .processes(playing), clockUID: outputUID, autoStart: false))
        }
        if let outputUID {
            list.append(TapConfig(kind: .globalStereo, clockUID: nil, autoStart: false))
            list.append(TapConfig(kind: .device(outputUID), clockUID: nil, autoStart: false))
        }
        return list
    }

    /// `preferred` is the id the self-test or an earlier recording found
    /// working. It goes first; the rest keep their order.
    static func systemStrategies(outputUID: String?, playing: [AudioObjectID], preferred: String?) -> [TapConfig] {
        var list = productionCandidates(outputUID: outputUID, playing: playing)
        if let preferred, let chosen = from(id: preferred, outputUID: outputUID, playing: playing) {
            list.removeAll { $0.id == chosen.id }
            list.insert(chosen, at: 0)
        }
        return list
    }

    static func from(id: String, outputUID: String?, playing: [AudioObjectID]) -> TapConfig? {
        let parts = id.split(separator: "/").map(String.init)
        guard parts.count >= 2 else { return nil }
        let clocked = parts[1] == "clocked"
        if clocked && outputUID == nil { return nil }
        let clock = clocked ? outputUID : nil
        let autoStart = parts.count > 2 && parts[2] == "autostart"
        let kind: TapKind
        switch parts[0] {
        case "global-stereo": kind = .globalStereo
        case "global-mono": kind = .globalMono
        case "global-no-exclude": kind = .globalNoExclude
        case "device":
            guard let outputUID else { return nil }
            kind = .device(outputUID)
        case "processes":
            guard !playing.isEmpty else { return nil }
            kind = .processes(playing)
        default:
            return nil
        }
        return TapConfig(kind: kind, clockUID: clock, autoStart: autoStart)
    }
}

/// Holds samples until the real rate of the stream is measured, then
/// releases them at that rate. A Bluetooth headset in call mode clocks the
/// aggregate at 16 kHz while the tap and the aggregate still report 48 kHz,
/// and the transcript would run three times too fast. HAL IO thread only.
final class StreamClock {
    let label: String
    private(set) var rate: Double
    private(set) var verified = false
    private let declared: Double
    private var startNanos: UInt64 = 0
    private var frames = 0
    private var pending: [Float] = []
    private static let verifySeconds = 1.0

    init(label: String, rate: Double) {
        self.label = label
        self.declared = rate
        self.rate = rate
    }

    func feed(_ samples: [Float], channels: Int) -> (samples: [Float], rate: Double)? {
        if verified { return (samples, rate) }
        let now = DispatchTime.now().uptimeNanoseconds
        pending.append(contentsOf: samples)
        if startNanos == 0 {
            startNanos = now
            return nil
        }
        frames += samples.count / max(1, channels)
        let elapsed = Double(now - startNanos) / 1_000_000_000
        guard elapsed >= StreamClock.verifySeconds else { return nil }
        verified = true
        let measured = Double(frames) / elapsed
        let snapped = StreamClock.nearestStandard(measured)
        if snapped > 0 && (rate <= 0 || abs(snapped - rate) / rate > 0.05) {
            rate = snapped
        }
        let name = label
        let declaredRate = declared
        let chosen = rate
        DispatchQueue.global(qos: .utility).async {
            CaptureLog.write(String(format: "%@ clock declared=%.0f Hz measured=%.0f Hz using=%.0f Hz",
                                    name, declaredRate, measured, chosen))
        }
        let released = pending
        pending = []
        return (released, rate)
    }

    static func nearestStandard(_ measured: Double) -> Double {
        guard measured > 0 else { return 0 }
        let rates: [Double] = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000]
        let best = rates.min { abs(log($0 / measured)) < abs(log($1 / measured)) } ?? measured
        return abs(best - measured) / measured < 0.08 ? best : measured.rounded()
    }
}

/// One process tap, its private aggregate device, and the IO proc that reads it.
final class SystemTap {
    let config: TapConfig
    /// Interleaved samples, sample rate, channel count. HAL IO thread.
    var onSamples: (([Float], Double, Int) -> Void)?

    private var tapID = noObject
    private var aggregateID = noObject
    private var procID: AudioDeviceIOProcID?
    private var format = AudioStreamBasicDescription()
    private var tapChannels = 2
    private var clock = StreamClock(label: "system", rate: 48000)
    private var loggedLayout = false

    init(config: TapConfig) {
        self.config = config
    }

    deinit { stop() }

    func start() throws {
        do {
            try openTap()
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
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

    private func openTap() throws {
        let id = config.id
        let (description, detail) = makeDescription()
        var tap = noObject
        let status = AudioHardwareCreateProcessTap(description, &tap)
        CaptureLog.write("tap \(id) create \(statusText(status)) \(detail) isExclusive=\(description.isExclusive)")
        guard status == noErr, tap != noObject else {
            throw CaptureError.tap(statusText(status))
        }
        tapID = tap

        let uuid = description.uuid.uuidString
        let reported = AudioHAL.stringProperty(tap, kAudioTapPropertyUID)
        let match = reported.map { $0.caseInsensitiveCompare(uuid) == .orderedSame }
        CaptureLog.write("tap \(id) uid description=\(uuid) object=\(reported ?? "unavailable") match=\(match.map { $0 ? "yes" : "no" } ?? "?")")

        guard let tapASBD = AudioHAL.tapFormat(tap), tapASBD.mSampleRate > 0, tapASBD.mChannelsPerFrame > 0 else {
            throw CaptureError.format
        }
        CaptureLog.write("tap \(id) format \(AudioHAL.describe(tapASBD))")
        tapChannels = Int(tapASBD.mChannelsPerFrame)
        format = tapASBD

        let aggregateStatus = createAggregate(tapUUID: uuid)
        guard aggregateStatus == noErr, aggregateID != noObject else {
            throw CaptureError.aggregate(statusText(aggregateStatus))
        }
        let nominal = AudioHAL.nominalRate(aggregateID)
        if let aggregateFormat = AudioHAL.streamFormat(aggregateID, scope: kAudioDevicePropertyScopeInput) {
            format = aggregateFormat
            CaptureLog.write("tap \(id) aggregate id=\(aggregateID) nominal=\(Int(nominal ?? 0)) Hz input \(AudioHAL.describe(aggregateFormat)) inputChannels=\(AudioHAL.channelCount(aggregateID, scope: kAudioDevicePropertyScopeInput))")
        }
        let declared = (nominal ?? 0) > 0 ? (nominal ?? 0) : format.mSampleRate
        clock = StreamClock(label: "system \(id)", rate: declared)

        var proc: AudioDeviceIOProcID?
        let created = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregateID, nil) { [weak self] _, inputData, _, _, _ in
            self?.consume(inputData)
        }
        guard created == noErr, let proc else {
            throw CaptureError.io("AudioDeviceCreateIOProcID \(statusText(created))")
        }
        procID = proc
        let started = AudioDeviceStart(aggregateID, proc)
        CaptureLog.write("tap \(id) AudioDeviceStart \(statusText(started))")
        guard started == noErr else {
            throw CaptureError.io("AudioDeviceStart \(statusText(started))")
        }
    }

    private func makeDescription() -> (CATapDescription, String) {
        let own = AudioProcesses.ownObject()
        let exclude: [AudioObjectID] = own.map { [$0] } ?? []
        let excludeText = own.map { "exclude=[\($0)]" } ?? "exclude=[] (listn has no process object yet)"
        let description: CATapDescription
        let detail: String
        switch config.kind {
        case .globalStereo:
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: exclude)
            detail = "stereo global " + excludeText
        case .globalMono:
            description = CATapDescription(monoGlobalTapButExcludeProcesses: exclude)
            detail = "mono global " + excludeText
        case .globalNoExclude:
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: [AudioObjectID]())
            detail = "stereo global exclude=[]"
        case .legacyIncludeOnly:
            description = CATapDescription(stereoGlobalTapButExcludeProcesses: exclude)
            description.isExclusive = false
            detail = "stereo global " + excludeText + " then isExclusive=false (old builds)"
        case .processes(let ids):
            description = CATapDescription(stereoMixdownOfProcesses: ids)
            detail = "stereo mixdown of processes \(ids)"
        case .device(let uid):
            description = CATapDescription(processes: [AudioObjectID](), deviceUID: uid, stream: 0)
            // An empty list with isExclusive is "exclude nobody": every
            // process playing to this device.
            description.isExclusive = true
            detail = "device uid=\(uid) stream=0"
        }
        description.name = "listn"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        return (description, detail)
    }

    private func createAggregate(tapUUID: String) -> OSStatus {
        var description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "listn",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: NSNumber(value: 1),
            kAudioAggregateDeviceIsStackedKey: NSNumber(value: 0),
            kAudioAggregateDeviceTapAutoStartKey: NSNumber(value: config.autoStart ? 1 : 0),
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUUID,
                    kAudioSubTapDriftCompensationKey: NSNumber(value: 1),
                ],
            ],
        ]
        if let clockUID = config.clockUID {
            description[kAudioAggregateDeviceMainSubDeviceKey] = clockUID
            description[kAudioAggregateDeviceClockDeviceKey] = clockUID
            description[kAudioAggregateDeviceSubDeviceListKey] = [
                [
                    kAudioSubDeviceUIDKey: clockUID,
                    kAudioSubDeviceDriftCompensationKey: NSNumber(value: 0),
                ],
            ]
        } else {
            description[kAudioAggregateDeviceSubDeviceListKey] = [] as [[String: Any]]
        }
        var aggregate = noObject
        let status = AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregate)
        CaptureLog.write("tap \(config.id) aggregate \(config.clockUID.map { "clocked to \($0)" } ?? "tap-only") autostart=\(config.autoStart) \(statusText(status))")
        if status == noErr, aggregate != noObject {
            aggregateID = aggregate
        }
        return status
    }

    private func consume(_ list: UnsafePointer<AudioBufferList>) {
        // The tap's channels come after any input channels of the clock
        // device. Count what actually arrived rather than trusting the
        // reported format.
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        var total = 0
        for buffer in buffers { total += Int(buffer.mNumberChannels) }
        let drop = total > tapChannels ? total - tapChannels : 0
        if !loggedLayout {
            loggedLayout = true
            let report = AudioHAL.bufferReport(list)
            let id = config.id
            let take = tapChannels
            DispatchQueue.global(qos: .utility).async {
                CaptureLog.write("tap \(id) first buffers channels=\(total) drop=\(drop) take=\(take) \(report)")
            }
        }
        let channels = max(1, tapChannels)
        let samples = AudioHAL.readFloat(list, format: format, dropChannels: drop, takeChannels: channels)
        guard !samples.isEmpty, let timed = clock.feed(samples, channels: channels) else { return }
        onSamples?(timed.samples, timed.rate, channels)
    }
}
