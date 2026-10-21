import AppKit
import CoreAudio
import Foundation

// System audio is a Core Audio process tap (macOS 14.2+) in a private
// aggregate device clocked by the default output. SystemTap builds one
// configuration; this engine tries them in order and, while some app is
// playing, replaces one that records only zeros with the next.
//
// The microphone is its own AudioDeviceIOProc on the chosen input device.
// It does not use AVAudioEngine and it does not share the tap aggregate, so
// a silent tap cannot zero the mic (or the other way around).

enum Prefs {
    /// Record the Mac's own microphone instead of the default input. A
    /// Bluetooth headset drops to call quality while its mic is open.
    static let useBuiltInMic = "useBuiltInMic"
    /// Id of the tap configuration that last delivered audio (TapConfig.id).
    static let tapStrategy = "systemTapStrategy"
}

struct AudioApp {
    var objectID: AudioObjectID
    var pid: pid_t
    var name: String
    var bundleID: String
}

func listAudioApps() -> [AudioApp] {
    let selfPID = ProcessInfo.processInfo.processIdentifier
    return AudioProcesses.all()
        .filter { $0.pid != selfPID }
        .map { AudioApp(objectID: $0.objectID, pid: $0.pid, name: $0.name, bundleID: $0.bundleID) }
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

struct LevelReport {
    var system: Float
    /// nil when this recording is not capturing the mic, so a system-only
    /// session is not reported as a silent microphone.
    var mic: Float?
    /// Other apps running audio output right now. nil when Core Audio does
    /// not say; an empty list means nothing is playing, so silence is expected.
    var playing: [String]?
}

final class CaptureEngine {
    var onPCM: ((UInt8, Double, Int, [Float]) -> Void)?
    var onLevels: ((LevelReport) -> Void)?
    var onWarning: ((String) -> Void)?
    var onDeviceChange: (() -> Void)?
    /// Fired once a track has delivered only zeros (for system audio: while
    /// some app was playing). The menu also watches levels.
    var onSilent: ((PrivacyPane) -> Void)?

    private let silenceWindow: Double = 2.5
    private var heard: [Int: Double] = [:]
    private var peaks: [Int: Float] = [:]
    private var reported: Set<Int> = []

    // System tap. Guarded by systemLock.
    private let systemLock = NSRecursiveLock()
    private var systemTap: SystemTap?
    private var strategies: [TapConfig] = []
    private var strategyIndex = 0
    private var systemWide = false
    private var exhausted = false

    // Guarded by levelLock.
    private var strategyPeak: Float = 0
    private var strategySeconds: Double = 0
    private var playingNow: [String]?

    // monitorQueue only.
    private let monitorQueue = DispatchQueue(label: "local.listn.capture.monitor")
    private var monitor: DispatchSourceTimer?
    private var monitorTicks = 0
    private var activeSilentTicks = 0
    private var confirmed = false

    private var micDevice = noObject
    private var micProc: AudioDeviceIOProcID?
    private var microphoneOpen = false

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

    /// The input the mic IOProc opens. `followsDefault` is false when the
    /// built-in mic is chosen over a different default input.
    static func chooseMicrophone() -> (device: AudioObjectID, followsDefault: Bool)? {
        let fallback = AudioHAL.defaultInput()
        if UserDefaults.standard.bool(forKey: Prefs.useBuiltInMic), let builtIn = AudioHAL.builtInInput() {
            return (builtIn, fallback == builtIn)
        }
        guard let fallback else { return nil }
        return (fallback, true)
    }

    /// Starts whichever side it can. A failure of the tap does not stop the
    /// microphone, and a failure of the microphone does not stop the tap.
    func start(appPID: pid_t?, appName: String?, mic: Bool) throws {
        stop()
        running = true
        monitorQueue.sync {
            monitorTicks = 0
            activeSilentTicks = 0
            confirmed = false
        }
        let output = AudioHAL.defaultOutput()
        openedOutputUID = output.flatMap(AudioHAL.uid)
        openedInputUID = nil
        if let output {
            CaptureLog.write("default output \(AudioHAL.describe(output)) bluetooth=\(AudioHAL.isBluetooth(output)) display=\(AudioHAL.isDisplayOutput(output))")
        } else {
            CaptureLog.write("default output missing")
        }
        if let input = AudioHAL.defaultInput() {
            CaptureLog.write("default input \(AudioHAL.describe(input)) bluetooth=\(AudioHAL.isBluetooth(input))")
        } else {
            CaptureLog.write("default input missing")
        }
        let processes = AudioProcesses.all()
        let own = ProcessInfo.processInfo.processIdentifier
        let playing = AudioProcesses.playing(in: processes, excludingPID: own)
        levelLock.lock()
        playingNow = playing.map { list in list.map(\.label) }
        levelLock.unlock()
        logProcesses(processes, reason: "start")

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
            try startSystem(appPID: appPID, appName: appName, playing: playing ?? [])
        } catch {
            problems.append("System audio: \(error.localizedDescription)")
            CaptureLog.write("system tap failed: \(error.localizedDescription)")
        }
        let systemOpen = systemRunning
        if micProc == nil && !systemOpen {
            running = false
            throw CaptureError.io(problems.joined(separator: " "))
        }
        if !problems.isEmpty {
            let message = problems.joined(separator: " ")
            DispatchQueue.main.async { [weak self] in self?.onWarning?(message) }
        }
        startMonitor()
        CaptureLog.write("capture running mic=\(micProc != nil) system=\(systemOpen)")
    }

    func stop() {
        running = false
        stopMonitor()
        levelLock.lock()
        heard = [:]
        peaks = [:]
        reported = []
        loggedFirst = []
        firstPeak = [:]
        firstSeconds = [:]
        systemMeter = StreamMeter()
        micMeter = StreamMeter()
        strategyPeak = 0
        strategySeconds = 0
        playingNow = nil
        levelLock.unlock()
        teardownMicrophone()
        systemLock.lock()
        systemTap?.stop()
        systemTap = nil
        strategies = []
        exhausted = false
        systemLock.unlock()
        systemDB = -120
        micDB = -120
    }

    /// Closes the system tap and keeps the microphone open. The self-test
    /// uses this to probe other tap configurations under the same conditions
    /// (a Bluetooth headset stays in call mode while its mic is open).
    func stopSystem() {
        systemLock.lock()
        systemTap?.stop()
        systemTap = nil
        systemLock.unlock()
        CaptureLog.write("system tap closed; microphone stays open")
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

    func playingNames() -> [String]? {
        levelLock.lock()
        defer { levelLock.unlock() }
        return playingNow
    }

    var currentStrategy: String? {
        systemLock.lock()
        defer { systemLock.unlock() }
        return systemTap?.config.id
    }

    deinit { stop() }

    private var systemRunning: Bool {
        systemLock.lock()
        defer { systemLock.unlock() }
        return systemTap != nil
    }

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

    // MARK: System tap

    private func startSystem(appPID: pid_t?, appName: String?, playing: [AudioProcess]) throws {
        systemLock.lock()
        defer { systemLock.unlock() }
        let outputUID = AudioHAL.defaultOutput().flatMap(AudioHAL.uid)
        if let appPID {
            guard let object = audioObjectID(for: appPID) else {
                throw CaptureError.noProcess(appName ?? "That app")
            }
            systemWide = false
            var list = [TapConfig(kind: .processes([object]), clockUID: outputUID, autoStart: false)]
            if outputUID != nil {
                list.append(TapConfig(kind: .processes([object]), clockUID: nil, autoStart: false))
            }
            strategies = list
        } else {
            systemWide = true
            let preferred = UserDefaults.standard.string(forKey: Prefs.tapStrategy)
            strategies = TapConfig.systemStrategies(outputUID: outputUID, playing: playing.map(\.objectID), preferred: preferred)
            CaptureLog.write("system tap strategies \(strategies.map(\.id)) saved=\(preferred ?? "none")")
        }
        var lastError: Error = CaptureError.tap("no tap configuration")
        for index in strategies.indices {
            do {
                try openStrategy(index)
                return
            } catch {
                lastError = error
                CaptureLog.write("tap \(strategies[index].id) failed: \(error.localizedDescription)")
            }
        }
        throw lastError
    }

    /// Caller holds systemLock.
    private func openStrategy(_ index: Int) throws {
        systemTap?.stop()
        systemTap = nil
        var config = strategies[index]
        if systemWide, case .processes = config.kind {
            let ids = (AudioProcesses.playingNow() ?? []).map(\.objectID)
            guard !ids.isEmpty else { throw CaptureError.tap("no app is playing audio") }
            config.kind = .processes(ids)
        }
        let tap = SystemTap(config: config)
        tap.onSamples = { [weak self] samples, rate, channels in
            self?.deliver(track: 0, samples: samples, rate: rate, channels: channels)
        }
        try tap.start()
        systemTap = tap
        strategyIndex = index
        levelLock.lock()
        strategyPeak = 0
        strategySeconds = 0
        levelLock.unlock()
        CaptureLog.write("system tap running with \(config.id) (\(index + 1) of \(strategies.count))")
    }

    private func startMonitor() {
        let timer = DispatchSource.makeTimerSource(queue: monitorQueue)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in
            self?.monitorTick()
        }
        monitor = timer
        timer.resume()
    }

    private func stopMonitor() {
        monitor?.cancel()
        monitor = nil
    }

    /// Once a second: who is playing, and whether the tap hears them.
    private func monitorTick() {
        guard running else { return }
        let processes = AudioProcesses.all()
        let playing = AudioProcesses.playing(in: processes, excludingPID: ProcessInfo.processInfo.processIdentifier)
        levelLock.lock()
        playingNow = playing.map { list in list.map(\.label) }
        let peak = strategyPeak
        let seconds = strategySeconds
        levelLock.unlock()
        monitorTicks += 1
        if monitorTicks % 5 == 0 {
            logProcesses(processes, reason: "\(monitorTicks)s")
        }
        if peak > 1e-5 {
            activeSilentTicks = 0
            if !confirmed {
                confirmed = true
                rememberWorkingStrategy()
            }
            return
        }
        guard !confirmed, seconds > 0, let active = playing, !active.isEmpty else {
            activeSilentTicks = 0
            return
        }
        activeSilentTicks += 1
        if activeSilentTicks >= 3 {
            activeSilentTicks = 0
            let names = active.map(\.label).joined(separator: ", ")
            advanceStrategy(reason: "only zeros for 3s while \(names) played")
        }
    }

    private func rememberWorkingStrategy() {
        systemLock.lock()
        let id = systemTap?.config.id
        let wide = systemWide
        systemLock.unlock()
        guard let id else { return }
        CaptureLog.write("system tap \(id) is delivering audio")
        if wide, UserDefaults.standard.string(forKey: Prefs.tapStrategy) != id {
            UserDefaults.standard.set(id, forKey: Prefs.tapStrategy)
            CaptureLog.write("saved \(id) as the system tap strategy")
        }
    }

    private func advanceStrategy(reason: String) {
        systemLock.lock()
        defer { systemLock.unlock() }
        guard running, !exhausted, let current = systemTap else { return }
        CaptureLog.write("system tap \(current.config.id) \(reason); trying the next configuration")
        var next = strategyIndex + 1
        while next < strategies.count {
            do {
                try openStrategy(next)
                return
            } catch {
                CaptureLog.write("tap \(strategies[next].id) failed: \(error.localizedDescription)")
                next += 1
            }
        }
        exhausted = true
        CaptureLog.write("every system tap configuration returned zeros while audio played; back to \(strategies.first?.id ?? "-")")
        if !strategies.isEmpty {
            do {
                try openStrategy(0)
            } catch {
                CaptureLog.write("reopening \(strategies[0].id) failed: \(error.localizedDescription)")
            }
        }
    }

    private func logProcesses(_ processes: [AudioProcess], reason: String) {
        CaptureLog.write("audio processes (\(reason)): \(AudioProcesses.summary(processes)) | \(AudioProcesses.outputSummary())")
    }

    // MARK: Microphone

    private func startMicrophone() throws {
        guard let choice = CaptureEngine.chooseMicrophone() else {
            throw CaptureError.io("no input device")
        }
        let device = choice.device
        guard let deviceFormat = AudioHAL.streamFormat(device, scope: kAudioDevicePropertyScopeInput) else {
            throw CaptureError.format
        }
        guard deviceFormat.mSampleRate > 0, deviceFormat.mChannelsPerFrame > 0 else {
            throw CaptureError.format
        }
        if choice.followsDefault { openedInputUID = AudioHAL.uid(device) }
        let preferBuiltIn = UserDefaults.standard.bool(forKey: Prefs.useBuiltInMic)
        CaptureLog.write("mic IOProc \(AudioHAL.describe(device)) bluetooth=\(AudioHAL.isBluetooth(device)) builtInPreferred=\(preferBuiltIn) followsDefault=\(choice.followsDefault) format \(AudioHAL.describe(deviceFormat))")
        let captured = deviceFormat
        let clock = StreamClock(label: "mic", rate: AudioHAL.nominalRate(device) ?? deviceFormat.mSampleRate)
        var proc: AudioDeviceIOProcID?
        let created = AudioDeviceCreateIOProcIDWithBlock(&proc, device, nil) { [weak self] _, inputData, _, _, _ in
            self?.consumeMicrophone(inputData, format: captured, clock: clock)
        }
        guard created == noErr, let proc else {
            throw CaptureError.io("mic IOProc \(statusText(created))")
        }
        micDevice = device
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

    private func consumeMicrophone(_ list: UnsafePointer<AudioBufferList>, format: AudioStreamBasicDescription, clock: StreamClock) {
        guard running else { return }
        let channels = max(1, Int(format.mChannelsPerFrame))
        let samples = AudioHAL.readFloat(list, format: format, dropChannels: 0, takeChannels: channels)
        guard !samples.isEmpty, let timed = clock.feed(samples, channels: channels) else { return }
        deliver(track: 1, samples: timed.samples, rate: timed.rate, channels: channels)
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

    // MARK: Delivery, meters, watchdog

    private func deliver(track: Int, samples: [Float], rate: Double, channels: Int) {
        guard running, !samples.isEmpty, rate > 0 else { return }
        var peak: Float = 0
        for sample in samples {
            let value = abs(sample)
            if value > peak { peak = value }
        }
        let seconds = Double(samples.count / max(1, channels)) / rate
        noteLevel(track: track, samples, peak: peak, seconds: seconds)
        watchSilence(track: track, peak: peak, seconds: seconds)
        noteFirstSecond(track: track, peak: peak, seconds: seconds)
        onPCM?(UInt8(track), rate, channels, samples)
    }

    private func watchSilence(track: Int, peak: Float, seconds: Double) {
        levelLock.lock()
        if reported.contains(track) {
            levelLock.unlock()
            return
        }
        // Zeros from the system tap are expected when nothing is playing.
        if track == 0, let playing = playingNow, playing.isEmpty {
            levelLock.unlock()
            return
        }
        heard[track, default: 0] += seconds
        peaks[track] = max(peaks[track] ?? 0, peak)
        let due = heard[track, default: 0] >= silenceWindow && (peaks[track] ?? 0) <= 1e-5
        if due { reported.insert(track) }
        let heardSeconds = heard[track] ?? 0
        let peakNow = peaks[track] ?? 0
        let playing = playingNow
        levelLock.unlock()
        guard due else { return }
        let name = track == 0 ? "system" : "mic"
        let who = playing.map { $0.joined(separator: ", ") } ?? "unknown"
        DispatchQueue.global(qos: .utility).async {
            CaptureLog.write(String(format: "watchdog %@ peak=%.6g heard=%.2fs playing=%@", name, peakNow, heardSeconds, who))
        }
        onSilent?(track == 0 ? .audioCapture : .microphone)
    }

    private func noteFirstSecond(track: Int, peak: Float, seconds: Double) {
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

    private func noteLevel(track: Int, _ samples: [Float], peak: Float, seconds: Double) {
        let db = rmsDB(samples)
        levelLock.lock()
        if track == 0 {
            systemDB = db
            systemMeter.add(samples)
            strategyPeak = max(strategyPeak, peak)
            strategySeconds += seconds
        } else {
            micDB = db
            micMeter.add(samples)
        }
        let due = Date().timeIntervalSince(lastLevel) >= 0.1
        if due { lastLevel = Date() }
        let report = LevelReport(system: systemDB, mic: microphoneOpen ? micDB : nil, playing: playingNow)
        levelLock.unlock()
        if due { onLevels?(report) }
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
