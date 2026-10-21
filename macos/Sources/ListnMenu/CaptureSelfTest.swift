import Foundation

/// `Listn --selftest [seconds]` and `open -a Listn.app --args --selftest`.
///
/// A direct launch is attributed to the parent process for TCC, so permission
/// checks lie. `open -a` makes this process the responsible one. Either way
/// the report is written to ~/Library/Application Support/listn/selftest.log
/// and appended to capture.log. stdout only exists for a direct launch.
///
/// 1. The production capture (mic + system) for `seconds`.
/// 2. Each tap configuration for `probeSeconds`, with the mic still open,
///    because a Bluetooth headset is in call mode only while its mic is.
/// 3. The two main configurations again with the mic closed.
/// The first production configuration that hears audio is saved and used
/// by the next recording.
///
/// Exit: 0 both tracks heard, 3 mic only, 4 system only, 2 neither, 1 start failed.
enum CaptureSelfTest {
    static let probeSeconds: Double = 3
    static let heardPeak: Float = 1e-4

    static func run(seconds: Double) -> Int32 {
        CaptureLog.write("selftest \(seconds)s, then tap probes of \(probeSeconds)s each")
        let engine = CaptureEngine()
        do {
            try engine.start(appPID: nil, appName: nil, mic: true)
        } catch {
            CaptureLog.write("selftest start failed: \(error.localizedDescription)")
            return 1
        }
        var micBest: Float = 0
        var systemBest: Float = 0
        let steps = max(1, Int(seconds.rounded(.up)))
        for second in 1...steps {
            Thread.sleep(forTimeInterval: 1)
            let meters = engine.takeMeters()
            micBest = max(micBest, meters.mic.peak)
            systemBest = max(systemBest, meters.system.peak)
            let playing = engine.playingNames().map { $0.isEmpty ? "nothing" : $0.joined(separator: ", ") } ?? "unknown"
            CaptureLog.write(String(
                format: "t=%ds system peak=%.6g rms=%.1fdB callbacks=%d | mic peak=%.6g rms=%.1fdB callbacks=%d | tap=%@ playing=%@",
                second,
                meters.system.peak, meters.system.rmsDB, meters.system.callbacks,
                meters.mic.peak, meters.mic.rmsDB, meters.mic.callbacks,
                engine.currentStrategy ?? "none", playing))
        }

        engine.stopSystem()
        var results: [ProbeResult] = []
        results += runProbes(label: "mic open", configs: probeConfigs(full: true))
        engine.stop()
        CaptureLog.write("microphone closed; waiting 2s for the output to leave call mode")
        Thread.sleep(forTimeInterval: 2)
        results += runProbes(label: "mic closed", configs: probeConfigs(full: false))

        for result in results { systemBest = max(systemBest, result.peak) }
        CaptureLog.write("probe summary:")
        for result in results { CaptureLog.write("  " + result.line) }
        saveWinner(results)

        let mic = micBest > heardPeak
        let system = systemBest > heardPeak
        let code: Int32
        switch (mic, system) {
        case (true, true): code = 0
        case (true, false): code = 3
        case (false, true): code = 4
        case (false, false): code = 2
        }
        if !system && results.allSatisfy({ ($0.playing ?? []).isEmpty }) {
            CaptureLog.write("no app was running audio output during the probes; play something (YouTube, Music) and run the self-test again")
        }
        CaptureLog.write("selftest done micPeak=\(micBest) systemPeak=\(systemBest) exit=\(code) log=\(CaptureLog.directory.path)/selftest.log")
        return code
    }

    /// (a) global exclude-self, (b) mixdown of every process playing,
    /// (c) a tap on the default output device, (d) clocked vs tap-only
    /// aggregate, (e) TapAutoStart, plus mono, an empty exclude list, and
    /// the include-only configuration of the old builds.
    private static func probeConfigs(full: Bool) -> [TapConfig] {
        let outputUID = AudioHAL.defaultOutput().flatMap(AudioHAL.uid)
        let playing = (AudioProcesses.playingNow() ?? []).map(\.objectID)
        var list = [TapConfig(kind: .globalStereo, clockUID: outputUID, autoStart: false)]
        if let outputUID {
            list.append(TapConfig(kind: .device(outputUID), clockUID: outputUID, autoStart: false))
        }
        guard full else { return list }
        if outputUID != nil {
            list.append(TapConfig(kind: .globalStereo, clockUID: nil, autoStart: false))
        }
        list.append(TapConfig(kind: .globalStereo, clockUID: outputUID, autoStart: true))
        list.append(TapConfig(kind: .globalNoExclude, clockUID: outputUID, autoStart: false))
        list.append(TapConfig(kind: .globalMono, clockUID: outputUID, autoStart: false))
        list.append(TapConfig(kind: .legacyIncludeOnly, clockUID: outputUID, autoStart: false))
        if playing.isEmpty {
            CaptureLog.write("probe processes/clocked skipped: no app is running audio output")
        } else {
            list.append(TapConfig(kind: .processes(playing), clockUID: outputUID, autoStart: false))
        }
        if let outputUID {
            list.append(TapConfig(kind: .device(outputUID), clockUID: nil, autoStart: false))
        }
        return list
    }

    private static func runProbes(label: String, configs: [TapConfig]) -> [ProbeResult] {
        CaptureLog.write("probes (\(label)): \(configs.map(\.id)) | \(AudioProcesses.outputSummary())")
        return configs.map { config in
            let result = CaptureSelfTest.probe(config, label: label)
            CaptureLog.write("probe " + result.line)
            return result
        }
    }

    /// Runs one configuration on its own thread so a start that never
    /// returns (TapAutoStart can wait for a client) cannot hang the test.
    private static func probe(_ config: TapConfig, label: String) -> ProbeResult {
        let processes = AudioProcesses.all()
        let playing = AudioProcesses.playing(in: processes, excludingPID: ProcessInfo.processInfo.processIdentifier)?.map(\.label)
        CaptureLog.write("probe [\(label)] \(config.id) begins | \(AudioProcesses.summary(processes))")
        let outcome = Locked<ProbeResult?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            outcome.value = CaptureSelfTest.measure(config, label: label, playing: playing)
            done.signal()
        }
        if done.wait(timeout: .now() + probeSeconds + 6) == .timedOut {
            return ProbeResult(label: label, id: config.id, peak: 0, callbacks: 0, frames: 0, rate: 0,
                               playing: playing, error: "did not return within \(Int(probeSeconds + 6))s (AudioDeviceStart hung)")
        }
        return outcome.value ?? ProbeResult(label: label, id: config.id, peak: 0, callbacks: 0, frames: 0, rate: 0,
                                            playing: playing, error: "no result")
    }

    private static func measure(_ config: TapConfig, label: String, playing: [String]?) -> ProbeResult {
        let stats = Locked(ProbeStats())
        let tap = SystemTap(config: config)
        tap.onSamples = { samples, rate, channels in
            var peak: Float = 0
            for sample in samples {
                let value = abs(sample)
                if value > peak { peak = value }
            }
            stats.withValue { (current: inout ProbeStats) -> Void in
                current.peak = max(current.peak, peak)
                current.callbacks += 1
                current.frames += samples.count / max(1, channels)
                current.rate = rate
            }
        }
        do {
            try tap.start()
        } catch {
            return ProbeResult(label: label, id: config.id, peak: 0, callbacks: 0, frames: 0, rate: 0,
                               playing: playing, error: error.localizedDescription)
        }
        Thread.sleep(forTimeInterval: probeSeconds)
        tap.stop()
        let totals = stats.value
        return ProbeResult(label: label, id: config.id, peak: totals.peak, callbacks: totals.callbacks,
                           frames: totals.frames, rate: totals.rate, playing: playing, error: nil)
    }

    private static func saveWinner(_ results: [ProbeResult]) {
        let order = [
            "global-stereo/clocked", "device/clocked", "global-stereo/tap-only", "device/tap-only",
            "processes/clocked", "global-no-exclude/clocked", "global-mono/clocked", "global-stereo/clocked/autostart",
        ]
        let heard = Set(results.filter { $0.peak > heardPeak }.map(\.id))
        guard let winner = order.first(where: { heard.contains($0) }) else {
            CaptureLog.write("no production tap configuration heard audio; the saved choice is unchanged (\(UserDefaults.standard.string(forKey: Prefs.tapStrategy) ?? "none"))")
            return
        }
        UserDefaults.standard.set(winner, forKey: Prefs.tapStrategy)
        CaptureLog.write("saved \(winner) as the system tap strategy for recordings")
    }
}

private struct ProbeStats {
    var peak: Float = 0
    var callbacks = 0
    var frames = 0
    var rate: Double = 0
}

private struct ProbeResult {
    var label: String
    var id: String
    var peak: Float
    var callbacks: Int
    var frames: Int
    var rate: Double
    var playing: [String]?
    var error: String?

    var line: String {
        let who = playing.map { $0.isEmpty ? "nothing" : $0.joined(separator: ", ") } ?? "unknown"
        if let error {
            return "[\(label)] \(id): FAILED \(error) | playing=\(who)"
        }
        let verdict = peak > CaptureSelfTest.heardPeak ? "AUDIO" : "zeros"
        return String(format: "[%@] %@: %@ peak=%.6g callbacks=%d frames=%d rate=%.0f | playing=%@",
                      label, id, verdict, peak, callbacks, frames, rate, who)
    }
}
