import Foundation

/// `Listn --selftest [seconds]` and `open -a Listn.app --args --selftest`.
///
/// A direct launch is attributed to the parent process for TCC, so permission
/// checks lie. `open -a` makes this process the responsible one. Either way
/// the report is written to ~/Library/Application Support/listn/selftest.log
/// and appended to capture.log. stdout only exists for a direct launch.
enum CaptureSelfTest {
    static func run(seconds: Double) -> Int32 {
        CaptureLog.write("selftest \(seconds)s")
        let engine = CaptureEngine()
        do {
            try engine.start(appPID: nil, appName: nil, mic: true)
        } catch {
            CaptureLog.write("selftest start failed: \(error.localizedDescription)")
            return 1
        }
        defer { engine.stop() }
        var best: Float = 0
        let steps = max(1, Int(seconds.rounded(.up)))
        for second in 1...steps {
            Thread.sleep(forTimeInterval: 1)
            let meters = engine.takeMeters()
            best = max(best, meters.system.peak, meters.mic.peak)
            CaptureLog.write(String(
                format: "t=%ds system peak=%.6g rms=%.1fdB callbacks=%d | mic peak=%.6g rms=%.1fdB callbacks=%d",
                second,
                meters.system.peak, meters.system.rmsDB, meters.system.callbacks,
                meters.mic.peak, meters.mic.rmsDB, meters.mic.callbacks))
        }
        let code: Int32 = best > 1e-4 ? 0 : 2
        CaptureLog.write("selftest done bestPeak=\(best) exit=\(code) log=\(CaptureLog.directory.path)/selftest.log")
        return code
    }
}
