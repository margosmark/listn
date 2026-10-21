import AppKit

/// `--selftest` skips the menu. `open -a Listn.app --args --selftest 8` is the
/// run whose permission prompts belong to listn; launching the binary from a
/// shell attributes TCC to the shell instead. Quit listn first: macOS does
/// not deliver arguments to an app that is already running.
private func selfTestSeconds(_ args: [String]) -> Double? {
    if let raw = args.first(where: { $0.hasPrefix("--selftest=") }) {
        return Double(raw.dropFirst("--selftest=".count))
    }
    guard let index = args.firstIndex(of: "--selftest") else { return nil }
    if index + 1 < args.count, let value = Double(args[index + 1]), !args[index + 1].hasPrefix("-") {
        return value
    }
    return 5
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let menu = MenuBarController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        menu.install()
    }

    func applicationWillTerminate(_ notification: Notification) {
        menu.shutdown()
    }
}

final class SelfTestDelegate: NSObject, NSApplicationDelegate {
    let seconds: Double
    init(seconds: Double) { self.seconds = seconds }

    func applicationDidFinishLaunching(_ notification: Notification) {
        CaptureLog.beginSelfTest()
        CaptureLog.write("selftest process pid=\(ProcessInfo.processInfo.processIdentifier)")
        Permissions.requestMicrophone { granted in
            CaptureLog.write("AVCaptureDevice microphone granted=\(granted) status=\(String(describing: Permissions.microphone()))")
            Permissions.requestSystemAudio { systemGranted in
                CaptureLog.write("TCC system audio granted=\(systemGranted) status=\(String(describing: Permissions.systemAudio()))")
                DispatchQueue.global(qos: .userInitiated).async { [seconds] in
                    let code = CaptureSelfTest.run(seconds: seconds)
                    DispatchQueue.main.async {
                        exit(code)
                    }
                }
            }
        }
    }
}

let app = NSApplication.shared
if let seconds = selfTestSeconds(CommandLine.arguments) {
    let delegate = SelfTestDelegate(seconds: min(30, max(1, seconds)))
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
} else {
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.accessory)
    app.run()
}
