import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let menu = MenuBarController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        menu.install()
    }

    func applicationWillTerminate(_ notification: Notification) {
        menu.shutdown()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
