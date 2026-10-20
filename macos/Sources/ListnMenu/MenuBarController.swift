import AppKit

private let preferredBundles = [
    "us.zoom.xos",
    "com.microsoft.teams2",
    "com.microsoft.teams",
    "com.google.Chrome",
    "com.apple.Safari",
    "company.thebrowser.Browser",
    "com.brave.Browser",
    "com.microsoft.edgemac",
    "org.mozilla.firefox",
    "com.tinyspeck.slackmacgap",
    "com.hnc.Discord",
]

private struct Line {
    var start: Double
    var speaker: String?
    var text: String
}

private enum Source {
    case system
    case app(pid: pid_t, name: String)
    case legacy
}

final class MenuBarController: NSObject, NSMenuDelegate {
    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    private let menu = NSMenu()
    private let sourceMenu = NSMenu()
    private let supervisor = PythonSupervisor()
    private let ipc = IPCClient()
    private let capture = CaptureEngine()

    private let startItem = NSMenuItem(title: "Start recording", action: #selector(toggleRecording), keyEquivalent: "")
    private let sourceItem = NSMenuItem(title: "Source: All system audio", action: nil, keyEquivalent: "")
    private let micItem = NSMenuItem(title: "Microphone", action: #selector(toggleMic), keyEquivalent: "")
    private let levelItem = NSMenuItem(title: "Level  ▯▯▯▯▯▯▯▯", action: nil, keyEquivalent: "")
    private let transcriptHeader = NSMenuItem(title: "Latest transcript", action: nil, keyEquivalent: "")
    private var lineItems: [NSMenuItem] = []
    private let errorItem = NSMenuItem(title: "", action: nil, keyEquivalent: "")
    private let libraryItem = NSMenuItem(title: "Open library", action: #selector(openLibrary), keyEquivalent: "")
    private let quitItem = NSMenuItem(title: "Quit listn", action: #selector(quit), keyEquivalent: "")

    private var source: Source = .system
    private var micOn = true
    private var phase: Phase = .idle
    private var lines: [Line] = []
    private var errorText: String?
    private var libraryPort = 8765
    private var localLevels = false
    private var shuttingDown = false
    private var captureGeneration = 0
    private let captureQueue = DispatchQueue(label: "local.listn.capture.control")
    private let gate = NSLock()

    private enum Phase {
        case idle, starting, recording, finishing
    }

    func install() {
        let image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "listn")
        image?.isTemplate = true
        statusItem.button?.image = image
        statusItem.button?.toolTip = "listn"

        for item in [startItem, micItem, libraryItem, quitItem] {
            item.target = self
        }
        levelItem.isEnabled = false
        transcriptHeader.isEnabled = false
        errorItem.isEnabled = false
        errorItem.isHidden = true
        sourceItem.submenu = sourceMenu
        micItem.state = .on

        lineItems = (0..<6).map { _ in
            let item = NSMenuItem(title: " ", action: nil, keyEquivalent: "")
            item.isHidden = true
            return item
        }

        menu.delegate = self
        menu.addItem(startItem)
        menu.addItem(sourceItem)
        menu.addItem(micItem)
        menu.addItem(levelItem)
        menu.addItem(.separator())
        menu.addItem(transcriptHeader)
        lineItems.forEach { menu.addItem($0) }
        menu.addItem(.separator())
        menu.addItem(errorItem)
        menu.addItem(libraryItem)
        menu.addItem(quitItem)
        statusItem.menu = menu
        rebuildSourceMenu()
        wireCapture()
    }

    func shutdown() {
        shuttingDown = true
        _ = bumpCapture()
        captureQueue.sync { capture.stop() }
        ipc.sendStop()
        ipc.close()
        supervisor.stop()
    }

    func menuWillOpen(_ menu: NSMenu) {
        if menu === self.menu {
            rebuildSourceMenu()
        }
    }

    @objc private func toggleRecording(_ sender: Any?) {
        if phase == .recording || phase == .starting {
            beginStop()
            return
        }
        if phase == .finishing { return }
        phase = .starting
        errorText = nil
        lines = []
        refresh()
        let chosen = source
        let mic = micOn
        supervisor.ensureRunning { [weak self] result in
            guard let self, !self.shuttingDown else { return }
            switch result {
            case .failure(let error):
                self.fail(error.localizedDescription)
            case .success(let info):
                self.libraryPort = info.port ?? self.supervisor.port
                self.beginCapture(info, source: chosen, mic: mic)
            }
        }
    }

    @objc private func toggleMic(_ sender: Any?) {
        guard phase == .idle, !isLegacy else { return }
        micOn.toggle()
        micItem.state = micOn ? .on : .off
    }

    @objc private func chooseSystem(_ sender: Any?) {
        guard phase == .idle else { return }
        source = .system
        refresh()
    }

    @objc private func chooseLegacy(_ sender: Any?) {
        guard phase == .idle else { return }
        source = .legacy
        refresh()
    }

    @objc private func chooseApp(_ sender: NSMenuItem) {
        guard phase == .idle, let info = sender.representedObject as? NSDictionary else { return }
        let pid = (info["pid"] as? NSNumber)?.int32Value ?? -1
        let name = info["name"] as? String ?? sender.title
        guard pid > 0 else { return }
        source = .app(pid: pid, name: name)
        refresh()
    }

    @objc private func openLibrary(_ sender: Any?) {
        guard let url = URL(string: "http://127.0.0.1:\(libraryPort)/") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func quit(_ sender: Any?) {
        shutdown()
        NSApp.terminate(nil)
    }

    private var isLegacy: Bool {
        if case .legacy = source { return true }
        return false
    }

    private func wireCapture() {
        capture.onPCM = { [weak self] track, rate, channels, samples in
            self?.ipc.sendAudio(track: track, rate: rate, channels: channels, samples: samples)
        }
        capture.onLevels = { [weak self] system, mic in
            DispatchQueue.main.async {
                guard let self, self.localLevels else { return }
                self.levelItem.title = "Level  " + meter(max(system, mic))
            }
        }
        capture.onWarning = { [weak self] message in
            DispatchQueue.main.async {
                self?.errorText = message
                self?.refresh()
            }
        }
        ipc.onEvent = { [weak self] event in
            self?.handle(event)
        }
        ipc.onError = { [weak self] message in
            self?.fail(message)
        }
        ipc.onDisconnect = { [weak self] in
            guard let self, !self.shuttingDown, self.phase != .idle else { return }
            _ = self.bumpCapture()
            self.captureQueue.async { self.capture.stop() }
            self.phase = .idle
            self.localLevels = false
            self.errorText = "Lost the transcription engine."
            self.refresh()
        }
    }

    private func beginCapture(_ info: ServerInfo, source: Source, mic: Bool) {
        guard let path = info.sock else {
            fail(EngineError.noSocket.localizedDescription)
            return
        }
        do {
            try ipc.connect(path)
        } catch {
            fail(error.localizedDescription)
            return
        }
        switch source {
        case .system:
            ipc.sendHelloAndStart(source: "system", appName: nil, appPID: nil, mic: mic)
        case .app(let pid, let name):
            ipc.sendHelloAndStart(source: "app", appName: name, appPID: pid, mic: mic)
        case .legacy:
            ipc.sendHelloAndStart(source: "device", appName: nil, appPID: nil, mic: false)
            localLevels = false
            phase = .recording
            refresh()
            return
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let token = self.bumpCapture()
            self.captureQueue.async {
                guard self.captureGenerationLocked() == token else { return }
                do {
                    switch source {
                    case .system:
                        try self.capture.start(appPID: nil, appName: nil, mic: mic)
                    case .app(let pid, let name):
                        try self.capture.start(appPID: pid, appName: name, mic: mic)
                    case .legacy:
                        break
                    }
                    DispatchQueue.main.async {
                        guard self.captureGenerationLocked() == token else {
                            self.captureQueue.async { self.capture.stop() }
                            return
                        }
                        self.localLevels = true
                        self.phase = .recording
                        self.refresh()
                    }
                } catch {
                    self.ipc.sendStop()
                    DispatchQueue.main.async {
                        self.fail(error.localizedDescription)
                    }
                }
            }
        }
    }

    private func beginStop() {
        phase = .finishing
        localLevels = false
        _ = bumpCapture()
        captureQueue.async { [weak self] in self?.capture.stop() }
        ipc.sendStop()
        refresh()
    }

    private func handle(_ event: [String: Any]) {
        guard let type = event["type"] as? String else { return }
        switch type {
        case "live.segments":
            guard let segments = event["segments"] as? [[String: Any]] else { return }
            for segment in segments {
                let text = segment["text"] as? String ?? ""
                guard !text.isEmpty else { continue }
                lines.append(Line(start: segment["start"] as? Double ?? 0, speaker: segment["speaker"] as? String, text: text))
            }
            if lines.count > 6 { lines.removeFirst(lines.count - 6) }
            refreshLines()
        case "live.level":
            if !localLevels, let db = event["level_db"] as? Double {
                levelItem.title = "Level  " + meter(Float(db))
            }
        case "live.stopped":
            _ = bumpCapture()
            captureQueue.async { [weak self] in self?.capture.stop() }
            localLevels = false
            phase = .idle
            levelItem.title = "Level  " + meter(-120)
            refresh()
        case "live.error":
            if let message = event["message"] as? String {
                errorText = message
                refresh()
            }
        default:
            break
        }
    }

    private func fail(_ message: String) {
        _ = bumpCapture()
        captureQueue.async { [weak self] in self?.capture.stop() }
        localLevels = false
        phase = .idle
        errorText = message
        refresh()
    }

    private func bumpCapture() -> Int {
        gate.lock()
        captureGeneration += 1
        let value = captureGeneration
        gate.unlock()
        return value
    }

    private func captureGenerationLocked() -> Int {
        gate.lock()
        let value = captureGeneration
        gate.unlock()
        return value
    }

    private func rebuildSourceMenu() {
        sourceMenu.removeAllItems()
        let system = NSMenuItem(title: "All system audio", action: #selector(chooseSystem), keyEquivalent: "")
        system.target = self
        system.state = isSystem ? .on : .off
        system.isEnabled = phase == .idle
        sourceMenu.addItem(system)

        let apps = listAudioApps()
        let ranked = apps.sorted { lhs, rhs in
            let left = preferredBundles.firstIndex(of: lhs.bundleID) ?? Int.max
            let right = preferredBundles.firstIndex(of: rhs.bundleID) ?? Int.max
            if left != right { return left < right }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
        if !ranked.isEmpty { sourceMenu.addItem(.separator()) }
        var seen: [String: Int] = [:]
        for app in ranked {
            seen[app.name, default: 0] += 1
        }
        for app in ranked {
            let title = (seen[app.name] ?? 0) > 1 ? "\(app.name) (\(app.pid))" : app.name
            let item = NSMenuItem(title: title, action: #selector(chooseApp), keyEquivalent: "")
            item.target = self
            item.representedObject = ["pid": NSNumber(value: app.pid), "name": app.name] as NSDictionary
            item.toolTip = app.bundleID
            if case .app(let pid, _) = source, pid == app.pid { item.state = .on }
            item.isEnabled = phase == .idle
            sourceMenu.addItem(item)
        }
        if ranked.isEmpty {
            let hint = NSMenuItem(title: "Apps show up once they play audio", action: nil, keyEquivalent: "")
            hint.isEnabled = false
            sourceMenu.addItem(hint)
        }
        sourceMenu.addItem(.separator())
        let legacy = NSMenuItem(title: "Legacy input device", action: #selector(chooseLegacy), keyEquivalent: "")
        legacy.target = self
        legacy.toolTip = "Record the input selected in the web UI. Use this for BlackHole or any other device."
        legacy.state = isLegacy ? .on : .off
        legacy.isEnabled = phase == .idle
        sourceMenu.addItem(legacy)
    }

    private var isSystem: Bool {
        if case .system = source { return true }
        return false
    }

    private func refresh() {
        switch phase {
        case .idle:
            startItem.title = "Start recording"
            startItem.isEnabled = true
        case .starting:
            startItem.title = "Starting…"
            startItem.isEnabled = false
        case .recording:
            startItem.title = "Stop recording"
            startItem.isEnabled = true
        case .finishing:
            startItem.title = "Finishing…"
            startItem.isEnabled = false
        }
        switch source {
        case .system:
            sourceItem.title = "Source: All system audio"
        case .app(_, let name):
            sourceItem.title = "Source: \(name)"
        case .legacy:
            sourceItem.title = "Source: Legacy input device"
        }
        micItem.state = micOn ? .on : .off
        micItem.isEnabled = phase == .idle && !isLegacy
        micItem.title = isLegacy ? "Microphone (use the legacy device)" : "Microphone"
        let busy = phase != .idle
        sourceItem.isEnabled = !busy
        if let errorText, !errorText.isEmpty {
            errorItem.title = String(errorText.prefix(180))
            errorItem.isHidden = false
        } else {
            errorItem.isHidden = true
        }
        let symbol = phase == .recording || phase == .finishing ? "record.circle" : "waveform"
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "listn")
        image?.isTemplate = true
        statusItem.button?.image = image
        statusItem.button?.toolTip = phase == .recording ? "listn — recording" : "listn"
        refreshLines()
        rebuildSourceMenu()
    }

    private func refreshLines() {
        for (index, item) in lineItems.enumerated() {
            if index < lines.count {
                let line = lines[index]
                let who = line.speaker.map { "\($0): " } ?? ""
                let text = "\(clockText(line.start))  \(who)\(line.text)"
                item.title = String(text.prefix(110))
                item.isHidden = false
            } else if index == 0 && lines.isEmpty {
                item.title = phase == .recording ? "Listening…" : "Transcript lines show up here"
                item.isHidden = false
            } else {
                item.isHidden = true
            }
        }
    }
}

private func clockText(_ seconds: Double) -> String {
    let whole = max(0, Int(seconds))
    return String(format: "%d:%02d", whole / 60, whole % 60)
}

private func meter(_ db: Float) -> String {
    let clamped = max(-50, min(0, db))
    let filled = Int(((clamped + 50) / 50) * 8)
    let bars = String(repeating: "▮", count: filled) + String(repeating: "▯", count: max(0, 8 - filled))
    if db <= -120 { return bars + "  silent" }
    return bars + String(format: "  %.0f dB", db)
}
