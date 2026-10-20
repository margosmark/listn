import AppKit

/// A small floating panel of the latest transcript lines.
///
/// It stays above other windows, including full screen spaces, and does not
/// steal focus when a line arrives. The menu bar turns it on; it is off until
/// then so a recording does not cover the meeting by itself.
final class CaptionsWindow {
    private let panel: NSPanel
    private let label: NSTextField

    var isVisible: Bool { panel.isVisible }

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 80, y: 72, width: 640, height: 132),
            styleMask: [.titled, .closable, .resizable, .nonactivatingPanel, .utilityWindow, .fullSizeContentView],
            backing: .buffered,
            defer: false)
        panel.title = "listn"
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.isFloatingPanel = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isMovableByWindowBackground = true
        panel.titlebarAppearsTransparent = true
        panel.backgroundColor = NSColor.black.withAlphaComponent(0.78)
        panel.minSize = NSSize(width: 320, height: 88)
        panel.isReleasedWhenClosed = false

        label = NSTextField(wrappingLabelWithString: "Captions appear here once someone speaks.")
        label.textColor = .white
        label.font = NSFont.systemFont(ofSize: 18, weight: .medium)
        label.maximumNumberOfLines = 4
        label.lineBreakMode = .byWordWrapping
        label.translatesAutoresizingMaskIntoConstraints = false
        if let content = panel.contentView {
            content.addSubview(label)
            NSLayoutConstraint.activate([
                label.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 16),
                label.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -16),
                label.topAnchor.constraint(equalTo: content.topAnchor, constant: 28),
                label.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -12),
            ])
        }
    }

    func show() {
        panel.orderFrontRegardless()
    }

    func hide() {
        panel.orderOut(nil)
    }

    func toggle() {
        if isVisible { hide() } else { show() }
    }

    func showLines(_ lines: [(speaker: String?, text: String)]) {
        let recent = lines.suffix(3)
        let text: String
        if recent.isEmpty {
            text = "Listening…"
        } else {
            text = recent.map { line in
                if let speaker = line.speaker, !speaker.isEmpty {
                    return "\(speaker): \(line.text)"
                }
                return line.text
            }.joined(separator: "\n")
        }
        label.stringValue = text
    }
}
