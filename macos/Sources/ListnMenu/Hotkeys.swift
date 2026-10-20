import Carbon
import Foundation

/// Control-Option-R starts or stops. Control-Option-B bookmarks the moment.
/// Carbon hotkeys do not need the Accessibility permission.
final class Hotkeys {
    static let shared = Hotkeys()
    var onStartStop: (() -> Void)?
    var onBookmark: (() -> Void)?
    private var refs: [EventHotKeyRef] = []
    private var installed = false

    func install() {
        guard !installed else { return }
        installed = true
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), hotkeyEvent, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), nil)
        register(id: 1, key: UInt32(kVK_ANSI_R))
        register(id: 2, key: UInt32(kVK_ANSI_B))
    }

    fileprivate func handle(_ id: UInt32) {
        DispatchQueue.main.async {
            if id == 1 { self.onStartStop?() }
            if id == 2 { self.onBookmark?() }
        }
    }

    private func register(id: UInt32, key: UInt32) {
        var ref: EventHotKeyRef?
        let hotKeyID = EventHotKeyID(signature: OSType(0x6C73746E), id: id)
        let status = RegisterEventHotKey(key, UInt32(controlKey | optionKey), hotKeyID, GetApplicationEventTarget(), 0, &ref)
        if status == noErr, let ref { refs.append(ref) }
    }
}

private func hotkeyEvent(_ next: EventHandlerCallRef?, _ event: EventRef?, _ user: UnsafeMutableRawPointer?) -> OSStatus {
    guard let event, let user else { return noErr }
    var hotKeyID = EventHotKeyID()
    let status = GetEventParameter(
        event,
        EventParamName(kEventParamDirectObject),
        EventParamType(typeEventHotKeyID),
        nil,
        MemoryLayout<EventHotKeyID>.size,
        nil,
        &hotKeyID)
    if status == noErr {
        Unmanaged<Hotkeys>.fromOpaque(user).takeUnretainedValue().handle(hotKeyID.id)
    }
    return noErr
}
