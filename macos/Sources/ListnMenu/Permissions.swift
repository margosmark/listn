import AppKit
import AVFoundation
import Darwin

/// Microphone and System Audio Recording access for this app.
///
/// Neither failure raises an error at capture time: macOS keeps the stream
/// running and delivers zeros. So access is checked before a recording
/// starts, and the capture engine also watches for digital silence.
enum PrivacyPane: String {
    case microphone
    case audioCapture = "audio-capture"

    var url: URL {
        switch self {
        case .microphone:
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
        case .audioCapture:
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!
        }
    }

    var message: String {
        switch self {
        case .microphone:
            return "listn isn't receiving audio from the microphone. Allow it in System Settings > Privacy & Security > Microphone."
        case .audioCapture:
            return "listn isn't receiving system audio. Allow it in System Settings > Privacy & Security > Screen & System Audio Recording."
        }
    }

    func open() {
        NSWorkspace.shared.open(url)
    }
}

enum Access {
    case granted, denied, undetermined
}

enum Permissions {
    static func microphone() -> Access {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return .granted
        case .notDetermined: return .undetermined
        default: return .denied
        }
    }

    /// Asks once if undetermined. The completion runs on the main queue.
    static func requestMicrophone(_ completion: @escaping (Bool) -> Void) {
        switch microphone() {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        }
    }

    // System Audio Recording has no public status API. TCC.framework's
    // preflight/request calls (the same ones Apple's own tools use for
    // kTCCServiceAudioCapture) are loaded at runtime. If they are missing the
    // check reports undetermined and the silence watchdog still applies.
    private typealias PreflightFn = @convention(c) (CFString, CFDictionary?) -> Int32
    private typealias RequestFn = @convention(c) (CFString, CFDictionary?, @escaping @convention(block) (Bool) -> Void) -> Void
    private static let tcc: UnsafeMutableRawPointer? = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW)
    private static let service = "kTCCServiceAudioCapture" as CFString

    static func systemAudio() -> Access {
        guard let tcc, let symbol = dlsym(tcc, "TCCAccessPreflight") else { return .undetermined }
        let preflight = unsafeBitCast(symbol, to: PreflightFn.self)
        switch preflight(service, nil) {
        case 0: return .granted
        case 1: return .denied
        default: return .undetermined
        }
    }

    static func requestSystemAudio(_ completion: @escaping (Bool) -> Void) {
        switch systemAudio() {
        case .granted:
            completion(true)
            return
        case .denied:
            completion(false)
            return
        case .undetermined:
            break
        }
        guard let tcc, let symbol = dlsym(tcc, "TCCAccessRequest") else {
            // Creating the tap shows the prompt; the watchdog catches a denial.
            completion(true)
            return
        }
        let request = unsafeBitCast(symbol, to: RequestFn.self)
        request(service, nil) { granted in
            DispatchQueue.main.async { completion(granted) }
        }
    }
}
