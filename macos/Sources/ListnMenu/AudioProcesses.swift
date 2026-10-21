import AppKit
import CoreAudio
import Foundation

/// One Core Audio client process, as the HAL reports it (macOS 14+).
struct AudioProcess {
    var objectID: AudioObjectID
    var pid: pid_t
    var bundleID: String
    var name: String
    /// nil when the HAL does not answer for this process.
    var runningOutput: Bool?
    var runningInput: Bool?

    var label: String { name }
}

enum AudioProcesses {
    static func all() -> [AudioProcess] {
        var result: [AudioProcess] = []
        for objectID in audioObjectIDs(systemObject, kAudioHardwarePropertyProcessObjectList) {
            guard let raw = readUInt32(objectID, kAudioProcessPropertyPID) else { continue }
            let pid = pid_t(bitPattern: raw)
            let application = NSRunningApplication(processIdentifier: pid)
            let bundleID = application?.bundleIdentifier
                ?? AudioHAL.stringProperty(objectID, kAudioProcessPropertyBundleID)
                ?? ""
            let name = application?.localizedName ?? (bundleID.isEmpty ? "pid \(pid)" : bundleID)
            let output = readUInt32(objectID, kAudioProcessPropertyIsRunningOutput).map { $0 != 0 }
            let input = readUInt32(objectID, kAudioProcessPropertyIsRunningInput).map { $0 != 0 }
            result.append(AudioProcess(objectID: objectID, pid: pid, bundleID: bundleID, name: name,
                                       runningOutput: output, runningInput: input))
        }
        return result
    }

    /// Other processes playing audio right now. nil when the HAL does not say
    /// for any of them, so callers can fall back to the old behaviour.
    static func playing(in processes: [AudioProcess], excludingPID pid: pid_t) -> [AudioProcess]? {
        let others = processes.filter { $0.pid != pid }
        if !others.isEmpty && others.allSatisfy({ $0.runningOutput == nil }) { return nil }
        return others.filter { $0.runningOutput == true }
    }

    static func playingNow() -> [AudioProcess]? {
        playing(in: all(), excludingPID: ProcessInfo.processInfo.processIdentifier)
    }

    /// listn's own process object, only if the HAL lists it. Before listn
    /// opens any audio the translation returns kAudioObjectUnknown, and an
    /// exclude list must never carry that.
    static func ownObject() -> AudioObjectID? {
        guard let object = audioObjectID(for: ProcessInfo.processInfo.processIdentifier) else { return nil }
        let listed = audioObjectIDs(systemObject, kAudioHardwarePropertyProcessObjectList)
        return listed.contains(object) ? object : nil
    }

    static func summary(_ processes: [AudioProcess]) -> String {
        let busy = processes.filter { $0.runningOutput == true || $0.runningInput == true }
        let parts = busy.map { process in
            "\(process.name) pid=\(process.pid) \(process.bundleID.isEmpty ? "-" : process.bundleID) " +
                "out=\(flag(process.runningOutput)) in=\(flag(process.runningInput))"
        }
        let head = parts.isEmpty ? "no process is running audio IO" : parts.joined(separator: "; ")
        return head + " (\(processes.count - busy.count) idle of \(processes.count))"
    }

    static func outputSummary() -> String {
        guard let output = AudioHAL.defaultOutput() else { return "default output missing" }
        let running = AudioHAL.isRunningSomewhere(output).map { $0 ? "yes" : "no" } ?? "?"
        return "default output \(AudioHAL.name(output) ?? "?") running somewhere=\(running)"
    }

    private static func flag(_ value: Bool?) -> String {
        guard let value else { return "?" }
        return value ? "yes" : "no"
    }
}
