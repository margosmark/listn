import AppKit
import CoreGraphics
import Foundation
import UserNotifications

/// Same rules as listn/meetings.py. Dedicated meeting apps match on the bundle
/// id. Slack matches only when a window title says Huddle. Browsers match only
/// when a window title looks like Meet, Zoom, Teams, or Webex.
enum MeetingRules {
    static let apps = [
        "us.zoom.xos": "Zoom",
        "com.microsoft.teams2": "Teams",
        "com.microsoft.teams": "Teams",
        "com.cisco.webexmeetings": "Webex",
        "com.webex.meetingmanager": "Webex",
        "Cisco-Systems.Spark": "Webex",
        "com.apple.FaceTime": "FaceTime",
        "com.apple.facetime": "FaceTime",
    ]
    static let browsers: Set<String> = [
        "com.google.Chrome", "com.apple.Safari", "company.thebrowser.Browser",
        "com.brave.Browser", "com.microsoft.edgemac", "org.mozilla.firefox",
    ]
    static let browserHints = ["google meet", "meet.google", "zoom meeting", "microsoft teams", "webex"]
    static let slack = "com.tinyspeck.slackmacgap"

    static func label(bundleID: String, appName: String, windowTitles: [String]) -> String? {
        if let known = apps[bundleID] { return known }
        let titles = windowTitles.joined(separator: " ").lowercased()
        let name = appName.lowercased()
        if bundleID == slack && (titles.contains("huddle") || name.contains("huddle")) {
            return "Slack huddle"
        }
        if browsers.contains(bundleID) {
            let blob = titles + " " + name
            if browserHints.contains(where: { blob.contains($0) }) { return "Browser meeting" }
        }
        return nil
    }
}

struct MeetingHit {
    var pid: pid_t
    var name: String
    var bundleID: String
    var label: String
}

func windowTitles(pid: pid_t) -> [String] {
    guard let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] else {
        return []
    }
    var titles: [String] = []
    for window in info {
        let owner = window[kCGWindowOwnerPID as String] as? pid_t ?? pid_t(window[kCGWindowOwnerPID as String] as? Int ?? -1)
        guard owner == pid, let title = window[kCGWindowName as String] as? String, !title.isEmpty else { continue }
        titles.append(title)
    }
    return titles
}

func currentMeetings() -> [MeetingHit] {
    var hits: [MeetingHit] = []
    for app in listAudioApps() {
        let titles = windowTitles(pid: app.pid)
        guard let label = MeetingRules.label(bundleID: app.bundleID, appName: app.name, windowTitles: titles) else { continue }
        hits.append(MeetingHit(pid: app.pid, name: app.name, bundleID: app.bundleID, label: label))
    }
    return hits
}

/// Polls process-tap clients. Calls back on the main queue when a meeting
/// appears or the one we are tracking goes away.
final class MeetingWatch: NSObject, UNUserNotificationCenterDelegate {
    var onOffer: ((MeetingHit) -> Void)?
    var onStart: ((MeetingHit) -> Void)?
    var onOfferEnd: ((MeetingHit) -> Void)?
    var onEnd: ((MeetingHit) -> Void)?
    var offer = true
    var autoStart = false
    private var timer: Timer?
    private var seen: [String: MeetingHit] = [:]
    private var tracked: MeetingHit?
    private var prompted: Set<String> = []

    func start() {
        let center = UNUserNotificationCenter.current()
        center.delegate = self
        center.requestAuthorization(options: [.alert, .sound]) { _, _ in }
        let start = UNNotificationAction(identifier: "record", title: "Record", options: [])
        let stop = UNNotificationAction(identifier: "stop", title: "Stop recording", options: [])
        center.setNotificationCategories([
            UNNotificationCategory(identifier: "MEETING_START", actions: [start], intentIdentifiers: [], options: []),
            UNNotificationCategory(identifier: "MEETING_END", actions: [stop], intentIdentifiers: [], options: []),
        ])
        timer?.invalidate()
        let timer = Timer(timeInterval: 2.0, repeats: true) { [weak self] _ in self?.poll() }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() { timer?.invalidate() }

    func track(_ hit: MeetingHit) { tracked = hit }

    func clearTrack() { tracked = nil }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse, withCompletionHandler completion: @escaping () -> Void) {
        let id = response.notification.request.content.threadIdentifier
        if response.actionIdentifier == "record", let hit = seen[id] {
            DispatchQueue.main.async { self.onStart?(hit) }
        } else if response.actionIdentifier == "stop", let hit = tracked {
            DispatchQueue.main.async { self.onEnd?(hit) }
        }
        completion()
    }

    private func poll() {
        let hits = currentMeetings()
        var now: [String: MeetingHit] = [:]
        for hit in hits { now[hit.bundleID] = hit }
        if offer {
            for (bundle, hit) in now where seen[bundle] == nil && !prompted.contains(bundle) {
                prompted.insert(bundle)
                if autoStart {
                    tracked = hit
                    onStart?(hit)
                } else {
                    notify(hit, ending: false)
                    onOffer?(hit)
                }
            }
        }
        if let tracked, now[tracked.bundleID] == nil, seen[tracked.bundleID] != nil {
            let ended = tracked
            self.tracked = nil
            if autoStart {
                onEnd?(ended)
            } else {
                notify(ended, ending: true)
                onOfferEnd?(ended)
            }
        }
        seen = now
    }

    private func notify(_ hit: MeetingHit, ending: Bool) {
        let content = UNMutableNotificationContent()
        content.threadIdentifier = hit.bundleID
        if ending {
            content.title = "\(hit.label) ended"
            content.body = "Stop the recording?"
            content.categoryIdentifier = "MEETING_END"
        } else if autoStart {
            content.title = "Recording \(hit.label)"
            content.body = "listn started because \(hit.name) is in a call."
        } else {
            content.title = "Record \(hit.label)?"
            content.body = "\(hit.name) started using audio."
            content.categoryIdentifier = "MEETING_START"
        }
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
    }
}
