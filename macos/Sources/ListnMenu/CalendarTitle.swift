import EventKit
import Foundation

struct CalendarMatch {
    var title: String
    var attendees: [String]
}

/// Shortest event that contains `moment`. Mirrors listn.meetings.pick_event.
func pickCalendarEvent(_ events: [EKEvent], moment: Date) -> EKEvent? {
    var best: EKEvent?
    var bestLength = TimeInterval.greatestFiniteMagnitude
    for event in events {
        guard let start = event.startDate, let end = event.endDate else { continue }
        if start <= moment && moment < end {
            let length = end.timeIntervalSince(start)
            if length < bestLength {
                best = event
                bestLength = length
            }
        }
    }
    return best
}

final class CalendarTitle {
    private let store = EKEventStore()

    func overlapping(completion: @escaping (CalendarMatch?) -> Void) {
        request { granted in
            guard granted else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let now = Date()
            let start = now.addingTimeInterval(-6 * 3600)
            let end = now.addingTimeInterval(12 * 3600)
            let predicate = self.store.predicateForEvents(withStart: start, end: end, calendars: nil)
            let events = self.store.events(matching: predicate)
            let match = pickCalendarEvent(events, moment: now).map { event in
                let names = (event.attendees ?? []).compactMap { $0.name }.filter { !$0.isEmpty }
                return CalendarMatch(title: event.title ?? "", attendees: names)
            }
            DispatchQueue.main.async { completion(match.flatMap { $0.title.isEmpty ? nil : $0 }) }
        }
    }

    private func request(_ completion: @escaping (Bool) -> Void) {
        if #available(macOS 14.0, *) {
            store.requestFullAccessToEvents { granted, _ in completion(granted) }
        } else {
            store.requestAccess(to: .event) { granted, _ in completion(granted) }
        }
    }
}
