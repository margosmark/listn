import Foundation
import IOKit.ps

/// Battery state for the notes and Whisper model choice.
///
/// A missing sensor reports "plugged in" so a desktop keeps the model you picked.
func currentPower() -> (onBattery: Bool, percent: Double?) {
    guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
        return (false, nil)
    }
    guard let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef] else {
        return (false, nil)
    }
    for source in sources {
        guard let description = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any] else {
            continue
        }
        let state = description[kIOPSPowerSourceStateKey as String] as? String
        let current = (description[kIOPSCurrentCapacityKey as String] as? NSNumber)?.doubleValue
        let maximum = (description[kIOPSMaxCapacityKey as String] as? NSNumber)?.doubleValue
        let onBattery = state == (kIOPSBatteryPowerValue as String)
        var percent: Double?
        if let current, let maximum, maximum > 0 {
            percent = current / maximum * 100
        }
        if state != nil {
            return (onBattery, percent)
        }
    }
    return (false, nil)
}
