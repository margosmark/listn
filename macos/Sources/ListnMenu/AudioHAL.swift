import CoreAudio
import Foundation

/// The SDK imports these as different integer types across versions. One
/// internal value so every file compares them the same way.
let systemObject = AudioObjectID(kAudioObjectSystemObject)
let noObject = AudioObjectID(kAudioObjectUnknown)

/// Core Audio device queries shared by capture and the self-test.
enum AudioHAL {
    static func defaultDevice(_ selector: AudioObjectPropertySelector) -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var device = noObject
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(systemObject, &address, 0, nil, &size, &device) == noErr,
              device != noObject else { return nil }
        return device
    }

    static func defaultInput() -> AudioObjectID? {
        defaultDevice(kAudioHardwarePropertyDefaultInputDevice)
    }

    static func defaultOutput() -> AudioObjectID? {
        defaultDevice(kAudioHardwarePropertyDefaultOutputDevice)
    }

    static func devices() -> [AudioObjectID] {
        audioObjectIDs(systemObject, kAudioHardwarePropertyDevices)
    }

    static func isBluetooth(_ device: AudioObjectID) -> Bool {
        let kind = transport(device)
        return kind == "Bluetooth" || kind == "Bluetooth LE"
    }

    /// The Mac's own microphone, whatever the system default input is.
    static func builtInInput() -> AudioObjectID? {
        let inputs = devices().filter { device in
            channelCount(device, scope: kAudioDevicePropertyScopeInput) > 0 && transport(device) == "built-in"
        }
        let named = inputs.first { device in
            (name(device) ?? "").localizedCaseInsensitiveContains("microphone")
        }
        return named ?? inputs.first
    }

    static func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        string(object, selector)
    }

    static func tapFormat(_ tap: AudioObjectID) -> AudioStreamBasicDescription? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        guard AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format) == noErr else { return nil }
        return format
    }

    /// kAudioDevicePropertyDeviceIsRunningSomewhere: some process is doing IO on it.
    static func isRunningSomewhere(_ device: AudioObjectID) -> Bool? {
        readUInt32(device, kAudioDevicePropertyDeviceIsRunningSomewhere).map { $0 != 0 }
    }

    static func uid(_ device: AudioObjectID) -> String? {
        string(device, kAudioDevicePropertyDeviceUID)
    }

    static func name(_ device: AudioObjectID) -> String? {
        string(device, kAudioObjectPropertyName)
    }

    static func streamFormat(_ device: AudioObjectID, scope: AudioObjectPropertyScope) -> AudioStreamBasicDescription? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamFormat,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain)
        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &format) == noErr else { return nil }
        return format
    }

    static func nominalRate(_ device: AudioObjectID) -> Double? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var rate: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &rate) == noErr else { return nil }
        return rate
    }

    static func channelCount(_ device: AudioObjectID, scope: AudioObjectPropertyScope) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    static func transport(_ device: AudioObjectID) -> String {
        guard let value = readUInt32(device, kAudioDevicePropertyTransportType) else { return "unknown" }
        switch value {
        case kAudioDeviceTransportTypeHDMI: return "HDMI"
        case kAudioDeviceTransportTypeDisplayPort: return "DisplayPort"
        case kAudioDeviceTransportTypeBuiltIn: return "built-in"
        case kAudioDeviceTransportTypeUSB: return "USB"
        case kAudioDeviceTransportTypeBluetooth: return "Bluetooth"
        case kAudioDeviceTransportTypeBluetoothLE: return "Bluetooth LE"
        case kAudioDeviceTransportTypeThunderbolt: return "Thunderbolt"
        case kAudioDeviceTransportTypeVirtual: return "virtual"
        case kAudioDeviceTransportTypePCI: return "PCI"
        case kAudioDeviceTransportTypeFireWire: return "FireWire"
        default: return fourCC(value)
        }
    }

    /// A monitor attached by HDMI or USB-C (those report DisplayPort). These
    /// devices have no volume scalar, and a global stereo tap often delivers
    /// zeros for them. `kAudioDevicePropertyDeviceIsRunningSomewhere` is not
    /// reliable on them either.
    static func isDisplayOutput(_ device: AudioObjectID) -> Bool {
        let kind = transport(device)
        if kind == "HDMI" || kind == "DisplayPort" { return true }
        let label = (name(device) ?? "").lowercased()
        return label.contains("hdmi") || label.contains("displayport")
    }

    static func hasVolume(_ device: AudioObjectID, scope: AudioObjectPropertyScope) -> Bool {
        for element in [kAudioObjectPropertyElementMain, AudioObjectPropertyElement(1)] {
            var address = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyVolumeScalar,
                mScope: scope,
                mElement: element)
            if AudioObjectHasProperty(device, &address) { return true }
        }
        return false
    }

    static func describe(_ device: AudioObjectID) -> String {
        let input = streamFormat(device, scope: kAudioDevicePropertyScopeInput)
        let output = streamFormat(device, scope: kAudioDevicePropertyScopeOutput)
        let rate = nominalRate(device) ?? input?.mSampleRate ?? output?.mSampleRate ?? 0
        return "\(name(device) ?? "device \(device)") uid=\(uid(device) ?? "?") " +
            "rate=\(Int(rate)) in=\(channelCount(device, scope: kAudioDevicePropertyScopeInput))ch " +
            "out=\(channelCount(device, scope: kAudioDevicePropertyScopeOutput))ch " +
            "transport=\(transport(device)) " +
            "volume=\(hasVolume(device, scope: kAudioDevicePropertyScopeOutput) ? "yes" : "no")"
    }

    static func describe(_ format: AudioStreamBasicDescription) -> String {
        var flags: [String] = []
        if format.mFormatFlags & kAudioFormatFlagIsFloat != 0 { flags.append("float") }
        if format.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0 { flags.append("int") }
        if format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0 { flags.append("non-interleaved") }
        else { flags.append("interleaved") }
        return "\(format.mSampleRate) Hz \(format.mChannelsPerFrame)ch \(format.mBitsPerChannel)bit " +
            "\(fourCC(format.mFormatID)) \(flags.joined(separator: ","))"
    }

    /// One line per buffer: channel count, byte size, and the peak if the
    /// bytes are readable as float32. Used to see whether an HDMI clock device
    /// inserted silent buffers in front of the tap.
    static func bufferReport(_ list: UnsafePointer<AudioBufferList>) -> String {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        if buffers.isEmpty { return "no buffers" }
        return buffers.enumerated().map { index, buffer in
            let frames = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
            var peak: Float = 0
            if let data = buffer.mData, frames > 0 {
                let pointer = data.assumingMemoryBound(to: Float.self)
                for offset in 0..<frames { peak = max(peak, abs(pointer[offset])) }
            }
            return "#\(index) ch=\(buffer.mNumberChannels) bytes=\(buffer.mDataByteSize) floatPeak=\(peak)"
        }.joined(separator: " | ")
    }

    /// Deinterleave `list` into float samples. `dropChannels` leading channels
    /// belong to the clock device (often zero on an output-only HDMI monitor);
    /// the tap's channels follow them.
    static func readFloat(_ list: UnsafePointer<AudioBufferList>, format: AudioStreamBasicDescription,
                          dropChannels: Int, takeChannels: Int) -> [Float] {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        var channels: [[Float]] = []
        let nonInterleaved = format.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        for buffer in buffers {
            guard let data = buffer.mData, buffer.mDataByteSize > 0 else { continue }
            let count = Int(buffer.mNumberChannels)
            if nonInterleaved || count <= 1 {
                if let samples = convert(data, byteCount: Int(buffer.mDataByteSize), format: format) {
                    channels.append(samples)
                }
            } else if let planar = convert(data, byteCount: Int(buffer.mDataByteSize), format: format) {
                let frames = planar.count / count
                guard frames > 0 else { continue }
                for channel in 0..<count {
                    var column = [Float]()
                    column.reserveCapacity(frames)
                    for frame in 0..<frames { column.append(planar[frame * count + channel]) }
                    channels.append(column)
                }
            }
        }
        let start = min(max(0, dropChannels), channels.count)
        let end = takeChannels > 0 ? min(channels.count, start + takeChannels) : channels.count
        guard start < end else { return [] }
        let kept = channels[start..<end]
        let frames = kept.map(\.count).min() ?? 0
        guard frames > 0 else { return [] }
        var interleaved = [Float](repeating: 0, count: frames * kept.count)
        for (channel, samples) in kept.enumerated() {
            for frame in 0..<frames {
                interleaved[frame * kept.count + channel] = samples[frame]
            }
        }
        return interleaved
    }
}

func readUInt32(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
    return value
}

func audioObjectIDs(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr, size > 0 else { return [] }
    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    var ids = [AudioObjectID](repeating: 0, count: count)
    guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &ids) == noErr else { return [] }
    return Array(ids.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

func statusText(_ status: OSStatus) -> String {
    if status == noErr { return "noErr" }
    return "\(fourCC(UInt32(bitPattern: status))) (\(status))"
}

func fourCC(_ value: UInt32) -> String {
    let bytes = [
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
    if bytes.allSatisfy({ $0 >= 32 && $0 < 127 }) {
        return "'" + (String(bytes: bytes, encoding: .ascii) ?? "") + "'"
    }
    return "0x" + String(format: "%08x", value)
}

private func string(_ device: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<CFString>.size)
    var value: CFString = "" as CFString
    let status = withUnsafeMutablePointer(to: &value) { pointer in
        AudioObjectGetPropertyData(device, &address, 0, nil, &size, UnsafeMutableRawPointer(pointer))
    }
    guard status == noErr else { return nil }
    let text = value as String
    return text.isEmpty ? nil : text
}

private func convert(_ data: UnsafeMutableRawPointer, byteCount: Int,
                     format: AudioStreamBasicDescription) -> [Float]? {
    if format.mFormatFlags & kAudioFormatFlagIsFloat != 0 && format.mBitsPerChannel == 32 {
        let frames = byteCount / MemoryLayout<Float>.size
        guard frames > 0 else { return nil }
        let pointer = data.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: frames))
    }
    if format.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0 && format.mBitsPerChannel == 16 {
        let frames = byteCount / MemoryLayout<Int16>.size
        guard frames > 0 else { return nil }
        let pointer = data.assumingMemoryBound(to: Int16.self)
        return (0..<frames).map { Float(pointer[$0]) / Float(32768) }
    }
    if format.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0 && format.mBitsPerChannel == 32 {
        let frames = byteCount / MemoryLayout<Int32>.size
        guard frames > 0 else { return nil }
        let pointer = data.assumingMemoryBound(to: Int32.self)
        return (0..<frames).map { Float(pointer[$0]) / Float(2147483648) }
    }
    return nil
}
