// swift-tools-version: 6.0
import PackageDescription

// Language mode v5: this target is AppKit plus Core Audio callbacks, which are
// not annotated for Swift 6 strict concurrency. The Swift 6.4 toolchain
// (Command Line Tools) still builds it. Process taps need macOS 14.2, which
// `.v14` cannot express, so the deployment target is given as a string.
let package = Package(
    name: "ListnMenu",
    platforms: [.macOS("14.2")],
    products: [
        .executable(name: "ListnMenu", targets: ["ListnMenu"]),
    ],
    targets: [
        .executableTarget(
            name: "ListnMenu",
            path: "Sources/ListnMenu",
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("AVFoundation"),
            ]
        ),
    ]
)
