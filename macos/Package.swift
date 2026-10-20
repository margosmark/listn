// swift-tools-version: 6.0
import PackageDescription

// Language mode v5: this target is AppKit plus Core Audio callbacks, which are
// not annotated for Swift 6 strict concurrency. The Swift 6.4 toolchain
// (Command Line Tools) still builds it. Deployment target 14.2 is passed by
// scripts/build-app.sh so CATapDescription is available without extra guards.
let package = Package(
    name: "ListnMenu",
    platforms: [.macOS(.v14)],
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
