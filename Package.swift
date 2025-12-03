// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Typogenesis",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "Typogenesis", targets: ["Typogenesis"])
    ],
    targets: [
        .executableTarget(
            name: "Typogenesis",
            path: "Typogenesis",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "TypogenesisTests",
            dependencies: ["Typogenesis"],
            path: "TypogenesisTests"
        )
    ]
)
