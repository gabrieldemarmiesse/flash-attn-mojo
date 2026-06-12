// swift-tools-version: 5.10
import PackageDescription

let package = Package(
  name: "bench_mfa",
  platforms: [
    .macOS(.v14),
  ],
  dependencies: [
    .package(path: "../../../metal-flash-attention"),
  ],
  targets: [
    .executableTarget(
      name: "bench_mfa",
      dependencies: [
        .product(name: "FlashAttention", package: "metal-flash-attention"),
      ]),
  ]
)
