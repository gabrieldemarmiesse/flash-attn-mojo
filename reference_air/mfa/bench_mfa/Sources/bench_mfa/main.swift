// bench_mfa: CLI wrapper around philipturner/metal-flash-attention's
// FORWARD attention kernel (single head, R x C x D, no mask, non-causal).
//
// Usage:
//   bench_mfa --seq 4096 --head-dim 128 --dtype fp16 --iters 20
//             [--warmup 5] [--dump-source /path/fwd.metal] [--check]
//
// Timing: ONE dispatch per MTLCommandBuffer; per-iteration GPU time is
// commandBuffer.gpuEndTime - commandBuffer.gpuStartTime (the same clock
// MFA's own SquareAttentionTest.testPerformance uses, except that test
// amortizes over 5 dispatches per command buffer and reports
// max GINSTRS over 5 trials).
//
// Output: one JSON object on stdout.

import FlashAttention
import Metal
import Foundation

// MARK: - Argument parsing

func parseArgs() -> (
  seq: Int, headDim: Int, dtype: String, iters: Int, warmup: Int,
  dispatches: Int, heads: Int, dumpSource: String?, check: Bool
) {
  var seq = 4096
  var headDim = 128
  var dtype = "fp16"
  var iters = 20
  var warmup = 5
  var dispatches = 5
  var heads = 1
  var dumpSource: String? = nil
  var check = false

  var args = Array(CommandLine.arguments.dropFirst())
  while !args.isEmpty {
    let flag = args.removeFirst()
    func value() -> String {
      guard !args.isEmpty else {
        FileHandle.standardError.write("missing value for \(flag)\n".data(using: .utf8)!)
        exit(2)
      }
      return args.removeFirst()
    }
    switch flag {
    case "--seq": seq = Int(value())!
    case "--head-dim": headDim = Int(value())!
    case "--dtype": dtype = value()
    case "--iters": iters = Int(value())!
    case "--warmup": warmup = Int(value())!
    case "--dispatches": dispatches = Int(value())!
    case "--heads": heads = Int(value())!
    case "--dump-source": dumpSource = value()
    case "--check": check = true
    default:
      FileHandle.standardError.write("unknown flag: \(flag)\n".data(using: .utf8)!)
      exit(2)
    }
  }
  return (seq, headDim, dtype, iters, warmup, dispatches, heads, dumpSource, check)
}

let opts = parseArgs()

guard opts.dtype == "fp16" || opts.dtype == "fp32" else {
  // MFA's reference implementation has no BF16 mode for Q/K/V inputs:
  // lowPrecisionInputs=true selects FP16 for Q/K/V (BF16 is only used for
  // dO and backward intermediates). O is ALWAYS FP32 in memory.
  FileHandle.standardError.write(
    "unsupported --dtype \(opts.dtype): MFA supports fp16 (lowPrecision) or fp32 inputs only\n"
      .data(using: .utf8)!)
  exit(2)
}
let lowPrecision = (opts.dtype == "fp16")

// MARK: - Kernel construction (mirrors SquareAttentionTest)

var attentionDesc = AttentionDescriptor()
attentionDesc.lowPrecisionInputs = lowPrecision
attentionDesc.lowPrecisionIntermediates = lowPrecision
attentionDesc.matrixDimensions = (
  row: UInt32(opts.seq),
  column: UInt32(opts.seq),
  head: UInt16(opts.headDim))
attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)

let kernelDesc = attentionDesc.kernelDescriptor(type: .forward)
let kernel = AttentionKernel(descriptor: kernelDesc)
var source = kernel.createSource()

// macOS 26 / Metal 4 compatibility: the Metal compiler now rejects __asm
// labels containing '.' ("illegal string literal in 'asm'"), which breaks
// MFA's declarations of the undocumented air.simdgroup_async_copy_* AIR
// intrinsics. Prefixing the label with \01 (LLVM's "use this symbol name
// literally" escape) bypasses the lexer restriction and emits the exact
// same AIR symbol.
source = source.replacingOccurrences(
  of: "__asm(\"air.", with: "__asm(\"\\01air.")

if let path = opts.dumpSource {
  // The runtime compiler (makeLibrary(source:)) implicitly includes
  // <metal_stdlib>; the offline CLI (xcrun metal -c) does not. Prepend it
  // so the dumped file is compilable offline. This include is the ONLY
  // difference from the string handed to makeLibrary below.
  let offline = "#include <metal_stdlib>\n" + source
  try! offline.write(toFile: path, atomically: true, encoding: .utf8)
}

let device = MTLContext.global.device
let commandQueue = MTLContext.global.commandQueue

// MFA compiles its runtime-generated MSL with DEFAULT MTLCompileOptions
// (options: nil): fast math ON, newest supported Metal language version.
let library = try! device.makeLibrary(source: source, options: nil)

let functionConstants = MTLFunctionConstantValues()
attentionDesc.setFunctionConstants(functionConstants)
let function = try! library.makeFunction(
  name: "attention", constantValues: functionConstants)

let pipelineDesc = MTLComputePipelineDescriptor()
pipelineDesc.computeFunction = function
pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
let pipeline = try! device.makeComputePipelineState(
  descriptor: pipelineDesc, options: [], reflection: nil)

// MARK: - Buffers

func randomArray(_ count: Int, seed: UInt64) -> [Float] {
  // Cheap xorshift; values in [-0.5, 0.5).
  var state = seed &* 6364136223846793005 &+ 1442695040888963407
  var output = [Float](repeating: 0, count: count)
  for i in 0..<count {
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    output[i] = Float(state % 1_000_000) / 1_000_000 - 0.5
  }
  return output
}

func createBuffer(
  _ array: [Float], _ precision: GEMMOperandPrecision
) -> MTLBuffer {
  switch precision {
  case .FP32:
    return device.makeBuffer(
      bytes: array, length: array.count * 4)!
  case .FP16:
    var converted = [Float16](repeating: 0, count: array.count)
    for i in array.indices { converted[i] = Float16(array[i]) }
    return converted.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: array.count * 2)!
    }
  case .BF16:
    var converted = [UInt16](repeating: 0, count: array.count)
    for i in array.indices {
      converted[i] = UInt16(truncatingIfNeeded: array[i].bitPattern >> 16)
    }
    return converted.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: array.count * 2)!
    }
  }
}

// MFA is a single-head R x C x D kernel; MHA is emulated the way its
// docs intend — one dispatch per head. Each head gets its OWN
// MTLBuffer set: heads are hazard-free against each other (they run
// concurrently inside one pass, like ccv's natively batched kernel),
// while back-to-back repeats of the same head hazard on that head's
// O/L buffers and serialize — which is what makes the
// dispatches-per-command-buffer timing trick work.
let memoryPrecisions = attentionDesc.memoryPrecisions
let operandSize = opts.seq * opts.headDim
var qArrays: [[Float]] = []
var kArrays: [[Float]] = []
var vArrays: [[Float]] = []
var buffersQ: [MTLBuffer] = []
var buffersK: [MTLBuffer] = []
var buffersV: [MTLBuffer] = []
var buffersO: [MTLBuffer] = []
var buffersL: [MTLBuffer] = []
for h in 0..<opts.heads {
  let q = randomArray(operandSize, seed: UInt64(42 + 3 * h))
  let k = randomArray(operandSize, seed: UInt64(43 + 3 * h))
  let v = randomArray(operandSize, seed: UInt64(44 + 3 * h))
  qArrays.append(q)
  kArrays.append(k)
  vArrays.append(v)
  buffersQ.append(createBuffer(q, memoryPrecisions[.Q]!))
  buffersK.append(createBuffer(k, memoryPrecisions[.K]!))
  buffersV.append(createBuffer(v, memoryPrecisions[.V]!))
  buffersO.append(createBuffer(
    [Float](repeating: 0, count: operandSize), memoryPrecisions[.O]!))
  buffersL.append(createBuffer(
    [Float](repeating: 0, count: opts.seq), memoryPrecisions[.L]!))
}

// MARK: - Dispatch

// One command buffer holding `dispatches` back-to-back dispatches of the
// same pipeline (Metal's automatic hazard tracking on bufferO/bufferL
// serializes them without CPU gaps — exactly what MFA's
// SquareAttentionTest.testPerformance does with dispatchCount=5).
// Returns the command buffer's GPU time divided by the dispatch count.
// Single-dispatch command buffers are NOT reliable on M-series: the GPU
// power-states down between command buffers and per-iteration times
// flap by 5-50x.
func runOnce() -> Double {
  let commandBuffer = commandQueue.makeCommandBuffer()!
  let encoder = commandBuffer.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipeline)
  encoder.setThreadgroupMemoryLength(
    Int(kernel.threadgroupMemoryAllocation), index: 0)

  let parallelization = Int(kernel.blockDimensions.parallelization)
  let blockCount = (opts.seq + parallelization - 1) / parallelization
  for _ in 0..<opts.dispatches {
    for h in 0..<opts.heads {
      encoder.setBuffer(buffersQ[h], offset: 0, index: 0)
      encoder.setBuffer(buffersK[h], offset: 0, index: 1)
      encoder.setBuffer(buffersV[h], offset: 0, index: 2)
      encoder.setBuffer(buffersO[h], offset: 0, index: 3)
      encoder.setBuffer(buffersL[h], offset: 0, index: 4)
      encoder.dispatchThreadgroups(
        MTLSize(width: blockCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
          width: Int(kernel.threadgroupSize), height: 1, depth: 1))
    }
  }
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()
  // Per-iteration time = one full pass over all heads.
  let latency = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
  return latency / Double(opts.dispatches)
}

for _ in 0..<opts.warmup { _ = runOnce() }
var gpuTimesUs: [Double] = []
for _ in 0..<opts.iters { gpuTimesUs.append(runOnce() * 1e6) }

// MARK: - Optional correctness check (naive CPU attention, fp32)

var maxError: Float = -1
if opts.check {
  let scale = 1 / Float(opts.headDim).squareRoot()
  let n = opts.seq
  let d = opts.headDim
  // Check a strided row subset of the first two heads (CPU latency).
  for head in 0..<min(2, opts.heads) {
    let qArray = qArrays[head]
    let kArray = kArrays[head]
    let vArray = vArrays[head]
    let oPointer = buffersO[head].contents().bindMemory(
      to: Float.self, capacity: operandSize)
    let rowStride = max(1, n / 64)
    for r in stride(from: 0, to: n, by: rowStride) {
      var scores = [Float](repeating: 0, count: n)
      var maxScore: Float = -.infinity
      for c in 0..<n {
        var dot: Float = 0
        for h in 0..<d { dot += qArray[r * d + h] * kArray[c * d + h] }
        scores[c] = dot * scale
        maxScore = max(maxScore, scores[c])
      }
      var sumExp: Float = 0
      for c in 0..<n {
        scores[c] = expf(scores[c] - maxScore)
        sumExp += scores[c]
      }
      for h in 0..<d {
        var acc: Float = 0
        for c in 0..<n { acc += scores[c] * vArray[c * d + h] }
        let expected = acc / sumExp
        let actual = oPointer[r * d + h]
        maxError = max(maxError, abs(expected - actual))
      }
    }
  }
}

// MARK: - Report

let minUs = gpuTimesUs.min()!
let sorted = gpuTimesUs.sorted()
let medianUs = sorted[sorted.count / 2]
// MFA's own metric: (2D + 5) * R * C "instructions" (forward).
let mfaInstrs = Double(2 * opts.headDim + 5) * Double(opts.seq) * Double(opts.seq)
  * Double(opts.heads)
// Conventional attention FLOPs: 2 GEMMs x 2 flops/MAC = 4*R*C*D (x heads).
let flops = 4.0 * Double(opts.seq) * Double(opts.seq) * Double(opts.headDim)
  * Double(opts.heads)

var json: [String: Any] = [
  "impl": "mfa",
  "kernel": "forward",
  "seq": opts.seq,
  "heads": opts.heads,
  "head_dim": opts.headDim,
  "dtype": opts.dtype,
  "iters": opts.iters,
  "warmup": opts.warmup,
  "dispatches_per_cb": opts.dispatches,
  "block_dims": [
    Int(kernel.blockDimensions.parallelization),
    Int(kernel.blockDimensions.traversal),
    Int(kernel.blockDimensions.head),
  ],
  "threadgroup_size": Int(kernel.threadgroupSize),
  "threadgroup_memory_bytes": Int(kernel.threadgroupMemoryAllocation),
  "gpu_time_us": gpuTimesUs.map { ($0 * 100).rounded() / 100 },
  "min_us": (minUs * 100).rounded() / 100,
  "median_us": (medianUs * 100).rounded() / 100,
  "ginstrs_mfa_formula": ((mfaInstrs / (minUs * 1e-6) / 1e9) * 10).rounded() / 10,
  "gflops_4rcd": ((flops / (minUs * 1e-6) / 1e9) * 10).rounded() / 10,
]
if opts.check {
  json["check_max_error"] = maxError
}

let data = try! JSONSerialization.data(
  withJSONObject: json, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
