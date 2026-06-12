// bench_ccv_attn.cpp — standalone forward-attention driver for ccv's MFA C++ port.
//
// Links directly against the AttentionKernel/AttentionDescriptor classes in
// ccv/lib/nnc/mfa/kernels (the C++ port of philipturner/metal-flash-attention v2,
// extended by liuliu with causal/mask/varlen/sinks/sliding-window/GQA).
//
// Modes:
//   bench (default): runs forward attention, kernel-only GPU time via
//                    MTLCommandBuffer GPUStartTime/GPUEndTime, JSON to stdout.
//   --dump-source F: writes the generated MSL for the selected config to F and exits.
//   --check:         also computes a CPU reference and reports max abs error.
//
// Usage:
//   bench_ccv_attn [--b B] [--r R] [--c C] [--hq H] [--hk H] [--d D]
//                  [--causal] [--iterations N] [--warmup W]
//                  [--check] [--dump-source out.metal]
//
// Buffer ABI (must match ccv_nnc_mfa_attention.cpp): Q=0, K=1, V=2, O=3, L=4.
// Inputs are FP16 (lowPrecisionInputs); O is written FP32 by the kernel
// (the MFA reference design always pages O as FP32; ccv casts afterwards).

#include "nnc/mfa/kernels/AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernel.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/PipelineValue.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

typedef __fp16 f16;

int main(int argc, char** argv) {
  uint32_t B = 1, R = 1024, C = 1024, Hq = 16, Hk = 16, D = 128;
  int iterations = 20, warmup = 5, dispatches = 5;
  bool causal = false, check = false;
  std::string dumpSource;

  for (int i = 1; i < argc; i++) {
    std::string a(argv[i]);
    auto next = [&]() -> const char* {
      if (++i >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); }
      return argv[i];
    };
    if (a == "--b") B = atoi(next());
    else if (a == "--r") R = atoi(next());
    else if (a == "--c") C = atoi(next());
    else if (a == "--hq") Hq = atoi(next());
    else if (a == "--hk") Hk = atoi(next());
    else if (a == "--d") D = atoi(next());
    else if (a == "--iterations") iterations = atoi(next());
    else if (a == "--warmup") warmup = atoi(next());
    else if (a == "--dispatches") dispatches = atoi(next());
    else if (a == "--causal") causal = true;
    else if (a == "--check") check = true;
    else if (a == "--dump-source") dumpSource = next();
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); exit(1); }
  }
  if (Hq % Hk != 0) { fprintf(stderr, "Hq must be divisible by Hk\n"); exit(1); }

  auto pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (!device) { fprintf(stderr, "no Metal device\n"); exit(1); }

  // Mirror the descriptor setup in ccv/lib/nnc/mfa/ccv_nnc_mfa_attention.cpp
  // (the params.type == 0 generic AttentionKernel forward path, FP16 inputs).
  AttentionDescriptor attentionDesc;
  attentionDesc.lowPrecisionInputs = true;          // FP16 Q/K/V
  attentionDesc.isBF16 = false;
  attentionDesc.lowPrecisionIntermediates = true;   // ccv default (no upcast flag)
  attentionDesc.matrixDimensions[0] = R;
  attentionDesc.matrixDimensions[1] = C;
  attentionDesc.matrixDimensions[2] = D;
  attentionDesc.transposeState[0] = false;
  attentionDesc.transposeState[1] = false;
  attentionDesc.transposeState[2] = false;
  attentionDesc.transposeState[3] = false;
  attentionDesc.Hq = Hq;
  attentionDesc.Hk = Hk;
  attentionDesc.batchDimension = B;
  attentionDesc.scale = 1.0f / sqrtf((float)D);
  attentionDesc.isCausal = causal;
  attentionDesc.masked = false;
  attentionDesc.isVarlen = false;
  attentionDesc.attentionSinks = false;
  attentionDesc.slidingWindow = 0;
  if (B > 1) {
    attentionDesc.batchStrides[AttentionOperand::Q] = R * D * Hq;
    attentionDesc.batchStrides[AttentionOperand::K] = C * D * Hk;
    attentionDesc.batchStrides[AttentionOperand::V] = C * D * Hk;
    attentionDesc.batchStrides[AttentionOperand::O] = R * D * Hq;
  }
  simd::uint4 leadingDimensions;
  leadingDimensions[0] = Hq * D;
  leadingDimensions[1] = Hk * D;
  leadingDimensions[2] = Hk * D;
  leadingDimensions[3] = Hq * D;
  attentionDesc.leadingDimensions = leadingDimensions;
  attentionDesc.type = AttentionKernelType::forward;

  DeviceProperties dprops = DeviceProperties();
  std::unordered_map<AttentionKernelDescriptor, std::unique_ptr<AttentionKernel>> libraryCache;
  auto found = attentionDesc.findKernel(device, dprops, /*binaryArchivesToRead=*/nullptr,
                                        /*binaryArchiveToWrite=*/nullptr, /*pathToWrite=*/"",
                                        &libraryCache);
  AttentionKernel* kernel = found.second->kernel;
  auto pipeline = found.second->pipeline;

  // Diagnose which compile path the constructor took: try compiling
  // kernel->source with the runtime compiler ourselves.
  bool runtimeCompileOK = false;
  {
    NS::Error* error = nil;
    auto string = NS::String::string(kernel->source.c_str(), NS::UTF8StringEncoding);
    auto probe = NS::TransferPtr(device->newLibrary(string, nil, &error));
    runtimeCompileOK = (probe.get() != nullptr && !error);
  }
  fprintf(stderr, "disable_async_copy=%d runtime_compile_ok=%d\n",
          (int)kernel->disableAsyncCopy, (int)runtimeCompileOK);

  if (!dumpSource.empty()) {
    std::ofstream out(dumpSource, std::ios::binary);
    out << "#include <metal_stdlib>\n\n";
    const std::string& src = kernel->source;
    out << (src.size() > 0 && src[0] == '\n' ? src.substr(1) : src) << "\n";
    out.close();
    fprintf(stderr, "wrote %s (blockDims %ux%ux%u, tgmem %u B, tgsize %u)\n",
            dumpSource.c_str(), kernel->blockDimensions[0], kernel->blockDimensions[1],
            kernel->blockDimensions[2], kernel->threadgroupMemoryAllocation,
            kernel->threadgroupSize);
    return 0;
  }

  // Host data: deterministic pseudo-random fp16 in [-0.5, 0.5).
  const size_t qElems = (size_t)B * R * Hq * D;
  const size_t kElems = (size_t)B * C * Hk * D;
  std::vector<f16> qHost(qElems), kHost(kElems), vHost(kElems);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (auto& x : qHost) x = (f16)dist(rng);
  for (auto& x : kHost) x = (f16)dist(rng);
  for (auto& x : vHost) x = (f16)dist(rng);

  auto qBuf = device->newBuffer(qHost.data(), qElems * sizeof(f16), MTL::ResourceStorageModeShared);
  auto kBuf = device->newBuffer(kHost.data(), kElems * sizeof(f16), MTL::ResourceStorageModeShared);
  auto vBuf = device->newBuffer(vHost.data(), kElems * sizeof(f16), MTL::ResourceStorageModeShared);
  // O is FP32 in memory for the low-precision-input forward kernel.
  auto oBuf = device->newBuffer(qElems * sizeof(float), MTL::ResourceStorageModeShared);
  auto lBuf = device->newBuffer((size_t)B * R * Hq * sizeof(float), MTL::ResourceStorageModeShared);

  auto queue = device->newCommandQueue();
  auto ceilDivide = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  MTL::Size gridSize(ceilDivide((int64_t)R, kernel->blockDimensions[0]) * Hq * B, 1, 1);
  MTL::Size groupSize((int64_t)kernel->threadgroupSize, 1, 1);

  // `dispatches` back-to-back dispatches per command buffer: Metal's
  // automatic hazard tracking on oBuf/lBuf serializes them without CPU
  // gaps, and the GPU stays at a steady power state (single-dispatch
  // command buffers flap 5-50x on M-series). Same methodology as
  // bench_mfa / MFA's own testPerformance.
  auto encodeOnce = [&](MTL::CommandBuffer* cmdbuf) {
    auto encoder = cmdbuf->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(qBuf, 0, AttentionOperand(AttentionOperand::Q).bufferIndex());
    encoder->setBuffer(kBuf, 0, AttentionOperand(AttentionOperand::K).bufferIndex());
    encoder->setBuffer(vBuf, 0, AttentionOperand(AttentionOperand::V).bufferIndex());
    encoder->setBuffer(oBuf, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
    encoder->setBuffer(lBuf, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
    for (int i = 0; i < dispatches; i++)
      encoder->dispatchThreadgroups(gridSize, groupSize);
    encoder->endEncoding();
  };

  auto runOnce = [&]() -> double {
    auto cmdbuf = queue->commandBuffer();
    encodeOnce(cmdbuf);
    cmdbuf->commit();
    cmdbuf->waitUntilCompleted();
    if (cmdbuf->status() == MTL::CommandBufferStatusError) {
      fprintf(stderr, "command buffer error\n");
      exit(1);
    }
    return (cmdbuf->GPUEndTime() - cmdbuf->GPUStartTime()) / dispatches;
  };

  for (int i = 0; i < warmup; i++) (void)runOnce();
  std::vector<double> times(iterations);
  for (int i = 0; i < iterations; i++) times[i] = runOnce();

  double mean = 0;
  for (double t : times) mean += t;
  mean /= iterations;
  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[iterations / 2];
  double tmin = sorted.front();

  // FLOP model: 2*R*C*D (QK^T) + 2*R*C*D (PV) per head per batch; /2 if causal.
  double flops = 4.0 * B * Hq * (double)R * (double)C * (double)D * (causal ? 0.5 : 1.0);

  double maxAbsErr = -1.0;
  if (check) {
    const float* oGpu = (const float*)oBuf->contents();
    const uint32_t ratio = Hq / Hk;
    const float scale = attentionDesc.scale;
    double worst = 0;
    for (uint32_t b = 0; b < B; b++)
      for (uint32_t h = 0; h < Hq; h++) {
        const uint32_t hk = h / ratio;
        for (uint32_t r = 0; r < R; r++) {
          const f16* qRow = &qHost[((size_t)b * R + r) * Hq * D + (size_t)h * D];
          // softmax over C (two-pass, double accum)
          double m = -1e30;
          std::vector<double> s(C, -1e30);
          const uint32_t cLim = causal ? std::min(C, r + 1 + (C > R ? C - R : 0)) : C;
          for (uint32_t c = 0; c < cLim; c++) {
            const f16* kRow = &kHost[((size_t)b * C + c) * Hk * D + (size_t)hk * D];
            double dot = 0;
            for (uint32_t d = 0; d < D; d++) dot += (double)(float)qRow[d] * (double)(float)kRow[d];
            s[c] = dot * scale;
            m = std::max(m, s[c]);
          }
          double l = 0;
          for (uint32_t c = 0; c < cLim; c++) { s[c] = exp(s[c] - m); l += s[c]; }
          for (uint32_t d = 0; d < D; d++) {
            double acc = 0;
            for (uint32_t c = 0; c < cLim; c++) {
              const f16* vRow = &vHost[((size_t)b * C + c) * Hk * D + (size_t)hk * D];
              acc += s[c] * (double)(float)vRow[d];
            }
            acc /= l;
            double got = oGpu[((size_t)b * R + r) * Hq * D + (size_t)h * D + d];
            worst = std::max(worst, fabs(acc - got));
          }
        }
      }
    maxAbsErr = worst;
  }

  printf("{\"impl\":\"ccv-mfa\",\"kind\":\"fwd\",\"B\":%u,\"R\":%u,\"C\":%u,\"Hq\":%u,\"Hk\":%u,"
         "\"D\":%u,\"dtype\":\"fp16\",\"causal\":%s,"
         "\"block_dims\":[%u,%u,%u],\"tg_size\":%u,\"tg_mem\":%u,"
         "\"warmup\":%d,\"iterations\":%d,\"dispatches_per_cb\":%d,",
         B, R, C, Hq, Hk, D, causal ? "true" : "false",
         kernel->blockDimensions[0], kernel->blockDimensions[1], kernel->blockDimensions[2],
         kernel->threadgroupSize, kernel->threadgroupMemoryAllocation,
         warmup, iterations, dispatches);
  printf("\"gpu_time_us\":[");
  for (int i = 0; i < iterations; i++)
    printf("%s%.2f", i ? "," : "", times[i] * 1e6);
  printf("],");
  printf("\"gpu_ms_mean\":%.6f,\"gpu_ms_median\":%.6f,\"gpu_ms_min\":%.6f,"
         "\"min_us\":%.2f,\"median_us\":%.2f,"
         "\"gflops_median\":%.2f",
         mean * 1e3, median * 1e3, tmin * 1e3,
         tmin * 1e6, median * 1e6,
         flops / median / 1e9);
  if (check) printf(",\"check_max_error\":%.3e", maxAbsErr);
  printf("}\n");

  pool->drain();
  return 0;
}
