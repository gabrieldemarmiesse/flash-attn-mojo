#!/bin/bash
# Build bench_ccv_attn against ccv's MFA C++ port (lib/nnc/mfa).
#
# Self-sufficient against a FRESH ccv clone: it (1) applies the macOS 26
# __asm("\01air.*") patch to GEMMHeaders.cpp if missing (the Metal 4
# compiler rejects asm labels containing dots; \01 is LLVM's
# literal-symbol escape and emits the identical AIR symbol), and
# (2) compiles each required object from source when the .o is missing
# or stale — no full ccv `./configure && make` needed.
set -euo pipefail
CCV=${CCV:-/Users/m1/flash-attn-mojo/ccv}
HERE="$(cd "$(dirname "$0")" && pwd)"

GH="$CCV/lib/nnc/mfa/kernels/GEMMHeaders.cpp"
if grep -q '__asm("air\.' "$GH"; then
  echo "patching $GH for macOS 26 (__asm air.* -> \\01air.*)"
  sed -i '' 's/__asm("air\./__asm("\\01air./g' "$GH"
  rm -f "${GH%.cpp}.o"  # force recompile below
fi

CXXFLAGS=(-std=c++17 -O3 -Wall
  -I"$CCV/lib" -I"$CCV/lib/nnc" -I"$CCV"
  -D HAVE_CBLAS -D HAVE_PTHREAD -D HAVE_ACCELERATE_FRAMEWORK
  -D USE_DISPATCH -D HAVE_MPS)

OBJS=(
  "$CCV/lib/nnc/mfa/kernels/AttentionKernel"
  "$CCV/lib/nnc/mfa/kernels/AttentionKernel+Precompiled"
  "$CCV/lib/nnc/mfa/kernels/AttentionDescriptor"
  "$CCV/lib/nnc/mfa/kernels/AttentionKernelDescriptor"
  "$CCV/lib/nnc/mfa/kernels/CodeWriter"
  "$CCV/lib/nnc/mfa/kernels/GEMMHeaders"
  "$CCV/lib/nnc/mfa/Metal"
  "$CCV/lib/nnc/mfa/ccv_nnc_mfa_error"
)
LINK=()
for base in "${OBJS[@]}"; do
  if [[ ! -f "$base.o" || "$base.cpp" -nt "$base.o" ]]; then
    echo "compiling ${base#"$CCV/"}.cpp"
    clang++ "${CXXFLAGS[@]}" -c "$base.cpp" -o "$base.o"
  fi
  LINK+=("$base.o")
done

clang++ -std=c++17 -O2 -Wall -I"$CCV/lib" \
  "$HERE/bench_ccv_attn.cpp" \
  "${LINK[@]}" \
  -framework Metal -framework Foundation -framework QuartzCore \
  -o "$HERE/bench_ccv_attn"
echo "built $HERE/bench_ccv_attn"
