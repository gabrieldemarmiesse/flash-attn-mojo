#!/bin/bash
# air_dis.sh — dump textual LLVM-IR (AIR) from a .air or .metallib to stdout.
#
# Usage:
#   scripts/air_dis.sh kernel.air                # full metal-objdump output
#   scripts/air_dis.sh kernel.metallib           # works on metallibs too
#   scripts/air_dis.sh --ll kernel.air           # strip objdump headers -> clean,
#                                                #   llvm-as/applegpu-nt-parseable .ll
#
# The AIR analog of our PTX dump: feed two outputs to an op-mix diff
# (grep -oE '^\s+(\w+)' histogramming works on the IR opcodes).
#
# Requires the Xcode Metal Toolchain component (xcodebuild -downloadComponent
# MetalToolchain). metal-objdump autodetects both LLVM-bitcode-wrapper (.air)
# and MetalLib container (.metallib) inputs.
set -euo pipefail

strip_headers=0
if [[ "${1:-}" == "--ll" ]]; then
  strip_headers=1
  shift
fi

if [[ $# -ne 1 || ! -f "${1:-}" ]]; then
  echo "usage: $0 [--ll] <file.air | file.metallib>" >&2
  exit 2
fi

if [[ $strip_headers -eq 1 ]]; then
  # Drop objdump's banner lines: "<file>:\tfile format ...",
  # "Disassembly of section ...", and per-symbol "0x... -- name:" markers.
  xcrun metal-objdump --disassemble "$1" \
    | grep -v -e ':	file format ' -e '^Disassembly of section' -e '^0x[0-9a-fA-F]* --'
else
  xcrun metal-objdump --disassemble "$1"
fi
