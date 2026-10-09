<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# Gemmini integration organization

Work belongs on `gemmini/bringup`. The isolated host build, synthetic frontend checks and ResNet50 v1.5 host diagnostics are verified on CPU. An opt-in Relax pass lowers a narrow integer matmul contract to an external C call, validated with a test-only host implementation. Its Gemmini C-library adapter passes numerical checks in the pinned functional simulator. Integrated TVM graph/device execution and pretrained qualification remain pending.

| Future location | Integration responsibility |
| --- | --- |
| `python/tvm/relax/backend/contrib/gemmini.py` | Implemented static integer matmul admission and external-call lowering; broader numerical/layout eligibility and fusion remain future work. |
| `src/relax/backend/contrib/gemmini/` | Emit Gemmini C-library wrappers, specialized objects and symbol bindings while preserving partition semantics. |
| `src/runtime/contrib/gemmini/` | TVM values/calling convention, buffer lifetimes, packing/workspace, dispatch, synchronization and result visibility. |
| `cmake/modules/contrib/Gemmini.cmake` | Optional codegen/runtime build wiring; no build module is installed. |
| `tests/python/relax/test_backend_contrib_gemmini.py` | Implemented host-reference admission, rejection, generated-call, control-flow and error-propagation checks; device checks remain future work. |

`apps/gemmini/` holds the host verifier and will hold standalone integration examples and build/run documentation as implementation becomes available. `verify_host.py` verifies checkout identity, LLVM code generation and CPU execution against NumPy, and records loaded-library/build identity. Use `examples/gemmini/comparisons/tvm/HOST_SETUP.md` in the parent comparison repository for the pinned environment, host verification commands and provenance.

The host recipe opts into `USE_HOST_ONLY_AUTO_COPY_GUARD` (OFF by default) because this checkout lacks the `LowerAutoCopy` implementation required by existing driver/MetaSchedule callers. `auto_copy_guard.cc` preserves unannotated IR and rejects automatic-copy markers; it is a validation-only pass, supplies no optimization, and must not coexist with the full implementation. The host verifier checks preservation and rejection through direct calls and ordinary `tvm.build`; record this local patch and enabled mode with the base commit.

`verify_onnx.py --output-dir DIR` checks matmul, batched matmul, convolution, LayerNorm and RMSNorm at opsets 17 and 18 against PyTorch and ONNX ReferenceEvaluator. `--suite importer` instead checks shape arithmetic, constant/runtime negative Gather indices, lower-rank Expand and scalar ConstantOfShape against independent NumPy expectations. Integer outputs require exact equality, including an int64 value beyond the int32 range. Both suites retain graphs, imported IR, arrays and dependency/library provenance.

Optional `--case`, `--opset` and `--importer-source` narrow checks or isolate a historical Python importer on the current runtime. Historical negative-Gather overrides retain imported IR but do not execute potentially unchecked indices; their status is `not_executed`, never a pass. The original v0.19 importer fails the opset-18 RMSNorm case numerically and fails the shape-array, lower-rank Expand and scalar ConstantOfShape cases during import. A local follow-up preserves the NumPy dtype when folding shape arithmetic, preventing the inherited binary fix from silently narrowing int64 output. These cases do not exercise every inherited change or qualify model/device behavior.

The selected backend route calls the Gemmini C operator library, following the comparison team's fairness guidance. Preserve shapes, layouts, quantization scales, rounding and output semantics; make supported shapes/layouts, argument contracts, numerical rules, workspace requirements and instruction policy explicit. Agustin requires every implementation, including this baseline, to avoid FSM/hardware-loop instructions. The handwritten compiler is the evaluated Phase-1/2 implementation, not a baseline dependency or configuration authority. The library owns instruction encoding; the intended optimized backend gives TVM control over tiling, transfers and reuse through library primitives. The whole-operation adapter remains a reference path. Pin headers to the actual hardware; a local `gemmini_params.h` must not be assumed to describe signed-int8 arithmetic.

Agustin permits baremetal or RTOS; this integration chooses baremetal. Host export/reload alone proves neither RISC-V nor baremetal deployment. The independent CPU probe establishes a bounded static generated-graph route under generic Spike, not general Relax runtime support. Confirm actual deployment ISA/ABI, startup and addressability; hardware revision, generated headers, quantization and timing boundaries remain explicit inputs. MX Gemmini requires a different hardware and numerical contract. Expose CPU work, data conversion, transfer and completion costs to the comparison harness.

## External matmul compiler boundary

Apply `tvm.relax.backend.contrib.gemmini.LowerGemminiMatmul()` before `LegalizeOps` and build for a CPU target. The pass admits positive, static, rank-two signed-int8 `[M,K] @ [K,N]` matmul with signed-int32 `[M,N]` output and `K <= 131071`. That bound makes every product and partial sum representable for all signed-int8 inputs. Dynamic shapes, batches, other dtypes and explicitly non-CPU tensor placement remain ordinary Relax operations. Existing `Codegen`/`Composite` functions are preserved. No quantization, bias, activation, scale or transpose fusion is implied, and a float ResNet graph receives no acceleration from this pass.

The generated TIR wrapper calls the synchronous ABI in [matmul.h](matmul.h), using compact row-major element strides `K,N,N`. Link an implementation when exporting the executable. Zero status means complete visible output; nonzero status raises a TVM runtime error. The call and status check remain active when optional TIR assertions are disabled, although callers should retain the normal shape/stride checks. Inputs must be compact zero-offset buffers, and the implementation must preserve them. The wrapper uses the ordinary TVM runtime error API; it does not establish a freestanding entrypoint.

The tests export/reload both bytecode and compiled Relax executables and explicitly link a C implementation embedded in the test file. That implementation accumulates independently in int64, records call counts and injects failures. It is a host oracle, not the Gemmini library's CPU fallback or a production device implementation. Missing external symbols do not provide an automatic CPU substitute for admitted calls. Unsupported operations retain TVM's normal CPU path. In an activated host environment, run:

```sh
python tests/python/relax/test_backend_contrib_gemmini.py -v
```

The test requires LLVM-enabled TVM, NumPy and `cc` on PATH. It covers rectangular shapes, negative/extreme inputs, the admitted reduction bound, repeated calls, frozen constants, wrapper reuse, CPU fallback, function/control-flow preservation and failure propagation. No accelerator headers, simulator or FPGA are used.

## Pinned C-library adapter

[matmul.c](matmul.c) implements the external ABI through `tiled_matmul` with primitive OS and fixed one-array I/J/K tiles, null bias, identity input/output scales, no activation and full int32 output. It rejects invalid dimensions, strides, unrepresentable byte spans and output/input overlap before issuing accelerator instructions. Callers must supply mapped, DMA-accessible buffers and exclusive accelerator ownership. CPU/compiler barriers surround the library's device call and fence; platform coherence remains a separate qualification.

Hardware loops are forbidden, with explicit `TVM_GEMMINI_FORBID_HW_LOOPS=1` and `TVM_GEMMINI_PE_OUTPUT_BITS=20`. The provisional Scala configuration has 20-bit PE output, which the C parameter header does not expose. Fixed DIM=16 and tile_K=1 keep each PE reduction representable for arbitrary int8 inputs, while outer K tiles accumulate in int32. An arbitrary automatic OS tile choice does not preserve this guarantee. The canonical integer header lacks `NORM_STAT_IDS`, which the operator header references even for no-activation calls. An integration-only fallback allows those unused branches to parse while rejecting `HAS_NORMALIZATIONS`. The canonical parameter header is unchanged and normalization is not supported.

[verify_matmul.py](verify_matmul.py) extracts only the selected Git blobs from existing local repositories. Its default source set is Gemmini `6ad65b90b1eb270c20ce4f04109e3dde0180b36f`, gemmini-rocc-tests `7c540b3adf1b86ad93d07f893abe3a73489b568e` and libgemmini `ea8f7ed7afd68e001fb06ddccad9a023c990961d`. It validates parameter/operator digests, stages the exact nested includes, cross-compiles RV64GC/lp64d and records resolved header, source, compiler, object and disassembly identities. It checks primitive-only instruction policy as well as rejection of incompatible headers. The final object must have no FSM instructions in any executable section; the same audit must be applied again to the final linked ELF. This verifies compilation, not a simulator/plugin build or numerical device execution.

From the comparison checkout, set the actual reference checkout and RISC-V toolchain paths:

```sh
GEMMINI_ROOT=/path/to/chipyard/generators/gemmini
RISCV_BIN=/path/to/riscv-tools/bin
TVM_ROOT="$PWD/third_party/baselines/tvm-gemmini"
BUILD_ROOT="$PWD/out/build/baselines/tvm-gemmini"
python "$TVM_ROOT/apps/gemmini/verify_matmul.py" --gemmini-repo "$GEMMINI_ROOT" --build-root "$BUILD_ROOT" \
  --cc "$RISCV_BIN/riscv64-unknown-elf-gcc" --objdump "$RISCV_BIN/riscv64-unknown-elf-objdump"
```

All required Git objects, including the pinned nested rocc-software commit, must already exist locally. No fetch, checkout, reference-header edit or TVM/RTL rebuild occurs. Use `--rocc-tests-repo` or `--rocc-software-repo` for separate checkouts and `--output-dir` for a fresh child of the selected build root. Link only the audited `matmul.o`: a relocatable link keeps the ABI entrypoint and its relocation dependencies while removing unused vendor WS code from `matmul.raw.o`. The raw object is diagnostic and must not be linked. Pass `--audit-elf /path/to/final.elf` to check all executable sections again after final linking. The no-FSM policy is confirmed. Jack identifies default Gemmini with a Rocket host; the exact FireSim build, matching generated headers and guest capacity remain unbound, and these source pins do not adopt gemmini-mlir as a baseline.

## TVM-owned primitive schedules

`LowerGemminiScheduledMatmul(tile_i=1, tile_j=1)` is a separate opt-in pass for the same static signed-int8/int32 matmul contract. Each macro dimension admits 1, 2 or 4 array tiles. The mathematical TensorIR template retains operand tiles across multiple outputs; genuine `TensorIntrin` substitutions replace copies, first-chunk overwrite, subsequent accumulation and stores with the C primitives. TVM emits the surrounding loops and row addresses. Reductions stay within 16 products per signed20 PE operation, with global accumulation in int32 SRAM.

`gemmini_schedule.make_semantic_matmul`, `tensorize_gemmini_matmul` and `make_gemmini_matmul` expose semantic IR, tensorized IR, the actual substitution trace and resource/call metadata. The trace replays tensorization of the generated tiled template; it is not a split/cache/reorder trace from an untiled TE matmul. Fixed tail descriptors preserve exact extents. `LowerMatchBuffer` must run before default buffer compaction to retain physical 16×16 slot strides. The pinned compiler's optional cached-region debug verifier fails on a partial-update case; normal structural matching, sref checks, IR well-formed checks and numerical tests remain active.

Eight native tests cover all nine tile choices, full/partial shapes, the maximum exact reduction, reuse, replay and rejection controls. Their independent C primitive emulator checks row bounds, initialization, operand rectangles and PE/accumulator overflow. For a 64×64×33 matmul, A and B loads each fall from 48 to 24 to 12 for 1×1, 2×2 and 4×4 choices, with identical computation and outputs. Counts are structural evidence, not timing. Run `python tests/python/relax/test_backend_contrib_gemmini_schedule.py` in the activated LLVM host environment.

Apply the pass before `LegalizeOps` and link the audited primitive adapter. Unsupported calls retain ordinary host lowering. This first pass introduces an external boundary before graph fusion; it does not schedule arbitrary fused contractions, implement overlap or perform learned tuning. The whole-call `LowerGemminiMatmul` remains the reference path. Full graph/operator optimization and target-measured search remain paper-fidelity requirements.

## Primitive adapter numerical verification

The header also exposes CPU-only whole-operation validation and `begin`, `load_a`, `load_b`, `compute`, `store` and `end` primitives for compiler-generated schedules. TVM supplies plain scratchpad/accumulator row allocations, tile extents and reduction ordering; these wrappers only configure the device and emit vendor-library primitive instructions. Validate all user buffers before `begin`, keep A/B immutable through `end`, and prove the row/dimension contract in [matmul.h](matmul.h). Each compute restarts a reduction of at most 16 products in the signed20 PE; later chunks add into int32 accumulator SRAM. First-chunk overwrite and subsequent accumulation are distinct operations. The adapter verifier retains and audits every public primitive, including uncalled ones; never link the raw object containing unused vendor FSM code.

[verify_matmul_runtime.py](verify_matmul_runtime.py) links the audited adapter object into baremetal programs and executes them with an explicitly selected, source-bound Gemmini Spike plugin. It requires passing adapter and simulator build receipts, verifies their source/tool/header bindings before and after execution, and audits every executable section of the final ELFs for prohibited FSM instructions. The simulator builder lives in the comparison repository at `examples/gemmini/comparisons/tvm/build_simulator.py`.

```sh
python "$TVM_ROOT/apps/gemmini/verify_matmul_runtime.py" --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$SIMULATOR_BUILD/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$SIMULATOR_BUILD/libgemmini.so" --output-dir "$BUILD_ROOT/matmul-runtime-check"
```

Set each path to the selected build/tool installation and use a fresh output directory. The checked suite covers 13 valid calls including K=131071, partial tiles, padded strides, input preservation, output guards, overwrite and retained outputs, plus 18 invalid calls and recovery. Missing-extension and deliberately wrong-oracle controls must fail. The numerical oracle uses independent int64 arithmetic. This is functional qualification of the separate C-library adapter; it establishes neither TVM graph integration, RTL bit-exactness, platform coherence nor timing.

## Bounded baremetal CPU proof

[baremetal.py](baremetal.py) provides the reusable `plan_graph` and `export_graph` interfaces for straight-line, static Relax `call_tir` graphs. Optimize and legalize the graph before export. It derives raw-pointer calls, a single constants blob, caller-owned outputs and an aligned workspace reused after each tensor's last use. Integer/FP32/bool tensors, multiple inputs and tuple outputs are supported; dynamic shapes, control flow and hidden scalar ABIs are rejected. Export writes the operator object, C orchestration, constant assembly/blob, IR and `graph.json` into the supplied output directory.

The generated `model_run(inputs, outputs, workspace, workspace_bytes)` validates alignment, byte spans and writable-buffer overlap before dispatch. The manifest's memory limit covers explicit tensors only; ELF sections, operator stack allocations and platform reservations must be added for deployment. Four native tests verify mixed dtypes, repeated calls, tuple retention/copies, workspace reuse, fused FP32 semantics and rejection paths. This exporter has not yet qualified an integrated Gemmini graph or full model. In the activated host environment, run `python tests/python/relax/test_backend_contrib_gemmini_baremetal.py`.

[verify_baremetal_cpu.py](verify_baremetal_cpu.py) compiles a fixed FP32 matmul/add/ReLU Relax graph into RISC-V CPU operations and derives static calls/constants from its legalized bindings. It links machine-mode startup, HTIF console/exit and caller-owned buffers without guest libc, libtvm, a C++ runtime or heap. Eight calls across five inputs must match an independent scalar oracle exactly, preserve input/constant/previous-output bytes and buffer canaries, and recover from a null-input rejection. A deliberately incorrect oracle must produce a failing target exit. Dynamic shapes, other dtypes, tuple outputs and unlegalized graphs are rejected.

This is an independent, narrow deployment proof, not a general Relax VM replacement or a complete-model exporter. Only this graph is numerically qualified. The verifier records generated IR, commands, tool/library/source identities, ELF memory bounds, one resident constant copy, reused workspace and a painted-stack high-water observation. It requests 256 MiB in generic Spike; that is not a statement about deployed DRAM. No Gemmini plugin or accelerator execution is involved.

In the activated host environment, from the comparison checkout, supply the actual installed tool paths:

```sh
TVM_ROOT="$PWD/third_party/baselines/tvm-gemmini"
BUILD_ROOT="$PWD/out/build/baselines/tvm-gemmini"
RISCV_BIN=/path/to/riscv-tools/bin
SPIKE_SUPPORT=/path/to/spike-dependencies
python "$TVM_ROOT/apps/gemmini/verify_baremetal_cpu.py" --tvm-source "$TVM_ROOT" --tvm-build "$BUILD_ROOT/host" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --output-dir "$BUILD_ROOT/baremetal-cpu-check"
```

The output directory must be new. Spike's selected library directory must provide its compatible `libstdc++.so.6`; the verifier supplies this path and the selected `dtc` to the Spike process without sourcing global Chipyard setup. Review the recorded limits before extending this proof to additional graphs or a real platform startup/transport.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Prove small matmul and convolution cases using the selected target's arithmetic before full-model device execution. The proposed signed-int8/int32 contract requires matching hardware and generated headers; it does not apply to a floating-point configuration. Keep model inputs and quality thresholds fixed through performance tuning; functional simulator evidence and qualified timing evidence serve distinct roles. Final comparison timing comes from FireSim, with platform and capture details coordinated by the comparison team. Keep full-model inputs, comparisons and measurement orchestration in the parent comparison repository. Build products, environments, model weights, captures and outputs stay outside this source tree, using the parent's configured `out/build/baselines/tvm-gemmini/` and other `out/` roots.
