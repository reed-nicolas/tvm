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

Work belongs on `gemmini/bringup`. The isolated host build, synthetic frontend checks and ResNet50 v1.5 host diagnostics are verified on CPU. An opt-in Relax pass now lowers a narrow integer matmul contract to an external C call, validated with a test-only host implementation. Gemmini C-library binding, target runtime integration, pretrained qualification and accelerator execution remain pending.

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

The selected backend route calls the Gemmini C operator library, following the comparison team's fairness guidance. Preserve shapes, layouts, quantization scales, rounding and output semantics; make supported shapes/layouts, argument contracts, numerical rules, workspace requirements and instruction policy explicit. The handwritten compiler remains a separate reference, and its zero hardware-loop policy does not automatically apply to this baseline. Hardware instruction encoding and device schedules belong to the selected C library. Pin headers to the actual hardware; a local `gemmini_params.h` must not be assumed to describe stock signed-int8 arithmetic.

The reference Gemmini deployment uses baremetal; whether this comparison shares that runtime is awaiting confirmation. Linux with the Relax VM is another unqualified option. Host export/reload proves neither RISC-V nor baremetal deployment. Confirm host ISA/ABI, runtime support and addressability; hardware revision, generated headers, quantization and timing boundaries remain explicit inputs. MX Gemmini requires a different hardware and numerical contract. Expose CPU work, data conversion, transfer and completion costs to the comparison harness.

## External matmul compiler boundary

Apply `tvm.relax.backend.contrib.gemmini.LowerGemminiMatmul()` before `LegalizeOps` and build for a CPU target. The pass admits positive, static, rank-two signed-int8 `[M,K] @ [K,N]` matmul with signed-int32 `[M,N]` output and `K <= 131071`. That bound makes every product and partial sum representable for all signed-int8 inputs. Dynamic shapes, batches, other dtypes and explicitly non-CPU tensor placement remain ordinary Relax operations. Existing `Codegen`/`Composite` functions are preserved. No quantization, bias, activation, scale or transpose fusion is implied, and a float ResNet graph receives no acceleration from this pass.

The generated TIR wrapper calls the synchronous ABI in [matmul.h](matmul.h), using compact row-major element strides `K,N,N`. Link an implementation when exporting the executable. Zero status means complete visible output; nonzero status raises a TVM runtime error. The call and status check remain active when optional TIR assertions are disabled, although callers should retain the normal shape/stride checks. Inputs must be compact zero-offset buffers, and the implementation must preserve them. The wrapper uses the ordinary TVM runtime error API; it does not establish a freestanding entrypoint.

The tests export/reload both bytecode and compiled Relax executables and explicitly link a C implementation embedded in the test file. That implementation accumulates independently in int64, records call counts and injects failures. It is a host oracle, not the Gemmini library's CPU fallback or a production device implementation. Missing external symbols do not provide an automatic CPU substitute for admitted calls. Unsupported operations retain TVM's normal CPU path. In an activated host environment, run:

```sh
python tests/python/relax/test_backend_contrib_gemmini.py -v
```

The test requires LLVM-enabled TVM, NumPy and `cc` on PATH. It covers rectangular shapes, negative/extreme inputs, the admitted reduction bound, repeated calls, frozen constants, wrapper reuse, CPU fallback, function/control-flow preservation and failure propagation. No accelerator headers, simulator or FPGA are used. Matching generated integer headers, library instruction policy and a proven target runtime are prerequisites for implementing the device side of this ABI.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Prove small matmul and convolution cases using the selected target's arithmetic before full-model device execution. The proposed signed-int8/int32 contract requires matching hardware and generated headers; it does not apply to a floating-point configuration. Keep model inputs and quality thresholds fixed through performance tuning; functional simulator evidence and qualified timing evidence serve distinct roles. Final comparison timing comes from FireSim, with platform and capture details coordinated by the comparison team. Keep full-model inputs, comparisons and measurement orchestration in the parent comparison repository. Build products, environments, model weights, captures and outputs stay outside this source tree, using the parent's configured `out/build/baselines/tvm-gemmini/` and other `out/` roots.
