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

Work belongs on `gemmini/bringup`. The isolated host compiler/runtime build, twenty synthetic frontend CPU checks and a full ResNet50 v1.5 architecture diagnostic with random weights and two synthetic images are verified on host CPU. Pretrained/full-session qualification and Gemmini execution remain pending; no Gemmini backend is registered. The source locations below are planned and unimplemented.

| Future location | Integration responsibility |
| --- | --- |
| `python/tvm/relax/backend/contrib/gemmini/` | Operation inventory, numerical/layout eligibility, and Relax partitioning/fusion based on backend capabilities, with explicit host fallback. |
| `src/relax/backend/contrib/gemmini/` | Emit Gemmini C-library wrappers, specialized objects and symbol bindings while preserving partition semantics. |
| `src/runtime/contrib/gemmini/` | TVM values/calling convention, buffer lifetimes, packing/workspace, dispatch, synchronization and result visibility. |
| `cmake/modules/contrib/Gemmini.cmake` | Optional codegen/runtime build wiring; no build module is installed. |
| `tests/python/contrib/test_gemmini/` | Frontend fidelity, partition admission/refusal, numerical/layout contracts, generated-call bindings and accelerator completion. |

`apps/gemmini/` holds the host verifier and will hold standalone integration examples and build/run documentation as implementation becomes available. `verify_host.py` verifies checkout identity, LLVM code generation and CPU execution against NumPy, and records loaded-library/build identity. Use `examples/gemmini/comparisons/tvm/HOST_SETUP.md` in the parent comparison repository for the pinned environment, host verification commands and provenance.

The host recipe opts into `USE_HOST_ONLY_AUTO_COPY_GUARD` (OFF by default) because this checkout lacks the `LowerAutoCopy` implementation required by existing driver/MetaSchedule callers. `auto_copy_guard.cc` preserves unannotated IR and rejects automatic-copy markers; it is a validation-only pass, supplies no optimization, and must not coexist with the full implementation. The host verifier checks preservation and rejection through direct calls and ordinary `tvm.build`; record this local patch and enabled mode with the base commit.

`verify_onnx.py --output-dir DIR` checks matmul, batched matmul, convolution, LayerNorm and RMSNorm at opsets 17 and 18 against PyTorch and ONNX ReferenceEvaluator. `--suite importer` instead checks shape arithmetic, constant/runtime negative Gather indices, lower-rank Expand and scalar ConstantOfShape against independent NumPy expectations. Integer outputs require exact equality, including an int64 value beyond the int32 range. Both suites retain graphs, imported IR, arrays and dependency/library provenance.

Optional `--case`, `--opset` and `--importer-source` narrow checks or isolate a historical Python importer on the current runtime. Historical negative-Gather overrides retain imported IR but do not execute potentially unchecked indices; their status is `not_executed`, never a pass. The original v0.19 importer fails the opset-18 RMSNorm case numerically and fails the shape-array, lower-rank Expand and scalar ConstantOfShape cases during import. A local follow-up preserves the NumPy dtype when folding shape arithmetic, preventing the inherited binary fix from silently narrowing int64 output. These cases do not exercise every inherited change or qualify model/device behavior.

The selected backend route calls the Gemmini C operator library, following the comparison team's fairness guidance. Preserve shapes, layouts, quantization scales, rounding and output semantics; make supported shapes/layouts, argument contracts, numerical rules, workspace requirements and instruction policy explicit. The handwritten compiler remains a separate reference, and its zero hardware-loop policy does not automatically apply to this baseline. Hardware instruction encoding and device schedules belong to the selected C library. Pin headers to the actual hardware; a local `gemmini_params.h` must not be assumed to describe stock signed-int8 arithmetic.

RISC-V Linux with the Relax VM is the preferred deployment proposal, subject to platform support and memory capacity. Confirm host ISA/ABI, Linux/runtime support and addressability; hardware revision, generated headers, quantization and timing boundaries remain explicit inputs. MX Gemmini requires a different hardware and numerical contract. Expose CPU work, data conversion, transfer and completion costs to the comparison harness.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Prove small matmul and convolution cases using the selected target's arithmetic before full-model device execution. The proposed signed-int8/int32 contract requires matching hardware and generated headers; it does not apply to a floating-point configuration. Keep model inputs and quality thresholds fixed through performance tuning; functional simulator evidence and qualified timing evidence serve distinct roles. Final comparison timing comes from FireSim, with platform and capture details coordinated by the comparison team. Keep full-model inputs, comparisons and measurement orchestration in the parent comparison repository. Build products, environments, model weights, captures and outputs stay outside this source tree, using the parent's configured `out/build/baselines/tvm-gemmini/` and other `out/` roots.
