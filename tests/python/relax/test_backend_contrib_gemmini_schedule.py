# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Independent CPU mathematics and primitive-contract tests, not device execution."""

import ctypes
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import LowerGemminiMatmul, LowerGemminiScheduledMatmul, prepare_gemmini_graph
from tvm.relax.backend.contrib.gemmini_schedule import make_gemmini_matmul, make_semantic_matmul, tensorize_gemmini_matmul

# This emulator interprets row-addressed primitive calls independently of TVM
# loop construction. It checks DMA rectangles, loaded bytes, initialized acc
# bytes, row alignment/capacity, PE20 bounds and final int32 bounds. It does not
# model queues, instruction encoding, translation or hardware timing/coherence.
EMULATOR_C = r"""
#include "matmul.h"
#include <limits.h>
#include <stddef.h>
#include <string.h>
static int8_t sp[16384][16];
static int32_t acc[1024][16];
static unsigned char loaded[16384][16], initialized[1024][16];
static const int8_t *abase, *bbase;
static int32_t *cbase;
static int64_t gm, gn, gk, astride, bstride, cstride;
static int active, admitted, faults, forced_status, wrong_mode, counts[7];
void emulator_reset(void) { memset(counts, 0, sizeof(counts)); faults = active = admitted = 0; forced_status = wrong_mode = 0; }
void emulator_fail(int status) { forced_status = status; }
void emulator_wrong(int mode) { wrong_mode = mode; }
int emulator_count(int kind) { return counts[kind]; }
int emulator_faults(void) { return faults; }
int emulator_active(void) { return active; }
static int check(int condition) { if (!condition) faults++; return condition; }
static int tile(uint32_t row, uint32_t rows, uint32_t cols, uint32_t limit) {
    return check(active && !(row % 16) && row + 16 <= limit && rows >= 1 && rows <= 16 && cols >= 1 && cols <= 16);
}
static int rectangle(const void *base, const void *ptr, int64_t height, int64_t width, int64_t stride, uint32_t rows, uint32_t cols, size_t bytes) {
    uintptr_t start = (uintptr_t)base, addr = (uintptr_t)ptr;
    if (!check(addr >= start && !((addr - start) % bytes))) return 0;
    uint64_t offset = (addr - start) / bytes;
    return check(offset / stride + rows <= (uint64_t)height && offset % stride + cols <= (uint64_t)width);
}
int32_t tvm_gemmini_validate_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t as, int64_t bs, int64_t cs) {
    counts[0]++;
    if (forced_status) return forced_status;
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || k > 131071 || as < k || bs < n || cs < n) return -1;
    abase = a; bbase = b; cbase = c; gm = m; gn = n; gk = k; astride = as; bstride = bs; cstride = cs;
    admitted = 1;
    return 0;
}
void tvm_gemmini_begin(int64_t as, int64_t bs, int64_t cs) {
    counts[1]++;
    check(admitted && !active && as == astride && bs == bstride && cs == cstride);
    active = 1; admitted = 0;
    memset(loaded, 0, sizeof(loaded)); memset(initialized, 0, sizeof(initialized));
}
void tvm_gemmini_load_a(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[2]++;
    if (!tile(row, rows, cols, 16384) || !rectangle(abase, src, gm, gk, astride, rows, cols, 1)) return;
    memset(loaded[row], 0, 16 * 16);
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        sp[row + i][j] = src[i * astride + j]; loaded[row + i][j] = 1;
    }
}
void tvm_gemmini_load_b(const int8_t* src, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[3]++;
    if (!tile(row, rows, cols, 16384) || !rectangle(bbase, src, gk, gn, bstride, rows, cols, 1)) return;
    memset(loaded[row], 0, 16 * 16);
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        sp[row + i][j] = src[i * bstride + j]; loaded[row + i][j] = 1;
    }
}
void tvm_gemmini_compute(uint32_t ar, uint32_t br, uint32_t cr, uint32_t m, uint32_t n, uint32_t k, int32_t update) {
    counts[4]++;
    if (!tile(ar, m, k, 16384) || !tile(br, k, n, 16384) || !tile(cr, m, n, 1024) || !check(ar != br && (update == 0 || update == 1))) return;
    for (uint32_t i = 0; i < m; i++) for (uint32_t j = 0; j < n; j++) {
        int64_t partial = 0;
        for (uint32_t r = 0; r < k; r++) {
            check(loaded[ar + i][r] && loaded[br + r][j]);
            int8_t bv = sp[br + r][j];
            if (wrong_mode == 2) bv ^= 1;
            partial += (int64_t)sp[ar + i][r] * bv;
        }
        check(partial >= -524288 && partial <= 524287);
        if (update) check(initialized[cr + i][j]);
        int64_t sum = partial + ((update && wrong_mode != 1) ? acc[cr + i][j] : 0);
        check(sum >= INT32_MIN && sum <= INT32_MAX);
        acc[cr + i][j] = (int32_t)sum; initialized[cr + i][j] = 1;
    }
}
void tvm_gemmini_store(int32_t* dst, uint32_t row, uint32_t rows, uint32_t cols) {
    counts[5]++;
    if (!tile(row, rows, cols, 1024) || !rectangle(cbase, dst, gm, gn, cstride, rows, cols, 4)) return;
    for (uint32_t i = 0; i < rows; i++) for (uint32_t j = 0; j < cols; j++) {
        check(initialized[row + i][j]); dst[i * cstride + j] = acc[row + i][j];
    }
}
void tvm_gemmini_end(void) { counts[6]++; check(active); active = 0; }
"""


def graph(m=17, n=19, k=33, dtype="int8", out_dtype="int32", twice=False):
    a = relax.Var("a", relax.TensorStructInfo((m, k), dtype))
    b = relax.Var("b", relax.TensorStructInfo((k, n), dtype))
    builder = relax.BlockBuilder()
    with builder.function("main", [a, b]):
        with builder.dataflow():
            product = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype))
            duplicate = builder.emit(relax.op.matmul(a, b, out_dtype=out_dtype)) if twice else product
            result = builder.emit_output(relax.op.add(product, duplicate))
        builder.emit_func_output(result)
    return builder.get()


def external_calls(func):
    result = []
    tir.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern") else None)
    return result


def quantized_graph(a_value=None):
    """Frozen weights, integer CPU transforms on both sides, three tuple outputs."""
    m, n, k = 17, 19, 33
    a = relax.Var("a", relax.TensorStructInfo((m, k), "int8"))
    w = relax.Var("w", relax.TensorStructInfo((k, n), "int8"))
    weight = ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype("int8")
    bias = np.arange(n, dtype="int32") - 5
    builder = relax.BlockBuilder()
    with builder.function("main", [a, w]):
        with builder.dataflow():
            wide = builder.emit(relax.op.astype(a, "int32"))
            offset = builder.emit(relax.op.add(wide, relax.const(np.int32(5))))
            bounded = builder.emit(relax.op.clip(offset, -128, 127))
            prepared = builder.emit(relax.op.astype(bounded, "int8"))
            constant_weight = builder.emit(relax.op.add(w, relax.const(np.zeros_like(weight))))
            product = builder.emit(relax.op.matmul(prepared, constant_weight, out_dtype="int32"))
            biased = builder.emit(relax.op.add(product, relax.const(bias)))
            activated = builder.emit(relax.op.nn.relu(biased))
            clipped = builder.emit(relax.op.clip(activated, 0, 300))
            shifted = builder.emit(relax.op.right_shift(clipped, relax.const(np.int32(3))))
            quantized = builder.emit(relax.op.astype(shifted, "int8"))
            output = builder.emit_output(relax.Tuple([quantized, product, prepared]))
        builder.emit_func_output(output)
    bindings = {"w": weight}
    if a_value is not None:
        bindings["a"] = a_value
    return relax.transform.BindParams("main", bindings)(builder.get()), weight, bias


def quantized_oracle(a, weight, bias):
    prepared = np.clip(a.astype("int64") + 5, -128, 127).astype("int8")
    product = prepared.astype("int64") @ weight.astype("int64")
    quantized = (np.clip(np.maximum(product + bias.astype("int64"), 0), 0, 300) >> 3).astype("int8")
    return quantized, product.astype("int32"), prepared


class GemminiScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-schedule-", ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.emulator = self.root / "primitive_emulator.c"
        self.emulator.write_text(EMULATOR_C)
        self.header = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"
        self.index = 0

    def build(self, mod, disable_assert=False):
        with tvm.transform.PassContext(config={"tir.disable_assert": disable_assert}):
            built = tvm.build(mod, "llvm")
        self.index += 1
        path = self.root / f"primitive_{self.index}.so"
        built.export_library(str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11", "-I" + str(self.header)])
        control = ctypes.CDLL(str(path))
        for name in ("emulator_count", "emulator_fail", "emulator_wrong"):
            getattr(control, name).argtypes = [ctypes.c_int]
        control.emulator_reset()
        return tvm.runtime.load_module(str(path)), control, built.get_source("ll")

    def build_graph(self, mod):
        executable = relax.build(mod, "llvm")
        self.index += 1
        path = self.root / f"graph_{self.index}.so"
        executable.export_library(str(path), addons=[str(self.emulator)], cc="cc", options=["-std=c11", "-I" + str(self.header)])
        control = ctypes.CDLL(str(path))
        for name in ("emulator_count", "emulator_fail", "emulator_wrong"):
            getattr(control, name).argtypes = [ctypes.c_int]
        control.emulator_reset()
        return relax.VirtualMachine(tvm.runtime.load_module(str(path)), tvm.cpu()), control

    def check_run(self, scheduled, runtime, control, a, b):
        m, k = a.shape
        n = b.shape[1]
        a_device, b_device = tvm.nd.array(a), tvm.nd.array(b)
        c_device = tvm.nd.array(np.full((m, n), 0x123456, dtype="int32"))
        control.emulator_reset()
        runtime(a_device, b_device, c_device)
        np.testing.assert_array_equal(c_device.numpy(), a.astype("int64") @ b.astype("int64"))
        np.testing.assert_array_equal(a_device.numpy(), a)
        np.testing.assert_array_equal(b_device.numpy(), b)
        self.assertEqual(control.emulator_faults(), 0)
        self.assertEqual(control.emulator_active(), 0)
        expected = [1, 1] + [scheduled.metadata[key] for key in ("load_a", "load_b", "compute", "store")] + [1]
        self.assertEqual([control.emulator_count(i) for i in range(7)], expected)
        return c_device

    def test_all_macro_tiles_full_and_tail_math_and_primitives(self):
        for tile_i, tile_j in itertools.product((1, 2, 4), repeat=2):
            for m, n, k in ((32, 32, 32), (33, 19, 49)):
                with self.subTest(tiles=(tile_i, tile_j), shape=(m, n, k)):
                    semantic = make_semantic_matmul(m, n, k, tile_i, tile_j)
                    self.assertTrue(tir.analysis.verify_well_formed(semantic["main"]))
                    a = ((np.arange(m * k).reshape(m, k) % 256) - 128).astype("int8")
                    b = ((np.arange(k * n).reshape(k, n) * 7 % 256) - 128).astype("int8")
                    semantic_output = tvm.nd.array(np.full((m, n), -111, dtype="int32"))
                    tvm.build(semantic, "llvm")(tvm.nd.array(a), tvm.nd.array(b), semantic_output)
                    np.testing.assert_array_equal(semantic_output.numpy(), a.astype("int64") @ b.astype("int64"))
                    scheduled = tensorize_gemmini_matmul(semantic)
                    self.assertTrue(tir.analysis.verify_well_formed(scheduled.scheduled_mod["main"]))
                    runtime, control, llvm = self.build(scheduled.scheduled_mod)
                    retained = self.check_run(scheduled, runtime, control, a, b)
                    saved = retained.numpy()
                    self.check_run(scheduled, runtime, control, np.zeros_like(a), b)
                    self.check_run(scheduled, runtime, control, a, b)
                    np.testing.assert_array_equal(retained.numpy(), saved)
                    self.assertNotIn("tvm_gemmini_matmul_i8_i32", llvm)
                    self.assertFalse(any("%" + name in llvm for name in ("AL", "BL", "CL")))
                    self.assertNotIn("alloca", llvm)
                    calls = external_calls(scheduled.scheduled_mod["main"])
                    for call in calls:
                        symbol = call.args[0].value
                        if symbol == "tvm_gemmini_compute":
                            self.assertEqual([str(arg.dtype) for arg in call.args[1:]], ["uint32"] * 6 + ["int32"])
                        if symbol in ("tvm_gemmini_load_a", "tvm_gemmini_load_b", "tvm_gemmini_store"):
                            self.assertEqual([str(arg.dtype) for arg in call.args[2:]], ["uint32"] * 3)
                    self.assertTrue(any(inst.kind.name == "Tensorize" for inst in scheduled.trace.insts))
                    self.assertLess(scheduled.lowering_steps.index("LowerMatchBuffer"), scheduled.lowering_steps.index("default TVM lowering"))

    def test_small_shapes_k_tails_and_maximum_exact_reduction(self):
        for m, n, k in ((1, 1, 1), (3, 7, 5), (1, 17, 17), (17, 1, 16), (1, 1, 131071)):
            with self.subTest(shape=(m, n, k)):
                scheduled = make_gemmini_matmul(m, n, k, 2, 2)
                a, b = np.full((m, k), -128, "int8"), np.full((k, n), -128, "int8")
                semantic_output = tvm.nd.array(np.zeros((m, n), "int32"))
                tvm.build(scheduled.semantic_mod, "llvm")(tvm.nd.array(a), tvm.nd.array(b), semantic_output)
                np.testing.assert_array_equal(semantic_output.numpy(), a.astype("int64") @ b.astype("int64"))
                runtime, control, _ = self.build(scheduled.scheduled_mod)
                self.check_run(scheduled, runtime, control, a, b)

    def test_real_reuse_and_distinct_traces(self):
        counts, traces = [], []
        for tile in (1, 2, 4):
            scheduled = make_gemmini_matmul(64, 64, 33, tile, tile)
            runtime, control, _ = self.build(scheduled.scheduled_mod)
            self.check_run(scheduled, runtime, control, np.ones((64, 33), "int8"), np.ones((33, 64), "int8"))
            counts.append((control.emulator_count(2), control.emulator_count(3), control.emulator_count(4), control.emulator_count(5)))
            traces.append(str(scheduled.trace))
        self.assertEqual(counts, [(48, 48, 48, 16), (24, 24, 48, 16), (12, 12, 48, 16)])
        self.assertEqual(len(set(traces)), 3)

    def test_trace_replay_and_mismatched_mathematics_rejection(self):
        semantic = make_semantic_matmul(17, 19, 33, 2, 2)
        result = tensorize_gemmini_matmul(semantic)
        replay = tir.Schedule(semantic, debug_mask=1)
        result.trace.apply_to_schedule(replay, remove_postproc=True)
        replayed = tir.transform.LowerMatchBuffer()(replay.mod)["main"]
        inner = result.scheduled_mod["main"].body.body.else_case.seq[1]
        tvm.ir.assert_structural_equal(replayed.body, inner, map_free_vars=True)

        def change_product(node):
            if isinstance(node, tir.Mul) and isinstance(node.a, tir.Cast) and isinstance(node.b, tir.Cast) and node.dtype == "int32":
                return node + tir.IntImm("int32", 1)
            return None

        changed = semantic["main"].with_body(tir.stmt_functor.ir_transform(semantic["main"].body, None, change_product, ["tir.Mul"]))
        with self.assertRaises(tir.schedule.ScheduleError):
            tensorize_gemmini_matmul(tvm.IRModule({"main": changed}))

    def test_wrong_update_and_operand_controls_fail_oracle(self):
        scheduled = make_gemmini_matmul(17, 19, 33, 2, 2)
        runtime, control, _ = self.build(scheduled.scheduled_mod)
        a, b = np.ones((17, 33), "int8"), np.ones((33, 19), "int8")
        for mode in (1, 2):
            control.emulator_reset()
            control.emulator_wrong(mode)
            output = tvm.nd.array(np.zeros((17, 19), "int32"))
            runtime(tvm.nd.array(a), tvm.nd.array(b), output)
            self.assertFalse(np.array_equal(output.numpy(), a.astype("int64") @ b.astype("int64")))
            self.assertEqual(control.emulator_faults(), 0)
        self.check_run(scheduled, runtime, control, a, b)

    def test_admission_precedes_device_calls_with_assertions_disabled(self):
        scheduled = make_gemmini_matmul(3, 7, 5)
        runtime, control, _ = self.build(scheduled.scheduled_mod, disable_assert=True)
        a, b = tvm.nd.array(np.ones((3, 5), "int8")), tvm.nd.array(np.ones((5, 7), "int8"))
        output = tvm.nd.array(np.full((3, 7), -123, "int32"))
        control.emulator_fail(-5)
        with self.assertRaisesRegex(tvm.error.TVMError, "scheduled matmul rejected"):
            runtime(a, b, output)
        self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(output.numpy(), -123)
        self.check_run(scheduled, runtime, control, a.numpy(), b.numpy())

    def test_shape_stride_capacity_and_tile_rejections(self):
        invalid = [(0, 1, 1, 1, 1), (1, -1, 1, 1, 1), (1, 1, 131072, 1, 1), (1, 1, True, 1, 1),
                   (1, 1, 1, 8, 1), (1, 1, 1, 1, 64), (1, 1, 1, True, 1), (1, 1 << 30, 1, 1, 1),
                   (1 << 63, 1, 1, 1, 1)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_semantic_matmul(*args)
        scheduled = make_gemmini_matmul(1, 1, 1, 4, 4)
        self.assertEqual((scheduled.metadata["spad_rows"], scheduled.metadata["acc_rows"]), (128, 256))

    def test_relax_pass_reuse_fallback_attributes_and_reference_route(self):
        original = graph(twice=True).with_attr("test_attribute", "preserved")
        lowered = LowerGemminiScheduledMatmul(2, 2)(original)
        self.assertTrue(relax.analysis.well_formed(lowered))
        self.assertEqual(str(lowered.attrs["test_attribute"]), "preserved")
        primitives = [func for func in lowered.functions.values() if isinstance(func, tir.PrimFunc)]
        self.assertEqual(len(primitives), 1)
        self.assertIn("tvm_gemmini_compute", primitives[0].script())
        self.assertNotIn("tvm_gemmini_matmul_i8_i32", primitives[0].script())
        tvm.ir.assert_structural_equal(LowerGemminiScheduledMatmul(2, 2)(lowered), lowered)
        host_adds = []
        relax.analysis.post_order_visit(lowered["main"].body, lambda node: host_adds.append(node) if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.add") else None)
        self.assertEqual(len(host_adds), 1)
        for mod in (graph(dtype="float32", out_dtype="float32"), graph(k=131072), graph(n=1 << 30)):
            tvm.ir.assert_structural_equal(LowerGemminiScheduledMatmul()(mod), mod)
        reference = LowerGemminiMatmul()(original)
        reference_primitives = [func for func in reference.functions.values() if isinstance(func, tir.PrimFunc)]
        self.assertEqual(len(reference_primitives), 1)
        self.assertIn("tvm_gemmini_matmul_i8_i32", reference_primitives[0].script())
        self.assertNotIn("tvm_gemmini_compute", reference_primitives[0].script())

    def test_graph_pipeline_frozen_weight_cpu_fusion_and_tuple_semantics(self):
        original, weight, bias = quantized_graph()
        original = original.with_attr("test_attribute", "preserved")
        plans = [prepare_gemmini_graph(original, 2, 2, optimize=value) for value in (False, True)]
        cpu_functions = []
        boundaries = []
        for mod in plans:
            self.assertTrue(relax.analysis.well_formed(mod))
            self.assertEqual(str(mod.attrs["test_attribute"]), "preserved")
            primitives = [func for func in mod.functions.values() if isinstance(func, tir.PrimFunc)]
            boundary = [func for func in primitives if any(call.args[0].value == "tvm_gemmini_compute" for call in external_calls(func))]
            self.assertEqual(len(boundary), 1)
            self.assertEqual(int(boundary[0].attrs["op_pattern"]), 8)
            boundaries.append(boundary[0])
            cpu_functions.append([func for func in primitives if func != boundary[0]])
        self.assertEqual(len(cpu_functions[1]), 2)
        self.assertGreater(len(cpu_functions[0]), len(cpu_functions[1]))
        cpu_module = tvm.IRModule({gv: func.with_attr("global_symbol", gv.name_hint) for gv, func in plans[1].functions_items() if isinstance(func, tir.PrimFunc) and func != boundaries[1]})
        cpu_llvm = tvm.build(cpu_module, "llvm").get_source("ll")
        for gv in cpu_module.get_global_vars():
            self.assertIn(str(gv.name_hint), cpu_llvm)
        self.assertNotIn("TVMBackendAllocWorkspace", cpu_llvm)
        self.assertNotIn("TVMBackendFreeWorkspace", cpu_llvm)
        tvm.ir.assert_structural_equal(boundaries[0], boundaries[1], map_free_vars=True)
        expected_body = make_gemmini_matmul(17, 19, 33, 2, 2).scheduled_mod["main"].body
        tvm.ir.assert_structural_equal(boundaries[1].body, expected_body, map_free_vars=True)
        calls = []
        relax.analysis.post_order_visit(plans[1]["main"].body, lambda node: calls.append(node) if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.call_tir") else None)
        boundary_call = next(call for call in calls if plans[1][call.args[0]] == boundaries[1])
        self.assertIsInstance(boundary_call.args[1].fields[1], relax.Constant)
        np.testing.assert_array_equal(boundary_call.args[1].fields[1].data.numpy(), weight)

        for mod in plans:
            vm, control = self.build_graph(mod)
            retained = None
            for a in (((np.arange(17 * 33).reshape(17, 33) % 256) - 128).astype("int8"), np.full((17, 33), -128, "int8"), np.full((17, 33), 127, "int8")):
                device_a = tvm.nd.array(a)
                control.emulator_reset()
                outputs = vm["main"](device_a)
                for actual, expected in zip(outputs, quantized_oracle(a, weight, bias)):
                    self.assertEqual(actual.dtype, str(expected.dtype))
                    np.testing.assert_array_equal(actual.numpy(), expected)
                np.testing.assert_array_equal(device_a.numpy(), a)
                self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 1, 6, 6, 12, 4, 1])
                self.assertEqual((control.emulator_faults(), control.emulator_active()), (0, 0))
                if retained is None:
                    retained = [(value, value.numpy()) for value in outputs]
                else:
                    for value, saved in retained:
                        np.testing.assert_array_equal(value.numpy(), saved)
            control.emulator_reset()
            control.emulator_fail(-5)
            with self.assertRaisesRegex(tvm.error.TVMError, "scheduled matmul rejected"):
                vm["main"](tvm.nd.array(np.zeros((17, 33), "int8")))
            self.assertEqual([control.emulator_count(i) for i in range(7)], [1, 0, 0, 0, 0, 0, 0])
            control.emulator_reset()
            recovery = np.zeros((17, 33), "int8")
            for actual, expected in zip(vm["main"](tvm.nd.array(recovery)), quantized_oracle(recovery, weight, bias)):
                np.testing.assert_array_equal(actual.numpy(), expected)

    def test_graph_pipeline_constant_only_folds_before_device_lowering(self):
        a = ((np.arange(17 * 33).reshape(17, 33) % 256) - 128).astype("int8")
        original, weight, bias = quantized_graph(a)
        optimized = prepare_gemmini_graph(original, 2, 2)
        self.assertTrue(relax.analysis.well_formed(optimized))
        self.assertEqual(len(optimized["main"].params), 0)
        self.assertFalse(any(isinstance(func, tir.PrimFunc) for func in optimized.functions.values()))
        vm, control = self.build_graph(optimized)
        for actual, expected in zip(vm["main"](), quantized_oracle(a, weight, bias)):
            np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual([control.emulator_count(i) for i in range(7)], [0] * 7)

    def test_graph_pipeline_rejects_device_input_before_constant_evaluation(self):
        semantic = graph()
        for device in (LowerGemminiScheduledMatmul()(semantic), LowerGemminiMatmul()(semantic)):
            with self.assertRaisesRegex(ValueError, "semantic graph before device lowering"):
                prepare_gemmini_graph(device)
        for kwargs in ({"tile_i": 8}, {"optimize": 1}):
            with self.assertRaises(ValueError):
                prepare_gemmini_graph(semantic, **kwargs)


if __name__ == "__main__":
    unittest.main()
