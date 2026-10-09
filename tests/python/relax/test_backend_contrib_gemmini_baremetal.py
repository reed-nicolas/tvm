# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native checks of graph-derived static calls, mixed dtypes and live storage."""

import ctypes
from _ctypes import dlclose
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tvm
from tvm import relax, tir

APP = Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"
sys.path.insert(0, str(APP))
from baremetal import export_graph, plan_graph


def aligned(shape, dtype):
    dtype = np.dtype(dtype)
    raw = np.empty(int(np.prod(shape)) * dtype.itemsize + 64, "uint8")
    start = -raw.ctypes.data % 64
    return raw[start:start + int(np.prod(shape)) * dtype.itemsize].view(dtype).reshape(shape)


def graph(dtype="int8", tuple_output=False, shape=(2, 3)):
    x = relax.Var("x", relax.TensorStructInfo(shape, dtype))
    w = np.arange(12, dtype=dtype).reshape(3, 4) - 5
    builder = relax.BlockBuilder()
    with builder.function("main", [x]):
        with builder.dataflow():
            mm = builder.emit(relax.op.matmul(x, relax.const(w), out_dtype="int32" if dtype == "int8" else dtype))
            a = builder.emit(relax.op.add(mm, relax.const(3, "int32" if dtype == "int8" else dtype)))
            b = builder.emit(relax.op.multiply(a, relax.const(2, "int32" if dtype == "int8" else dtype)))
            c = builder.emit(relax.op.subtract(b, relax.const(1, "int32" if dtype == "int8" else dtype)))
            output = builder.emit_output(relax.Tuple([mm, c, x, c]) if tuple_output else c)
        builder.emit_func_output(output)
    return relax.transform.LegalizeOps()(builder.get())


def custom_graph(*, alignment=64, multi_output=False, attribute=None, storage_alignment=None):
    buffers = [tir.decl_buffer((8,), "int32", name=name, data_alignment=alignment) for name in ("x", "dead", "live")]
    x, dead, live = buffers
    index = tir.Var("i", "int32")
    def write(buffer, amount):
        body = tir.BufferStore(buffer, tir.BufferLoad(x, [index]) + amount, [index])
        return tir.For(index, 0, 8, tir.ForKind.SERIAL, body)
    selected = buffers if multi_output else [x, live]
    body = tir.SeqStmt([write(live, 11), write(dead, -37)]) if multi_output else write(live, 11)
    if storage_alignment is not None:
        body = tir.AttrStmt(x.data, "storage_alignment", storage_alignment, body)
    primitive = tir.PrimFunc([buffer.data for buffer in selected], body,
                            buffer_map={buffer.data: buffer for buffer in selected}).with_attr("tir.noalias", True)
    if attribute:
        primitive = primitive.with_attr(*attribute)
    value = relax.Var("x", relax.TensorStructInfo((8,), "int32"))
    builder = relax.BlockBuilder()
    gv = builder.add_func(primitive, "producer")
    with builder.function("main", [value]):
        with builder.dataflow():
            info = relax.TensorStructInfo((8,), "int32")
            result = builder.emit(relax.call_tir(gv, [value], [info, info] if multi_output else info))
            live_result = builder.emit(relax.TupleGetItem(result, 1)) if multi_output else result
            output = builder.emit_output(relax.op.add(live_result, relax.const(1, "int32")))
        builder.emit_func_output(output)
    return relax.transform.LegalizeOps()(builder.get())


class StaticGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gemmini-static-graph-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def compile(self, mod):
        report = export_graph(mod, self.root, target="llvm")
        library = self.root / "model.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-std=c11", "-O2", self.root / "model.c", self.root / "constants.S", self.root / "operators.o", "-o", library], check=True, capture_output=True)
        loaded = ctypes.CDLL(str(library))
        self.addCleanup(dlclose, loaded._handle)
        loaded.model_run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64]
        loaded.model_run.restype = ctypes.c_int32
        inputs = [aligned(item["shape"], item["dtype"]) for item in report["inputs"]]
        outputs = [aligned(item["shape"], item["dtype"]) for item in report["outputs"]]
        workspace = aligned((max(report["workspace_bytes"], 64),), "uint8")
        return report, loaded, inputs, outputs, workspace

    def invoke(self, loaded, inputs, outputs, workspace, size=None):
        ins = (ctypes.c_void_p * len(inputs))(*(array.ctypes.data for array in inputs))
        outs = (ctypes.c_void_p * len(outputs))(*(array.ctypes.data for array in outputs))
        return loaded.model_run(ins, outs, workspace.ctypes.data, workspace.nbytes if size is None else size)

    def test_mixed_dtype_calls_reuse_and_repeated_execution(self):
        report, loaded, inputs, outputs, workspace = self.compile(graph())
        self.assertEqual(len(report["calls"]), 4)
        self.assertEqual(report["workspace_bytes"], 128)
        offsets = [t["offset"] for t in report["intermediates"] if t["origin"] == "workspace"]
        self.assertEqual(offsets, [0, 64, 0])
        weights = np.arange(12, dtype="int64").reshape(3, 4) - 5
        for seed in (0, 1, 0):
            value = (np.arange(6).reshape(2, 3) - 3 + seed).astype("int8")
            inputs[0][:] = value
            outputs[0][:] = -777
            self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
            np.testing.assert_array_equal(outputs[0], (value.astype("int64") @ weights + 3) * 2 - 1)
            np.testing.assert_array_equal(inputs[0], value)

    def test_retained_tuple_results_alias_copies_and_workspace_rejection(self):
        report, loaded, inputs, outputs, workspace = self.compile(graph(tuple_output=True))
        value = np.arange(6, dtype="int8").reshape(2, 3) - 3
        inputs[0][:] = value
        for output in outputs:
            output.fill(17)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace, size=0), -1)
        self.assertTrue(all(np.all(output == 17) for output in outputs))
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = value.astype("int64") @ (np.arange(12).reshape(3, 4) - 5)
        for actual, wanted in zip(outputs, [expected, (expected + 3) * 2 - 1, value, (expected + 3) * 2 - 1]):
            np.testing.assert_array_equal(actual, wanted)
        copies = [output.copy() for output in outputs]
        self.assertEqual(self.invoke(loaded, inputs, [outputs[0]] * 4, workspace), -2)
        for output, before in zip(outputs, copies):
            np.testing.assert_array_equal(output, before)
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        self.assertFalse(report["heap_required_by_orchestration"])

    def test_fused_fp32_graph_keeps_mathematics(self):
        mod = graph(dtype="float32")
        optimized = relax.get_pipeline("zero")(mod)
        report, loaded, inputs, outputs, workspace = self.compile(optimized)
        inputs[0][:] = np.arange(6, dtype="float32").reshape(2, 3) - 3
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        expected = (inputs[0] @ (np.arange(12, dtype="float32").reshape(3, 4) - 5) + 3) * 2 - 1
        np.testing.assert_array_equal(outputs[0], expected)
        self.assertLess(len(report["calls"]), 4)

    def test_large_unscheduled_fused_buffer_requires_rejected_runtime_allocator(self):
        # Large internal buffers lower to indirect runtime callbacks; never load
        # or execute this object without the TVM runtime initializing them.
        optimized = relax.get_pipeline("zero")(graph(dtype="float32", shape=(128, 3)))
        with self.assertRaisesRegex(ValueError, "runtime workspace allocator"):
            export_graph(optimized, self.root, target="llvm")
        self.assertIn("@__TVMBackendAllocWorkspace", (self.root / "operators.ll").read_text())
        self.assertFalse((self.root / "operators.o").exists())
        self.assertFalse((self.root / "graph.json").exists())

    def test_unused_multi_output_sibling_cannot_alias_live_result(self):
        report, loaded, inputs, outputs, workspace = self.compile(custom_graph(multi_output=True))
        first, second = report["intermediates"][:2]
        inputs[0][:] = np.arange(8, dtype="int32")
        self.assertEqual(self.invoke(loaded, inputs, outputs, workspace), 0)
        np.testing.assert_array_equal(outputs[0], inputs[0] + 12)
        self.assertNotEqual(first["offset"], second["offset"])

    def test_stronger_external_alignment_is_rejected(self):
        for mod in (custom_graph(alignment=128), custom_graph(storage_alignment=128)):
            with self.subTest(), self.assertRaisesRegex(ValueError, "alignment"):
                plan_graph(mod)

    def test_unsupported_function_abi_and_target_are_rejected(self):
        for attribute in (("calling_conv", 1), ("calling_conv", 2), ("target", tvm.target.Target("cuda -arch=sm_50"))):
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "calling convention|target"):
                plan_graph(custom_graph(attribute=attribute))

    def test_unknown_shapes_dtypes_and_memory_are_rejected(self):
        for mod in (graph(dtype="float64"), graph(shape=(tir.Var("batch", "int64"), 3))):
            with self.assertRaises(ValueError):
                plan_graph(mod)
        with self.assertRaises(ValueError):
            export_graph(graph(), self.root, target="llvm", memory_limit_bytes=1)
        self.assertFalse((self.root / "model.c").exists())


if __name__ == "__main__":
    unittest.main()
