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
"""Check synthetic PyTorch exports and focused ONNX regressions against Relax CPU execution."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


CASES = ("matmul", "batched_matmul", "conv2d", "layer_norm", "rms_norm")
IMPORTER_CASES = ("shape_add", "gather_negative", "gather_negative_constant", "expand_leading_dims", "constant_of_shape_scalar")
RTOL, ATOL = 1e-4, 1e-5


def allowed_path(value):
    """Reject restricted components before resolving links or opening any file."""
    path = Path(os.path.abspath(value))
    for component in path.parts:
        if any(word in component.lower() for word in ("hammer", "vlsi")):
            raise ValueError("Restricted path component")
    # Walk links individually so an intermediate forbidden target is never traversed.
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            target = Path(os.readlink(current))
            current = allowed_path(target if target.is_absolute() else current.parent / target)
    return current


def identity(path):
    path = allowed_path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensors(values):
    return [{"shape": list(value.shape), "dtype": str(value.dtype)} for value in values]


def compare(actual, expected, np):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"Output mismatch: {actual.shape}/{actual.dtype} versus {expected.shape}/{expected.dtype}")
    delta = np.abs(actual - expected)
    errors = {"max_absolute_error": float(delta.max()), "max_relative_error": float((delta / np.maximum(np.abs(expected), ATOL)).max())}
    integer = np.issubdtype(expected.dtype, np.integer)
    errors["comparison"] = "exact_integer" if integer else "floating_tolerance"
    errors["passed"] = bool(np.array_equal(actual, expected) if integer else np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False))
    return errors


def make_case(name, torch, np):
    rng = np.random.default_rng(194)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if name == "conv2d":
                self.conv = torch.nn.Conv2d(2, 3, 3, stride=(2, 1), padding=(1, 1))
                with torch.no_grad():
                    self.conv.weight.copy_(torch.from_numpy(rng.uniform(-0.7, 0.7, (3, 2, 3, 3)).astype("float32")))
                    self.conv.bias.copy_(torch.tensor([-0.4, 0.2, 0.7]))
            if name in ("layer_norm", "rms_norm"):
                self.register_buffer("weight", torch.tensor([0.7, 1.3, -0.8, 1.8, 0.4]))
                self.register_buffer("bias", torch.tensor([0.3, -0.5, 0.2, 0.6, -0.1]))

        def forward(self, value, right=None):
            if name in ("matmul", "batched_matmul"):
                return torch.matmul(value, right)
            if name == "conv2d":
                return self.conv(value)
            if name == "layer_norm":
                return torch.nn.functional.layer_norm(value, (5,), self.weight, self.bias, eps=1e-5)
            return value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-5) * self.weight

    if name == "matmul":
        shapes = [(3, 5), (5, 4)]
    elif name == "batched_matmul":
        shapes = [(2, 3, 5), (2, 5, 4)]
    elif name == "conv2d":
        shapes = [(1, 2, 5, 6)]
    else:
        shapes = [(2, 3, 5)]
    arrays = [rng.normal(0.2, 0.8, shape).astype("float32") for shape in shapes]
    if name in ("layer_norm", "rms_norm"):
        arrays[0] = arrays[0] * np.array([[0.2, 2.0, 5.0], [3.0, 0.4, 1.5]], dtype="float32")[..., None] + np.array([[-2.0, 0.3, 4.0], [1.0, -3.0, 0.7]], dtype="float32")[..., None]
    return Model().eval(), arrays


def make_importer_case(name, opset, onnx, np):
    """Build schema-valid ONNX cases with independent NumPy expectations."""
    helper, tensor, numpy_helper = onnx.helper, onnx.TensorProto, onnx.numpy_helper
    value = np.arange(6, dtype="float32").reshape(2, 3) - 2
    inputs, initializers = {"value": value}, []
    if name == "shape_add":
        offset = np.array([1, 3, 2**40 + 7], dtype="int64")
        initializers = [numpy_helper.from_array(np.array(0, dtype="int64"), "index"), numpy_helper.from_array(offset, "offset")]
        nodes = [helper.make_node("Shape", ["value"], ["shape"]), helper.make_node("Gather", ["shape", "index"], ["rows"], axis=0), helper.make_node("Add", ["rows", "offset"], ["output"])]
        expected = offset + value.shape[0]
    elif name in ("gather_negative", "gather_negative_constant"):
        inputs["value"] = np.arange(12, dtype="float32").reshape(4, 3) - 5
        indices = np.array([-1, 0, -2], dtype="int64")
        if name == "gather_negative":
            inputs["indices"] = indices
        else:
            initializers = [numpy_helper.from_array(indices, "indices")]
        nodes = [helper.make_node("Gather", ["value", "indices"], ["output"], axis=0)]
        expected = np.take(inputs["value"], indices, axis=0)
    elif name == "expand_leading_dims":
        initializers = [numpy_helper.from_array(np.array([3], dtype="int64"), "shape")]
        nodes = [helper.make_node("Expand", ["value", "shape"], ["output"])]
        expected = value.copy()
    else:
        inputs["value"] = np.array(-1.25, dtype="float32")
        initializers = [numpy_helper.from_array(np.array([], dtype="int64"), "shape")]
        fill = numpy_helper.from_array(np.array([2.5], dtype="float32"), "fill")
        nodes = [helper.make_node("ConstantOfShape", ["shape"], ["filled"], value=fill), helper.make_node("Add", ["value", "filled"], ["output"])]
        expected = inputs["value"] + np.array(2.5, dtype="float32")
    graph_inputs = [helper.make_tensor_value_info(key, tensor.INT64 if value.dtype == np.int64 else tensor.FLOAT, list(value.shape)) for key, value in inputs.items()]
    output = helper.make_tensor_value_info("output", tensor.INT64 if expected.dtype == np.int64 else tensor.FLOAT, list(expected.shape))
    graph = helper.make_graph(nodes, name, graph_inputs, [output], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="gemmini_frontend_verifier"), inputs, expected


def provenance(tvm, torch, onnx, np, override):
    import google.protobuf as protobuf
    from tvm._ffi.base import _LIB
    from tvm.relax.frontend.onnx import onnx_frontend

    root = allowed_path(Path(__file__).parents[2])
    python_path, library_path = allowed_path(tvm.__file__), allowed_path(_LIB._name)
    if not python_path.is_relative_to(root / "python"):
        raise RuntimeError(f"Wrong TVM Python package: {python_path}")
    library_dirs = [allowed_path(value) for value in os.environ.get("TVM_LIBRARY_PATH", "").split(os.pathsep) if value]
    if not library_dirs or library_path.parent not in library_dirs:
        raise RuntimeError(f"Loaded library {library_path} does not match explicit TVM_LIBRARY_PATH")
    if not tvm.runtime.enabled("llvm"):
        raise RuntimeError("Loaded TVM lacks LLVM support")
    importer = onnx_frontend
    if override is not None:
        source = allowed_path(override)
        spec = importlib.util.spec_from_file_location("tvm.relax.frontend.onnx._verification_importer", source)
        importer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(importer)
    marker = tvm.get_global_func("tir.transform.HostOnlyAutoCopyGuardEnabled", allow_missing=True)
    info = tvm.support.libinfo()
    return importer.from_onnx, {
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__, "protobuf": protobuf.__version__, "tvm": tvm.__version__},
        "dependency_locations": {name: str(allowed_path(module.__file__)) for name, module in (("torch", torch), ("onnx", onnx), ("numpy", np), ("protobuf", protobuf))},
        "tvm_python_location": str(python_path), "loaded_libtvm_path": str(library_path), "tvm_library_path": os.environ["TVM_LIBRARY_PATH"],
        "llvm_enabled": True, "build_info": {key: str(info[key]) for key in ("LLVM_VERSION", "USE_LLVM", "GIT_COMMIT_HASH") if key in info},
        "host_only_auto_copy_guard": {"available": marker is not None, "enabled": bool(marker()) if marker is not None else False},
        "frontend_source": identity(importer.__file__), "normal_frontend_source": identity(onnx_frontend.__file__), "importer_override": override is not None,
    }


def run_case(name, opset, output, from_onnx, torch, onnx, np, tvm):
    from onnx.reference import ReferenceEvaluator
    from tvm import relax

    record = {"case": name, "opset": opset, "status": "failed", "stage": "inputs", "rtol": RTOL, "atol": ATOL}
    try:
        directory = allowed_path(output / f"{name}_opset{opset}")
        directory.mkdir(parents=True, exist_ok=True)
        graph_path = allowed_path(directory / "model.onnx")
        if name in IMPORTER_CASES:
            graph, inputs, expected = make_importer_case(name, opset, onnx, np)
            names, arrays, reference_name = list(inputs), list(inputs.values()), "numpy"
            record["stage"] = "onnx_construction"
            onnx.save(graph, str(graph_path))
        else:
            model, arrays = make_case(name, torch, np)
            arguments = tuple(torch.from_numpy(value) for value in arrays)
            names, reference_name = ["value", "right"][:len(arrays)], "torch"
            with torch.no_grad():
                expected = model(*arguments).numpy()
            record["stage"] = "torch_export"
            torch.onnx.export(model, arguments, str(graph_path), input_names=names, output_names=["output"], opset_version=opset, dynamo=False, do_constant_folding=True)
            graph = onnx.load(str(graph_path))
        inputs = dict(zip(names, arrays))
        saved = {reference_name: expected, **inputs}
        record["reference"] = reference_name
        record["inputs"] = dict(zip(names, tensors(arrays)))
        record[f"{reference_name}_output"] = tensors([expected])[0]
        np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
        record["onnx_graph"] = identity(graph_path)
        record["operators"] = dict(sorted(Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in graph.graph.node).items()))
        record["stage"] = "onnx_reference"
        onnx.checker.check_model(graph, full_check=True)
        reference = ReferenceEvaluator(graph).run(None, inputs)[0]
        record["onnx_output"] = tensors([reference])[0]
        saved["onnx"] = reference
        np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
        record[f"onnx_vs_{reference_name}"] = compare(reference, expected, np)
        if not record[f"onnx_vs_{reference_name}"]["passed"]:
            raise AssertionError(f"ONNX reference differs from {reference_name}")
        record["stage"] = "relax_import"
        mod = from_onnx(graph, shape_dict={key: list(value.shape) for key, value in zip(names, arrays)}, keep_params_in_input=False)
        ir_path = allowed_path(directory / "imported_relax.py")
        ir_path.write_text(mod.script(), encoding="utf-8")
        record["relax_ir"] = identity(ir_path)
        if name in ("gather_negative", "gather_negative_constant") and from_onnx.__module__.endswith("._verification_importer"):
            record.update(status="not_executed", stage="ablation_index_safety", reason="Historical Gather may leave negative take indices unchecked; imported IR retained without executing it")
            return record
        record["stage"] = "relax_build"
        executable = relax.build(mod, target="llvm")
        record["stage"] = "relax_execute"
        device = tvm.cpu()
        result = relax.VirtualMachine(executable, device)["main"](*[tvm.nd.array(value, device) for value in arrays])
        actual = result.numpy() if hasattr(result, "numpy") else result[0].numpy()
        record["relax_output"] = tensors([actual])[0]
        saved["relax"] = actual
        np.savez(allowed_path(directory / "numerical_outputs.npz"), **saved)
        record["stage"] = "numerical_compare"
        record[f"relax_vs_{reference_name}"] = compare(actual, expected, np)
        record["relax_vs_onnx"] = compare(actual, reference, np)
        if not all(record[key]["passed"] for key in (f"relax_vs_{reference_name}", "relax_vs_onnx")):
            raise AssertionError("Relax output differs from reference")
        record.update(status="passed", stage="complete")
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", choices=("torch", "importer"), default="torch", help="Default case set; --case overrides this selection")
    parser.add_argument("--case", choices=CASES + IMPORTER_CASES, action="append")
    parser.add_argument("--opset", choices=(17, 18), type=int, action="append")
    parser.add_argument("--importer-source", type=Path, help="Isolated importer ablation; does not replace the normal module")
    args = parser.parse_args()
    output = allowed_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "scope": "Synthetic frontend checks, LLVM host CPU only; no models or Gemmini execution", "suite": args.suite, "seed": 194, "cases": [], "verifier_source": identity(__file__)}
    try:
        import numpy as np
        import onnx
        import torch
        import tvm

        torch.set_num_threads(1)
        torch.manual_seed(194)
        from_onnx, metadata = provenance(tvm, torch, onnx, np, args.importer_source)
        report.update(metadata)
        selected = args.case or (CASES if args.suite == "torch" else IMPORTER_CASES)
        for name in selected:
            for opset in args.opset or (17, 18):
                record = run_case(name, opset, output, from_onnx, torch, onnx, np, tvm)
                report["cases"].append(record)
                print(f"{name} opset {opset}: {record['status']} ({record['stage']})", flush=True)
        if all(record["status"] == "passed" for record in report["cases"]):
            report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report_path = allowed_path(output / "results.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
