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
"""Export straight-line, static Relax call_tir graphs without a guest VM or heap.

Graph optimization and accelerator scheduling must precede export. This module
only derives calls, constants and buffer lifetimes; it does not implement tensor
operators. Unsupported control flow, dynamic tensors and hidden scalar ABIs fail
explicitly. The caller supplies aligned, disjoint input/output/workspace storage.
Each call_tir operator must synchronously complete its outputs, preserve inputs
and constants, and retain no buffer pointers after return. These are compiler/
operator contracts, not a static proof of arbitrary TIR or external-call effects.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
import tvm
from tvm import relax, tir
from verify_baremetal_cpu import checked_path

ALIGNMENT = 64
DTYPES = {"int8": "int8_t", "uint8": "uint8_t", "int32": "int32_t", "int64": "int64_t", "float32": "float", "bool": "uint8_t"}
RV64_TARGET = "llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c -mabi=lp64d"


def _aligned(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(eq=False)
class Tensor:
    name: str
    shape: tuple
    dtype: str
    origin: str
    birth: int = -1
    last_use: int = -1
    offset: int = 0
    value: object = None

    @property
    def nbytes(self):
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize

    def metadata(self):
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype, "bytes": self.nbytes, "origin": self.origin, "offset": self.offset, "birth": self.birth, "last_use": self.last_use}


def _tensor(info, name, origin, birth=-1):
    if not isinstance(info, relax.TensorStructInfo) or info.dtype not in DTYPES:
        raise ValueError("unsupported static tensor dtype or structure")
    if not isinstance(info.shape, relax.ShapeExpr) or any(not isinstance(d, tir.IntImm) or int(d) <= 0 for d in info.shape):
        raise ValueError("tensor dimensions must be positive static integers")
    if info.vdevice is not None and info.vdevice.target.kind.name not in ("llvm", "c"):
        raise ValueError("only host-addressed tensor buffers are supported")
    result = Tensor(name, tuple(int(d) for d in info.shape), info.dtype, origin, birth=birth, last_use=birth)
    if result.nbytes > (1 << 63) - 1:
        raise ValueError("tensor byte extent overflows the target pointer ABI")
    return result


def _flatten(value):
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, tuple):
        return [tensor for item in value for tensor in _flatten(item)]
    raise ValueError("only tensor and nested tensor-tuple values are supported")


def _allocate_workspace(tensors):
    """Reuse aligned storage only after a tensor's final consuming call."""
    free, active, high_water = [], [], 0
    for tensor in tensors:
        survivors = []
        for previous in active:
            if previous.last_use < tensor.birth:
                free.append((previous.offset, _aligned(previous.nbytes)))
            else:
                survivors.append(previous)
        active = survivors
        merged = []
        for offset, size in sorted(free):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        free = merged
        size = _aligned(tensor.nbytes)
        suitable = [(extent, offset, index) for index, (offset, extent) in enumerate(free) if extent >= size]
        if suitable:
            extent, offset, index = min(suitable)
            free.pop(index)
            if extent > size:
                free.append((offset + size, extent - size))
        else:
            offset = high_water
            high_water += size
        tensor.offset = offset
        active.append(tensor)
    return high_water


def plan_graph(mod):
    """Validate the graph's raw-pointer ABI and derive a reusable storage plan."""
    function = mod["main"]
    if not isinstance(function, relax.Function) or not isinstance(function.body, relax.SeqExpr):
        raise ValueError("require a straight-line Relax main function")
    values, inputs, constants, tensors, calls, functions = {}, [], [], [], [], {}
    for index, param in enumerate(function.params):
        tensor = _tensor(param.struct_info, f"input_{index}", "input")
        values[param] = tensor
        inputs.append(tensor)

    def resolve(expr):
        if isinstance(expr, relax.Var) and expr in values:
            return values[expr]
        if isinstance(expr, relax.Constant):
            if expr not in values:
                tensor = _tensor(expr.struct_info, f"constant_{len(constants)}", "constant")
                tensor.value = expr.data.numpy()
                values[expr] = tensor
                constants.append(tensor)
            return values[expr]
        if isinstance(expr, relax.Tuple):
            return tuple(resolve(field) for field in expr.fields)
        if isinstance(expr, relax.TupleGetItem):
            source = resolve(expr.tuple_value)
            if not isinstance(source, tuple) or not 0 <= expr.index < len(source):
                raise ValueError("invalid static tuple selection")
            return source[expr.index]
        raise ValueError("unsupported graph value or unbound variable")

    def output_value(info, call_index):
        if isinstance(info, relax.TupleStructInfo):
            return tuple(output_value(field, call_index) for field in info.fields)
        tensor = _tensor(info, f"tensor_{len(tensors)}", "workspace", call_index)
        tensors.append(tensor)
        return tensor

    for block in function.body.blocks:
        if not isinstance(block, (relax.BindingBlock, relax.DataflowBlock)):
            raise ValueError("unsupported graph binding block")
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                raise ValueError("unsupported graph binding")
            call = binding.value
            if not isinstance(call, relax.Call):
                values[binding.var] = resolve(call)
                continue
            if call.op != tvm.ir.Op.get("relax.call_tir") or len(call.args) != 2:
                raise ValueError("export requires legalized buffer-only call_tir operations")
            gv, arguments = call.args
            if not isinstance(gv, tvm.ir.GlobalVar) or not isinstance(arguments, relax.Tuple):
                raise ValueError("unsupported call_tir signature")
            symbol = gv.name_hint
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol):
                raise ValueError("PrimFunc symbol is not a C identifier")
            primitive = mod[gv]
            if not isinstance(primitive, tir.PrimFunc) or len(primitive.params) != len(primitive.buffer_map):
                raise ValueError("PrimFunc requires an unsupported scalar or hidden workspace ABI")
            if primitive.attrs and int(primitive.attrs.get("calling_conv", 0)) != 0:
                raise ValueError("PrimFunc requires the default calling convention")
            if primitive.attrs and "target" in primitive.attrs:
                raise ValueError("PrimFunc target must be supplied by the exporter")
            actual_inputs = [resolve(arg) for arg in arguments.fields]
            if any(not isinstance(arg, Tensor) for arg in actual_inputs):
                raise ValueError("call_tir operands must be individual tensors")
            result = output_value(binding.var.struct_info, len(calls))
            actual_outputs = _flatten(result)
            if len(primitive.params) != len(actual_inputs) + len(actual_outputs):
                raise ValueError("PrimFunc buffer count differs from call_tir")
            for param, tensor in zip(primitive.params, [*actual_inputs, *actual_outputs]):
                buffer = primitive.buffer_map[param]
                if buffer.dtype != tensor.dtype or any(not isinstance(dim, tir.IntImm) for dim in buffer.shape) or tuple(int(dim) for dim in buffer.shape) != tensor.shape:
                    raise ValueError("PrimFunc tensor shape/dtype ABI mismatch")
                if buffer.strides or not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset) != 0:
                    raise ValueError("only compact zero-offset PrimFunc buffers are supported")
                if buffer.scope() != "global":
                    raise ValueError("PrimFunc parameters must use host-addressed storage")
                if buffer.data_alignment <= 0 or ALIGNMENT % buffer.data_alignment:
                    raise ValueError("PrimFunc external-buffer alignment exceeds the storage guarantee")
            external_data = {buffer.data for buffer in primitive.buffer_map.values()}
            def check_alignment(node):
                if isinstance(node, tir.AttrStmt) and node.attr_key == "storage_alignment" and node.node in external_data:
                    if not isinstance(node.value, tir.IntImm) or int(node.value) <= 0 or ALIGNMENT % int(node.value):
                        raise ValueError("PrimFunc external-buffer storage alignment exceeds the storage guarantee")
            tir.stmt_functor.post_order_visit(primitive.body, check_alignment)
            for tensor in actual_inputs:
                tensor.last_use = len(calls)
            values[binding.var] = result
            functions[gv] = primitive.with_attr("global_symbol", symbol)
            calls.append((symbol, actual_inputs, actual_outputs))
    outputs = _flatten(resolve(function.body.body))
    if not outputs or not calls:
        raise ValueError("require a nonempty tensor program and output")
    unique_outputs = set(outputs)
    for index, tensor in enumerate(outputs):
        tensor.last_use = len(calls)
        # Duplicate returns and input/constant returns are copied at completion.
        if tensor.origin == "workspace":
            tensor.origin = "output"
            tensor.offset = index
    workspace = [tensor for tensor in tensors if tensor not in unique_outputs]
    workspace_bytes = _allocate_workspace(workspace)
    constants_bytes = 0
    for tensor in constants:
        tensor.offset = constants_bytes
        constants_bytes = _aligned(constants_bytes + tensor.nbytes)
    return {"inputs": inputs, "outputs": outputs, "constants": constants, "tensors": tensors, "calls": calls, "functions": functions, "workspace_bytes": workspace_bytes, "constants_bytes": constants_bytes}


def export_graph(mod, output_dir, *, target=RV64_TARGET, memory_limit_bytes=1 << 30):
    """Write operators, one constant blob, C orchestration and a storage manifest.

    The limit covers explicit tensor storage only; executable sections, generated
    operator stack allocations, startup and platform reservations are additional.
    Caller-supplied buffers must each contain the manifest's full byte envelope.
    """
    plan = plan_graph(mod)
    tensors_bytes = plan["workspace_bytes"] + plan["constants_bytes"] + sum(t.nbytes for t in [*plan["inputs"], *plan["outputs"]])
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int) or memory_limit_bytes <= 0 or tensors_bytes > memory_limit_bytes:
        raise ValueError("explicit tensor storage exceeds the supplied memory budget")
    output_dir = checked_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    constant_path = output_dir / "constants.bin"
    with constant_path.open("wb") as stream:
        for tensor in plan["constants"]:
            stream.seek(tensor.offset)
            stream.write(tensor.value.astype(np.dtype(tensor.dtype).newbyteorder("<"), copy=False).tobytes())
        stream.truncate(max(plan["constants_bytes"], 1))
    include_path = str(constant_path.absolute()).replace("\\", "\\\\").replace('"', '\\"')
    if any(character in include_path for character in ("\n", "\r")):
        raise ValueError("constant artifact path contains an unsupported line break")
    (output_dir / "constants.S").write_text(f'.section .rodata.model_constants,"a",@progbits\n.balign {ALIGNMENT}\n.globl model_constants\nmodel_constants:\n.incbin "{include_path}"\n.section .note.GNU-stack,"",@progbits\n')

    def address(tensor):
        if tensor.origin == "input":
            return f"inputs[{plan['inputs'].index(tensor)}]"
        if tensor.origin == "output":
            return f"outputs[{tensor.offset}]"
        if tensor.origin == "constant":
            return f"(model_constants + {tensor.offset})"
        return f"(workspace + {tensor.offset})"

    lines = ["#include <stdint.h>", "#include <stddef.h>", "extern const uint8_t model_constants[];", "static const char *model_last_error;", "void TVMAPISetLastError(const char *message) { model_last_error = message; }", "const char *model_get_last_error(void) { return model_last_error; }"]
    for symbol, args, results in plan["calls"]:
        ctypes = [DTYPES[tensor.dtype] + "*" for tensor in [*args, *results]]
        declaration = "extern int32_t " + symbol + "(" + ", ".join(ctypes) + ");"
        if declaration not in lines:
            lines.append(declaration)
    lines.extend([
        "static int model_span(const void *pointer, uint64_t bytes) { return pointer && !((uintptr_t)pointer % 64) && bytes <= UINTPTR_MAX - (uintptr_t)pointer; }",
        "static int model_overlap(const void *a, uint64_t an, const void *b, uint64_t bn) { return an && bn && (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an; }",
        "int32_t model_run(const void *const *inputs, void *const *outputs, uint8_t *workspace, uint64_t workspace_bytes) {",
        "  model_last_error = 0;",
        "  if (!inputs || !outputs) return -1;",
        f"  if (workspace_bytes < UINT64_C({plan['workspace_bytes']})) return -1;",
    ])
    if plan["workspace_bytes"]:
        lines.append(f"  if (!model_span(workspace, {plan['workspace_bytes']})) return -1;")
    for group in ("inputs", "outputs"):
        for index, tensor in enumerate(plan[group]):
            lines.append(f"  if (!model_span({group}[{index}], {tensor.nbytes})) return -1;")
    read_buffers = [(f"inputs[{i}]", t.nbytes) for i, t in enumerate(plan["inputs"])] + [("model_constants", plan["constants_bytes"])]
    write_buffers = [(f"outputs[{i}]", t.nbytes) for i, t in enumerate(plan["outputs"])]
    if plan["workspace_bytes"]:
        write_buffers.append(("workspace", plan["workspace_bytes"]))
    for index, (pointer, size) in enumerate(write_buffers):
        for other, other_size in [*read_buffers, *write_buffers[:index]]:
            if other_size:
                lines.append(f"  if (model_overlap({pointer}, {size}, {other}, {other_size})) return -2;")
    for symbol, args, results in plan["calls"]:
        arguments = ["(" + DTYPES[tensor.dtype] + "*)" + address(tensor) for tensor in [*args, *results]]
        lines.append(f"  if ({symbol}({', '.join(arguments)})) return -3;")
    for index, tensor in enumerate(plan["outputs"]):
        if tensor.origin != "output" or tensor.offset != index:
            lines.append(f"  for (uint64_t i = 0; i < UINT64_C({tensor.nbytes}); ++i) ((uint8_t*)outputs[{index}])[i] = ((const uint8_t*){address(tensor)})[i];")
    lines.extend(["  return 0;", "}"])
    (output_dir / "model.c").write_text("\n".join(lines) + "\n")
    primitive_mod = tvm.IRModule(plan["functions"]).with_attr("executor", tvm.relay.backend.Executor("aot", {"unpacked-api": True, "interface-api": "c"}))
    module = tvm.build(primitive_mod, target=target)
    module.save(str(output_dir / "operators.o"))
    (output_dir / "operators.ll").write_text(module.get_source("ll"))
    (output_dir / "operators.tir.py").write_text(primitive_mod.script(show_meta=True))
    (output_dir / "graph.relax.py").write_text(mod.script(show_meta=True))
    manifest = {
        "target": str(target), "alignment_bytes": ALIGNMENT, "heap_required_by_orchestration": False,
        "workspace_bytes": plan["workspace_bytes"], "constants_bytes": plan["constants_bytes"], "explicit_tensor_bytes": tensors_bytes,
        "memory_limit_bytes": memory_limit_bytes, "memory_scope": "tensor storage only; add ELF sections, operator stack/workspace, startup and platform reservations",
        "inputs": [tensor.metadata() for tensor in plan["inputs"]], "outputs": [tensor.metadata() for tensor in plan["outputs"]],
        "constants": [tensor.metadata() for tensor in plan["constants"]], "intermediates": [tensor.metadata() for tensor in plan["tensors"]],
        "calls": [{"symbol": name, "inputs": [t.name for t in args], "outputs": [t.name for t in results]} for name, args, results in plan["calls"]],
        "constants_sha256": hashlib.sha256(constant_path.read_bytes()).hexdigest(), "full_model_qualified": False, "timing_qualified": False,
    }
    (output_dir / "graph.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
