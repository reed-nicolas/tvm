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

"""Opt-in Relax matmul lowering to the proposed Gemmini external C ABI.

This pass establishes a compiler boundary, not accelerator execution. Supply and
link an implementation of apps/gemmini/matmul.h when exporting the executable.
Run before LegalizeOps. Unsupported calls remain ordinary Relax operations.
"""

import tvm
from tvm import relax, tir
from tvm.relax.expr_functor import PyExprMutator, mutator

_SYMBOL = "tvm_gemmini_matmul_i8_i32"
_MAX_K = ((1 << 31) - 1) // (128 * 128)
_MAX_BYTES = (1 << 63) - 1


def _shape(info, dtype):
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != dtype or info.ndim != 2:
        return None
    if not isinstance(info.shape, relax.ShapeExpr):
        return None
    if info.vdevice is not None and info.vdevice.target.kind.name not in ("llvm", "c"):
        return None
    if any(not isinstance(dim, tir.IntImm) or dim.value <= 0 for dim in info.shape):
        return None
    return tuple(int(dim.value) for dim in info.shape)


def _eligible(call):
    if not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.matmul":
        return None
    left, right = (_shape(arg.struct_info, "int8") for arg in call.args)
    output = _shape(call.struct_info, "int32")
    if left is None or right is None or output is None:
        return None
    m, k = left
    inner, n = right
    if inner != k or output != (m, n) or k > _MAX_K:
        return None
    if max(m * k, k * n, m * n * 4) > _MAX_BYTES:
        return None
    return m, n, k


def _wrapper(m, n, k):
    a = tir.decl_buffer((m, k), "int8", name="a")
    b = tir.decl_buffer((k, n), "int8", name="b")
    c = tir.decl_buffer((m, n), "int32", name="c")
    dims = [tir.IntImm("int64", value) for value in (m, n, k, k, n, n)]
    result = tir.call_extern("int32", _SYMBOL, a.data, b.data, c.data, *dims)
    status = tir.Var("status", "int32")
    error = tir.SeqStmt([tir.Evaluate(tir.call_extern("void", "TVMAPISetLastError", tir.StringImm("Gemmini external matmul returned an error"))), tir.Evaluate(tir.tvm_throw_last_error())])
    body = tir.LetStmt(status, result, tir.IfThenElse(status != 0, error, None))
    return tir.PrimFunc([a.data, b.data, c.data], body, buffer_map={a.data: a, b.data: b, c.data: c}).with_attr("tir.noalias", True)


@mutator
class _MatmulLowerer(PyExprMutator):
    def __init__(self, mod):
        super().__init__(mod)
        self.wrappers = {}

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        shape = _eligible(call)
        if shape is None:
            return call
        if shape not in self.wrappers:
            self.wrappers[shape] = self.builder_.add_func(_wrapper(*shape), "gemmini_matmul")
        return self.builder_.normalize(relax.call_tir(self.wrappers[shape], tuple(call.args), call.struct_info))

    def visit_function_(self, func):
        if func.attrs and ("Codegen" in func.attrs or "Composite" in func.attrs):
            return func
        return super().visit_function_(func)


@tvm.transform.module_pass(opt_level=0, name="LowerGemminiMatmul")
class LowerGemminiMatmul:
    """Lower positive static 2D signed-int8 matmul with signed-int32 output.

    The external function must synchronously write the complete output, preserve
    inputs, and honor the C ABI. No bias, scaling, activation or batching is fused.
    The reduction bound guarantees exact int32 sums for every int8 input value.
    DLTensor arguments must be compact; generated TIR wrappers check that ABI.
    This opt-in pass does not choose hardware headers, runtime or instruction policy.
    """

    def transform_module(self, mod, _ctx):
        lowerer = _MatmulLowerer(mod)
        for gv, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        return lowerer.builder_.get()
