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
"""Qualify an exported static graph against frozen independent fixture bytes.

This is a functional Spike/plugin check, not timing, RTL, deployed hardware or
application-quality qualification. The caller provides reference provenance;
this verifier does not generate expected answers or import the reference model.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from verify_baremetal_cpu import STARTUP, LINKER, GUEST_BASE, GUEST_BYTES, STACK_BYTES, checked_path, identity
from verify_matmul import audit_instructions
from verify_matmul_runtime import bind_runtime


RUNNER = r'''
#include <stdint.h>
#include <stddef.h>
#include "matmul.h"
#include "fixture.h"
#ifndef BAD_ORACLE
#define BAD_ORACLE 0
#endif
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)));
static void puts_htif(const char*s){static volatile uint64_t request[4] __attribute__((aligned(8)));uint64_t n=0;while(s[n])++n;request[0]=64;request[1]=1;request[2]=(uintptr_t)s;request[3]=n;asm volatile("fence rw,rw" ::: "memory");tohost=(uintptr_t)request;while(!fromhost){}fromhost=0;asm volatile("fence rw,rw" ::: "memory");}
static void number(uint64_t n){char text[32];unsigned i=0;do{text[i++]=(char)('0'+n%10);n/=10;}while(n);for(unsigned j=0;j<i/2;++j){char t=text[j];text[j]=text[i-1-j];text[i-1-j]=t;}text[i]=0;puts_htif(text);}
int printf(const char*format,...){puts_htif("VENDOR_DIAGNOSTIC\n");return 0;}
__attribute__((noreturn)) void exit(int code){tohost=((uint64_t)(uint32_t)(code?code:99)<<1)|1;for(;;){}}
extern int32_t model_run(const void*const*,void*const*,uint8_t*,uint64_t);
extern const uint8_t model_constants[];
static uint64_t loads_a,loads_b,computes,stores,admissions;
int32_t __real_tvm_gemmini_validate_matmul_i8_i32(const int8_t*,const int8_t*,int32_t*,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
int32_t __wrap_tvm_gemmini_validate_matmul_i8_i32(const int8_t*a,const int8_t*b,int32_t*c,int64_t m,int64_t n,int64_t k,int64_t as,int64_t bs,int64_t cs){++admissions;return __real_tvm_gemmini_validate_matmul_i8_i32(a,b,c,m,n,k,as,bs,cs);}
void __real_tvm_gemmini_load_a(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_load_b(const int8_t*,uint32_t,uint32_t,uint32_t);
void __real_tvm_gemmini_compute(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,int32_t);
void __real_tvm_gemmini_store(int32_t*,uint32_t,uint32_t,uint32_t);
void __wrap_tvm_gemmini_load_a(const int8_t*p,uint32_t r,uint32_t m,uint32_t n){++loads_a;__real_tvm_gemmini_load_a(p,r,m,n);}
void __wrap_tvm_gemmini_load_b(const int8_t*p,uint32_t r,uint32_t m,uint32_t n){++loads_b;__real_tvm_gemmini_load_b(p,r,m,n);}
void __wrap_tvm_gemmini_compute(uint32_t a,uint32_t b,uint32_t c,uint32_t m,uint32_t n,uint32_t k,int32_t add){++computes;__real_tvm_gemmini_compute(a,b,c,m,n,k,add);}
void __wrap_tvm_gemmini_store(int32_t*p,uint32_t r,uint32_t m,uint32_t n){++stores;__real_tvm_gemmini_store(p,r,m,n);}
static uint64_t hash(const void*p,uint64_t n){const volatile uint8_t*b=p;uint64_t h=UINT64_C(1469598103934665603);for(uint64_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);return h;}
static void fill(uint8_t*p,uint64_t n,uint8_t value){for(uint64_t i=0;i<n;++i)p[i]=value;}
static int equal(const uint8_t*a,const uint8_t*b,uint64_t n,int wrong){for(uint64_t i=0;i<n;++i)if(a[i]!=(uint8_t)(b[i]^((wrong&&i==0)?1:0)))return 0;return 1;}
static int guard(const uint8_t*head,const uint8_t*tail){for(unsigned i=0;i<64;++i)if(head[i]!=0x6d||tail[i]!=0x6d)return 0;return 1;}
static int padding(const uint8_t*value,uint64_t n,uint64_t extent,uint8_t poison){for(uint64_t i=n;i<extent;++i)if(value[i]!=poison)return 0;return 1;}
DECLARATIONS
int main(void){
  INITIALIZE_GUARDS
  const void*inputs[INPUT_COUNT];void*outputs[OUTPUT_COUNT];
  INITIAL_POINTERS
  if(model_run(0,outputs,workspace.value,WORKSPACE_BYTES)!=-1||computes)return 10;
  if(WORKSPACE_BYTES&&model_run(inputs,outputs,workspace.value,0)!=-1)return 10;
  puts_htif("EXPORTED_GRAPH_INPUT_REJECTION_PASS\n");
  uint64_t constants_before=hash(model_constants,CONSTANT_BYTES);
  uint64_t retained[OUTPUT_COUNT];unsigned call=0;
  for(unsigned repeat=0;repeat<REPEATS;++repeat)for(unsigned sample=0;sample<SAMPLES;++sample,++call){
    unsigned slot=call%2;uint8_t poison=(uint8_t)(0xa5^call);
    PREPARE_INPUTS
    PREPARE_OUTPUTS
    fill(workspace.value,sizeof(workspace.value),poison);
    loads_a=loads_b=computes=stores=admissions=0;
    if(model_run(inputs,outputs,workspace.value,WORKSPACE_BYTES))return 21;
    CHECK_OUTPUTS
    CHECK_INPUTS
    CHECK_GUARDS
    if(!padding(workspace.value,WORKSPACE_BYTES,sizeof(workspace.value),poison))return 28;
    if(constants_before!=hash(model_constants,CONSTANT_BYTES))return 29;
    if(loads_a!=EXPECTED_LOADS_A||loads_b!=EXPECTED_LOADS_B||computes!=EXPECTED_COMPUTES||stores!=EXPECTED_STORES||admissions!=EXPECTED_ADMISSIONS)return 30;
    puts_htif("EXPORTED_GRAPH_CALL_PASS sample=");number(sample);puts_htif(" repeat=");number(repeat);puts_htif(" primitive_counts=1 exact_bytes=1\n");
  }
  extern uint8_t _stack_bottom[],_stack_top[];
  uint8_t*first=_stack_bottom;while(first<_stack_top&&*first==(uint8_t)(UINT64_C(0x55aa55aa55aa55aa)>>(((uintptr_t)first-(uintptr_t)_stack_bottom)%8*8)))++first;
  uint64_t used=(uint64_t)(_stack_top-first);if(!used||used>=STACK_BYTES)return 31;
  puts_htif("STACK_HIGH_WATER_BYTES=");number(used);puts_htif("\n");
  puts_htif("TVM_EXPORTED_GRAPH_PASS calls=");number(call);puts_htif(" exact_bytes=1 guards=1 inputs=1 constants=1 retained=1 poisoned_workspace=1 primitive_counts=1\n");return 0;
}
'''


def aligned_size(size):
    return (size + 63) // 64 * 64


def validate_tensor(tensor):
    """Refuse declared byte extents that could underallocate a typed buffer."""
    dtypes = {"int8", "uint8", "int32", "int64", "float32", "bool"}
    if not isinstance(tensor, dict) or type(tensor.get("dtype")) is not str or tensor["dtype"] not in dtypes:
        raise ValueError("Unsupported manifest tensor dtype")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("Manifest tensor shape must contain positive static integers")
    required = math.prod(shape) * np.dtype(tensor["dtype"]).itemsize
    if required > (1 << 63) - 1 or type(tensor.get("bytes")) is not int or tensor["bytes"] != required:
        raise ValueError("Manifest tensor bytes must equal dtype size times shape")
    return required


def validate_manifest(manifest):
    """Validate every declared tensor and the explicit storage envelope."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    for group in ("inputs", "outputs", "constants", "intermediates"):
        tensors = manifest.get(group)
        if not isinstance(tensors, list) or (group in ("inputs", "outputs") and not tensors):
            raise ValueError("Manifest requires complete input/output/tensor lists")
        for tensor in tensors:
            validate_tensor(tensor)
    for name in ("workspace_bytes", "constants_bytes", "explicit_tensor_bytes"):
        if type(manifest.get(name)) is not int or not 0 <= manifest[name] <= (1 << 63) - 1:
            raise ValueError("Manifest storage bytes must be nonnegative integers")
    required = sum(tensor["bytes"] for group in ("inputs", "outputs") for tensor in manifest[group]) + manifest["workspace_bytes"] + manifest["constants_bytes"]
    if manifest["explicit_tensor_bytes"] != required:
        raise ValueError("Manifest explicit tensor bytes do not match its storage envelope")
    for group, limit in (("constants", manifest["constants_bytes"]), ("intermediates", manifest["workspace_bytes"])):
        for tensor in manifest[group]:
            if group == "intermediates" and tensor.get("origin") != "workspace":
                continue
            offset = tensor.get("offset")
            if type(offset) is not int or offset < 0 or offset % 64 or offset + tensor["bytes"] > limit:
                raise ValueError("Manifest tensor lies outside its aligned storage envelope")


def load_fixture(path, manifest):
    """Require explicit exact dtype/shape for each complete input/output stream."""
    for group in ("inputs", "outputs"):
        for tensor in manifest[group]:
            validate_tensor(tensor)
    arrays, count = {}, None
    with np.load(checked_path(path), allow_pickle=False) as fixture:
        expected = {f"{kind}_{index}" for kind in ("input", "output") for index in range(len(manifest[kind + "s"]))}
        if set(fixture.files) != expected:
            raise ValueError("Fixture must contain exactly the graph input_N and output_N arrays")
        for kind in ("input", "output"):
            for index, tensor in enumerate(manifest[kind + "s"]):
                name = f"{kind}_{index}"
                value = fixture[name]
                if value.dtype != np.dtype(tensor["dtype"]) or value.ndim != len(tensor["shape"]) + 1 or tuple(value.shape[1:]) != tuple(tensor["shape"]) or value.shape[0] < 1:
                    raise ValueError(f"Fixture dtype/shape mismatch: {name}")
                if count is None:
                    count = value.shape[0]
                if value.shape[0] != count or (value.dtype.kind == "f" and not np.isfinite(value).all()):
                    raise ValueError("Fixture streams must have equal positive counts and finite values")
                arrays[name] = np.ascontiguousarray(value)
    return arrays, count


def make_guest(out, manifest, arrays, samples, repeats, counts):
    """Generate guarded caller storage and compare independent frozen byte blobs."""
    validate_manifest(manifest)
    declarations, initialization, initial_pointers = [], [], []
    prepare_inputs, prepare_outputs, check_inputs, check_outputs, check_guards = [], [], [], [], []
    assembly = []
    for kind in ("input", "output"):
        for index, tensor in enumerate(manifest[kind + "s"]):
            name = f"{kind}_{index}"
            extent, size = aligned_size(tensor["bytes"]), tensor["bytes"]
            blob = out / (name + ".bin")
            if any(character in str(blob) for character in ("\n", "\r")):
                raise ValueError("Fixture assembly path contains an unsupported line break")
            with blob.open("wb") as stream:
                for value in arrays[name]:
                    stream.write(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
                    stream.write(bytes(extent - size))
            assembly.extend([f'.section .rodata.fixture_{name},"a",@progbits', '.balign 64', f'.globl fixture_{name}', f'.type fixture_{name},@object', f'fixture_{name}:', f'.incbin "{str(blob).replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"', f'.size fixture_{name},.-fixture_{name}'])
            declarations.extend([f"extern const uint8_t fixture_{name}[];", f"struct {name}_storage {{uint8_t head[64];uint8_t value[{extent}];uint8_t tail[64];}};", f"static struct {name}_storage {name}{'[2]' if kind == 'output' else ''} __attribute__((aligned(64)));"])
            if kind == "input":
                initialization.append(f"fill({name}.head,64,0x6d);fill({name}.tail,64,0x6d);")
                initial_pointers.append(f"inputs[{index}]={name}.value;")
                prepare_inputs.append(f"fill({name}.value,{extent},0x37);for(uint64_t i=0;i<{size};++i){name}.value[i]=fixture_{name}[(uint64_t)sample*{extent}+i];")
                check_inputs.append(f"if(!equal({name}.value,fixture_{name}+(uint64_t)sample*{extent},{size},0)||!padding({name}.value,{size},{extent},0x37))return 24;")
                check_guards.append(f"if(!guard({name}.head,{name}.tail))return 27;")
            else:
                initialization.append(f"for(unsigned b=0;b<2;++b){{fill({name}[b].head,64,0x6d);fill({name}[b].tail,64,0x6d);}}")
                initial_pointers.append(f"outputs[{index}]={name}[0].value;")
                prepare_outputs.append(f"outputs[{index}]={name}[slot].value;fill({name}[slot].value,{extent},poison);")
                check_outputs.extend([f"if(!equal({name}[slot].value,fixture_{name}+(uint64_t)sample*{extent},{size},BAD_ORACLE&&call==0&&{index}==0)){{puts_htif(\"EXPORTED_GRAPH_NUMERICAL_FAILURE\\n\");return 22;}}", f"if(!padding({name}[slot].value,{size},{extent},poison))return 23;", f"if(call&&retained[{index}]!=hash({name}[1-slot].value,{extent}))return 26;retained[{index}]=hash({name}[slot].value,{extent});"])
                check_guards.append(f"for(unsigned b=0;b<2;++b)if(!guard({name}[b].head,{name}[b].tail))return 27;")
    workspace = max(64, aligned_size(manifest["workspace_bytes"]))
    declarations.extend([f"struct workspace_storage {{uint8_t head[64];uint8_t value[{workspace}];uint8_t tail[64];}};", "static struct workspace_storage workspace __attribute__((aligned(64)));"])
    initialization.append("fill(workspace.head,64,0x6d);fill(workspace.tail,64,0x6d);")
    check_guards.append("if(!guard(workspace.head,workspace.tail))return 27;")
    replacements = {"DECLARATIONS": declarations, "INITIALIZE_GUARDS": initialization, "INITIAL_POINTERS": initial_pointers,
                    "PREPARE_INPUTS": prepare_inputs, "PREPARE_OUTPUTS": prepare_outputs, "CHECK_INPUTS": check_inputs,
                    "CHECK_OUTPUTS": check_outputs, "CHECK_GUARDS": check_guards}
    runner = RUNNER
    for name, lines in replacements.items():
        runner = runner.replace(name, "\n  ".join(lines))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (out / "fixture.S").write_text("\n".join(assembly) + "\n")
    (out / "runner.c").write_text(runner)
    definitions = {"INPUT_COUNT": len(manifest["inputs"]), "OUTPUT_COUNT": len(manifest["outputs"]), "SAMPLES": samples, "REPEATS": repeats,
                   "WORKSPACE_BYTES": manifest["workspace_bytes"], "CONSTANT_BYTES": manifest["constants_bytes"], "STACK_BYTES": STACK_BYTES,
                   **{"EXPECTED_" + name.upper(): value for name, value in counts.items()}}
    (out / "fixture.h").write_text("\n".join(f"#define {name} UINT64_C({value})" for name, value in definitions.items()) + "\n")
    (out / "start.S").write_text(STARTUP)
    (out / "link.ld").write_text(LINKER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph-dir", "fixture", "fixture-provenance", "tvm-source", "tvm-build", "adapter-receipt", "simulator-receipt", "riscv-gcc", "spike", "dtc", "spike-library-dir", "plugin", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.timeout_seconds <= 86400:
        parser.error("repeats must be at least two; timeout must be in [1,86400]")
    paths = {name: checked_path(value) for name, value in vars(args).items() if isinstance(value, Path)}
    out, app = paths["output_dir"], checked_path(Path(__file__).parent)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": __doc__, "commands": [], "checks": [], "timing_qualified": False, "application_quality_qualified": False, "deployed_hardware_verified": False, "rtl_bit_exact": False, "invocation": [sys.executable, *sys.argv]}
    immutable = {}
    def save():
        (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    def bind(path, expected=None):
        record = identity(checked_path(path))
        if expected is not None and record["sha256"] != expected:
            raise ValueError("Identity mismatch: " + str(path))
        immutable[record["path"]] = record["sha256"]
        return record
    def command(argv, label, env=None, expected=0):
        entry = {"argv": list(map(str, argv)), "expected": expected, "log": label + ".log"}
        report["commands"].append(entry)
        save()
        start = time.monotonic()
        print(label, "started", flush=True)
        try:
            with (out / entry["log"]).open("w") as stream:
                process = subprocess.run(entry["argv"], stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, env=env)
            stdout = (out / entry["log"]).read_text()
        except subprocess.TimeoutExpired as error:
            entry.update(timed_out=True, wall_seconds=time.monotonic() - start)
            save()
            raise RuntimeError(label + " timed out") from error
        entry.update(exit_code=process.returncode, wall_seconds=time.monotonic() - start)
        save()
        print(label, "exit", process.returncode, flush=True)
        if (expected == "nonzero" and process.returncode == 0) or (expected != "nonzero" and process.returncode != expected):
            raise RuntimeError(label + " failed; see " + entry["log"])
        return stdout
    try:
        adapter, gcc, tools, opcode, forbidden = bind_runtime(paths, app, report, bind)
        report["runtime_helpers"].append(report["verifier_source"])
        report["verifier_source"] = bind(__file__)
        graph_dir = paths["graph_dir"]
        report["graph_artifacts"] = [bind(graph_dir / name) for name in ("graph.json", "graph.relax.py", "operators.tir.py", "operators.o", "operators.ll", "model.c", "constants.S", "constants.bin")]
        manifest = json.loads((graph_dir / "graph.json").read_text())
        validate_manifest(manifest)
        if not manifest.get("target", "").startswith("llvm -mtriple=riscv64-unknown-elf") or manifest.get("runtime_workspace_allocator_required") is not False:
            raise ValueError("Require a RV64 static export without runtime workspace allocation")
        if (graph_dir / "constants.bin").stat().st_size != max(manifest["constants_bytes"], 1):
            raise ValueError("Constant blob size differs from its declared storage envelope")
        if bind(graph_dir / "constants.bin")["sha256"] != manifest["constants_sha256"]:
            raise ValueError("Exported constants do not match their manifest")
        report["fixture"], report["fixture_provenance_artifact"] = bind(paths["fixture"]), bind(paths["fixture_provenance"])
        provenance = json.loads(paths["fixture_provenance"].read_text())
        if provenance.get("fixture_sha256") != report["fixture"]["sha256"] or not isinstance(provenance.get("reference_sources"), list) or not provenance["reference_sources"] or not isinstance(provenance.get("reference_contract"), str) or not provenance["reference_contract"].strip():
            raise ValueError("Fixture provenance must bind fixture_sha256, reference_sources and reference_contract")
        for source in provenance["reference_sources"]:
            bind(source["path"], source["sha256"])
        report["fixture_provenance"] = provenance
        arrays, samples = load_fixture(paths["fixture"], manifest)
        sample_ids = provenance.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != samples or any(not isinstance(value, str) or not value.strip() for value in sample_ids) or len(set(sample_ids)) != samples:
            raise ValueError("Fixture provenance requires distinct sample_ids for the complete selected stream")
        report["samples"], report["repeats"] = samples, args.repeats
        source, build = paths["tvm_source"], paths["tvm_build"]
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI="ctypes", PYTHONDONTWRITEBYTECODE="1")
        sys.path.insert(0, str(checked_path(source / "python")))
        import tvm
        from tvm._ffi.base import _LIB
        if not checked_path(tvm.__file__).is_relative_to(source / "python") or checked_path(_LIB._name).parent != build:
            raise ValueError("TVM source/library do not match selected paths")
        report["compiler_library"] = bind(_LIB._name)
        for name in ("apps/gemmini/baremetal.py", "python/tvm/relax/backend/contrib/gemmini_schedule.py"):
            bind(source / name)
        primitive_mod = tvm.script.from_source((graph_dir / "operators.tir.py").read_text())
        counts = dict.fromkeys(("loads_a", "loads_b", "computes", "stores", "admissions"), 0)
        shapes = []
        # Derive counts from the selected module's declared schedule, not model names.
        from tvm.relax.backend.contrib.gemmini_schedule import _DIM
        for call in manifest["calls"]:
            function = primitive_mod[call["symbol"]]
            if function.attrs and "gemmini.m" in function.attrs:
                m, n, k, ti, tj = (int(function.attrs["gemmini." + name]) for name in ("m", "n", "k", "tile_i", "tile_j"))
                mt, nt, kt = ((value + _DIM - 1) // _DIM for value in (m, n, k))
                row = {"symbol": call["symbol"], "m": m, "n": n, "k": k, "macs": m * n * k,
                       "loads_a": mt * ((n + tj * _DIM - 1) // (tj * _DIM)) * kt,
                       "loads_b": nt * ((m + ti * _DIM - 1) // (ti * _DIM)) * kt,
                       "computes": mt * nt * kt, "stores": mt * nt, "admissions": 1}
                shapes.append(row)
                for key in counts:
                    counts[key] += row[key]
        if not shapes:
            raise ValueError("No scheduled contraction with declared primitive-count metadata")
        report["schedule"] = {"shapes": shapes, "expected_counts_per_invocation": counts, "semantic_macs": sum(row["macs"] for row in shapes)}
        make_guest(out, manifest, arrays, samples, args.repeats, counts)
        report["generated_fixtures"] = [bind(out / f"{kind}_{index}.bin") for kind in ("input", "output") for index in range(len(manifest[kind + "s"]))]
        report["generated_guest_sources"] = [bind(out / name) for name in ("runner.c", "fixture.h", "fixture.S", "start.S", "link.ld")]
        env = dict(os.environ)
        for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_AUDIT"):
            env.pop(name, None)
        env["PATH"] = str(gcc.parent) + ":/usr/bin:/bin"
        flags = ["-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-msmall-data-limit=0", "-O2", "-ffreestanding", "-fno-builtin", "-fno-stack-protector", "-fno-tree-loop-distribute-patterns", "-nostdlib", "-nostartfiles", "-static", "-Wl,--no-relax", "-Wl,-T," + str(out / "link.ld"), "-I" + str(app), "-I" + str(out)]
        wrappers = ["-Wl,--wrap=tvm_gemmini_" + name for name in ("load_a", "load_b", "compute", "store", "validate_matmul_i8_i32")]
        programs = {}
        for name, wrong in (("valid", 0), ("bad-oracle", 1)):
            elf = out / (name + ".elf")
            command([gcc, *flags, *wrappers, "-DBAD_ORACLE=" + str(wrong), "-Wl,-Map," + str(out / (name + ".map")), out / "start.S", out / "runner.c", out / "fixture.S", graph_dir / "model.c", graph_dir / "constants.S", graph_dir / "operators.o", adapter, "-o", elf], "link-" + name, env)
            audit = audit_instructions(elf.read_bytes(), opcode, forbidden)
            if audit["prohibited_hits"]:
                raise ValueError("FSM instructions in final executable")
            report["checks"].append({"name": name + "_no_fsm", "passed": True, **audit})
            if command([tools["nm"], "-u", elf], "undefined-" + name).strip():
                raise ValueError("Unresolved runtime symbols in guest")
            programs[name] = elf
        headers = command([tools["readelf"], "-W", "-l", programs["valid"]], "program-headers")
        symbols_text = command([tools["nm"], "-S", "--defined-only", programs["valid"]], "symbols")
        segments, symbols, symbol_sizes = [], {}, {}
        for line in headers.splitlines():
            fields = line.split()
            if fields and fields[0] == "LOAD":
                segments.append({"start": int(fields[2], 16), "file_bytes": int(fields[4], 16), "memory_bytes": int(fields[5], 16)})
        for line in symbols_text.splitlines():
            fields = line.split()
            if len(fields) in (3, 4):
                try:
                    symbols[fields[-1]] = int(fields[0], 16)
                    symbol_sizes[fields[-1]] = int(fields[1], 16) if len(fields) == 4 else 0
                except ValueError:
                    pass
        if not segments or any(segment["start"] < GUEST_BASE or segment["start"] + segment["memory_bytes"] > GUEST_BASE + GUEST_BYTES for segment in segments):
            raise ValueError("ELF load extent exceeds requested memory")
        if symbols["_stack_top"] - symbols["_stack_bottom"] != STACK_BYTES or symbols["_image_end"] > GUEST_BASE + GUEST_BYTES:
            raise ValueError("Stack/image reservation mismatch")
        storage = ["model_constants", "workspace", *(f"input_{i}" for i in range(len(manifest["inputs"]))), *(f"output_{i}" for i in range(len(manifest["outputs"])))]
        if any(symbols[name] % 64 for name in storage):
            raise ValueError("Linked tensor storage is not aligned")
        fixture_symbols = []
        for kind in ("input", "output"):
            for index, tensor in enumerate(manifest[kind + "s"]):
                name = f"fixture_{kind}_{index}"
                extent = samples * aligned_size(tensor["bytes"])
                address = symbols.get(name)
                if address is None or address % 64 or symbol_sizes[name] != extent or not any(segment["start"] <= address and address + extent <= segment["start"] + segment["file_bytes"] for segment in segments):
                    raise ValueError("Linked fixture symbol extent does not match its frozen blob")
                fixture_symbols.append({"name": name, "address": address, "bytes": extent})
        report["fixture_symbols"] = fixture_symbols
        report["memory"] = {"load_segments": segments, "reserved_image_bytes": symbols["_image_end"] - GUEST_BASE,
                            "requested_guest_bytes": GUEST_BYTES, "stack_reserved_bytes": STACK_BYTES,
                            "workspace_bytes": manifest["workspace_bytes"], "constants_bytes": manifest["constants_bytes"],
                            "explicit_tensor_bytes": manifest["explicit_tensor_bytes"], "parameter_copies": 1,
                            "scope": "requested generic Spike map; not deployed hardware capacity"}
        simulator_env = dict(env, LD_LIBRARY_PATH=str(paths["spike_library_dir"]), PATH=str(paths["dtc"].parent) + ":/usr/bin:/bin")
        base = [paths["spike"], "--isa=rv64gc", "-m0x80000000:0x10000000", "-p1"]
        missing = command([*base, programs["valid"]], "without-extension", simulator_env, expected="nonzero")
        if "EXPORTED_GRAPH_INPUT_REJECTION_PASS" not in missing:
            raise ValueError("Missing-extension control did not reach device dispatch")
        extension = [*base, "--extlib=" + str(paths["plugin"]), "--extension=gemmini"]
        valid = command([*extension, programs["valid"]], "valid", simulator_env)
        gate = f"TVM_EXPORTED_GRAPH_PASS calls={samples * args.repeats} exact_bytes=1 guards=1 inputs=1 constants=1 retained=1 poisoned_workspace=1 primitive_counts=1"
        if gate not in valid or valid.count("EXPORTED_GRAPH_CALL_PASS") != samples * args.repeats:
            raise ValueError("Complete stream numerical/lifetime gate absent")
        high_water = next((line.partition("=")[2] for line in valid.splitlines() if line.startswith("STACK_HIGH_WATER_BYTES=")), None)
        if high_water is None or not 0 < int(high_water) < STACK_BYTES:
            raise ValueError("Stack high-water evidence absent or out of bounds")
        report["memory"]["stack_high_water_bytes"] = int(high_water)
        wrong = command([*extension, programs["bad-oracle"]], "bad-oracle", simulator_env, expected=22)
        if "EXPORTED_GRAPH_NUMERICAL_FAILURE" not in wrong:
            raise ValueError("Wrong-oracle control did not reach numerical gate")
        report["checks"].append({"name": "exported_graph_runtime", "passed": True, "samples": samples, "repeats": args.repeats,
                                 "invocations": samples * args.repeats, "exact_bytes": True, "primitive_counts": True,
                                 "inputs_preserved": True, "constants_preserved": True, "retained_outputs": True,
                                 "workspace_poisoned_each_call": True, "storage_canaries": True, "bad_oracle_exit": 22})
        for path, digest in immutable.items():
            if identity(path)["sha256"] != digest:
                raise ValueError("Source/tool/fixture changed during verification")
        report["immutable_bindings"] = immutable
        report["artifacts"] = [identity(out / name) for name in ("runner.c", "fixture.h", "fixture.S", "valid.elf", "bad-oracle.elf", "start.S", "link.ld")]
        report.update(status="passed", simulator_executed=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], "receipt": str(out / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
