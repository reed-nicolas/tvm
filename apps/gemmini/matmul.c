/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "matmul.h"

#include <stddef.h>
#include <stdint.h>

#ifndef TVM_GEMMINI_FORBID_HW_LOOPS
#define TVM_GEMMINI_FORBID_HW_LOOPS 0
#endif
#if TVM_GEMMINI_FORBID_HW_LOOPS != 1
#error "The OS C-library adapter requires explicit TVM_GEMMINI_FORBID_HW_LOOPS=1"
#endif
#if !defined(TVM_GEMMINI_PE_OUTPUT_BITS) || TVM_GEMMINI_PE_OUTPUT_BITS != 20
#error "Bind the source-verified 20-bit PE output using TVM_GEMMINI_PE_OUTPUT_BITS=20"
#endif
#if !defined(__riscv) || __riscv_xlen != 64
#error "The Gemmini adapter requires an RV64 target"
#endif

/* Load the selected generated header before the library. Do not change its bytes.
 * The pinned library parses normalization branches even when hardware support is
 * absent. This parsing fallback enables no feature: the only admitted activation
 * below is NO_ACTIVATION, and HAS_NORMALIZATIONS stays absent.
 */
#include "include/gemmini_params.h"
#ifdef HAS_NORMALIZATIONS
#error "This Gemmini adapter does not admit a normalization configuration"
#endif
#ifndef NORM_STAT_IDS
#define NORM_STAT_IDS 2
#endif
#include "include/gemmini.h"

#if defined(ELEM_T_IS_FLOAT) || defined(ELEM_T_IS_LOWPREC_FLOAT)
#error "The Gemmini adapter requires signed integer operands"
#endif
#if !defined(HAS_MVIN_SCALE) || !defined(ACC_SCALE_T_IS_FLOAT)
#error "The Gemmini adapter requires floating-point identity input/output scales"
#endif
#ifndef ACC_READ_FULL_WIDTH
#error "The Gemmini adapter requires full-width accumulator readout"
#endif

_Static_assert(CHAR_BIT == 8 && sizeof(elem_t) == 1 && (elem_t)-1 < 0, "elem_t must be signed int8");
_Static_assert(sizeof(acc_t) == 4 && (acc_t)-1 < 0, "acc_t must be signed int32");
_Static_assert(_Generic((scale_t)0, float: 1, default: 0), "scale_t must be float");
_Static_assert(_Generic((acc_scale_t)0, float: 1, default: 0), "acc_scale_t must be float");
_Static_assert(sizeof(size_t) == 8 && sizeof(uintptr_t) == 8, "RV64 pointer/size ABI required");
_Static_assert(DIM == 16, "the bounded OS schedule requires DIM=16");
_Static_assert(BANK_NUM * BANK_ROWS >= 2 * DIM && ACC_ROWS >= DIM, "insufficient resources for an OS tile");

/* Conservative byte envelopes include stride padding. Callers must own and map
 * the entire envelope; this validates representability, not allocation or DMA
 * accessibility. The selected runtime must serialize use of the accelerator.
 */
static int span(const void* data, uint64_t rows, uint64_t cols, uint64_t stride, size_t element_bytes, uintptr_t* begin, uintptr_t* end) {
  uint64_t max_elements = (uint64_t)PTRDIFF_MAX / element_bytes;
  if (data == NULL || stride < cols || cols > max_elements || rows - 1 > (max_elements - cols) / stride) {
    return 0;
  }
  uint64_t bytes = ((rows - 1) * stride + cols) * element_bytes;
  *begin = (uintptr_t)data;
  if (bytes > UINTPTR_MAX - *begin) {
    return 0;
  }
  *end = *begin + bytes;
  return 1;
}

int32_t tvm_gemmini_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k,
                                int64_t a_stride, int64_t b_stride, int64_t c_stride) {
  if (m <= 0 || n <= 0 || k <= 0 || k > 131071 || a_stride <= 0 || b_stride <= 0 || c_stride <= 0 || (uintptr_t)c % _Alignof(int32_t) != 0) {
    return -1;
  }
  /* CONFIG_ST packs its byte stride into 32 bits; null D still gets configured.
   * Apply the same conservative limit to input DMA strides.
   */
  if ((uint64_t)a_stride > UINT32_MAX || (uint64_t)b_stride > UINT32_MAX ||
      (uint64_t)c_stride > UINT32_MAX / sizeof(acc_t) || (uint64_t)n > UINT32_MAX / sizeof(acc_t)) {
    return -2;
  }
  uintptr_t a_begin, a_end, b_begin, b_end, c_begin, c_end;
  if (!span(a, m, k, a_stride, sizeof(elem_t), &a_begin, &a_end) ||
      !span(b, k, n, b_stride, sizeof(elem_t), &b_begin, &b_end) ||
      !span(c, m, n, c_stride, sizeof(acc_t), &c_begin, &c_end)) {
    return -2;
  }
  if ((c_begin < a_end && a_begin < c_end) || (c_begin < b_end && b_begin < c_end)) {
    return -3;
  }
  if (MVIN_SCALE_IDENTITY != 1.0f || ACC_SCALE_IDENTITY != 1.0f) {
    return -4;
  }

  /* Exclusive accelerator ownership is a caller obligation. Flush stale address
   * translations before this invocation; the runtime must make buffers DMA
   * accessible and provide the platform's completion/coherence contract.
   * The vendor fence lacks a compiler memory clobber. These barriers order CPU
   * accesses around its device fence; they cannot establish platform coherence.
   */
  asm volatile("fence rw,rw" ::: "memory");
  gemmini_flush(0);
  /* Fixed tile_K=1 bounds each PE reduction to 16 products: at most 262144 in
   * magnitude, within the source-verified signed 20-bit PE output. The library
   * accumulates successive K tiles in int32 SRAM. Auto tiling can exceed this
   * PE bound; do not substitute tiled_matmul_auto here.
   */
  tiled_matmul((size_t)m, (size_t)n, (size_t)k, a, b, NULL, c, (size_t)a_stride, (size_t)b_stride, (size_t)n, (size_t)c_stride,
               MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 0, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
               false, 1, 1, 1, false, false, true, false, 0, OS);
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}
