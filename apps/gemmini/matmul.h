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
 * See the License for the specific language governing permissions
 * and limitations under the License.
 */
#ifndef TVM_APPS_GEMMINI_MATMUL_H_
#define TVM_APPS_GEMMINI_MATMUL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C[m,n] = A[m,k] * B[k,n], signed exact arithmetic, no bias or scaling.
 * Strides count elements, with row-major nonoverlapping output storage.
 * Return 0 only after all output writes are complete and visible to the caller.
 * Preserve A/B. Nonzero signals failure; callers must not consume partial output.
 * LowerGemminiMatmul emits positive dimensions, k <= 131071, compact row strides.
 * Hardware/library implementations must independently enforce their capabilities.
 */
int32_t tvm_gemmini_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t a_stride, int64_t b_stride, int64_t c_stride);

/* CPU-only whole-operation admission. Call before issuing any primitive.
 * The return values and memory contract match tvm_gemmini_matmul_i8_i32.
 */
int32_t tvm_gemmini_validate_matmul_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, int64_t m, int64_t n, int64_t k, int64_t a_stride, int64_t b_stride, int64_t c_stride);

/* Trusted compiler-generated primitive contract: validation above passed; the
 * caller owns the accelerator exclusively from begin through end. TVM owns all
 * tile loops, pointer offsets, local allocation, reuse, and reduction ordering.
 * Begin uses the validated strides; every DMA pointer is a proven subview of
 * the validated A/B/C envelopes. Strides count elements. Local addresses are plain DIM-aligned row indices;
 * reserve 16 rows per tile within 16384 scratchpad / 1024 accumulator rows.
 * A/B tile slots are disjoint. Every tile dimension is in [1,16], and accumulate
 * is exactly 0 (overwrite) or 1 (add to initialized int32 accumulator SRAM).
 * Each compute starts a fresh <=16-product PE reduction; global K <=131071.
 * A/B and any DMA source staging buffers remain immutable until end. Store is
 * asynchronous: read C only after end. These functions do no runtime admission.
 * The runtime must supply DMA mapping, completion, and platform coherence.
 */
void tvm_gemmini_begin(int64_t a_stride, int64_t b_stride, int64_t c_stride);
void tvm_gemmini_load_a(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols);
void tvm_gemmini_load_b(const int8_t* src, uint32_t spadrow, uint32_t rows, uint32_t cols);
void tvm_gemmini_compute(uint32_t arow, uint32_t brow, uint32_t crow, uint32_t m, uint32_t n, uint32_t k, int32_t accumulate);
void tvm_gemmini_store(int32_t* dst, uint32_t crow, uint32_t rows, uint32_t cols);
void tvm_gemmini_end(void);

#ifdef __cplusplus
}
#endif
#endif  /* TVM_APPS_GEMMINI_MATMUL_H_ */
