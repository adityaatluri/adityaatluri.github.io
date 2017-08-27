---
title: "Musings on GEMM"
layout: post
date: 2017-08-24 20:48
image: /assets/images/markdown.jpg
headerImage: false
tag:
- sse
- avx
- avx-512
- intel
- amd
category: blog
author: adityaatluri
description: Musings on GEMM
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

# Introduction
With lot of emphasis on gemm performance for deep-learning, people found that implementing gemm on hardware can bring more performance than using latency optimized cores. This is not limited to just the cores, the memory access patterns are hard wired into the silicon there by getting data to cores as fast as possible. In this blog we discuss about implementing high performing gemm operation on different available hardware. We don't try to implement the whole gemm implementation but just matrix multiplication of 4x4 sub-matrix.
There are 2 ways of implementing GEMM operations:
1. Inner Product
2. Outer Product

## Inner Product
This is something anyone who did basic algebra would know. It is a row multiplied with a column. Given 1x4 (row) multiplied with 4x1 (column) gives 1x1 element.

![Markdowm Image][1]{: class="bigger-image" }
<figcaption class="caption">Inner Product</figcaption>

> The number of math to load ops are: (4 MACs)/(4+4) = 0.5

## Outer Product
An outer product is where a 1x4 (column) multiplied with 1x4 (row) giving a 4x4 matrix.

![Markdowm Image][2]{: class="bigger-image" }
<figcaption class="caption">Outer Product</figcaption>

> The number of math to load ops are: (16 MACs)/(4+4) = 2

### Analysis
In terms of load registers, the number is same for both A and B, but the output for inner product requires only 1 register where as outer product requires 16 registers (assuming datatype is float). It is a trade-off that user should decide on whether to have less compute per load or use more registers to do better computer per load. Generally, using outer product is the best way to do gemm as most of the SIMD architectures have big enough register files to store the C matrix (AVX, AMD-GPU, NV-GPU)

In this blog we implement outer product on different SIMD architectures: X86, AMDGPU, NVGPU

# X86
X86 CPUs support SIMD instructions like `SSE`, `AVX`, `AVX512` and the latest `AVX512-4FMAPS`
## SSE
SSE are SIMD instructions operating on 128-bit wide data. Each SSE register (xmm0-xmm15) can hold 4 32-bit or 2 64-bit data types and operate on floats, int and double. For x86, we focus on just floats as halfs are emulated using shorts and doubles are over kill for machine learning applications.

{% highlight cpp %}
__m128 c0 = _mm_load_ps(C.data() + 0);
__m128 c1 = _mm_load_ps(C.data() + 4);
__m128 c2 = _mm_load_ps(C.data() + 8);
__m128 c3 = _mm_load_ps(C.data() + 12);

for (int j = 0; j < 4; j++) {
	__m128 a0 = _mm_broadcast_ss(A.data() + j);
	__m128 a1 = _mm_broadcast_ss(A.data() + 4 + j);
	__m128 a2 = _mm_broadcast_ss(A.data() + 8 + j);
	__m128 a3 = _mm_broadcast_ss(A.data() + 12 + j);

	__m128 b = _mm_load_ps(B.data() + j * 4);

	c0 = _mm_add_ps(_mm_mul_ps(a0, b), c0);
	c1 = _mm_add_ps(_mm_mul_ps(a1, b), c1);
	c2 = _mm_add_ps(_mm_mul_ps(a2, b), c2);
	c3 = _mm_add_ps(_mm_mul_ps(a3, b), c3);
}

_mm_store_ps(C.data() + 0, c0);
_mm_store_ps(C.data() + 4, c1);
_mm_store_ps(C.data() + 8, c2);
_mm_store_ps(C.data() + 12, c3);

{% endhighlight %}

As `__m128` can store 4 floats, we need 4 `__m128`s to store the 4x4 output matrix (c0,c1,c2,c3). The first row of the output matrix needs just a0 and first row of b matrix. We broadcast a0 to the 4 SIMD lanes and do multiply-and-add between broadcasted-a0 and b. Do the same for rest of the `a` elements (a1, a2 and a3).

## AVX
{% highlight cpp %}
__m256 c0 = _mm256_load_ps(C.data() + 0);
__m256 c1 = _mm256_load_ps(C.data() + 8);
__m256 c2 = _mm256_load_ps(C.data() + 16);
__m256 c3 = _mm256_load_ps(C.data() + 24);

__m256 c4 = _mm256_load_ps(C.data() + 32);
__m256 c5 = _mm256_load_ps(C.data() + 40);
__m256 c6 = _mm256_load_ps(C.data() + 48);
__m256 c7 = _mm256_load_ps(C.data() + 56);

for (int j = 0; j < 8; j++) {
	__m256 a0 = _mm256_broadcast_ss(A.data() + 0 * height + j);
	__m256 a1 = _mm256_broadcast_ss(A.data() + 1 * height + j);
	__m256 a2 = _mm256_broadcast_ss(A.data() + 2 * height + j);
	__m256 a3 = _mm256_broadcast_ss(A.data() + 3 * height + j);

	__m256 a4 = _mm256_broadcast_ss(A.data() + 4 * height + j);
	__m256 a5 = _mm256_broadcast_ss(A.data() + 5 * height + j);
	__m256 a6 = _mm256_broadcast_ss(A.data() + 6 * height + j);
	__m256 a7 = _mm256_broadcast_ss(A.data() + 7 * height + j);

	__m256 b = _mm256_load_ps(B.data() + j * 8);

	c0 = _mm256_add_ps(_mm256_mul_ps(a0, b), c0);
	c1 = _mm256_add_ps(_mm256_mul_ps(a1, b), c1);
	c2 = _mm256_add_ps(_mm256_mul_ps(a2, b), c2);
	c3 = _mm256_add_ps(_mm256_mul_ps(a3, b), c3);

	c4 = _mm256_add_ps(_mm256_mul_ps(a4, b), c4);
	c5 = _mm256_add_ps(_mm256_mul_ps(a5, b), c5);
	c6 = _mm256_add_ps(_mm256_mul_ps(a6, b), c6);
	c7 = _mm256_add_ps(_mm256_mul_ps(a7, b), c7);
}


_mm256_storeu_ps(C.data() + 0,  c0);
_mm256_storeu_ps(C.data() + 8,  c1);
_mm256_storeu_ps(C.data() + 16, c2);
_mm256_storeu_ps(C.data() + 24, c3);

_mm256_storeu_ps(C.data() + 32, c0);
_mm256_storeu_ps(C.data() + 40, c1);
_mm256_storeu_ps(C.data() + 48, c2);
_mm256_storeu_ps(C.data() + 56, c3);

{% endhighlight %}

## AVX512


# AMDGPU

## Float GEMM

{% highlight cpp %}

typedef float float4 __attribute__((ext_vector_type(4)));

__global__ void GEMM(float4 *A, float4 *B, float4 *C) {
	int tx = threadIdx.x;
	float4 a, b, c[4];
	float4 *Aptr = A + (tx / 2);
	float4 *Bptr = B + (tx % 2);
	c[0] = C[0 + (tx / 2)*8 + (tx % 2)];
	c[1] = C[2 + (tx / 2)*8 + (tx % 2)];
	c[2] = C[4 + (tx / 2)*8 + (tx % 2)];
	c[3] = C[6 + (tx / 2)*8 + (tx % 2)];
	for (int i = 0; i < A_WIDTH; i++) {
		a = *(Aptr + i * 2);
		b = *(Bptr + i * 2);
		c[0].x += a.x * b.x;
		c[0].y += a.x * b.y;
		c[0].z += a.x * b.z;
		c[0].w += a.x * b.w;

		c[1].x += a.y * b.x;
		c[1].y += a.y * b.y;
		c[1].z += a.y * b.z;
		c[1].w += a.y * b.w;

		c[2].x += a.z * b.x;
		c[2].y += a.z * b.y;
		c[2].z += a.z * b.z;
		c[2].w += a.z * b.w;

		c[3].x += a.w * b.x;
		c[3].y += a.w * b.y;
		c[3].z += a.w * b.z;
		c[3].w += a.w * b.w;
	}
	C[0 + (tx / 2)*8 + (tx % 2)] = c[0];
	C[2 + (tx / 2)*8 + (tx % 2)] = c[1];
	C[4 + (tx / 2)*8 + (tx % 2)] = c[2];
	C[6 + (tx / 2)*8 + (tx % 2)] = c[3];
}

{% endhighlight %}

## Half2 GEMM

{% highlight cpp %}

typedef __fp16 __half4 __attribute__((ext_vector_type(4)));

typedef struct {
    __half4 p, q;
} __half8;

#define A_WIDTH 16
#define A_HEIGHT 16
#define B_WIDTH 16
#define B_HEIGHT 16
#define C_WIDTH 16
#define C_HEIGHT 16

extern "C" __half4 __v_pk_fma_f16(__half4, __half4, __half4) __asm("llvm.fma.v2f16");

__global__ void GEMM(__half8 *A, __half8 *B, __half8 *C) {
    int tx = threadIdx.x;
    __half4 a0, a1, b0, b1, c[16];
    __half8 *Aptr = A + (tx / 2);
    __half8 *Bptr = B + (tx % 2);
    for(int i=0;i<8;i++){
        c[2*i+0]  = C[i*2 + (tx / 2) * 8 + (tx % 2)].p;
        c[2*i+1]  = C[i*2 + (tx / 2) * 8 + (tx % 2)].q;
    }

    for (int i = 0; i < A_WIDTH; i++) {
        a0 = *(Aptr + i * 2).p;
        a1 = *(Aptr + i * 2).q;
        b0 = *(Bptr + i * 2).p;
        b1 = *(Bptr + i * 2).q;
        c[0].xy = __v_pk_fma_f16(a0.xx, b0.xy, c[0].xy);
        c[0].zw = __v_pk_fma_f16(a0.xx, b0.zw, c[0].zw);

        c[1].xy = __v_pk_fma_f16(a0.xx, b1.xy, c[1].xy);
        c[1].zw = __v_pk_fma_f16(a0.xx, b1.zw, c[1].zw);

        c[2].xy = __v_pk_fma_f16(a0.yy, b0.xy, c[2].xy);
        c[2].zw = __v_pk_fma_f16(a0.yy, b0.zw, c[2].zw);

        c[3].xy = __v_pk_fma_f16(a0.yy, b1.xy, c[3].xy);
        c[3].zw = __v_pk_fma_f16(a0.yy, b1.zw, c[3].zw);

        c[4].xy = __v_pk_fma_f16(a0.zz, b0.xy, c[4].xy);
        c[4].zw = __v_pk_fma_f16(a0.zz, b0.zw, c[4].zw);

        c[5].xy = __v_pk_fma_f16(a0.zz, b1.xy, c[5].xy);
        c[5].zw = __v_pk_fma_f16(a0.zz, b1.zw, c[5].zw);

        c[6].xy = __v_pk_fma_f16(a0.ww, b0.xy, c[6].xy);
        c[6].zw = __v_pk_fma_f16(a0.ww, b0.zw, c[6].zw);

        c[7].xy = __v_pk_fma_f16(a0.ww, b1.xy, c[7].xy);
        c[7].zw = __v_pk_fma_f16(a0.ww, b1.zw, c[7].zw);

        c[8].xy = __v_pk_fma_f16(a1.xx, b0.xy, c[8].xy);
        c[8].zw = __v_pk_fma_f16(a1.xx, b0.zw, c[8].zw);

        c[9].xy = __v_pk_fma_f16(a1.xx, b1.xy, c[9].xy);
        c[9].zw = __v_pk_fma_f16(a1.xx, b1.zw, c[9].zw);

        c[10].xy = __v_pk_fma_f16(a1.yy, b0.xy, c[10].xy);
        c[10].zw = __v_pk_fma_f16(a1.yy, b0.zw, c[10].zw);

        c[11].xy = __v_pk_fma_f16(a1.yy, b1.xy, c[11].xy);
        c[11].zw = __v_pk_fma_f16(a1.yy, b1.zw, c[11].zw);

        c[12].xy = __v_pk_fma_f16(a1.zz, b0.xy, c[12].xy);
        c[12].zw = __v_pk_fma_f16(a1.zz, b0.zw, c[12].zw);

        c[13].xy = __v_pk_fma_f16(a1.zz, b1.xy, c[13].xy);
        c[13].zw = __v_pk_fma_f16(a1.zz, b1.zw, c[13].zw);

        c[14].xy = __v_pk_fma_f16(a1.ww, b0.xy, c[14].xy);
        c[14].zw = __v_pk_fma_f16(a1.ww, b0.zw, c[14].zw);

        c[15].xy = __v_pk_fma_f16(a0.ww, b1.xy, c[15].xy);
        c[15].zw = __v_pk_fma_f16(a0.ww, b1.zw, c[15].zw);
    }

    for(int i=0;i<8;i++){
        C[2*i + (tx / 2) * 8 + (tx % 2)].p = c[2*i+0];
        C[2*i + (tx / 2) * 8 + (tx % 2)].q = c[2*i+1];
    }

}


{% endhighlight %}

## Mixed Precision GEMM

{% highlight cpp %}

#define HEIGHT 8
#define WIDTH 8

__global__ void GEMM(__half4 *A, __half4 *B, float *C) {
    int tx = threadIdx.x;
    __half4 a, b;
    float c[16];
    __half4 *Aptr = A + (tx / 2);
    __half4 *Bptr = B + (tx % 2);

    for(int i=0;i<4;i++){
        c[0 + i*4] = C[0 + i * 8 + (tx / 2) * 32 + (tx % 2)*4];
        c[1 + i*4] = C[1 + i * 8 + (tx / 2) * 32 + (tx % 2)*4];
        c[2 + i*4] = C[2 + i * 8 + (tx / 2) * 32 + (tx % 2)*4];
        c[3 + i*4] = C[3 + i * 8 + (tx / 2) * 32 + (tx % 2)*4];
    }

    for (int i = 0; i < WIDTH; i++) {
        a = *(Aptr + i * 2);
        b = *(Bptr + i * 2);

        // c[0] = a.x * b.x + c[0]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[0,1,1]": "=v"(c[0]): "v"(a.xy), "v"(b.xy), "v"(c[0]));
        // c[1] = a.x * b.y + c[1]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,1,1]": "=v"(c[1]): "v"(a.xy), "v"(b.xy), "v"(c[1]));
        // c[2] = a.x * b.z + c[2]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[0,1,1]": "=v"(c[2]): "v"(a.xy), "v"(b.zw), "v"(c[2]));
        // c[3] = a.x * b.w + c[2]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,1,1]": "=v"(c[3]): "v"(a.xy), "v"(b.zw), "v"(c[3]));

        // c[4] = a.y * b.x + c[4]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,1]": "=v"(c[4]): "v"(a.xy), "v"(b.xy), "v"(c[4]));
        // c[5] = a.y * b.y + c[5]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,1] op_sel_hi:[0,1,1]": "=v"(c[5]): "v"(a.xy), "v"(b.xy), "v"(c[5]));
        // c[6] = a.y * b.z + c[6]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,1]": "=v"(c[6]): "v"(a.xy), "v"(b.zw), "v"(c[6]));
        // c[7] = a.y * b.w + c[7]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,1] op_sel_hi:[0,1,1]": "=v"(c[7]): "v"(a.xy), "v"(b.zw), "v"(c[7]));

        // c[8] = a.z * b.x + c[8]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[0,1,1]": "=v"(c[8]): "v"(a.zw), "v"(b.xy), "v"(c[8]));
        // c[9] = a.z * b.y + c[9]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,1,1]": "=v"(c[9]): "v"(a.zw), "v"(b.xy), "v"(c[9]));
        // c[10] = a.z * b.z + c[10]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[0,1,1]": "=v"(c[10]): "v"(a.zw), "v"(b.zw), "v"(c[10]));
        // c[11] = a.z * b.w + c[11]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,1,1]": "=v"(c[11]): "v"(a.zw), "v"(b.zw), "v"(c[11]));

        // c[12] = a.w * b.x + c[12]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,1]": "=v"(c[12]): "v"(a.zw), "v"(b.xy), "v"(c[12]));
        // c[13] = a.w * b.y + c[13]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,1] op_sel_hi:[0,1,1]": "=v"(c[13]): "v"(a.zw), "v"(b.xy), "v"(c[13]));
        // c[14] = a.w * b.z + c[14]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,1]": "=v"(c[14]): "v"(a.zw), "v"(b.zw), "v"(c[14]));
        // c[15] = a.w * b.w + c[15]
        asm volatile("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,1] op_sel_hi:[0,1,1]": "=v"(c[15]): "v"(a.zw), "v"(b.zw), "v"(c[15]));

    }

    for(int i=0;i<4;i++){
        C[0 + i * 8 + (tx / 2) * 32 + (tx % 2)*4] = c[0 + i*4];
        C[1 + i * 8 + (tx / 2) * 32 + (tx % 2)*4] = c[1 + i*4];
        C[2 + i * 8 + (tx / 2) * 32 + (tx % 2)*4] = c[2 + i*4];
        C[3 + i * 8 + (tx / 2) * 32 + (tx % 2)*4] = c[3 + i*4];
    }

}


{% endhighlight %}

# NVGPU

## Float GEMM
{% highlight cpp %}

__global__ void GEMM(float4 *A, float4 *B, float4 *C) {
	int tx = threadIdx.x;
	float4 a, b, c[4];
	float4 *Aptr = A + (tx / 2);
	float4 *Bptr = B + (tx % 2);
	c[0] = C[0 + (tx / 2)*8 + (tx % 2)];
	c[1] = C[2 + (tx / 2)*8 + (tx % 2)];
	c[2] = C[4 + (tx / 2)*8 + (tx % 2)];
	c[3] = C[6 + (tx / 2)*8 + (tx % 2)];
	for (int i = 0; i < A_WIDTH; i++) {
		a = *(Aptr + i * 2);
		b = *(Bptr + i * 2);
		c[0].x += a.x * b.x;
		c[0].y += a.x * b.y;
		c[0].z += a.x * b.z;
		c[0].w += a.x * b.w;

		c[1].x += a.y * b.x;
		c[1].y += a.y * b.y;
		c[1].z += a.y * b.z;
		c[1].w += a.y * b.w;

		c[2].x += a.z * b.x;
		c[2].y += a.z * b.y;
		c[2].z += a.z * b.z;
		c[2].w += a.z * b.w;

		c[3].x += a.w * b.x;
		c[3].y += a.w * b.y;
		c[3].z += a.w * b.z;
		c[3].w += a.w * b.w;
	}
	C[0 + (tx / 2)*8 + (tx % 2)] = c[0];
	C[2 + (tx / 2)*8 + (tx % 2)] = c[1];
	C[4 + (tx / 2)*8 + (tx % 2)] = c[2];
	C[6 + (tx / 2)*8 + (tx % 2)] = c[3];
}

{% endhighlight %}

## Half2 GEMM

{% highlight cpp %}

typedef struct {
	__half2 x, y, z, w;
}__half8;

__global__ void GEMM(__half8 *A, __half8 *B, __half8 *C) {
	int tx = threadIdx.x;
	__half8 a, b, c[4];
	__half8 *Aptr = A + (tx / 2);
	__half8 *Bptr = B + (tx % 2);
	c[0] = C[0 + (tx / 2) * 8 + (tx % 2)];
	c[1] = C[2 + (tx / 2) * 8 + (tx % 2)];
	c[2] = C[4 + (tx / 2) * 8 + (tx % 2)];
	c[3] = C[6 + (tx / 2) * 8 + (tx % 2)];
	for (int i = 0; i < A_WIDTH; i++) {
		a = *(Aptr + i * 2);
		b = *(Bptr + i * 2);
		c[0].x = __hfma2(a.x, b.x, c[0].x);
		c[0].y = __hfma2(a.x, b.y, c[0].y);
		c[0].z = __hfma2(a.x, b.z, c[0].z);
		c[0].w = __hfma2(a.x, b.w, c[0].w);

		c[1].x = __hfma2(a.y, b.x, c[1].x);
		c[1].y = __hfma2(a.y, b.y, c[1].y);
		c[1].z = __hfma2(a.y, b.z, c[1].z);
		c[1].w = __hfma2(a.y, b.w, c[1].w);

		c[2].x = __hfma2(a.z, b.x, c[2].x);
		c[2].y = __hfma2(a.z, b.y, c[2].y);
		c[2].z = __hfma2(a.z, b.z, c[2].z);
		c[2].w = __hfma2(a.z, b.w, c[2].w);

		c[3].x = __hfma2(a.w, b.x, c[3].x);
		c[3].y = __hfma2(a.w, b.y, c[3].y);
		c[3].z = __hfma2(a.w, b.z, c[3].z);
		c[3].w = __hfma2(a.w, b.w, c[3].w);
	}
	C[0 + (tx / 2) * 8 + (tx % 2)] = c[0];
	C[2 + (tx / 2) * 8 + (tx % 2)] = c[1];
	C[4 + (tx / 2) * 8 + (tx % 2)] = c[2];
	C[6 + (tx / 2) * 8 + (tx % 2)] = c[3];
}


{% endhighlight %}

## Int4 GEMM

{% highlight cpp %}

__global__ void GEMM(int4 *A, int4 *B, int4 *C) {
	int tx = threadIdx.x;
	int4 a, b, c[4];
	int4 *Aptr = A + (tx / 2);
	int4 *Bptr = B + (tx % 2);
	c[0] = C[0 + (tx / 2) * 8 + (tx % 2)];
	c[1] = C[2 + (tx / 2) * 8 + (tx % 2)];
	c[2] = C[4 + (tx / 2) * 8 + (tx % 2)];
	c[3] = C[6 + (tx / 2) * 8 + (tx % 2)];
	for (int i = 0; i < A_WIDTH; i++) {
		a = *(Aptr + i * 2);
		b = *(Bptr + i * 2);
		c[0].x = __dp4a(a.x, b.x, c[0].x);
		c[0].y = __dp4a(a.x, b.y, c[0].y);
		c[0].z = __dp4a(a.x, b.z, c[0].z);
		c[0].w = __dp4a(a.x, b.w, c[0].w);

		c[1].x = __dp4a(a.y, b.x, c[1].x);
		c[1].y = __dp4a(a.y, b.y, c[1].y);
		c[1].z = __dp4a(a.y, b.z, c[1].z);
		c[1].w = __dp4a(a.y, b.w, c[1].w);

		c[2].x = __dp4a(a.z, b.x, c[2].x);
		c[2].y = __dp4a(a.z, b.y, c[2].y);
		c[2].z = __dp4a(a.z, b.z, c[2].z);
		c[2].w = __dp4a(a.z, b.w, c[2].w);

		c[3].x = __dp4a(a.w, b.x, c[3].x);
		c[3].y = __dp4a(a.w, b.y, c[3].y);
		c[3].z = __dp4a(a.w, b.z, c[3].z);
		c[3].w = __dp4a(a.w, b.w, c[3].w);
	}
	C[0 + (tx / 2) * 8 + (tx % 2)] = c[0];
	C[2 + (tx / 2) * 8 + (tx % 2)] = c[1];
	C[4 + (tx / 2) * 8 + (tx % 2)] = c[2];
	C[6 + (tx / 2) * 8 + (tx % 2)] = c[3];
}

{% endhighlight %}

## WMMA GEMM

Will add once CUDA 9 gets public

[1]: https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide1.JPG
[2]: https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide2.JPG
