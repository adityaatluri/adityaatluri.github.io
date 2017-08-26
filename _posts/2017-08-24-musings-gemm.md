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

## Half2 GEMM

## Mixed Precision GEMM

# NVGPU

## Float GEMM
{% highlight cpp %}
__global__ void GEMM512x512(float4 *A, float4 *B, float *C){
	int tx = threadIdx.x;
	__shared__ float4 sA[];
}
{% endhighlight %}

## Half2 GEMM

## Int4 GEMM

## WMMA GEMM

[1]: https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide1.JPG
[2]: https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide2.JPG
