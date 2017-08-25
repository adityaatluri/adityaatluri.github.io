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
With lot of emphasis on gemm performance for deep-learning, people found that implementing gemm on hardware can bring more performance than using latency optimized cores. This is not limited to just the cores, the memory access patterns are hard wired into the silicon there by getting data to cores as fast as possible.
There are 2 ways of implementing GEMM operations:
1. Inner Product
2. Outer Product

## Inner Product
This is something anyone who did basic algebra would know. It is a row multiplied with a column. Given 1x4 (row) multiplied with 4x1 (column) gives 1x1 element.
![Markdowm Image][https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide1.JPG]
<figcaption class="caption">Inner Product</figcaption>
The number of math to load ops are: (4 MACs)/(4+4). This a bad ratio if you are doing gemm on throughput-optimized cores (GPUs). Then, how are BLAS libraries on GPUs able to achieve peak throughput? The answer is **Outer Product**

## Outer Product
An outer product is where a 1x4 (column) multiplied with 1x4 (row) giving a 4x4 matrix.
![Markdowm Image](https://github.com/adityaatluri/adityaatluri.github.io/raw/master/assets/images/Slide2.JPG)
<figcaption class="caption">Outer Product</figcaption>
The number of math to load ops are: (16 MACs)/(4+4). This a good ratio for vector/simd processors (GPUs).

In this blog we implement outer product on different SIMD architectures (SSE, AVX, AVX512, AVX512-4FMAPS, AMD-GPU, NV-GPU, NV-TensorCores)

### SSE
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

### AVX
{% highlight cpp %}

{% endhighlight %}
