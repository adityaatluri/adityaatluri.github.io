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

# SSE
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
