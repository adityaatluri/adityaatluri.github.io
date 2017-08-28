---
title: "Parallel Programming Patterns on AMDGPU - Part 1"
layout: post
date: 2017-08-28 22:48
image: /assets/images/markdown.jpg
headerImage: false
tag:
- amd
- vega
- rx vega
- polaris
category: blog
author: adityaatluri
description: Implement Parallel Programming Patterns on AMDGPU - 1
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

# Introduction
If you are working on a parallel machine, you are using one of the parallel programming pattern or the other. For example, you are watching video on a webpage, you are doing *fork-join* managing the events of the media player, *map*, *reduce*, *stencil* to do the image processing in the video. In this blog, we try to implement multiple patterns on latest AMD GPUs (Vega and Polaris).
The patterns we implement are:
1. Map
2. Stencil
3. Reduction
4. Scan
5. Pack
6. Gather
7. Scatter
8. Expand

Throughout this blog, we build a library *arcticfox* which can operate these patterns on gpu device arrays. First, we create a `vector` which allocates device memory and use it to apply patterns.

# Map
{% highlight cpp %}
template<typename T, typename BinaryOp>
__global__ void Map(T *Out, T *In1, T *In2, size_t count, BinaryOp Func) {
  size_t tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if(tid < count) {
    Out[tid] = Func(In1[tid], In2[tid]);
  }
}
{% endhighlight %}

> Map implementation

{% highlight cpp %}
#include"arcticfox/map.h"

template<typename T>
struct add{
  __host__ __device__ T operator(const T& rhs, const T& lhs) const {
    return lhs + rhs;
  }
};

namespace arcticfox {
  template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOp>
  void map(InputIterator1 In1Begin, InputIterator1 In1End, InputIterator2 In2Begin, OutputIterator OutBegin, BinaryOp Op){
    size_t len = (In1End - In1Begin) / sizeof(*In1Begin);
    hipLaunchKernelGGL((Map<In1Begin->Type, BinaryOp>), );
  }
}

int main() {

}

{% endhighlight %}
