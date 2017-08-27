#include<iostream>
#include<vector>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

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

#define TX 4

extern "C" __half4 __v_pk_fma_f16(__half4, __half4, __half4) __asm("llvm.fma.v2f16");

__global__ void GEMM(__half8 *A, __half8 *B, __half8 *C) {
    int tx = threadIdx.x;
    __half4 a0, a1, b0, b1, c[16];
    __half8 *Aptr = A + (tx / 2);
    __half8 *Bptr = B + (tx % 2);
    c[0]  = C[0 + (tx / 2) * 8 + (tx % 2)].p;
    c[1]  = C[0 + (tx / 2) * 8 + (tx % 2)].q;
    c[2]  = C[2 + (tx / 2) * 8 + (tx % 2)].p;
    c[3]  = C[2 + (tx / 2) * 8 + (tx % 2)].q;
    c[4]  = C[4 + (tx / 2) * 8 + (tx % 2)].p;
    c[5]  = C[4 + (tx / 2) * 8 + (tx % 2)].q;
    c[6]  = C[6 + (tx / 2) * 8 + (tx % 2)].p;
    c[7]  = C[6 + (tx / 2) * 8 + (tx % 2)].q;
    c[8]  = C[8 + (tx / 2) * 8 + (tx % 2)].p;
    c[9]  = C[8 + (tx / 2) * 8 + (tx % 2)].q;
    c[10] = C[10 + (tx / 2) * 8 + (tx % 2)].p;
    c[11] = C[10 + (tx / 2) * 8 + (tx % 2)].q;
    c[12] = C[12 + (tx / 2) * 8 + (tx % 2)].p;
    c[13] = C[12 + (tx / 2) * 8 + (tx % 2)].q;
    c[14] = C[14 + (tx / 2) * 8 + (tx % 2)].p;
    c[15] = c[14 + (tx / 2) * 8 + (tx % 2)].q;
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

    C[0 + (tx / 2) * 8 + (tx % 2)].p = c[0];
    C[0 + (tx / 2) * 8 + (tx % 2)].q = c[1];
    C[2 + (tx / 2) * 8 + (tx % 2)].p = c[2];
    C[2 + (tx / 2) * 8 + (tx % 2)].q = c[3];
    C[4 + (tx / 2) * 8 + (tx % 2)].p = c[4];
    C[4 + (tx / 2) * 8 + (tx % 2)].q = c[5];
    C[6 + (tx / 2) * 8 + (tx % 2)].p = c[6];
    C[6 + (tx / 2) * 8 + (tx % 2)].q = c[7];
    C[8 + (tx / 2) * 8 + (tx % 2)].p = c[8];
    C[8 + (tx / 2) * 8 + (tx % 2)].q = c[9];
    C[10 + (tx / 2) * 8 + (tx % 2)].p = c[10];
    C[10 + (tx / 2) * 8 + (tx % 2)].q = c[11];
    C[12 + (tx / 2) * 8 + (tx % 2)].p = c[12];
    C[12 + (tx / 2) * 8 + (tx % 2)].q = c[13];
    C[14 + (tx / 2) * 8 + (tx % 2)].p = c[14];
    C[14 + (tx / 2) * 8 + (tx % 2)].q = c[15];

}
