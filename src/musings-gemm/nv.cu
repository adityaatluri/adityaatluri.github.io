#include<iostream>
#include<vector>
#include<cuda.h>
#include"cuda_runtime.h"
#include"sm_61_intrinsics.h"
#include<cuda_fp16.h>

#define A_WIDTH 8
#define A_HEIGHT 8
#define B_WIDTH 8
#define B_HEIGHT 8

#define TX 4

#define SIZE A_WIDTH * A_HEIGHT * sizeof(float)

typedef struct {
	__half2 x, y, z, w;
}__half8;

__global__ void GEMM(float4 *A, float4 *B, float4 *C) {
	int tx = threadIdx.x;
	float4 a, b, c[4];
	float4* Aptr = A + (tx / 2);
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

__global__ void GEMM(int4 *A, int4 *B, int4 *C) {
	int tx = threadIdx.x;
	int4 a, b, c[4];
	int4* Aptr = A + (tx / 2);
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

void fGEMM() {
	std::vector<float> A(A_WIDTH*A_HEIGHT);
	std::vector<float> B(B_WIDTH*B_HEIGHT);
	std::vector<float> C(A_HEIGHT*B_WIDTH);

	std::fill(A.begin(), A.end(), 1.0f);
	std::fill(B.begin(), B.end(), 1.0f);
	std::fill(C.begin(), C.end(), 0.0f);

	float *Ad, *Bd, *Cd;

	cudaMalloc(&Ad, SIZE);
	cudaMalloc(&Bd, SIZE);
	cudaMalloc(&Cd, SIZE);

	cudaMemcpy(Ad, A.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Cd, C.data(), SIZE, cudaMemcpyHostToDevice);

	GEMM << <dim3(1, 1, 1), dim3(TX, 1, 1) >> > ((float4*)Ad, (float4*)Bd, (float4*)Cd);
	cudaDeviceSynchronize();

	cudaMemcpy(C.data(), Cd, SIZE, cudaMemcpyDeviceToHost);

	for (int j = 0; j < A_HEIGHT; j++) {
		for (int i = 0; i < A_WIDTH; i++) {
			std::cout << C[i + j*A_WIDTH] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void cGEMM() {
	std::vector<char> A(A_WIDTH*A_HEIGHT*4);
	std::vector<char> B(B_WIDTH*B_HEIGHT*4);
	std::vector<int> C(A_HEIGHT*B_WIDTH);

	std::fill(A.begin(), A.end(), char(1));
	std::fill(B.begin(), B.end(), char(1));
	std::fill(C.begin(), C.end(), int(0));

	float *Ad, *Bd, *Cd;

	cudaMalloc(&Ad, SIZE);
	cudaMalloc(&Bd, SIZE);
	cudaMalloc(&Cd, SIZE);

	cudaMemcpy(Ad, A.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Cd, C.data(), SIZE, cudaMemcpyHostToDevice);

	GEMM << <dim3(1, 1, 1), dim3(TX, 1, 1) >> > ((int4*)Ad, (int4*)Bd, (int4*)Cd);
	cudaDeviceSynchronize();

	cudaMemcpy(C.data(), Cd, SIZE, cudaMemcpyDeviceToHost);

	for (int j = 0; j < A_HEIGHT; j++) {
		for (int i = 0; i < A_WIDTH; i++) {
			std::cout << C[i + j*A_WIDTH] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void hGEMM() {
	std::vector<__half> A(A_WIDTH*A_HEIGHT * 2);
	std::vector<__half> B(B_WIDTH*B_HEIGHT * 2);
	std::vector<__half> C(A_HEIGHT*B_WIDTH * 2);

	for (int i = 0; i < A_WIDTH*A_HEIGHT*2; i++) {
		A[i].x = 1;
		B[i].x = 1;
		C[i].x = 0;
	}

	float *Ad, *Bd, *Cd;

	cudaMalloc(&Ad, SIZE);
	cudaMalloc(&Bd, SIZE);
	cudaMalloc(&Cd, SIZE);

	cudaMemcpy(Ad, A.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B.data(), SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Cd, C.data(), SIZE, cudaMemcpyHostToDevice);

	GEMM << <dim3(1, 1, 1), dim3(TX, 1, 1) >> > ((__half8*)Ad, (__half8*)Bd, (__half8*)Cd);
	cudaDeviceSynchronize();

	cudaMemcpy(C.data(), Cd, SIZE, cudaMemcpyDeviceToHost);

	for (int j = 0; j < A_HEIGHT; j++) {
		for (int i = 0; i < A_WIDTH*2; i++) {
			std::cout << C[i + j*A_WIDTH*2].x << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main() {
	fGEMM();
	cGEMM();
	hGEMM();
}
