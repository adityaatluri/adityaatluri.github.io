#include<iostream>
#include<vector>
#include<chrono>
#include<mmintrin.h>


void Do4x4SSE() {
	std::vector<float> A(4 * 4);
	std::vector<float> B(4 * 4);
	std::vector<float> C(4 * 4);
	std::fill(A.begin(), A.end(), 1.0f);
	std::fill(B.begin(), B.end(), 1.0f);
	std::fill(C.begin(), C.end(), 0.0f);

	static const int ITER = 1024 * 1024;

	auto start = std::chrono::high_resolution_clock::now();

	__m128 c0 = _mm_load_ps(C.data() + 0);
	__m128 c1 = _mm_load_ps(C.data() + 4);
	__m128 c2 = _mm_load_ps(C.data() + 8);
	__m128 c3 = _mm_load_ps(C.data() + 12);


	for (int iter = 0; iter < ITER; iter++) {
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
	}
	_mm_store_ps(C.data() + 0, c0);
	_mm_store_ps(C.data() + 4, c1);
	_mm_store_ps(C.data() + 8, c2);
	_mm_store_ps(C.data() + 12, c3);

	auto stop = std::chrono::high_resolution_clock::now();
	double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
	unsigned long long numOps = ITER * 4 * 4 * 4;
	double throughput = numOps / 1.0E9 / elapsedSec;
	std::cout << "Elapsed Time: " << elapsedSec << std::endl;
	std::cout << "Throughput: " << throughput << " GFLOPs"<<std::endl;
	std::cout << " " << std::endl;
}

void Do8x8AVX() {
	static const int width = 8;
	static const int height = 8;
	std::vector<float> A(width * height);
	std::vector<float> B(width * height);
	std::vector<float> C(width * height);
	
	std::fill(A.begin(), A.end(), 1.0f);
	std::fill(B.begin(), B.end(), 1.0f);
	std::fill(C.begin(), C.end(), 0.0f);

	static const int ITER = 1024 * 1024;

	auto start = std::chrono::high_resolution_clock::now();

	__m256 c0 = _mm256_load_ps(C.data() + 0);
	__m256 c1 = _mm256_load_ps(C.data() + 8);
	__m256 c2 = _mm256_load_ps(C.data() + 16);
	__m256 c3 = _mm256_load_ps(C.data() + 24);

	__m256 c4 = _mm256_load_ps(C.data() + 32);
	__m256 c5 = _mm256_load_ps(C.data() + 40);
	__m256 c6 = _mm256_load_ps(C.data() + 48);
	__m256 c7 = _mm256_load_ps(C.data() + 56);

	for (int iter = 0; iter < ITER; iter++) {
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
	}


	_mm256_storeu_ps(C.data() + 0,  c0);
	_mm256_storeu_ps(C.data() + 8,  c1);
	_mm256_storeu_ps(C.data() + 16, c2);
	_mm256_storeu_ps(C.data() + 24, c3);

	_mm256_storeu_ps(C.data() + 32, c0);
	_mm256_storeu_ps(C.data() + 40, c1);
	_mm256_storeu_ps(C.data() + 48, c2);
	_mm256_storeu_ps(C.data() + 56, c3);


	auto stop = std::chrono::high_resolution_clock::now();
	double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
	unsigned long long numOps = ITER * 8 * 8 * 8;
	double throughput = numOps / 1.0E9 / elapsedSec;
	std::cout << "Elapsed Time: " << elapsedSec << std::endl;
	std::cout << "Throughput: " << throughput << " GFLOPs" << std::endl;
	std::cout << " " << std::endl;
}

int main() {
	Do4x4SSE();
	Do8x8AVX();

}
