#include "gpu.cuh"

#include <stdio.h>

__device__ int int_max(int a, int b) {
	return a > b ? a : b;
}

__global__ void find_array_max(int *array, size_t size, int *max, int *mutex) {
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	size_t step = gridDim.x * blockDim.x;
	
	// Разделяемая между потоками память
	// для хранения максимумов, найденных в каждом блоке
	__shared__ int cache[256];
	
	size_t index_offset = 0;
	int temp = 0;

	while(index + index_offset < size){
		temp = int_max(temp, array[index + index_offset]);
    index_offset += step;
	}

	cache[threadIdx.x] = temp;
	__syncthreads();

	// Свертка найденных максимумов
	for (size_t i = blockDim.x / 2; i > 0; i >>= 1) {
		if (threadIdx.x < i) {
			cache[threadIdx.x] = int_max(cache[threadIdx.x], cache[threadIdx.x + i]);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		// Запись в глобальный максимум с блокировкой мьютекса
		while(atomicCAS(mutex, 0, 1) != 0);
		*max = int_max(*max, cache[0]);
		atomicExch(mutex, 0);
	}
}

__global__ void divide_array_elements(int *array, size_t size, int divider) {
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	size_t step = gridDim.x * blockDim.x;
	size_t index_offset = 0;

	while(index + index_offset < size){
		array[index + index_offset] /= divider;
		index_offset += step;
	}
}