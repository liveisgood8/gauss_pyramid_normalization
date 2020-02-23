#include "main.cuh"

#include <time.h>
#include <cuda_runtime.h>

#include "gpu.cuh"
#include "test.cuh"

#define CUDA_BLOCKS 256
#define CUDA_THREADS_PER_BLOCK 256

FILE *output_file = stdout;

int* make_host_array(size_t rows, size_t columns, int* real_max) {
	const size_t size = rows * columns;
	int *arr = (int*)malloc(size * sizeof(int));

	int tmp_realmax = 0;
	for (size_t i = 0; i < size; i++) {
		int value = rand() % 100;
		if (value > tmp_realmax) {
			tmp_realmax = value;
		}
		arr[i] = value;
	}

	if (real_max) {
		*real_max = tmp_realmax;
	}

	return arr;
}

int** make_gauss_pyramid(size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids, int* real_max) {
	int **arr_pyramid = (int**)malloc(number_of_pyramids * sizeof(int*));

	int tmp_gauss_max = 0;
	for (size_t i = 0; i < number_of_pyramids; i++) {
		int arr_max = 0;
		arr_pyramid[i] = make_host_array(base_rows, base_columns, &arr_max);
		
		if (arr_max > tmp_gauss_max) {
			tmp_gauss_max = arr_max;
		}

		base_rows /= 2;
		base_columns /= 2;
	}

	if (real_max) {
		*real_max = tmp_gauss_max;
	}

	return arr_pyramid;
}

void free_gauss_pyramid(int **arrays, size_t number_of_pyramids) {
	for (size_t i = 0; i < number_of_pyramids; i++) {
		free(arrays[i]);
	}
	free(arrays);
}

int* copy_host_array_to_dev(const int *host_arr, size_t rows, size_t columns) {
	int *dev_arr;
	const size_t size = rows * columns;

	cudaMalloc((void**)&dev_arr, size * sizeof(int));
	cudaMemcpy(dev_arr, host_arr, size * sizeof(int), cudaMemcpyHostToDevice);

	return dev_arr;
}

void copy_dev_max_to_host(const int *dev_max, int *host_max) {
	cudaMemcpy(host_max, dev_max, sizeof(int), cudaMemcpyDeviceToHost);
}

void print_host_array(const int *arr, size_t rows, size_t columns) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < columns; j++) {
			fprintf(output_file, "%d,", *(arr + i * columns + j));
		}
		fprintf(output_file, "\n");
	}
}

void print_host_gauss_piramid(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids) {
	for (size_t i = 0; i < number_of_pyramids; i++) {
		print_host_array(arrays[i], base_rows, base_columns);
		base_rows /= 2;
		base_columns /= 2;
		fprintf(output_file, "---\n");
	}
}

int find_array_max_on_dev(int *host_arr, size_t rows, size_t columns) {
	int *host_max = (int*)malloc(sizeof(int));

	int *dev_max;
	int *dev_mutex;
	int *dev_arr = copy_host_array_to_dev(host_arr, rows, columns);
	cudaMalloc((void**)&dev_max, sizeof(int));
	cudaMalloc((void**)&dev_mutex, sizeof(int));
	cudaMemset(dev_max, 0, sizeof(int));
	cudaMemset(dev_mutex, 0, sizeof(int));

	dim3 blocks = min(rows * columns, (size_t)CUDA_BLOCKS);
	dim3 threads_per_block = CUDA_THREADS_PER_BLOCK;
	find_array_max<<<blocks, threads_per_block>>>(dev_arr, rows * columns, 
		dev_max, dev_mutex);

	copy_dev_max_to_host(dev_max, host_max);
	
	int host_max_copy = *host_max;

	free(host_max);
	cudaFree(dev_arr);
	cudaFree(dev_max);
	cudaFree(dev_mutex);

	return host_max_copy;
}

void divide_array_on_dev(int *array, size_t rows, size_t columns, int divider) {
	int *dev_arr = copy_host_array_to_dev(array, rows, columns);

	dim3 blocks = min(rows * columns, (size_t)CUDA_BLOCKS);
	dim3 threads_per_block = CUDA_THREADS_PER_BLOCK;
	divide_array_elements<<<blocks, threads_per_block>>>(dev_arr, rows * columns, divider);

	cudaMemcpy(array, dev_arr, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_arr);
}

int find_gauss_pyramid_max_on_dev(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids) {
	int max = 0;
	for (size_t i = 0; i < number_of_pyramids; i++) {
		int arr_max = find_array_max_on_dev(arrays[i], base_rows, base_columns);
		if (arr_max > max) max = arr_max;
		base_rows /= 2;
		base_columns /= 2;
	}
	return max;
}

void divide_gauss_pyramid_on_dev(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids, int divider) {
	for (size_t i = 0; i < number_of_pyramids; i++) {
		divide_array_on_dev(arrays[i], base_rows, base_columns, divider);
		base_rows /= 2;
		base_columns /= 2;
	}
}

void open_report_file() {
	if ((output_file = fopen("output.txt", "w")) == 0) {
		output_file = stdout;
		fprintf(output_file, "cannot open output file, printf to stdout");
	}
}

int main() {
	open_report_file();
	run_tests();
}
