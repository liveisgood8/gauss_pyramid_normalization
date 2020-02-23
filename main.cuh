#ifndef MAIN
#define MAIN

#include <stddef.h>
#include <stdio.h>

extern FILE *output_file;

int* make_host_array(size_t rows, size_t columns, int* real_max = nullptr);

int** make_gauss_pyramid(size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids, int* real_max = nullptr);
void free_gauss_pyramid(int **arrays, size_t number_of_pyramids);

int* copy_host_array_to_dev(const int *host_arr, size_t rows, size_t columns);
void copy_dev_max_to_host(const int *dev_max, int *host_max);

void print_host_array(const int *arr, size_t rows, size_t columns);
void print_host_gauss_piramid(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids);

int find_array_max_on_dev(int *host_arr, size_t rows, size_t columns);
void divide_array_on_dev(int *array, size_t rows, size_t columns, int divider);
int find_gauss_pyramid_max_on_dev(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids);
void divide_gauss_pyramid_on_dev(int **arrays, size_t base_rows, size_t base_columns, 
	size_t number_of_pyramids, int divider);
  
#endif /* MAIN */
