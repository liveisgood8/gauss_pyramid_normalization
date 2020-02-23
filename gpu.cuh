#ifndef GPU
#define GPU

__global__ void find_array_max(int *array, size_t size, int *max, int *mutex);
__global__ void divide_array_elements(int *array, size_t size, int divider);

#endif /* GPU */
