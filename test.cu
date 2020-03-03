#include "main.cuh"

#include <stdio.h>
#include <math.h>

#define TIMED_FUNCTION(func) \
  { \
    clock_t start, end;  \
    start = clock(); \
    func; \
    end = clock(); \
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC); \
    fprintf(output_file, #func " finished in %lf seconds\n", time_taken); \
  }

bool isMatrixOutputEnabled = false;

void test_gauss_pyramid_max() {
  fprintf(output_file, "* test_gauss_pyramid_max calculated on host and device\n");

  const size_t base_rows = (size_t)pow(2, 8);
  const size_t base_columns = (size_t)pow(2, 8);
  const size_t number_of_pyramids = 6;

  for (size_t i = 0; i < 5; i++) {
    int real_pyramid_max = 0;

    int **arr_pyramid = make_gauss_pyramid(base_rows, base_columns, number_of_pyramids, &real_pyramid_max);
    int pyramid_max = find_gauss_pyramid_max_on_dev(arr_pyramid, base_rows, base_columns, 
      number_of_pyramids);
    
    free_gauss_pyramid(arr_pyramid, number_of_pyramids);

    if (real_pyramid_max != pyramid_max) {
      fprintf(output_file, "!!! test_gauss_pyramid_max failed\n");
      break;
    }
  }
}

void test_alg(size_t base_rows, size_t base_columns, 
  size_t number_of_pyramids, const char *test_name) {
    fprintf(output_file, "* %s\n", test_name);

  int **arr_pyramid = make_gauss_pyramid(base_rows, base_columns, number_of_pyramids);
  if (isMatrixOutputEnabled) {
    fprintf(output_file, "--- before ---\n");
    print_host_gauss_piramid(arr_pyramid, base_rows, base_columns, number_of_pyramids);
  }

	int pyramid_max = find_gauss_pyramid_max_on_dev(arr_pyramid, base_rows, base_columns, 
    number_of_pyramids);
  
  if (isMatrixOutputEnabled) {
    fprintf(output_file, "max element: %d\n", pyramid_max);
  }
    
  divide_gauss_pyramid_on_dev(arr_pyramid, base_rows, base_columns, 
    number_of_pyramids, pyramid_max);
  if (isMatrixOutputEnabled) {
    fprintf(output_file, "--- after ---\n");
    print_host_gauss_piramid(arr_pyramid, base_rows, base_columns, number_of_pyramids);
  }

  free_gauss_pyramid(arr_pyramid, number_of_pyramids);
  // fprintf(stderr,"GPU last error: %d\n", cudaGetLastError());
}

void test_1() {
  test_alg(
    (size_t)pow(2, 11),
    (size_t)pow(2, 11),
    6,
    "test alg with 2^11 rows and 2^11 columns for 6 level gauss pyramid"
  );
}

void test_2() {
  test_alg(
    (size_t)pow(2, 12),
    (size_t)pow(2, 12),
    6,
    "test alg with 2^12 rows and 2^12 columns for 6 level gauss pyramid"
  );
}

void test_3() {
  test_alg(
    (size_t)pow(2, 13),
    (size_t)pow(2, 13),
    6,
    "test alg with 2^13 rows and 2^13 columns for 6 level gauss pyramid"
  );
}


void test_4() {
  test_alg(
    (size_t)pow(2, 14),
    (size_t)pow(2, 14),
    6,
    "test alg with 2^14 rows and 2^14 columns for 6 level gauss pyramid"
  );
}

void run_tests() {
  srand(time(0));
  test_gauss_pyramid_max();
  TIMED_FUNCTION(test_1());
  TIMED_FUNCTION(test_2());
  TIMED_FUNCTION(test_3());
}
