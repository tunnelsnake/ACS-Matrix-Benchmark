#include <cassert>
#include <iostream>
#include <chrono>
#include "matrix.h"
#include "ssecheck.h"

int en_sse = 0;
int en_avx = 0;
int en_avx2 = 0;


void test_matrix() {  
  matrix<unsigned int> m1(10, 10);

  // Test uninitialized read
  assert(m1.get(0, 0) == 0);

  // Test set and get
  m1.set(1, 1, 8);
  m1.set(9, 7, 2);
  assert(m1.get(1, 1) == 8);
  assert(m1.get(9, 7) == 2);

  // Test printing
  std::cout << m1 << std::endl;

  // Test fill_zeroes
  m1.fill_zeroes();
  assert(m1.get(1, 1) == 0);
  assert(m1.get(9, 7) == 0);

  // Test printing
  std::cout << m1 << std::endl;

  // Test transposition
  matrix<unsigned int> m2(5, 10);
  m2.set(1, 2, 32);
  m2.set(4, 3, 64);
  std::cout << m2 << std::endl;
  m2.transpose();
  std::cout << m2 << std::endl;
  assert(m2.cols == 5);
  assert(m2.rows == 10);
  assert(m2.get(2, 1) == 32);
  assert(m2.get(3, 4) == 64);

  // Test Identity
  matrix<unsigned int> m3(5, 5);
  m3.apply_identity();
  std::cout << m3 << std::endl;
  m3.print();

  // Test CPU multiplication
  matrix<unsigned int> * m4;
  m4 = matmul_cpu(&m2, &m3);
  std::cout << m4 << std::endl;
  std::cout << "Matrix test successful" << std::endl;
}

int main(int argc, char ** argv) {
    test_matrix();
    en_sse = sse_enabled();
    en_avx = avx_enabled();
    en_avx2 = avx2_enabled();
    std::cout << "SSE:  " << en_sse  << std::endl;
    std::cout << "AVX:  " << en_avx << std::endl;
    std::cout << "AVX2: " << en_avx2 << std::endl;

}