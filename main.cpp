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
  m4 = matmul_cpu_cache_block(&m2, &m3, 1);
  std::cout << *m4 << std::endl;
  std::cout << "Matrix test successful" << std::endl;
}

void stress_test() {
  matrix<unsigned int> m1(250, 250);
  matrix<unsigned int> m2(250, 250);
  matrix<unsigned int> * m3;

  unsigned int num_trials = 10;
  unsigned int cumulative_time = 0;
  double avg_time = 0;

  for (int i = 0; i < num_trials; i++) {
    auto before = std::chrono::high_resolution_clock::now();
    m3 = matmul_cpu_cache_block(&m1, &m2, 1);
    auto after = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
    cumulative_time += duration.count();
  }

  avg_time = cumulative_time / num_trials;
  std::cout << "With Cache Blocking: " << avg_time << " microseconds" << std::endl;
  cumulative_time = 0;

  for (int i = 0; i < num_trials; i++) {
    auto before = std::chrono::high_resolution_clock::now();
    m3 = matmul_cpu(&m1, &m2);
    auto after = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
    cumulative_time += duration.count();
  }

  avg_time = cumulative_time / num_trials;
  std::cout << "Without Cache Blocking: " << avg_time << " microseconds" << std::endl;
}

int main(int argc, char ** argv) {
    // test_matrix();
    // en_sse = sse_enabled();
    // en_avx = avx_enabled();
    // en_avx2 = avx2_enabled();
    // std::cout << "SSE:  " << en_sse  << std::endl;
    // std::cout << "AVX:  " << en_avx << std::endl;
    // std::cout << "AVX2: " << en_avx2 << std::endl;

    stress_test();

}