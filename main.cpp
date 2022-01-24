#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
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
  m4 = matmul_cpu_cache_block(&m2, &m3, 3);
  std::cout << *m4 << std::endl;
  std::cout << "Matrix test successful" << std::endl;
}

void large_matrix_test_float() {
    int large_matrix_size = 10000;
    matrix<float> m1(10000, 10000);
    matrix<float> m2(10000, 10000);
    std::cout << "Starting Large Floating Point Matrix Test. Size: " << large_matrix_size << " x " << large_matrix_size << std::endl;
    std::cout << "Testing SSE, AVX, and AVX2" << std::endl;

    auto before = std::chrono::high_resolution_clock::now();
    auto m3 = matmul_cpu_sse(&m1, &m2);
    auto after = std::chrono::high_resolution_clock::now();
    delete m3;
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(after - before);
    std::cout << "SSE: " << duration.count() << std::endl;

    before = std::chrono::high_resolution_clock::now();
    m3 = matmul_cpu_avx(&m1, &m2);
    after = std::chrono::high_resolution_clock::now();
    delete m3;
    duration = std::chrono::duration_cast<std::chrono::seconds>(after - before);
    std::cout << "AVX: " << duration.count() << std::endl;

    before = std::chrono::high_resolution_clock::now();
    m3 = matmul_cpu_avxfma(&m1, &m2);
    after = std::chrono::high_resolution_clock::now();
    delete m3;
    duration = std::chrono::duration_cast<std::chrono::seconds>(after - before);
    std::cout << "AVX FMA: " << duration.count() << std::endl;
}

void large_matrix_test_fixed() {
    int large_matrix_size = 10000;
    matrix<uint32_t> m1(10000, 10000);
    matrix<uint32_t> m2(10000, 10000);
    std::cout << "Starting Large Fixed Point Matrix Test. Size: " << large_matrix_size << " x " << large_matrix_size << std::endl;
    std::cout << "Testing SSE (16 Bit), SSE (32 Bit)" << std::endl;

    auto before = std::chrono::high_resolution_clock::now();
    auto m3 = matmul_cpu_sse(&m1, &m2);
    auto after = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(after - before);
    delete m3;
    std::cout << "SSE (32 Bit): " << duration.count() << std::endl;

    matrix<uint16_t> m4(10000, 10000);
    matrix<uint16_t> m5(10000, 10000);
    before = std::chrono::high_resolution_clock::now();
    auto m6 = matmul_cpu_sse(&m4, &m5);
    after = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(after - before);
    delete m6;
    std::cout << "SSE (16 Bit): " << duration.count() << std::endl;

}


void fixed_point_stress_test() {
  std::ofstream f("fixed_data.txt", std::ofstream::out);
  f << "matsize,method,trials,time,"<< std::endl;

  for (int i = 10; i < 650; i++) {
    std::cout << "Trial #" << i << std::endl;
    matrix<uint16_t> m1_16(i, i);
    matrix<uint16_t> m2_16(i, i);

    matrix<uint32_t> m1_32(i, i);
    matrix<uint32_t> m2_32(i, i);

    unsigned int num_trials = 100;
    unsigned int cumulative_time = 0;
    double avg_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu(&m1_16, &m2_16);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }
    avg_time = cumulative_time / num_trials;
    // std::cout << "Vanilla: " << avg_time << " microseconds" << std::endl;
    f << i << ",vanilla16," << num_trials << "," << avg_time << "," << std::endl;
    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu(&m1_32, &m2_32);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }
    avg_time = cumulative_time / num_trials;
    // std::cout << "Vanilla: " << avg_time << " microseconds" << std::endl;
    f << i << ",vanilla32," << num_trials << "," << avg_time << "," << std::endl;
    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_cache_block(&m1_16, &m2_16, 100);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "Cache Blocking: " << avg_time << " microseconds" << std::endl;
    f << i << ",cacheblock16," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_cache_block(&m1_32, &m2_32, 100);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "Cache Blocking: " << avg_time << " microseconds" << std::endl;
    f << i << ",cacheblock32," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_sse(&m1_16, &m2_16);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "With SSE: " << avg_time << " microseconds" << std::endl;
    f << i << ",sse16," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_sse(&m1_32, &m2_32);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "With SSE: " << avg_time << " microseconds" << std::endl;
    f << i << ",sse32," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    f.flush();
  }
  f.close();
}


void floating_point_stress_test() {
  std::ofstream f("float_data.txt", std::ofstream::out);
  f << "matsize,method,trials,time,"<< std::endl;
  for (int i = 10; i < 650; i++) {
    std::cout << "Trial #" << i << std::endl;
    matrix<float> m1(i, i);
    matrix<float> m2(i, i);
    // matrix<float> * m3;

    unsigned int num_trials = 100;
    unsigned int cumulative_time = 0;
    double avg_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu(&m1, &m2);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }
    avg_time = cumulative_time / num_trials;
    std::cout << "Vanilla: " << avg_time << " microseconds" << std::endl;
    f << i << ",vanilla," << num_trials << "," << avg_time << "," << std::endl;
    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_cache_block(&m1, &m2, 100);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    std::cout << "Cache Blocking: " << avg_time << " microseconds" << std::endl;
    f << i << ",cacheblock," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_sse(&m1, &m2);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    std::cout << "With SSE: " << avg_time << " microseconds" << std::endl;
    f << i << ",sse," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_avx(&m1, &m2);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "With AVX: " << avg_time << " microseconds" << std::endl;
    f << i << ",avx," << num_trials << "," << avg_time << "," << std::endl;

    cumulative_time = 0;

    for (int i = 0; i < num_trials; i++) {
      auto before = std::chrono::high_resolution_clock::now();
      auto m3 = matmul_cpu_avxfma(&m1, &m2);
      auto after = std::chrono::high_resolution_clock::now();
      delete m3;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
      cumulative_time += duration.count();
    }

    avg_time = cumulative_time / num_trials;
    // std::cout << "With AVX Multiply Accumulate: " << avg_time << " microseconds" << std::endl;
    f << i << ",avxmla," << num_trials << "," << avg_time << "," << std::endl;
    cumulative_time = 0;

    f.flush();
  }
  f.close();
}

int main(int argc, char ** argv) {
    test_matrix();
    en_sse = sse_enabled();
    en_avx = avx_enabled();
    en_avx2 = avx2_enabled();
    std::cout << "SSE:  " << en_sse  << std::endl;
    std::cout << "AVX:  " << en_avx << std::endl;
    std::cout << "AVX2: " << en_avx2 << std::endl;

    large_matrix_test_float();
    large_matrix_test_fixed();
    // floating_point_stress_test();
    // fixed_point_stress_test();

}