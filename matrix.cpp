#include "matrix.h"


// SSE used only a single data type for XMM registers:

//     four 32-bit single-precision floating-point numbers

// SSE2 would later expand the usage of the XMM registers to include:

//     two 64-bit double-precision floating-point numbers or
//     two 64-bit integers or
//     four 32-bit integers or
//     eight 16-bit short integers or
//     sixteen 8-bit bytes or characters.

// Multiply two matrices using only manual multiply accumulate
// SSE 1.0 has support for multiplying four single precision
// floats within one of its XMM Registers.  Because of this,
// it is supported as a template specialization.
// see: https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/
// template <>
matrix<float> * matmul_cpu_sse(matrix<float> * m1, matrix<float> * m2) {
  // Change this to double
  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<float>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m128 sum = _mm_setzero_ps();
      __m128 m1_row_seg;
      __m128 m2_col_seg;
      float buf[4];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 4); k += 4) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm_loadu_ps(&(m1->_elements->at(i)[k]));
        m2_col_seg = _mm_loadu_ps(&(m2->_elements_col_maj->at(j)[k]));
        sum = _mm_add_ps(sum, _mm_mul_ps(m1_row_seg, m2_col_seg));
      }
      _mm_storeu_ps(buf, sum);
      acc = buf[0] + buf[1] + buf[2] + buf[3];

      unsigned int simd_remainder = m1->cols % 4;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      // res->set(i, j, acc);
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}

matrix<double> * matmul_cpu_sse(matrix<double> * m1, matrix<double> * m2) {

  double acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<double>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m128d sum = _mm_setzero_pd();
      __m128d m1_row_seg;
      __m128d m2_col_seg;
      double buf[2];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 2); k += 2) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm_loadu_pd(&(m1->_elements->at(i)[k]));
        m2_col_seg = _mm_loadu_pd(&(m2->_elements_col_maj->at(j)[k]));
        sum = _mm_add_pd(sum, _mm_mul_pd(m1_row_seg, m2_col_seg));
      }
      _mm_storeu_pd(buf, sum);
      acc = buf[0] + buf[1];

      unsigned int simd_remainder = m1->cols % 2;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      // res->set(i, j, acc);
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}

matrix<uint32_t> * matmul_cpu_sse(matrix<uint32_t> * m1, matrix<uint32_t> * m2) {
  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<uint32_t>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m128i sum = _mm_setzero_si128();
      __m128i m1_row_seg;
      __m128i m2_col_seg;
      uint32_t buf[4];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 4); k += 4) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm_loadu_si128((__m128i_u *)(&(m1->_elements->at(i)[k])));
        m2_col_seg = _mm_loadu_si128((__m128i_u *)(&(m2->_elements_col_maj->at(j)[k])));
        sum = _mm_add_epi32(sum, _mm_mullo_epi32(m1_row_seg, m2_col_seg));
      }
      _mm_storeu_si128((__m128i_u *) buf, sum);
      acc = buf[0] + buf[1] + buf[2] + buf[3];

      unsigned int simd_remainder = m1->cols % 4;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      // res->set(i, j, acc);
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}

matrix<uint16_t> * matmul_cpu_sse(matrix<uint16_t> * m1, matrix<uint16_t> * m2) {

  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<uint16_t>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m128i sum = _mm_setzero_si128();
      __m128i m1_row_seg;
      __m128i m2_col_seg;
      float buf[4];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 8); k += 8) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm_loadu_si128((__m128i_u *)(&(m1->_elements->at(i)[k])));
        m2_col_seg = _mm_loadu_si128((__m128i_u *)(&m2->_elements_col_maj->at(j)[k]));
        sum = _mm_add_epi16(sum, _mm_mullo_epi16(m1_row_seg, m2_col_seg));
      }
      _mm_storeu_si128((__m128i_u *) buf, sum);
      acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

      unsigned int simd_remainder = m1->cols % 8;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      // res->set(i, j, acc);
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}


matrix<float> * matmul_cpu_avx(matrix<float> * m1, matrix<float> * m2) {
  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<float>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m256 sum = _mm256_setzero_ps();
      __m256 m1_row_seg;
      __m256 m2_col_seg;
      float buf[8];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 8); k += 8) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm256_loadu_ps(&(m1->_elements->at(i)[k]));
        m2_col_seg = _mm256_loadu_ps(&(m2->_elements_col_maj->at(j)[k]));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(m1_row_seg, m2_col_seg));
      }
      _mm256_storeu_ps(buf, sum);
      acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

      unsigned int simd_remainder = m1->cols % 8;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      // res->set(i, j, acc);
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}

matrix<float> * matmul_cpu_avxfma(matrix<float> * m1, matrix<float> * m2) {
  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an MxP matrix
  const auto res = new matrix<float>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;
      __m256 sum = _mm256_setzero_ps();
      __m256 m1_row_seg;
      __m256 m2_col_seg;
      float buf[8];
      // do the dot product of m1 row with m2 column
      for (int k = 0; k < ceil(m1->cols / 8); k += 8) {
        // acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        m1_row_seg = _mm256_loadu_ps(&(m1->_elements->at(i)[k]));
        m2_col_seg = _mm256_loadu_ps(&(m2->_elements_col_maj->at(j)[k]));
        sum = _mm256_fmadd_ps(m1_row_seg, m2_col_seg, sum);
      }
      _mm256_storeu_ps(buf, sum);
      acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

      unsigned int simd_remainder = m1->cols % 8;
      if (simd_remainder != 0) {
        for (int k = m1->cols - simd_remainder; k < m1->cols; k++) {
            acc += m1->_elements->at(i)[k] * m2->_elements->at(j)[k];
        }
      }
      res->_elements->at(i)[j] = acc;
    }
  }
  return res;
}