#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cassert>
#include <type_traits>
#include <iostream>

template <class T>
class matrix {

  public:
    unsigned int rows; 
    unsigned int cols;

    matrix (unsigned int nRows, unsigned int nCols);
    ~matrix();

    T get(unsigned int row, unsigned int col) const;
    void set(unsigned int row, unsigned int col, T val);

    void transpose();
    void fill_zeroes();
    void apply_identity();

    void print();

    template <class K>
    friend matrix<K> * matmul_cpu_cache_block(matrix<K> * m1, matrix<K> * m2, size_t block_size);

  private:
    std::vector<T> * _internal_getRow(unsigned int row);
    std::vector<T> * _internal_getCol(unsigned int col);
    void _transpose(std::vector<std::vector<T>> * dest, std::vector<std::vector<T>> * src);
    void _internal_populate_col_maj();

    unsigned int _nElementBytes;
    std::vector<std::vector<T>> * _elements;
    std::vector<std::vector<T>> * _elements_col_maj;
};

template <class T>
matrix<T>::matrix(unsigned int nRows, unsigned int nCols) {
  this->_elements = new std::vector<std::vector<T>>;
  this->_elements_col_maj = new std::vector<std::vector<T>>;
  this->rows = nRows;
  this->cols = nCols;
  _elements->resize(nRows);
  _elements_col_maj->resize(nCols);
  for (int i = 0; i < nRows; i++) (*_elements)[i].resize(nCols);
  for (int i = 0; i < nCols; i++) (*_elements_col_maj)[i].resize(nRows);
}

template <class T>
matrix<T>::~matrix() {

}

template <class T>
T matrix<T>::get(unsigned int row, unsigned int col) const {
    assert(row < this->rows && col < this->cols);
    return (*_elements)[row][col];
}

template <class T>
void matrix<T>::set(unsigned int row, unsigned int col, T val) {
    assert(row < this->rows && col < this->cols);
    (*_elements)[row][col] = val;
    (*_elements_col_maj)[col][row] = val;
}

template <class T>
void matrix<T>::_internal_populate_col_maj() {
    this->_transpose(this->_elements_col_maj, this->_elements);
}

// Used to pull a row from the _elements variable (row major)
// Does not require that _elements_col_major be up to date
// Will use the regular element storage in _elements
template <class T>
std::vector<T> * matrix<T>::_internal_getRow(unsigned int row) {
  assert(row >= 0 && row < this->rows);
  return &(this->_elements->at(row));
}

// Used to pull a column from the _elements_col_major variable
// Note that the callee must first ensure that _elements_col_major
// is up to date by calling _internal_populate_col_major
template <class T>
std::vector<T> * matrix<T>::_internal_getCol(unsigned int col) {
  assert(col >= 0 && col < this-> cols);
  return &(this->_elements_col_maj->at(cols));
}

// Internal transpose function that returns pointer to the transposed
// elements.  Used by the regular transpose function.
template <class T>
void matrix<T>::_transpose(std::vector<std::vector<T>> * dest, std::vector<std::vector<T>> * src) {
  dest->resize(this->cols);
  for (int i = 0; i < this->cols; i++) (*dest)[i].resize(this->rows);
  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      (*dest)[j][i] = (*src)[i][j];
    }
  }
}


template <class T>
void matrix<T>::fill_zeroes() {
  int orig_rows = this->rows;
  int orig_cols = this->cols;
  for (int i = 0; i < this->rows; i++) {
    (*this->_elements)[i].clear();
    (*this->_elements)[i].resize(this->cols);
  }

  // Ensure that the row sizes and column sizes don't change
  assert(orig_rows == this->rows);
  assert(orig_cols == this->cols);
}

  // A function used to produce an identity matrix.
  // Only valid for arithmetic types (except bool)
  // See https://en.cppreference.com/w/cpp/types/is_arithmetic
  // Additionally, an identity matrix is always square
  template <class T>
  void matrix<T>::apply_identity() {
    assert(std::is_arithmetic_v<T> == true);
    assert((std::is_same<T,bool>::value != true));
    assert(this->rows == this->cols);
    this->fill_zeroes();
    for (int i = 0; i < this->rows; i++) {
      this->set(i, i, 1);
    }
  }

  template <class T>
  void matrix<T>::print() {
    std::cout << *this << std::endl;
  }

template <class T>
void matrix<T>::transpose() {
//   std::vector<std::vector<T>> t;
//     t.resize(this->cols);
//   for (int i = 0; i < this->cols; i++) t[i].resize(this->rows);
//   for (int i = 0; i < this->rows; i++) {
//     for (int j = 0; j < this->cols; j++) {
//       t[j][i] = this->get(i, j);
//     }
//   }
_transpose(this->_elements_col_maj, this->_elements);

  unsigned int tmpCols = this->cols;
  this->cols = this->rows;
  this->rows = tmpCols;

  const auto tmpElem = this->_elements;
  this->_elements = _elements_col_maj;
  this->_elements_col_maj = tmpElem;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const matrix<T>& m1) {
  os << "Matrix [" << m1.rows << ", " << m1.cols  << "]\n";
  for (int i = 0; i < m1.rows; i++) {
    os << "[";
    for (int j = 0; j < m1.cols; j++) {
      os << m1.get(i, j) << ", ";
    }
    os << "]\n";
  }
  os << std::flush;
  return os;
}

// Multiply two matrices using only manual multiply accumulate
// No HW extensions or optimizations are used
template <class T>
matrix<T> * matmul_cpu(matrix<T> * m1, matrix<T> * m2) {
  long long int acc;

  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);
  
  // An MxN * NxP yields an NxP matrix
  const auto res = new matrix<T>(m1->rows, m2->cols);
  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      // For every index in the matrix
      // Compute the new value.  It is the dot product of col from 1
      // and row from 2. Only one for loop required as they are both 
      // the same vector length.
      acc = 0;

      // do the dot product of m1 row with m2 column
      for (int k = 0; k < m1->cols; k++) {
        acc += m1->get(i, k) * m2->get(j, k);
      }
      res->set(i, j, acc);
    }
  }
  return res;
}

// Multiply two matrices using only manual multiply accumulate
// cache optimization is used in this algorithm.  A copy of
// the row-major data is created and transposed and is stored
// internally. No preloading is done, only "blocking."
// See: https://www.youtube.com/watch?v=G92BCtfTwOE
// Note: This function is a friend of matrix - private members are used.
template <class T>
matrix<T> * matmul_cpu_cache_block(matrix<T> * m1, matrix<T> * m2, size_t block_size) {
  // Make sure that matrices match 1's cols to 2's rows
  assert(m1->cols == m2->rows);

  // TODO: Make this work for all block sizes
  // Make sure that the block size evenly divides the shared dimension
  assert(m1->cols % block_size == 0);
  long long int acc = 0;

  // Force the second matrix to generate column major data
  // m2->_internal_populate_col_maj();

  // An MxN * NxP yields an NxP matrix
  const auto res = new matrix<T>(m1->rows, m2->cols);
  for (int row = 0; row < m1->rows; row += block_size) {
    for (int col = 0; col < m2->cols; col += block_size) {

      // This might at first seem not efficient but the gained efficiency comes
      // From the fact that we are better utilizing cache lines since we have
      // both row major and column major copies of the data.
      // In m1, we only use row major data. In m2, we only use column major data.
      for (int rowBlockIndex = 0; rowBlockIndex < block_size; rowBlockIndex++) {
        for (int colBlockIndex = 0; colBlockIndex < block_size; colBlockIndex++) {

          // do the dot product of m1 row with m2 column
          for (int k = 0; k < m1->cols; k++) {
              acc += (*m1->_elements)[row + rowBlockIndex][k] * (*m2->_elements)[col + colBlockIndex][k];
          }
          res->set(row + rowBlockIndex, col + colBlockIndex, acc);
          acc = 0;
        }
      }
    }
  }
  return res;
}



#endif //MATRIX_H