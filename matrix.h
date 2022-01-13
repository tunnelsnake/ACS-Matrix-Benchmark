#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cassert>
#include <type_traits>

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

  private:
    unsigned int _nElementBytes;
    std::vector<std::vector<T>> _elements;
};

template <class T>
matrix<T>::matrix(unsigned int nRows, unsigned int nCols) {
  this->rows = nRows;
  this->cols = nCols;
  _elements.resize(nRows);
  for (int i = 0; i < nRows; i++) _elements[i].resize(nCols);
}

template <class T>
matrix<T>::~matrix() {

}

template <class T>
T matrix<T>::get(unsigned int row, unsigned int col) const {
    assert(row < this->rows && col < this->cols);
    return _elements[row][col];
}

template <class T>
void matrix<T>::set(unsigned int row, unsigned int col, T val) {
    assert(row < this->rows && col < this->cols);
    _elements[row][col] = val;
}

template <class T>
void matrix<T>::fill_zeroes() {
  int orig_rows = this->rows;
  int orig_cols = this->cols;
  for (int i = 0; i < this->rows; i++) {
    this->_elements[i].clear();
    this->_elements[i].resize(this->cols);
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
  std::vector<std::vector<T>> t;
    t.resize(this->cols);
  for (int i = 0; i < this->cols; i++) t[i].resize(this->rows);

  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      t[j][i] = this->get(i, j);
    }
  }

  unsigned int tmp = this->cols;
  this->cols = this->rows;
  this->rows = tmp;
  this->_elements = t;
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


#endif //MATRIX_H