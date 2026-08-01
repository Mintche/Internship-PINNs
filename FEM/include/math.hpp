#ifndef MATH_HPP
#define MATH_HPP
#include <iostream>
#include <map>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <complex>
typedef std::complex<double> complexe;
//---------------------------------------------------------------------------
//  Helper for conjugation (generic T vs complex<T>)
//---------------------------------------------------------------------------
template<typename T> T conjugate(const T& v) { return v; }
template<typename T> std::complex<T> conjugate(const std::complex<T>& v) { return std::conj(v); }

//---------------------------------------------------------------------------
//  Operations on vector<T>
//---------------------------------------------------------------------------
template<typename T> std::vector<T> operator+(const std::vector<T>& u, const std::vector<T>& v)
{
    std::vector<T> w(u);
    auto itv = v.begin();
    for(auto itw=w.begin(); itw!=w.end(); ++itw, ++itv) *itw+=*itv;
    return w;
}
template<typename T> std::vector<T> operator-(const std::vector<T>& u, const std::vector<T>& v)
{
    std::vector<T> w(u);
    auto itv = v.begin();
    for(auto itw=w.begin(); itw!=w.end(); ++itw, ++itv) *itw-=*itv;
    return w;
}
template<typename T> std::vector<T> operator*(const std::vector<T>& u, const T& s)
{
    std::vector<T> w(u);
    for(auto& wi : w) wi*=s;
    return w;
}
template<typename T> std::vector<T> operator*(const T& s, const std::vector<T>& u)
{
    std::vector<T> w(u);
    for(auto& wi : w) wi*=s;
    return w;
}
template<typename T> std::vector<T> operator/(const std::vector<T>& u, const T& s)
{
    std::vector<T> w(u);
    for(auto& wi : w) wi/=s;
    return w;
}

template<typename T> T operator|(const std::vector<T>& u, const std::vector<T>& v)
{
    T s= T(0);
    auto itv = v.begin();
    for(auto itu=u.begin(); itu!=u.end(); ++itu, ++itv) s+=*itu * *itv;
    return s;
}

template<typename T> double normesup(const std::vector<T>& u)
{
    double normesup= 0.;
    for(auto const& it : u){
        if (std::abs(it) > normesup) normesup = std::abs(it);
    }
    return normesup;
}

template<>
inline complexe operator|(const std::vector<complexe>& u, const std::vector<complexe>& v)
{
    complexe s= complexe(0);
    auto itv = v.begin();
    for(auto itu=u.begin(); itu!=u.end(); ++itu, ++itv) s+=std::conj(*itu) * (*itv);
    return s;
}

template<typename T> 
double norm(const std::vector<T>&u)
{
    return std::sqrt(std::abs(u|u));
}
template<typename T> std::ostream& operator<<(std::ostream& os,const std::vector<T>& v)
{
  os<<"(";
  auto itv=v.begin();
  for(;itv!=v.end()-1;++itv) os<<(*itv)<<",";
  os<<(*itv)<<")";
  return os;
}

//---------------------------------------------------------------------------
//     classe FullMatrix
//---------------------------------------------------------------------------

template <typename T>
class FullMatrix {

protected:

    int n_rows;
    int n_cols;
    std::vector<T> coefs;
    bool is_ldlt_factorized = false;
    
public:
    

    FullMatrix(int n, int m) : n_rows(n), n_cols(m), coefs(static_cast<std::size_t>(n) * m, T(0)) {}

    // Access

    T& operator()(int i, int j) { return coefs[i * n_cols + j]; }
    const T& operator()(int i, int j) const { return coefs[i * n_cols + j]; }

    // Operators
    
    std::vector<T> operator*(const std::vector<T>& x) const {
        // Check the vector size.
        if (x.size() != static_cast<std::size_t>(n_cols)){
            throw std::invalid_argument("Error: The matrix column count must equal the vector size.");
        }
        std::vector<T> res(this->n_rows, T(0));
        for(int i = 0; i < n_rows; ++i) {
            for(int j = 0; j < n_cols; ++j) {
                res[i] += (*this)(i, j) * x[j];
            }
        }
        return res;
    }

    FullMatrix<T> operator*(const FullMatrix<T>& M) const {
        if (n_cols != M.n_rows) {
            throw std::invalid_argument("Error: Incompatible dimensions for matrix multiplication.");
        }
        FullMatrix<T> res(n_rows, M.n_cols);
        for(int i = 0; i < n_rows; ++i) {
            for(int j = 0; j < M.n_cols; ++j) {
                T sum = T(0);
                for(int k = 0; k < n_cols; ++k) {
                    sum += (*this)(i, k) * M(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
    }
    

    void operator+=(const FullMatrix<T>& M){
        // Check the dimensions.
        if (n_rows != M.n_rows || n_cols != M.n_cols){
            throw std::invalid_argument("Error: Cannot add matrices with different dimensions.");
        }
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i] += M.coefs[i];
        }
    }

    void operator-=(const FullMatrix<T>& M){
        // Check the dimensions.
        if (n_rows != M.n_rows || n_cols != M.n_cols){
            throw std::invalid_argument("Error: Cannot subtract matrices with different dimensions.");
        }
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i] -= M.coefs[i];
        }
    }

    void operator*=(const T s){
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i]*=s;
        }
    }

    // Fill with a value.
    void fill(T val) {
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i] = val;
        }
    }

    // Solve with Gaussian pivoting.

    // In-place LDL* factorization for Hermitian matrices.
    // The lower triangle stores L (without its unit diagonal),
    // and the diagonal stores D.

    void factorize() {
        if (is_ldlt_factorized) return;
        if (n_rows != n_cols) {
            throw std::logic_error("Error: LDLT factorization requires a square matrix.");
        }
        
        int n = n_rows;

        for (int j = 0; j < n; ++j) {
            // Compute D_jj.
            T d_val = (*this)(j, j);
            for (int k = 0; k < j; ++k) {
                // d_val -= L_jk * conj(L_jk) * D_kk
                d_val -= (*this)(j, k) * conjugate((*this)(j, k)) * (*this)(k, k);
            }
            
            if (std::abs(d_val) < 1e-14) {
                 throw std::runtime_error("Error: Singular matrix or zero pivot in LDL* factorization.");
            }
            (*this)(j, j) = d_val;

            // Compute column j of L.
            T inv_d_val = T(1.0) / d_val;
            for (int i = j + 1; i < n; ++i) {
                T l_val = (*this)(i, j);
                for (int k = 0; k < j; ++k) {
                    l_val -= (*this)(i, k) * conjugate((*this)(j, k)) * (*this)(k, k);
                }
                (*this)(i, j) = l_val * inv_d_val;
            }
        }
        is_ldlt_factorized = true;
    }

    void solve(std::vector<T>& x, const std::vector<T>& b){
        if (is_ldlt_factorized) {
            // Solve with the LDL* factorization (A x = b -> L D L* x = b).
            if (b.size() != static_cast<std::size_t>(n_rows)) {
                throw std::invalid_argument("Error: Vector b must match the matrix row count.");
            }

            int n = n_rows;
            x = b;

            // 1. Forward substitution: L z = b (z is stored in x).
            for (int i = 0; i < n; ++i) {
                T sum = T(0);
                for (int j = 0; j < i; ++j) {
                    sum += (*this)(i, j) * x[j];
                }
                x[i] -= sum;
            }

            // 2. Diagonal solve: D y = z (y is stored in x).
            for (int i = 0; i < n; ++i) {
                x[i] /= (*this)(i, i);
            }

            // 3. Back substitution: L* x = y (the final x is stored in x).
            for (int i = n - 1; i >= 0; --i) {
                T sum = T(0);
                for (int j = i + 1; j < n; ++j) {
                    sum += conjugate((*this)(j, i)) * x[j]; // L*_ij = conj(L_ji)
                }
                x[i] -= sum;
            }
            return;
        }

        // Gaussian pivoting requires a square matrix.
        if (n_rows != n_cols){
            throw std::logic_error("Error: solve() requires a square matrix.");
        }
        if (b.size() != static_cast<std::size_t>(n_rows)) {
            throw std::invalid_argument("Error: Vector b must match the matrix row count.");
        }

        int n = n_rows;
        FullMatrix<T> A = *this; 
        x = b; 

        for (int k = 0; k < n; ++k) {
            
            // --- PIVOTING ---
            int pivot = k;
            auto max_val = std::abs(A(k,k));
            
            // Find the best pivot in column k.
            for(int i = k + 1; i < n; ++i) {
                if(std::abs(A(i,k)) > max_val) {
                    max_val = std::abs(A(i,k));
                    pivot = i;
                }
            }
            
            // A zero pivot means that the matrix is singular.
            if (max_val < 1e-12) { 
                // Raise an exception for a singular matrix.
                throw std::runtime_error("Error: Singular or nearly singular matrix.");
            }

            // Swap rows in A.
            if (pivot != k) {
                for (int j = k; j < n; ++j) { // Entries before j=k are zero.
                    std::swap(A(k,j), A(pivot,j));
                }
                // Swap entries in the right-hand-side vector x.
                std::swap(x[k], x[pivot]);
            }

            // Continue the elimination.
            for (int i = k + 1; i < n; ++i) {
                T factor = A(i, k) / A(k, k);
                for (int j = k; j < n; ++j) { // Start at j=k.
                    A(i, j) -= factor * A(k, j);
                }
                x[i] -= factor * x[k];
            }
        }

        // Back substitution.
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                x[i] -= A(i, j) * x[j];
            }
            x[i] /= A(i, i);
        }
    }

    // Transpose.
    FullMatrix<T> transpose() const {
        FullMatrix<T> res(n_cols, n_rows);
        for(int i = 0; i < n_rows; ++i) {
            for(int j = 0; j < n_cols; ++j) {
                res(j, i) = (*this)(i, j);
            }
        }
        return res;
    }

    FullMatrix<T> adjoint() const;


    // Inverse (computed one column at a time with solve).
    FullMatrix<T> inverse() const {
        if (n_rows != n_cols) throw std::logic_error("Error: Cannot invert a non-square matrix.");
        FullMatrix<T> res(n_rows, n_cols);
        FullMatrix<T> tmp = *this; 
        if (n_rows == n_cols) tmp.factorize(); // Dramatically speeds up inversion.
        
        std::vector<T> b(n_rows, T(0)), x(n_rows);
        for(int j = 0; j < n_cols; ++j) {
            b[j] = T(1); // Column j of the identity matrix.
            tmp.solve(x, b); // Solve A * x = e_j.
            for(int i = 0; i < n_rows; ++i) res(i, j) = x[i];
            b[j] = T(0); // Reset for the next iteration.
        }
        return res;
    }

    std::size_t rows (){
        return n_rows;
    }

    void print(std::ostream& os) const {
        os << "(";
        for (int i = 0; i < n_rows; i++){
            for (int j = 0; j < n_cols; j++){
                os << (*this)(i,j) << " ";
            }
            os << std::endl;
        }
        os << ")";
    }
};

template<typename T>
FullMatrix<T> operator+(const FullMatrix<T>& A, const FullMatrix<T>& B){
    FullMatrix<T> R = A;
    R += B;
    return R;
}

template<typename T>
FullMatrix<T> operator*(const FullMatrix<T>& A, const T s){
    FullMatrix<T> R = A;
    R *= s;
    return R;
}

template<typename T>
FullMatrix<T> operator*(const T s, const FullMatrix<T>& A){
    return A*s;
}

template<typename T>
FullMatrix<T> operator-(const FullMatrix<T>& A, const FullMatrix<T>& B){
    FullMatrix<T> R = A;
    R -= B;
    return R;
}

template<>
inline FullMatrix<complexe> FullMatrix<complexe>::adjoint() const {
        FullMatrix<complexe> res(n_cols, n_rows);
        for(int i = 0; i < n_rows; ++i) {
            for(int j = 0; j < n_cols; ++j) {
                res(j, i) = std::conj((*this)(i, j));
            }
        }
        return res;
}

//---------------------------------------------------------------------------
//     ProfileMatrix class
//---------------------------------------------------------------------------

template <typename T>
class ProfileMatrix {
private:

    int n;
    std::vector<std::size_t> p;
    std::vector<T> coefs;
    std::vector<std::size_t> q;
    std::vector<std::size_t> offsets;
    bool is_factorized;

public:

    ProfileMatrix(const std::vector<std::size_t>& p_in) : n(p_in.size()), p(p_in), is_factorized(false) {
        q.resize(n);
        offsets.resize(n);
        std::size_t total_size = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); i++) {
            offsets[i] = total_size;
            q[i] = i - p[i] + 1;
            total_size += q[i];
        }
        coefs.resize(total_size, T(0));
    }

    // Access

    T operator()(int i, int j) const {
        if (i < j) return (*this)(j, i); // Symmetry.
        if (j < static_cast<int>(p[i])) return T(0);
        return coefs[offsets[i] + j - p[i]];
    }
    
    T& operator()(int i, int j) {
        if (i < j) return (*this)(j, i);
        if (j < static_cast<int>(p[i])) throw std::runtime_error("Outside matrix profile");
        return coefs[offsets[i] + j - p[i]];
    }

    // Operators

    std::vector<T> operator*(const std::vector<T>& x) const {
        if (x.size() != static_cast<std::size_t>(n)) {
            // dimensions
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication.");
        }
        
        std::vector<T> res(n, T(0));
        std::size_t index_offset = 0;

        for (size_t i = 0; i < static_cast<size_t>(n); i++) {
            for (size_t j = p[i]; j < i; j++) {
                const T& val = coefs[index_offset + j - p[i]];
                res[i] += val * x[j];
                res[j] += val * x[i];
            }
            const T& diag_val = coefs[index_offset + i - p[i]];
            res[i] += diag_val * x[i];

            index_offset += q[i];
        }
        return res;
    }

    void operator+=(const ProfileMatrix<T>& M){
        for (size_t i = 0;i<coefs.size();i++){
            coefs[i] += M.coefs[i];
        }
    }

    void operator-=(const ProfileMatrix<T>& M){
        for (size_t i = 0;i<coefs.size();i++){
            coefs[i] -= M.coefs[i];
        }
    }

    // In-place LDL^T factorization.
    void factorize() {
        if(is_factorized) return;
        
        std::vector<T> diag(n);
        for(int i=0; i<n; ++i) {
            diag[i] = coefs[offsets[i] + i - static_cast<int>(p[i])];
        }

        for (int i = 0; i < n; ++i) {
            T d_val = T(0);
            int pi = static_cast<int>(p[i]);
            
            for (int j = pi; j < i; ++j) {
                T sum = T(0);
                int pj = static_cast<int>(p[j]);
                int start_k = std::max(pi, pj);

                const T* row_i = &coefs[offsets[i] - pi];
                const T* row_j = &coefs[offsets[j] - pj];

                for (int k = start_k; k < j; ++k) {
                    // sum += L_ik * L_jk * D_k
                    sum += row_i[k] * row_j[k] * diag[k];
                }
                
                T& A_ij = coefs[offsets[i] + j - pi];
                A_ij = (A_ij - sum) / diag[j];
                
                d_val += A_ij * A_ij * diag[j];
            }
            diag[i] -= d_val;
            coefs[offsets[i] + i - pi] = diag[i];
            
            if (std::abs(diag[i]) < 1e-14) throw std::runtime_error("Zero pivot in LDLT");
        }
        is_factorized = true;
    }

    // Fast solve using the existing factorization.
    void solve(std::vector<T>& x, const std::vector<T>& b) {
        if (!is_factorized) factorize(); // Factorize automatically if needed.
        if (b.size() != static_cast<std::size_t>(n)) throw std::invalid_argument("Invalid size");
        
        x = b;
        
        // 1. Forward substitution L z = b (L has an implicit unit diagonal).
        // In profile LDL storage, L_ij is stored at (i,j) for i>j.
        for (int i = 0; i < n; ++i) {
            T sum = T(0);
            int pi = static_cast<int>(p[i]);
            for (int j = pi; j < i; ++j) 
                sum += (*this)(i,j) * x[j];
            x[i] -= sum; // L_ii is 1, so no division is needed.
        }

        // 2. Diagonal solve D y = z.
        for (int i = 0; i < n; ++i) {
            x[i] /= (*this)(i,i);
        }

        // 3. Back substitution L^T x = y.
        for (int i = n - 1; i >= 0; --i) {
            int pi = static_cast<int>(p[i]);
            for (int j = pi; j < i; ++j) {
                // x[j] -= L_ji^T * x[i] => x[j] -= L_ij * x[i]
                x[j] -= (*this)(i,j) * x[i];
            }
        }
    }
};

template<typename T>

ProfileMatrix<T> operator+(const ProfileMatrix<T>& A,const ProfileMatrix<T>& B){
    ProfileMatrix<T> R(A);
    R += B;
    return R;
}

template<typename T>

ProfileMatrix<T> operator-(const ProfileMatrix<T>& A,const ProfileMatrix<T>& B){
    ProfileMatrix<T> R(A);
    R -= B;
    return R;
}

#endif //MATH_HPP
