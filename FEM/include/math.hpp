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
//  Opérations sur vector<T>
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

    // Accès

    T& operator()(int i, int j) { return coefs[i * n_cols + j]; }
    const T& operator()(int i, int j) const { return coefs[i * n_cols + j]; }

    // Opérateurs
    
    std::vector<T> operator*(const std::vector<T>& x) const {
        // Ajout d'une vérification de la taille du vecteur
        if (x.size() != static_cast<std::size_t>(n_cols)){
            throw std::invalid_argument("Erreur: Le nombre de colonnes de la matrice doit être égal à la taille du vecteur.");
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
            throw std::invalid_argument("Erreur: Dimensions incompatibles pour la multiplication matricielle.");
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
        // vérification de la taille
        if (n_rows != M.n_rows || n_cols != M.n_cols){
            throw std::invalid_argument("Erreur: Addition de matrices de tailles différentes.");
        }
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i] += M.coefs[i];
        }
    }

    void operator-=(const FullMatrix<T>& M){
        // vérification de la taille
        if (n_rows != M.n_rows || n_cols != M.n_cols){
            throw std::invalid_argument("Erreur: Soustraction de matrices de tailles différentes.");
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

    // Remplissage avec une valeur
    void fill(T val) {
        for (size_t i = 0; i < coefs.size(); i++){
            coefs[i] = val;
        }
    }

    // Résolution via pivot de Gauss

    // Factorisation LDL* (pour matrices hermitiennes) en place.
    // La partie triangulaire inférieure stocke L (sans la diagonale unité)
    // et la diagonale stocke D.

    void factorize() {
        if (is_ldlt_factorized) return;
        if (n_rows != n_cols) {
            throw std::logic_error("Erreur: La factorisation LDLT ne s'applique qu'aux matrices carrées.");
        }
        
        int n = n_rows;

        for (int j = 0; j < n; ++j) {
            // Calcul de D_jj
            T d_val = (*this)(j, j);
            for (int k = 0; k < j; ++k) {
                // d_val -= L_jk * conj(L_jk) * D_kk
                d_val -= (*this)(j, k) * conjugate((*this)(j, k)) * (*this)(k, k);
            }
            
            if (std::abs(d_val) < 1e-14) {
                 throw std::runtime_error("Erreur: Matrice singulière ou pivot nul dans la factorisation LDL*.");
            }
            (*this)(j, j) = d_val;

            // Calcul de la colonne j de L
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
            // Résolution avec la factorisation LDL* (A x = b -> L D L* x = b)
            if (b.size() != static_cast<std::size_t>(n_rows)) {
                throw std::invalid_argument("Erreur: La taille du vecteur b doit correspondre au nombre de lignes de la matrice.");
            }

            int n = n_rows;
            x = b;

            // 1. Descente : L z = b  (z est stocké dans x)
            for (int i = 0; i < n; ++i) {
                T sum = T(0);
                for (int j = 0; j < i; ++j) {
                    sum += (*this)(i, j) * x[j];
                }
                x[i] -= sum;
            }

            // 2. Diagonale : D y = z (y est stocké dans x)
            for (int i = 0; i < n; ++i) {
                x[i] /= (*this)(i, i);
            }

            // 3. Remontée : L* x = y (x final est stocké dans x)
            for (int i = n - 1; i >= 0; --i) {
                T sum = T(0);
                for (int j = i + 1; j < n; ++j) {
                    sum += conjugate((*this)(j, i)) * x[j]; // L*_ij = conj(L_ji)
                }
                x[i] -= sum;
            }
            return;
        }

        // La résolution par pivot de Gauss ne s'applique qu'aux matrices carrées
        if (n_rows != n_cols){
            throw std::logic_error("Erreur: solve() ne peut être appelée que pour des matrices carrées.");
        }
        if (b.size() != static_cast<std::size_t>(n_rows)) {
            throw std::invalid_argument("Erreur: La taille du vecteur b doit correspondre au nombre de lignes de la matrice.");
        }

        int n = n_rows;
        FullMatrix<T> A = *this; 
        x = b; 

        for (int k = 0; k < n; ++k) {
            
            // --- AJOUT PIVOT ---
            int pivot = k;
            auto max_val = std::abs(A(k,k));
            
            // Recherche du meilleur pivot dans la colonne k
            for(int i = k + 1; i < n; ++i) {
                if(std::abs(A(i,k)) > max_val) {
                    max_val = std::abs(A(i,k));
                    pivot = i;
                }
            }
            
            // Si le pivot est nul, la matrice est singulière
            if (max_val < 1e-12) { 
                // Lancer une exception pour une matrice singulière
                throw std::runtime_error("Erreur: Matrice singulière ou presque singulière.");
            }

            // Échange des lignes dans A
            if (pivot != k) {
                for (int j = k; j < n; ++j) { // On peut commencer à j=k (avant c'est 0)
                    std::swap(A(k,j), A(pivot,j));
                }
                // Échange dans le vecteur second membre x
                std::swap(x[k], x[pivot]);
            }

            // Suite normale de l'élimination
            for (int i = k + 1; i < n; ++i) {
                T factor = A(i, k) / A(k, k);
                for (int j = k; j < n; ++j) { // Optimisation: commencer à j=k
                    A(i, j) -= factor * A(k, j);
                }
                x[i] -= factor * x[k];
            }
        }

        // Substitution arrière
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                x[i] -= A(i, j) * x[j];
            }
            x[i] /= A(i, i);
        }
    }

    // Transposée
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


    // Inverse (calculée colonne par colonne via solve)
    FullMatrix<T> inverse() const {
        if (n_rows != n_cols) throw std::logic_error("Erreur: Inversion d'une matrice non carrée.");
        FullMatrix<T> res(n_rows, n_cols);
        FullMatrix<T> tmp = *this; 
        if (n_rows == n_cols) tmp.factorize(); // Accélère drastiquement l'inversion
        
        std::vector<T> b(n_rows, T(0)), x(n_rows);
        for(int j = 0; j < n_cols; ++j) {
            b[j] = T(1); // Colonne j de la matrice identité
            tmp.solve(x, b); // Résout A * x = e_j
            for(int i = 0; i < n_rows; ++i) res(i, j) = x[i];
            b[j] = T(0); // Remise à zéro pour la prochaine itération
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
//     Classe ProfileMatrix
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

    // Accès

    T operator()(int i, int j) const {
        if (i < j) return (*this)(j, i); // Symétrie
        if (j < static_cast<int>(p[i])) return T(0);
        return coefs[offsets[i] + j - p[i]];
    }
    
    T& operator()(int i, int j) {
        if (i < j) return (*this)(j, i);
        if (j < static_cast<int>(p[i])) throw std::runtime_error("Hors profil");
        return coefs[offsets[i] + j - p[i]];
    }

    // Opérateurs

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

    // Factorisation LDL^T en place
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
            
            if (std::abs(diag[i]) < 1e-14) throw std::runtime_error("Pivot nul dans LDLT");
        }
        is_factorized = true;
    }

    // Résolution rapide utilisant la factorisation existante
    void solve(std::vector<T>& x, const std::vector<T>& b) {
        if (!is_factorized) factorize(); // Auto-factorisation si oublié
        if (b.size() != static_cast<std::size_t>(n)) throw std::invalid_argument("Taille invalide");
        
        x = b;
        
        // 1. Descente L z = b (L a des 1 sur la diagonale, implicites ou non, ici stockés différemment)
        // Note: Dans LDL stocké en profil, L_ij est stocké à (i,j) pour i>j.
        for (int i = 0; i < n; ++i) {
            T sum = T(0);
            int pi = static_cast<int>(p[i]);
            for (int j = pi; j < i; ++j) 
                sum += (*this)(i,j) * x[j];
            x[i] -= sum; // L_ii est 1, donc pas de division
        }

        // 2. Diagonale D y = z
        for (int i = 0; i < n; ++i) {
            x[i] /= (*this)(i,i);
        }

        // 3. Remontée L^T x = y
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
