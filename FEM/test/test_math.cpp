#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "math.hpp"

using namespace std;

// Fonction utilitaire pour vérifier les résultats (double)
bool check(const vector<double>& res, const vector<double>& expected, double tol = 1e-9) {
    double err = 0.0;
    for(size_t i=0; i<res.size(); ++i) err += abs(res[i] - expected[i]);
    return err < tol;
}

// Fonction utilitaire pour vérifier les résultats (complex)
bool check(const vector<complex<double>>& res, const vector<complex<double>>& expected, double tol = 1e-9) {
    double err = 0.0;
    for(size_t i=0; i<res.size(); ++i) err += abs(res[i] - expected[i]);
    return err < tol;
}

int main() {
    cout << "===========================================" << endl;
    cout << "   TEST UNITAIRE : PROFILE MATRIX (LDL^T)  " << endl;
    cout << "===========================================" << endl;

    // -------------------------------------------------------
    // TEST 1 : Système Réel (Laplacien 1D 3x3)
    // -------------------------------------------------------
    // Matrice A :
    //  2 -1  0
    // -1  2 -1
    //  0 -1  2
    // Solution pour b = (1, 0, 1) -> x = (1, 1, 1)
    
    cout << "\n--- Test 1 : Matrice Reelle 3x3 ---" << endl;
    try {
        // Définition du profil
        // Ligne 0 (diag 0) -> p=0
        // Ligne 1 (0, 1)   -> p=0
        // Ligne 2 (1, 2)   -> p=1
        vector<size_t> p = {0, 0, 1};
        ProfileMatrix<double> A(p);

        // Remplissage (Symétrique)
        A(0,0) = 2.0; 
        A(1,0) = -1.0; A(1,1) = 2.0;
        A(2,1) = -1.0; A(2,2) = 2.0;

        vector<double> b = {1.0, 0.0, 1.0};
        vector<double> x(3);

        A.solve(x, b);

        cout << "Solution obtenue : " << x << endl;
        vector<double> expected = {1.0, 1.0, 1.0};
        
        if(check(x, expected)) cout << "[OK] Test Reel valide." << endl;
        else cout << "[ECHEC] Resultat incorrect." << endl;

    } catch(const exception& e) {
        cout << "[ERREUR] " << e.what() << endl;
    }

    // -------------------------------------------------------
    // TEST 2 : Système Complexe (Type Helmholtz)
    // -------------------------------------------------------
    // Matrice C :
    //  i  1
    //  1  i
    // Solution pour b = (1+i, 1+i) -> x = (1, 1)
    
    cout << "\n--- Test 2 : Matrice Complexe 2x2 ---" << endl;
    try {
        vector<size_t> p_c = {0, 0};
        ProfileMatrix<complex<double>> C(p_c);

        C(0,0) = {0.0, 1.0}; // i
        C(1,0) = {1.0, 0.0}; // 1
        C(1,1) = {0.0, 1.0}; // i

        vector<complex<double>> b_c = {{1.0, 1.0}, {1.0, 1.0}};
        vector<complex<double>> x_c(2);

        C.solve(x_c, b_c);

        cout << "Solution obtenue : " << x_c << endl;
        vector<complex<double>> expected_c = {{1.0, 0.0}, {1.0, 0.0}};

        if(check(x_c, expected_c)) cout << "[OK] Test Complexe valide." << endl;
        else cout << "[ECHEC] Resultat incorrect." << endl;

    } catch(const exception& e) {
        cout << "[ERREUR] " << e.what() << endl;
    }

    // -------------------------------------------------------
    // TEST 3 : Vérification Hors Profil
    // -------------------------------------------------------
    cout << "\n--- Test 3 : Acces Hors Profil ---" << endl;
    try {
        vector<size_t> p = {0, 1}; // 2x2, A(1,0) hors profil
        ProfileMatrix<double> A(p);
        A(1, 0) = 5.0; // Doit lancer une exception
        cout << "[ECHEC] L'exception n'a pas ete lancee." << endl;
    } catch(const exception& e) {
        cout << "[OK] Exception capturee : " << e.what() << endl;
    }

    // -------------------------------------------------------
    // TEST 4 : FullMatrix Operations
    // -------------------------------------------------------
    cout << "\n--- Test 4 : FullMatrix Operations ---" << endl;
    
    // Test Inverse 2x2
    FullMatrix<double> M(2, 2);
    M(0,0) = 4.0; M(0,1) = 3.0;
    M(1,0) = 3.0; M(1,1) = 2.0;
    
    // Det = 8 - 9 = -1. Inv = 1/-1 * [2 -3; -3 4] = [-2 3; 3 -4]
    FullMatrix<double> InvM = M.inverse();
    
    if(abs(InvM(0,0) - (-2.0)) < 1e-9 && abs(InvM(1,1) - (-4.0)) < 1e-9)
        cout << "[OK] Inversion FullMatrix 2x2 correcte." << endl;
    else
        cout << "[ECHEC] Inversion FullMatrix 2x2 incorrecte." << endl;
    
    // Test Solve
    vector<double> b_full = {1.0, 2.0};
    vector<double> x_full(2);
    M.solve(x_full, b_full); // x = InvM * b = [-2 3; 3 -4]*[1;2] = [4; -5]
    
    if(abs(x_full[0] - 4.0) < 1e-9 && abs(x_full[1] - (-5.0)) < 1e-9)
        cout << "[OK] Resolution FullMatrix Ax=b correcte." << endl;
    else
        cout << "[ECHEC] Resolution FullMatrix Ax=b incorrecte." << endl;

    return 0;
}