#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>

#include "fem.hpp"

using namespace std;
using namespace usim;

// Utilitaires de vérification
bool check_close(double val, double expected, double tol = 1e-9) {
    return std::abs(val - expected) < tol;
}

bool check_close(complexe val, complexe expected, double tol = 1e-9) {
    return std::abs(val - expected) < tol;
}

void test_shape_functions() {
    cout << "--- Test 1 : Fonctions de forme P2 ---" << endl;
    vector<double> phi(6);
    
    // Test au sommet 0 (0,0) -> phi[0] doit valoir 1
    Fem::evaluate_shape_functions(0.0, 0.0, phi);
    if(check_close(phi[0], 1.0) && check_close(phi[1], 0.0) && check_close(phi[3], 0.0))
        cout << "[OK] Valeurs au sommet 0 correctes." << endl;
    else
        cout << "[ECHEC] Valeurs au sommet 0 incorrectes." << endl;

    // Test au milieu de l'arête 0-1 (0.5, 0) -> phi[3] doit valoir 1
    Fem::evaluate_shape_functions(0.5, 0.0, phi);
    if(check_close(phi[3], 1.0) && check_close(phi[0], 0.0))
        cout << "[OK] Valeurs au noeud milieu correctes." << endl;
    else
        cout << "[ECHEC] Valeurs au noeud milieu incorrectes." << endl;

    // Partition de l'unité : somme(phi) = 1 partout
    Fem::evaluate_shape_functions(0.33, 0.33, phi);
    double sum = 0;
    for(double v : phi) sum += v;
    if(check_close(sum, 1.0))
        cout << "[OK] Partition de l'unite respectee." << endl;
    else
        cout << "[ECHEC] Partition de l'unite violee (somme=" << sum << ")." << endl;
}

void test_gradients() {
    cout << "\n--- Test 2 : Gradients P2 ---" << endl;
    FullMatrix<double> grads(6, 2);
    
    // La somme des gradients doit être nulle (grad(1) = 0)
    Fem::evaluate_gradients(0.25, 0.25, grads);
    double sum_dx = 0, sum_dy = 0;
    for(int i=0; i<6; ++i) {
        sum_dx += grads(i, 0);
        sum_dy += grads(i, 1);
    }
    
    if(check_close(sum_dx, 0.0) && check_close(sum_dy, 0.0))
        cout << "[OK] Somme des gradients nulle." << endl;
    else
        cout << "[ECHEC] Somme des gradients non nulle." << endl;
}

void test_assembly_single_triangle() {
    cout << "\n--- Test 3 : Assemblage (Rigidite & Masse) sur 1 Triangle ---" << endl;
    
    // Création manuelle d'un maillage avec 1 triangle (Triangle de référence)
    MeshP2 mesh;
    mesh.nodes.resize(6);
    // Sommets
    mesh.nodes[0] = {0.0, 0.0, 0};
    mesh.nodes[1] = {1.0, 0.0, 1};
    mesh.nodes[2] = {0.0, 1.0, 2};
    // Milieux
    mesh.nodes[3] = {0.5, 0.0, 3};
    mesh.nodes[4] = {0.5, 0.5, 4};
    mesh.nodes[5] = {0.0, 0.5, 5};
    
    TriangleP2 tri;
    tri.node_ids[0]=0; tri.node_ids[1]=1; tri.node_ids[2]=2;
    tri.node_ids[3]=3; tri.node_ids[4]=4; tri.node_ids[5]=5;
    tri.ref = 0;
    mesh.triangles.push_back(tri);
    
    // Calcul du profil
    vector<size_t> p = Fem::compute_profile(mesh);
    ProfileMatrix<complexe> K(p);
    ProfileMatrix<complexe> M(p);
    
    // Assemblage
    Fem::A_matrix(mesh, K, 1.0);
    Fem::B_matrix(mesh, M, 1.0, 1.0, 1.0); // k0=1, k_d=1
    
    // Vérification Rigidité : K * 1 = 0 (car grad(1) = 0)
    vector<complexe> ones(6, 1.0);
    vector<complexe> res_K = K * ones;
    bool k_ok = true;
    for(auto v : res_K) {
        if(abs(v) > 1e-9) k_ok = false;
    }
    if(k_ok) cout << "[OK] Matrice de Rigidite (noyau contient les constantes)." << endl;
    else cout << "[ECHEC] Matrice de Rigidite incorrecte." << endl;

    // Vérification Masse : 1^T * M * 1 = Int(1*1) = Aire = 0.5
    vector<complexe> res_M = M * ones;
    complexe total_mass = 0;
    for(auto v : res_M) total_mass += v;
    
    if(check_close(total_mass, complex<double>(0.5,0.0))) 
        cout << "[OK] Matrice de Masse (integrale correcte)." << endl;
    else 
        cout << "[ECHEC] Matrice de Masse incorrecte (Masse=" << total_mass << ", attendu 0.5)." << endl;
}

void test_boundary_assembly() {
    cout << "\n--- Test 4 : Assemblage Bord (Matrice E) ---" << endl;
    
    // Maillage 1 triangle, l'arête 0 (noeuds 0-1) est sur le bord (tag 10)
    MeshP2 mesh;
    mesh.Ly = 1.0; // Nécessaire pour compute_E
    mesh.nodes.resize(6);
    mesh.nodes[0] = {0.0, 0.0, 0};
    mesh.nodes[1] = {1.0, 0.0, 1};
    mesh.nodes[2] = {0.0, 1.0, 2};
    mesh.nodes[3] = {0.5, 0.0, 3};
    mesh.nodes[4] = {0.5, 0.5, 4};
    mesh.nodes[5] = {0.0, 0.5, 5};
    
    TriangleP2 tri;
    tri.node_ids[0]=0; tri.node_ids[1]=1; tri.node_ids[2]=2;
    tri.node_ids[3]=3; tri.node_ids[4]=4; tri.node_ids[5]=5;
    
    // On marque l'arête 0 (entre noeuds 0 et 1) avec le tag 10
    tri.edge_ref[0] = 10; 
    tri.edge_ref[1] = 0; 
    tri.edge_ref[2] = 0;
    mesh.triangles.push_back(tri);
    
    // compute_E retourne une FullMatrix (Ndof x Nmodes)
    // On teste avec 1 mode (n=0), tag 10, k0=1.0
    FullMatrix<complexe> E = Fem::compute_E(mesh, 1, 10, 1.0);
    
    // L'intégrale sur le bord (longueur 1) de la fonction constante 1 doit valoir 1.
    // Pour n=0, c_0 = sqrt(1/Ly) = 1.
    // Somme des éléments de la colonne 0 de E = Int(sum(phi_i) * c_0) = Int(1 * 1) = 1.
    
    complexe sum = 0;
    for(int i=0; i<static_cast<int>(E.rows()); ++i) sum += E(i, 0);
    
    if(check_close(sum, complexe(1.0,0.0)))
        cout << "[OK] Matrice de Bord (integrale longueur correcte)." << endl;
    else
        cout << "[ECHEC] Matrice de Bord incorrecte (Somme=" << sum << ", attendu 1.0)." << endl;
}

void test_l2_error() {
    cout << "\n--- Test 5 : Erreur L2 sur milieu homogene (Geo 0 a Lx) ---" << endl;

    // 1. Chargement
    string filename = "../data/test_ultrasound_sans_defaut.msh"; // Assurez-vous que c'est le bon nom
    MeshP2 mesh;
    mesh.read_msh_v2_ascii(filename, {}); // Pas de défauts pour ce test

    // 2. Parametres
    double k0 = 5.0; // Vérifiez que c'est cohérent avec la fréquence voulue
    double h = mesh.Ly; 
    
    // IMPORTANT : On identifie la position physique du bord gauche
    // D'après le .geo, xmin sera 0.0.
    double x_source_location = mesh.xmin; 

    int tag_left = 11;
    int tag_right = 12;
    int n_mode = 0; 
    int N_modes = 5;

    // 3. Assemblage Matrices (Inchangé)
    vector<size_t> profile = Fem::compute_profile_enhanced(mesh, {tag_left, tag_right});
    ProfileMatrix<complexe> K(profile);
    Fem::A_matrix(mesh, K, 1.0);
    Fem::B_matrix(mesh, K, k0, k0, -1.0); 

    FullMatrix<complexe> E_minus = Fem::compute_E(mesh, N_modes, tag_left, k0);
    FullMatrix<complexe> E_plus  = Fem::compute_E(mesh, N_modes, tag_right, k0);
    FullMatrix<complexe> D(N_modes, N_modes);
    Fem::compute_D(D, N_modes, h, k0);

    Fem::T_matrix(K, E_minus, D, h, tag_left, -1.0);
    Fem::T_matrix(K, E_plus, D, h, tag_right, -1.0);

    K.factorize();

    // 4. Vecteur Source (CORRECTION)
    // On dit à la fonction : "La source est située à la coordonnée x_source_location"
    // Comme x_source_location = 0, le terme de phase exp(...) vaudra 1.
    vector<complexe> G_minus = Fem::assemble_source_vector(mesh, E_minus, n_mode, k0, x_source_location, -1.0);
    
    // Résolution
    vector<complexe> U_fem(mesh.ndof());
    K.solve(U_fem, G_minus);

    // 5. Solution Exacte
    // Onde plane : Phi(x) = exp(i * beta * x)
    // Comme le maillage commence à 0, à l'entrée (x=0), exp(0) = 1.
    // Cela correspond parfaitement à la source imposée (phase nulle à l'origine).
    complexe beta = Fem::compute_beta(k0, h, n_mode);
    vector<complexe> U_ex(mesh.ndof());
    
    for(const auto& node : mesh.nodes) {
        double c0 = Fem::evaluate_c_1d(node.y, h, n_mode);
        // node.x est dans [0, 1], donc la phase évolue de 0 à beta*1
        U_ex[node.id] = c0 * exp(complexe(0, 1) * beta * node.x);
    }

    // 6. Calcul Erreur
    ProfileMatrix<complexe> M_pure(profile);
    Fem::B_matrix(mesh, M_pure, 1.0, 1.0, 1.0); 

    vector<complexe> error = U_fem - U_ex;
    double l2_error = sqrt(std::abs(error | (M_pure * error)));
    double l2_sol = sqrt(std::abs(U_ex | (M_pure * U_ex)));
    double rel_error = l2_error / l2_sol;

    cout << "Erreur L2 relative: " << rel_error * 100.0 << " %" << endl;
}

int main() {
    test_shape_functions();
    test_gradients();
    test_assembly_single_triangle();
    test_boundary_assembly();
    test_l2_error();
    return 0;
}