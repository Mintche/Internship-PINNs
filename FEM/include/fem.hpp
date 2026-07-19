#ifndef FEM_HPP
#define FEM_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "math.hpp"
#include "mesh.hpp"
#include <queue>
#include <set>
#include <map>
#include <cstdio>

// Structure pour stocker un point d'intégration (Gauss)
struct QuadraturePoint {
    double x, y; // Coordonnées sur le triangle de référence
    double w;    // Poids associé
};

namespace Fem {
    // -------------------------------------------------------------------------
    // Fonctions de base P2 (Shape Functions) sur le triangle de référence
    // -------------------------------------------------------------------------

    void evaluate_shape_functions(double x, double y, std::vector<double>& phi);

    // -------------------------------------------------------------------------
    // Gradients des fonctions de base
    // -------------------------------------------------------------------------

    void evaluate_gradients(double x, double y, FullMatrix<double>& grads);

    // -------------------------------------------------------------------------
    // Quadrature de Gauss
    // -------------------------------------------------------------------------

    std::vector<QuadraturePoint> get_quadrature_points();

    // -------------------------------------------------------------------------
    // Algorithme Reverse Cuthill-McKee (RCM) pour réduire la largeur de bande
    // -------------------------------------------------------------------------

    void reorder_mesh_rcm(usim::MeshP2& mesh);

    // -------------------------------------------------------------------------
    // Calcul du profil de la matrice
    // -------------------------------------------------------------------------

    std::vector<std::size_t> compute_profile(const usim::MeshP2& mesh);

    std::vector<std::size_t> compute_profile_enhanced(const usim::MeshP2& mesh, const std::vector<int>& boundary_tags);

    // -------------------------------------------------------------------------
    // Analyse géométrique du maillage
    // -------------------------------------------------------------------------

    double get_max_edge_length(const usim::MeshP2& mesh);

    // -------------------------------------------------------------------------
    // Assemblage de la Matrice de Rigidité A
    // -------------------------------------------------------------------------

    void A_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& A, double factor = 1.0);

    // -------------------------------------------------------------------------
    // Assemblage de la matrice de masse geometrique M
    // -------------------------------------------------------------------------

    void M_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& M, double factor = 1.0);

    // -------------------------------------------------------------------------
    // Assemblage du terme de masse pondere B = k(x)^2 M
    // -------------------------------------------------------------------------

    void B_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0, double k_d_val, double factor = -1.0);

    void B_matrix_by_tag(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0, const std::map<int, double>& tag_contrasts, double factor = -1.0);

    void B_matrix_from_nodal_sound_speed(const usim::MeshP2& mesh,
                                         ProfileMatrix<complexe>& B,
                                         double omega,
                                         const std::vector<double>& sound_speed,
                                         double factor = -1.0);

    // -------------------------------------------------------------------------
    // Quadrature 1D (Gauss-Legendre) pour les bords
    // -------------------------------------------------------------------------

    std::vector<QuadraturePoint> get_quadrature_points_1d();

    // -------------------------------------------------------------------------
    // Fonctions de forme 1D P2 sur [-1, 1] et calcul de c_n
    // -------------------------------------------------------------------------

    void evaluate_shape_functions_1d(double t, std::vector<double>& phi);

    double evaluate_c_1d(double y, double h, int n);

    // -------------------------------------------------------------------------
    // Calcul de Beta_n
    // -------------------------------------------------------------------------

    complexe compute_beta(double k0, double h, int n);

    // -------------------------------------------------------------------------
    // Assemblage matrice E
    // -------------------------------------------------------------------------

    FullMatrix<complexe> compute_E(const usim::MeshP2& mesh, int N_modes, int boundary_tag, double k0);

    // -------------------------------------------------------------------------
    // Assemblage matrice D
    // -------------------------------------------------------------------------

    void compute_D(FullMatrix<complexe>& D, int N_modes, double h, double k0);

    // -------------------------------------------------------------------------
    // Assemblage matrice T
    // -------------------------------------------------------------------------

    void T_matrix(ProfileMatrix<complexe>& K, FullMatrix<complexe>& E, FullMatrix<complexe>& D, double h, int boundary_tag, double factor = -1.0);

    // -------------------------------------------------------------------------
    // Assemblage vecteur G
    // -------------------------------------------------------------------------

    std::vector<complexe> assemble_source_vector(const usim::MeshP2& mesh, const FullMatrix<complexe>& E,int n_inc, double k0, double L, double coef);
};

#endif
