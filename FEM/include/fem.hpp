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

// Stores a Gauss integration point.
struct QuadraturePoint {
    double x, y; // Coordinates on the reference triangle.
    double w;    // Associated weight.
};

namespace Fem {
    // -------------------------------------------------------------------------
    // P2 shape functions on the reference triangle.
    // -------------------------------------------------------------------------

    void evaluate_shape_functions(double x, double y, std::vector<double>& phi);

    // -------------------------------------------------------------------------
    // Shape-function gradients.
    // -------------------------------------------------------------------------

    void evaluate_gradients(double x, double y, FullMatrix<double>& grads);

    // -------------------------------------------------------------------------
    // Gauss quadrature.
    // -------------------------------------------------------------------------

    std::vector<QuadraturePoint> get_quadrature_points();

    // -------------------------------------------------------------------------
    // Reverse Cuthill-McKee (RCM) algorithm to reduce the matrix bandwidth.
    // -------------------------------------------------------------------------

    void reorder_mesh_rcm(usim::MeshP2& mesh);

    // -------------------------------------------------------------------------
    // Compute the matrix profile.
    // -------------------------------------------------------------------------

    std::vector<std::size_t> compute_profile(const usim::MeshP2& mesh);

    std::vector<std::size_t> compute_profile_enhanced(const usim::MeshP2& mesh, const std::vector<int>& boundary_tags);

    // -------------------------------------------------------------------------
    // Mesh geometry analysis.
    // -------------------------------------------------------------------------

    double get_max_edge_length(const usim::MeshP2& mesh);

    // -------------------------------------------------------------------------
    // Assemble the stiffness matrix A.
    // -------------------------------------------------------------------------

    void A_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& A, double factor = 1.0);

    // -------------------------------------------------------------------------
    // Assemble the geometric mass matrix M.
    // -------------------------------------------------------------------------

    void M_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& M, double factor = 1.0);

    // -------------------------------------------------------------------------
    // Assemble the weighted mass term B = k(x)^2 M.
    // -------------------------------------------------------------------------

    void B_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0, double k_d_val, double factor = -1.0);

    void B_matrix_by_tag(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0, const std::map<int, double>& tag_contrasts, double factor = -1.0);

    void B_matrix_from_nodal_sound_speed(const usim::MeshP2& mesh,
                                         ProfileMatrix<complexe>& B,
                                         double omega,
                                         const std::vector<double>& sound_speed,
                                         double factor = -1.0);

    // -------------------------------------------------------------------------
    // One-dimensional Gauss-Legendre quadrature on boundaries.
    // -------------------------------------------------------------------------

    std::vector<QuadraturePoint> get_quadrature_points_1d();

    // -------------------------------------------------------------------------
    // One-dimensional P2 shape functions on [-1, 1] and c_n evaluation.
    // -------------------------------------------------------------------------

    void evaluate_shape_functions_1d(double t, std::vector<double>& phi);

    double evaluate_c_1d(double y, double h, int n);

    // -------------------------------------------------------------------------
    // Compute beta_n.
    // -------------------------------------------------------------------------

    complexe compute_beta(double k0, double h, int n);

    // -------------------------------------------------------------------------
    // Assemble matrix E.
    // -------------------------------------------------------------------------

    FullMatrix<complexe> compute_E(const usim::MeshP2& mesh, int N_modes, int boundary_tag, double k0);

    // -------------------------------------------------------------------------
    // Assemble matrix D.
    // -------------------------------------------------------------------------

    void compute_D(FullMatrix<complexe>& D, int N_modes, double h, double k0);

    // -------------------------------------------------------------------------
    // Assemble matrix T.
    // -------------------------------------------------------------------------

    void T_matrix(ProfileMatrix<complexe>& K, FullMatrix<complexe>& E, FullMatrix<complexe>& D, double h, int boundary_tag, double factor = -1.0);

    // -------------------------------------------------------------------------
    // Assemble vector G.
    // -------------------------------------------------------------------------

    std::vector<complexe> assemble_source_vector(const usim::MeshP2& mesh, const FullMatrix<complexe>& E,int n_inc, double k0, double L, double coef);
};

#endif
