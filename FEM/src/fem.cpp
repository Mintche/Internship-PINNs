#include "fem.hpp"

namespace {

template <typename WavenumberForTriangle>
void assemble_B_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B,
                       double factor, WavenumberForTriangle wavenumber_for_triangle) {
    auto qp = Fem::get_quadrature_points();
    std::vector<double> phi(6);

    for (const auto& tri : mesh.triangles) {
        usim::Point2D p0 = mesh.nodes[tri.node_ids[0]];
        usim::Point2D p1 = mesh.nodes[tri.node_ids[1]];
        usim::Point2D p2 = mesh.nodes[tri.node_ids[2]];

        double detJac = (p1.x-p0.x)*(p2.y-p0.y)-(p1.y-p0.y)*(p2.x-p0.x);
        double k = wavenumber_for_triangle(tri);

        for (const auto& q : qp) {
            Fem::evaluate_shape_functions(q.x, q.y, phi);
            double weight = q.w * std::abs(detJac);

            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double val = phi[i] * phi[j] * weight * (k * k);
                    B(tri.node_ids[i], tri.node_ids[j]) += val * factor;
                }
            }
        }
    }
}

} // namespace

namespace Fem {

// Node order: 0,1,2 (vertices), then 3,4,5 (midpoints).

void evaluate_shape_functions(double x, double y, std::vector<double>& phi) {

    double lambda1 = 1.0 - x - y;
    double lambda2 = x;
    double lambda3 = y;

    phi[0] = lambda1 * (2.0 * lambda1 - 1.0); // Vertex 0.
    phi[1] = lambda2 * (2.0 * lambda2 - 1.0); // Vertex 1.
    phi[2] = lambda3 * (2.0 * lambda3 - 1.0); // Vertex 2.

    phi[3] = 4.0 * lambda1 * lambda2; // Midpoint of edge 0-1.
    phi[4] = 4.0 * lambda2 * lambda3; // Midpoint of edge 1-2.
    phi[5] = 4.0 * lambda1 * lambda3; // Midpoint of edge 0-2.
}

void evaluate_gradients(double x, double y, FullMatrix<double>& grads) {

    grads(0,0) = 4.0*x + 4.0*y - 3.0;
    grads(0,1) = 4.0*x + 4.0*y - 3.0;

    grads(1,0) = 4.0*x - 1.0;
    grads(1,1) = 0.0;

    grads(2,0) = 0.0;
    grads(2,1) = 4.0*y - 1.0;

    grads(3,0) = 4.0 - 8.0*x - 4.0*y;
    grads(3,1) = -4.0*x;

    grads(4,0) = 4.0*y;
    grads(4,1) = 4.0*x;

    grads(5,0) = -4.0*y;
    grads(5,1) = 4.0 - 4.0*x - 8.0*y;
}

std::vector<QuadraturePoint> get_quadrature_points() {

    double s0 = 1/3.;
    double s1 = (6-std::sqrt(15.0))/21.;
    double s2 = (6+std::sqrt(15.0))/21.;
    double s3 = (9+2*std::sqrt(15.0))/21.;
    double s4 = (9-2*std::sqrt(15.0))/21.;

    double eta0 = 9/80.;
    double eta1 = (155-std::sqrt(15.0))/2400.;
    double eta2 = (155+std::sqrt(15.0))/2400.;

    std::vector<QuadraturePoint> qp(7);

    qp[0].x = s0;
    qp[0].y = s0;
    qp[0].w = eta0;

    qp[1].x = s1;
    qp[1].y = s1;
    qp[1].w = eta1;

    qp[2].x = s1;
    qp[2].y = s3;
    qp[2].w = eta1;

    qp[3].x = s3;
    qp[3].y = s1;
    qp[3].w = eta1;

    qp[4].x = s2;
    qp[4].y = s2;
    qp[4].w = eta2;

    qp[5].x = s2;
    qp[5].y = s4;
    qp[5].w = eta2;

    qp[6].x = s4;
    qp[6].y = s2;
    qp[6].w = eta2;
        
    return qp;

}

void reorder_mesh_rcm(usim::MeshP2& mesh) {
    int n = mesh.nodes.size();
    if (n == 0) return;

    // 1. Build the adjacency graph.
    std::vector<std::set<int>> adj(n);
    for(const auto& tri : mesh.triangles) {
        for(int i=0; i<6; ++i) {
            for(int j=i+1; j<6; ++j) {
                int u = tri.node_ids[i];
                int v = tri.node_ids[j];
                adj[u].insert(v);
                adj[v].insert(u);
            }
        }
    }

    // 2. Find a pseudo-peripheral starting node.
    // Heuristic: minimum-degree node, then BFS to find the farthest node.
    int start_node = 0;
    std::size_t min_deg = n + 1;
    for(int i=0; i<n; ++i) {
        if(adj[i].size() < min_deg) {
            min_deg = adj[i].size();
            start_node = i;
        }
    }

    auto get_farthest = [&](int start) {
        std::queue<int> q; q.push(start);
        std::vector<int> dist(n, -1); dist[start] = 0;
        int farthest = start;
        while(!q.empty()) {
            int u = q.front(); q.pop();
            if(dist[u] > dist[farthest]) farthest = u;
            for(int v : adj[u]) {
                if(dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
            }
        }
        return farthest;
    };

    int P = get_farthest(start_node);
    int Q = get_farthest(P);
    start_node = Q;

    // 3. Parcours Cuthill-McKee
    std::vector<int> perm; perm.reserve(n);
    std::vector<bool> visited(n, false);

    for(int i=0; i<n; ++i) {
        // Handle disconnected components (if the mesh has multiple pieces).
        int root = (i == 0) ? start_node : i;
        if(visited[root]) continue;

        std::queue<int> q;
        q.push(root);
        visited[root] = true;
        perm.push_back(root);

        while(!q.empty()) {
            int u = q.front(); q.pop();

            // Collect unvisited neighbors.
            std::vector<int> neighbors;
            for(int v : adj[u]) {
                if(!visited[v]) neighbors.push_back(v);
            }
            // Sort by increasing degree.
            std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b){
                return adj[a].size() < adj[b].size();
            });

            for(int v : neighbors) {
                visited[v] = true;
                perm.push_back(v);
                q.push(v);
            }
        }
    }

    // 4. Reverse (RCM)
    std::reverse(perm.begin(), perm.end());

    // 5. Apply the permutation to the mesh.
    std::vector<int> old_to_new(n);
    std::vector<usim::Point2D> new_nodes(n);
    for(int i=0; i<n; ++i) {
        int old_id = perm[i];
        old_to_new[old_id] = i;
        new_nodes[i] = mesh.nodes[old_id];
        new_nodes[i].id = i;
    }
    mesh.nodes = std::move(new_nodes);
    for(auto& tri : mesh.triangles) {
        for(int k=0; k<6; ++k) tri.node_ids[k] = old_to_new[tri.node_ids[k]];
    }
    std::printf("RCM reordering applied.\n");
}

    std::vector<std::size_t> compute_profile(const usim::MeshP2& mesh){
        std::size_t ndof = mesh.ndof();
        std::vector<std::size_t> p(ndof);
        
        // Initialization: p[i] = i (at least the diagonal).
        for(std::size_t i=0; i<ndof; ++i) p[i] = i;

        // Traverse elements to update the bandwidth.
        for(const auto& tri : mesh.triangles) {
            for(int i=0; i<6; ++i) {
                int u = tri.node_ids[i]; // Potential row.
                for(int j=0; j<6; ++j) {
                    int v = tri.node_ids[j]; // Potential column.
                    // If v < p[u], widen the profile.
                    if (v < static_cast<int>(p[u])) {
                        p[u] = v;
                    }
                }
            }
        }
        return p;
    }

std::vector<std::size_t> compute_profile_enhanced(const usim::MeshP2& mesh, const std::vector<int>& boundary_tags){

    // 1. Standard profile based on triangles.
    std::vector<std::size_t> p = compute_profile(mesh);

    // 2. Widen the profile for boundary terms.

    for (int tag : boundary_tags) {
        std::vector<int> boundary_nodes;
        // Traverse edges to find nodes on the boundary with the given tag.
        for(const auto& tri : mesh.triangles) {
            for(int i=0; i<3; ++i) {
                if(tri.edge_ref[i] == tag) {
                    // The three edge nodes (two vertices and one midpoint).
                    boundary_nodes.push_back(tri.node_ids[i]);
                    boundary_nodes.push_back(tri.node_ids[(i+1)%3]);
                    boundary_nodes.push_back(tri.node_ids[i+3]);
                }
            }
        }
        // Remove duplicates.
        std::sort(boundary_nodes.begin(), boundary_nodes.end());
        boundary_nodes.erase(std::unique(boundary_nodes.begin(), boundary_nodes.end()), boundary_nodes.end());

        // For every node pair (u, v) on this boundary, update the profile
        // because matrix T creates a non-zero coefficient between them.
        for (int u : boundary_nodes) {
            for (int v : boundary_nodes) {
                if (u > v) { // Store only the lower triangle.
                    if (v < static_cast<int>(p[u])) {
                        p[u] = v;
                    }
                }
            }
        }
    }
    return p;
}

double get_max_edge_length(const usim::MeshP2& mesh) {
    double h_max = 0.0;
    for (const auto& tri : mesh.triangles) {
        // Check the three main triangle edges (vertices 0-1, 1-2, 2-0).
        int vertices[3] = {tri.node_ids[0], tri.node_ids[1], tri.node_ids[2]};
        for (int i = 0; i < 3; ++i) {
            usim::Point2D p1 = mesh.nodes[vertices[i]];
            usim::Point2D p2 = mesh.nodes[vertices[(i + 1) % 3]];
            double dist = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
            if (dist > h_max) h_max = dist;
        }
    }
    return h_max;
}

void A_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& A, double factor){
    std::vector<QuadraturePoint> qp = get_quadrature_points();

    FullMatrix<double> dphi_ref(6,2);
    for (const auto& tri : mesh.triangles) {

        usim::Point2D p0 = mesh.nodes[tri.node_ids[0]];
        usim::Point2D p1 = mesh.nodes[tri.node_ids[1]];
        usim::Point2D p2 = mesh.nodes[tri.node_ids[2]];

        // Reference-to-physical map F(S) = B_l*S + b_l, with b_l = S0 and
        // B_l = [S1-S0, S2-S0].

        double J00 = p1.x-p0.x; double J01 = p2.x-p0.x;
        double J10 = p1.y-p0.y; double J11 = p2.y-p0.y;
        double detJac = J00*J11 - J01*J10;
        double invDet = 1.0 / detJac;
        
        double iJ00 =  J11 * invDet;
        double iJ01 = -J01 * invDet;
        double iJ10 = -J10 * invDet;
        double iJ11 =  J00 * invDet;

    // Loop over quadrature points.
        for (const auto& q : qp) {
            evaluate_gradients(q.x, q.y, dphi_ref);
            double weight = q.w * std::abs(detJac) * factor;

            double G[6][2];
            for(int i=0; i<6; ++i) {
                G[i][0] = dphi_ref(i,0)*iJ00 + dphi_ref(i,1)*iJ10;
                G[i][1] = dphi_ref(i,0)*iJ01 + dphi_ref(i,1)*iJ11;
            }
            
            for(int i=0; i<6; ++i) {
                for(int j=0; j<=i; ++j) {
                    double dot = G[i][0]*G[j][0] + G[i][1]*G[j][1];
                    A(tri.node_ids[i], tri.node_ids[j]) += complexe(dot * weight, 0.0);
                }
            }
        }
    }
}

void M_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& M, double factor){
    const auto qp = get_quadrature_points();
    std::vector<double> phi(6);

    for (const auto& tri : mesh.triangles) {
        const usim::Point2D p0 = mesh.nodes[tri.node_ids[0]];
        const usim::Point2D p1 = mesh.nodes[tri.node_ids[1]];
        const usim::Point2D p2 = mesh.nodes[tri.node_ids[2]];

        const double detJac =
            (p1.x-p0.x)*(p2.y-p0.y) - (p1.y-p0.y)*(p2.x-p0.x);

        for (const auto& q : qp) {
            evaluate_shape_functions(q.x, q.y, phi);
            const double weight = q.w * std::abs(detJac) * factor;

            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    M(tri.node_ids[i], tri.node_ids[j]) +=
                        complexe(phi[i] * phi[j] * weight, 0.0);
                }
            }
        }
    }
}

void B_matrix(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0, double k_d_val, double factor){
    assemble_B_matrix(mesh, B, factor, [=](const usim::TriangleP2& tri) {
        return tri.is_defect ? k_d_val : k0;
    });
}

void B_matrix_by_tag(const usim::MeshP2& mesh, ProfileMatrix<complexe>& B, double k0,
                     const std::map<int, double>& tag_contrasts, double factor) {
    assemble_B_matrix(mesh, B, factor, [&](const usim::TriangleP2& tri) {
        const auto match = tag_contrasts.find(tri.ref);
        const double contrast = (match == tag_contrasts.end()) ? 1.0 : match->second;
        return k0 / contrast;
    });
}

void B_matrix_from_nodal_sound_speed(const usim::MeshP2& mesh,
                                     ProfileMatrix<complexe>& B,
                                     double omega,
                                     const std::vector<double>& sound_speed,
                                     double factor) {
    if (sound_speed.size() != mesh.ndof()) {
        throw std::invalid_argument(
            "B_matrix_from_nodal_sound_speed: sound-speed size must equal mesh ndof");
    }
    if (!(omega > 0.0) || !std::isfinite(omega)) {
        throw std::invalid_argument(
            "B_matrix_from_nodal_sound_speed: omega must be finite and positive");
    }
    std::vector<double> squared_slowness(sound_speed.size());
    for (std::size_t index = 0; index < sound_speed.size(); ++index) {
        const double speed = sound_speed[index];
        if (!(speed > 0.0) || !std::isfinite(speed)) {
            throw std::invalid_argument(
                "B_matrix_from_nodal_sound_speed: sound speeds must be finite and positive");
        }
        squared_slowness[index] = 1.0 / (speed * speed);
    }

    const auto quadrature_points = get_quadrature_points();
    std::vector<double> phi(6);
    for (const auto& triangle : mesh.triangles) {
        const auto& p0 = mesh.nodes[triangle.node_ids[0]];
        const auto& p1 = mesh.nodes[triangle.node_ids[1]];
        const auto& p2 = mesh.nodes[triangle.node_ids[2]];
        const double determinant =
            (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);

        for (const auto& point : quadrature_points) {
            evaluate_shape_functions(point.x, point.y, phi);
            double material = 0.0;
            for (int local = 0; local < 6; ++local) {
                material += phi[local] * squared_slowness[triangle.node_ids[local]];
            }
            if (!(material > 0.0) || !std::isfinite(material)) {
                throw std::runtime_error(
                    "Non-positive interpolated squared slowness at a quadrature point");
            }
            const double weight = point.w * std::abs(determinant);
            const double coefficient = omega * omega * material * weight * factor;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    B(triangle.node_ids[i], triangle.node_ids[j]) +=
                        coefficient * phi[i] * phi[j];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// One-dimensional Gauss-Legendre quadrature on boundaries.
// -------------------------------------------------------------------------
std::vector<QuadraturePoint> get_quadrature_points_1d(){
    std::vector<QuadraturePoint> qp(3);
    double sqrt35 = std::sqrt(3.0/5.0);
    
    qp[0].x = -sqrt35; qp[0].w = 5.0/9.0;
    qp[1].x = 0.0;     qp[1].w = 8.0/9.0;
    qp[2].x = sqrt35;  qp[2].w = 5.0/9.0;
    
    return qp;
}

// phi[0]: t=-1 (left), phi[1]: t=1 (right), phi[2]: t=0 (midpoint).
void evaluate_shape_functions_1d(double t, std::vector<double>& phi){
    phi[0] = 0.5 * t * (t - 1.0); // Left endpoint.
    phi[1] = 0.5 * t * (t + 1.0); // Right endpoint.
    phi[2] = 1.0 - t * t;         // Midpoint.
}

double evaluate_c_1d(double y, double h, int n){
    if (n == 0) return std::sqrt(1.0 / h);
    return std::sqrt(2.0 / h) * std::cos(n * M_PI * y / h);
}

complexe compute_beta(double k0, double h, int n){
    return std::sqrt(complexe(k0*k0 - (M_PI*n/h)*(M_PI*n/h), 0.0));
}

FullMatrix<complexe> compute_E(const usim::MeshP2& mesh, int N_modes, int boundary_tag, [[maybe_unused]] double k0) {
    
    int ndof = mesh.ndof();
    // E is an (Ndof x N_modes) matrix.
    FullMatrix<complexe> E(ndof, N_modes); 

    auto qp = get_quadrature_points_1d(); // One-dimensional Gauss quadrature.
    std::vector<double> phi_1d(3);
    double h = mesh.Ly; // Waveguide height.

    // Traverse all triangles to find boundary edges.
    for (const auto& tri : mesh.triangles) {
        for (int edge_i = 0; edge_i < 3; ++edge_i) {
            
            // If the edge belongs to the requested boundary (for example, tag 1
            // for the left side or tag 2 for the right side).
            if (tri.edge_ref[edge_i] == boundary_tag) {
                
                // Local node indices for the edge endpoints (A, B) and midpoint (M).
                int idx_A = edge_i;
                int idx_B = (edge_i + 1) % 3;
                int idx_M = edge_i + 3;

                // Global matrix indices.
                int nodes_global[3] = {
                    tri.node_ids[idx_A],
                    tri.node_ids[idx_B], 
                    tri.node_ids[idx_M]
                };

                // Physical coordinates used to compute the edge length (Jacobian).
                usim::Point2D A = mesh.nodes[nodes_global[0]];
                usim::Point2D B = mesh.nodes[nodes_global[1]];
                
                // Edge length (vertical segment).
                double edge_length = std::sqrt(std::pow(B.x - A.x, 2) + std::pow(B.y - A.y, 2));
                double detJac = edge_length / 2.0;

                // Loop over integration points.
                for (const auto& q : qp) {
                    double t = q.x; // Reference coordinate in [-1, 1].
                    double w = q.w;
                    
                    // 1. One-dimensional shape functions at t.
                    evaluate_shape_functions_1d(t, phi_1d);

                    // 2. Physical Y coordinate at the integration point.

                    double y_phys = 0.5 * ((B.y + A.y) + t * (B.y - A.y)) - mesh.ymin; // Shift if the mesh is not centered.

                    // 3. Fill E.
                    for (int i = 0; i < 3; ++i) {
                        for (int n = 0; n < N_modes; ++n) {

                            double cn_val = evaluate_c_1d(y_phys, h, n);

                            std::complex<double> val = phi_1d[i] * cn_val * w * detJac;
                            
                            E(nodes_global[i], n) += val;
                        }
                    }
                }
            }
        }
    }
    return E;
}

void compute_D(FullMatrix<complexe>& D, int N_modes, double h, double k0){
        for (int j =0; j<N_modes; j++) {
            D(j,j) = std::complex<double>(0.0,1.0)* compute_beta(k0, h, j);
        }       
}

void T_matrix(ProfileMatrix<complexe>& K, FullMatrix<complexe>& E, FullMatrix<complexe>& D, [[maybe_unused]] double h, [[maybe_unused]] int boundary_tag, double factor){

    int Ndof = E.rows();
    int Nmodes = D.rows();

    // Optimization: identify active DOFs (those on the boundary).
    // E is sparse (non-zero only on the boundary), so avoid an N^2 loop.
    std::vector<int> active_dofs;
    active_dofs.reserve(Ndof / 10); // Estimate.
    for(int i=0; i<Ndof; ++i) {
        for(int n=0; n<Nmodes; ++n) {
            if(std::abs(E(i,n)) > 1e-14) { active_dofs.push_back(i); break; }
        }
    }

    for (int i : active_dofs) {
        
        for (int j : active_dofs) { 
            if (j > i) continue; // Symmetry: j <= i.
            complexe val_T_ij = 0.0;
            
            for (int n = 0; n < Nmodes; ++n) {
                // Compute on the fly to avoid a temporary ED matrix.
                val_T_ij += E(i, n) * D(n, n) * E(j, n);
            }

            if (std::abs(val_T_ij) > 1e-14) {
                K(i, j) += factor * val_T_ij;
            }
        }
    }
}

std::vector<complexe> assemble_source_vector(const usim::MeshP2& mesh, const FullMatrix<complexe>& E,int n_inc, double k0, double L, double coef) {

    int Ndof = mesh.ndof();
    std::vector<complexe> G(Ndof, 0.0);
    
    complexe beta = compute_beta(k0, mesh.Ly, n_inc);
    complexe coeff = -2.0 * std::complex<double>(0,1) * beta * std::exp(coef*std::complex<double>(0,1) * beta * L);

    // Fill the vector.

    for (int i = 0; i < Ndof; ++i) {
        G[i] = coeff * E(i, n_inc); 
    }
    return G;
}

} // namespace Fem
