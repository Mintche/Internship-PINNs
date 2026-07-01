#ifndef MESH_HPP
#define MESH_HPP

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "math.hpp"

namespace usim {

struct Point2D {
    double x = 0.0;
    double y = 0.0;
    int    id = -1;   // global index (0-based)
    int    ref = 0;   // optional node reference (0 if none)
};

struct TriangleP2 {
    // Local ordering compatible with the statement:
    // 1-2-3 are vertices, 4 is midpoint(1,2), 5 is midpoint(2,3), 6 is midpoint(1,3).
    std::array<int,6> node_ids{{-1,-1,-1,-1,-1,-1}};
    int ref = 0;                // triangle "physical" tag (e.g. defect zone id)
    bool is_defect = false;     // user can set from ref via set_defect_tags(...)
    std::array<int,3> edge_ref{{0,0,0}}; // for edges (1,2), (2,3), (3,1): boundary tag or 0
};

class MeshP2 {
public:
    // Public data for simplicity (project-style).
    std::vector<Point2D>    nodes;      // includes generated mid-edge nodes
    std::vector<TriangleP2> triangles;  // P2 triangles

    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double Lx   = 0.0, Ly   = 0.0;
    double h_max = 0.0;  // longueur caractéristique max (calculée)

    double compute_h_max(bool store = true);
    void compute_bbox_and_dims();

    // Read a .msh (MSH v2 ASCII) and enrich it to P2.
    // defect_tags: if not empty, triangles with ref in defect_tags are marked as defect.
    void read_msh_v2_ascii(const std::string& filename, const std::vector<int>& defect_tags = {});

    // Mark defects from triangle refs
    void set_defect_tags(const std::vector<int>& defect_tags);

    // Convenience: number of P2 dofs (one per node in this simplified project)
    std::size_t ndof() const { return nodes.size(); }

    // Export to a Matlab/Octave .m script (mesh only).
    // Creates variables:
    //   Coor: (Nbpt x 2)
    //   Tri : (Nbtri x 6)   (1-based indices for Matlab)
    //   RefTri: (Nbtri x 1)
    //   EdgeRef: (Nbtri x 3)
    void write_matlab_mesh_m(const std::string& out_m_file) const;

    // Export des triangles du défaut pour Python (format: x1 y1 x2 y2 x3 y3)
    void write_defect_coords_txt(const std::string& filename) const;

    // Export nodal complex field to Matlab/Octave .m script.
    // Creates variables:
    //   Ure, Uim (Nbpt x 1), and optionally Umag.
    void write_matlab_field_m(const std::string& out_m_file,
                              const std::vector<complexe>& U,
                              const std::string& var_prefix="U") const;

    // Export a simple ASCII point cloud: x y Re(U) Im(U)
    void write_point_cloud_txt(const std::string& out_txt,
                               const std::vector<complexe>& U) const;
};

}

#endif // MESH_HPP
