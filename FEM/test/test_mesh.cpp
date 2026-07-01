#include "mesh.hpp"
using namespace usim;

int main(){
    MeshP2 mesh;

    // si ta zone "défaut" est taggée 2 dans Gmsh :
    mesh.read_msh_v2_ascii("test_ultrasound.msh", {2});

    mesh.write_matlab_mesh_m("mesh_out.m");

    // si tu as un champ nodal complexe U (taille = mesh.nodes.size()):
    // std::vector<complexe> U(mesh.ndof());
    // mesh.write_matlab_field_m("field_out.m", U);

    return 0;
}
