#include "mesh.hpp"

namespace usim {

namespace detail {

std::string ltrim(std::string s){
    size_t i=0; while(i<s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}
std::string rtrim(std::string s){
    if(s.empty()) return s;
    size_t i=s.size();
    while(i>0 && std::isspace(static_cast<unsigned char>(s[i-1]))) --i;
    return s.substr(0,i);
}
std::string trim(std::string s){ return rtrim(ltrim(std::move(s))); }

bool starts_with(const std::string& s, const std::string& p){
    return s.size()>=p.size() && s.compare(0,p.size(),p)==0;
}

std::vector<long long> split_ll(const std::string& line){
    std::stringstream ss(line);
    std::vector<long long> out;
    long long v;
    while(ss>>v) out.push_back(v);
    return out;
}

struct EdgeKey {
    int a=0, b=0;
    EdgeKey() = default;
    EdgeKey(int i, int j){
        if(i<j){ a=i; b=j; }
        else { a=j; b=i; }
    }
    bool operator==(const EdgeKey& o) const { return a==o.a && b==o.b; }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const noexcept {
        return (static_cast<std::size_t>(e.a) << 32) ^ static_cast<std::size_t>(e.b);
    }
};

} // namespace detail

double MeshP2::compute_h_max(bool store) {
    if (triangles.empty() || nodes.empty()) {
        if (store) h_max = 0.0;
        return 0.0;
    }
    auto dist = [&](int ia, int ib) -> double {
        const auto& A = nodes[ia];
        const auto& B = nodes[ib];
        const double dx = A.x - B.x;
        const double dy = A.y - B.y;
        return std::sqrt(dx*dx + dy*dy);
    };
    double hm = 0.0;
    for (const auto& t : triangles) {
        const int i1 = t.node_ids[0];
        const int i2 = t.node_ids[1];
        const int i3 = t.node_ids[2];
        const double e12 = dist(i1, i2);
        const double e23 = dist(i2, i3);
        const double e31 = dist(i3, i1);
        const double ht = std::max(e12, std::max(e23, e31));
        if (ht > hm) hm = ht;
    }
    if (store) h_max = hm;
    return hm;
}

void MeshP2::compute_bbox_and_dims(){
    if(nodes.empty()){
        xmin = xmax = ymin = ymax = 0.0;
        Lx = Ly = 0.0;
        return;
    }
    xmin = xmax = nodes[0].x;
    ymin = ymax = nodes[0].y;
    for(const auto& p : nodes){
        if(p.x < xmin) xmin = p.x;
        if(p.x > xmax) xmax = p.x;
        if(p.y < ymin) ymin = p.y;
        if(p.y > ymax) ymax = p.y;
    }
    Lx = xmax - xmin;
    Ly = ymax - ymin;
}

void MeshP2::read_msh_v2_ascii(const std::string& filename, const std::vector<int>& defect_tags) {
    std::ifstream in(filename);
    if(!in) throw std::runtime_error("MeshP2: cannot open file: " + filename);
    std::string line;
    double msh_version = 2.2;
    bool is_binary = false;
    while(std::getline(in,line)){
        line = detail::trim(line);
        if(line=="$MeshFormat"){
            std::getline(in,line);
            {
                std::stringstream ss(line);
                ss >> msh_version;
                int file_type=0, data_size=0;
                ss >> file_type >> data_size;
                is_binary = (file_type==1);
            }
            while(std::getline(in,line)){
                line = detail::trim(line);
                if(line=="$EndMeshFormat") break;
            }
            break;
        }
        if(line=="$Nodes" || line=="$Elements"){
            in.clear();
            in.seekg(0, std::ios::beg);
            break;
        }
    }
    if(is_binary) throw std::runtime_error("MeshP2: binary .msh not supported (export ASCII).");
    if(msh_version >= 4.0){
        throw std::runtime_error("MeshP2: .msh version >= 4 detected. Export your mesh as 'MSH 2.2 ASCII' in Gmsh.");
    }
    in.clear();
    in.seekg(0, std::ios::beg);
    std::unordered_map<long long, int> tag_to_idx;
    nodes.clear();
    triangles.clear();
    std::unordered_map<detail::EdgeKey, int, detail::EdgeKeyHash> boundary_ref;
    while(std::getline(in,line)){
        line = detail::trim(line);
        if(line=="$Nodes") break;
    }
    if(!in) throw std::runtime_error("MeshP2: missing $Nodes section");
    std::getline(in,line);
    long long nb_nodes = std::stoll(detail::trim(line));
    nodes.reserve(static_cast<size_t>(nb_nodes));
    for(long long i=0;i<nb_nodes;i++){
        std::getline(in,line);
        long long tag = 0;
        double x=0.0, y=0.0, z=0.0;
        {
            std::stringstream ss(line);
            if(!(ss >> tag >> x >> y >> z)) throw std::runtime_error("MeshP2: invalid node line: " + line);
        }
        Point2D p; p.x=x; p.y=y; p.id=static_cast<int>(nodes.size());
        nodes.push_back(p);
        tag_to_idx[tag]=p.id;
    }
    while(std::getline(in,line)){
        line = detail::trim(line);
        if(line=="$EndNodes") break;
    }
    compute_bbox_and_dims();
    while(std::getline(in,line)){
        line = detail::trim(line);
        if(line=="$Elements") break;
    }
    if(!in) throw std::runtime_error("MeshP2: missing $Elements section");
    std::getline(in,line);
    long long nb_elems = std::stoll(detail::trim(line));
    struct TriP1Tmp{ int v1,v2,v3; int ref; };
    std::vector<TriP1Tmp> tri_p1;
    tri_p1.reserve(static_cast<size_t>(nb_elems));
    for(long long e=0;e<nb_elems;e++){
        std::getline(in,line);
        if(!in) break;
        auto toks = detail::split_ll(line);
        if(toks.size()<4) continue;
        int elm_type = static_cast<int>(toks[1]);
        int ntags    = static_cast<int>(toks[2]);
        int physical = 0;
        if(static_cast<int>(toks.size()) >= 3 + ntags + 1){
            if(ntags>0) physical = static_cast<int>(toks[3]);
        }
        int start = 3 + ntags;
        if(elm_type==1){
            int a = tag_to_idx.at(toks[start]);
            int b = tag_to_idx.at(toks[start+1]);
            boundary_ref[detail::EdgeKey(a,b)] = physical;
            nodes[a].ref = (nodes[a].ref==0? physical : nodes[a].ref);
            nodes[b].ref = (nodes[b].ref==0? physical : nodes[b].ref);
        } else if(elm_type==2){
            int v1 = tag_to_idx.at(toks[start]);
            int v2 = tag_to_idx.at(toks[start+1]);
            int v3 = tag_to_idx.at(toks[start+2]);
            tri_p1.push_back({v1,v2,v3,physical});
        }
    }
    std::unordered_map<detail::EdgeKey, int, detail::EdgeKeyHash> edge_mid;
    auto get_mid = [&](int a, int b)->int{
        detail::EdgeKey k(a,b);
        auto it = edge_mid.find(k);
        if(it!=edge_mid.end()) return it->second;
        const auto& pa = nodes[a];
        const auto& pb = nodes[b];
        Point2D pm;
        pm.x = 0.5*(pa.x+pb.x); pm.y = 0.5*(pa.y+pb.y);
        pm.id = static_cast<int>(nodes.size());
        auto itb = boundary_ref.find(k);
        if(itb!=boundary_ref.end()) pm.ref = itb->second;
        nodes.push_back(pm);
        edge_mid[k]=pm.id;
        return pm.id;
    };
    triangles.reserve(tri_p1.size());
    for(const auto& t : tri_p1){
        TriangleP2 T;
        T.ref = t.ref;
        T.node_ids[0]=t.v1; T.node_ids[1]=t.v2; T.node_ids[2]=t.v3;
        T.node_ids[3]=get_mid(t.v1,t.v2);
        T.node_ids[4]=get_mid(t.v2,t.v3);
        T.node_ids[5]=get_mid(t.v1,t.v3);
        auto e12 = boundary_ref.find(detail::EdgeKey(t.v1,t.v2));
        auto e23 = boundary_ref.find(detail::EdgeKey(t.v2,t.v3));
        auto e31 = boundary_ref.find(detail::EdgeKey(t.v3,t.v1));
        T.edge_ref[0] = (e12==boundary_ref.end()? 0 : e12->second);
        T.edge_ref[1] = (e23==boundary_ref.end()? 0 : e23->second);
        T.edge_ref[2] = (e31==boundary_ref.end()? 0 : e31->second);
        triangles.push_back(T);
    }
    set_defect_tags(defect_tags);
}

void MeshP2::set_defect_tags(const std::vector<int>& defect_tags){
    if(defect_tags.empty()){
        for(auto& t: triangles) t.is_defect=false;
        return;
    }
    std::unordered_map<int,bool> is_def;
    for(int r: defect_tags) is_def[r]=true;
    for(auto& t: triangles) t.is_defect = (is_def.find(t.ref)!=is_def.end());
}

void MeshP2::write_matlab_mesh_m(const std::string& out_m_file) const {
    std::ofstream out(out_m_file);
    if(!out) throw std::runtime_error("MeshP2: cannot write: " + out_m_file);
    out << "% Auto-generated by MeshP2::write_matlab_mesh_m\nCoor = [ ...\n";
    for(const auto& p: nodes) out << p.x << " " << p.y << ";\n";
    out << "];\n\nTri = [ ...\n";
    for(const auto& t: triangles)
        out << (t.node_ids[0]+1) << " " << (t.node_ids[1]+1) << " " << (t.node_ids[2]+1) << " "
            << (t.node_ids[3]+1) << " " << (t.node_ids[4]+1) << " " << (t.node_ids[5]+1) << ";\n";
    out << "];\n\nRefTri = [ ...\n";
    for(const auto& t: triangles) out << t.ref << ";\n";
    out << "];\n\nIsDef = zeros(" << triangles.size() << ",1);\n";
    for (size_t t = 0; t < triangles.size(); ++t) out << "IsDef(" << (t+1) << ") = " << (triangles[t].is_defect ? 1 : 0) << ";\n";
    out << "\nEdgeRef = [ ...\n";
    for(const auto& t: triangles) out << t.edge_ref[0] << " " << t.edge_ref[1] << " " << t.edge_ref[2] << ";\n";
    out << "];\n\n";
}

void MeshP2::write_defect_coords_txt(const std::string& filename) const {
    std::ofstream out(filename);
    out << "x,y\n";
    for(const auto& t : triangles) {
        if(t.is_defect) {
            for(int k=0; k<3; ++k) out << nodes[t.node_ids[k]].x << "," << nodes[t.node_ids[k]].y << "\n";
        }
    }
}

void MeshP2::write_matlab_field_m(const std::string& out_m_file, const std::vector<complexe>& U, const std::string& var_prefix) const {
    if(U.size()!=nodes.size()) throw std::invalid_argument("write_matlab_field_m: U size must equal Nbpt.");
    std::ofstream out(out_m_file);
    if(!out) throw std::runtime_error("MeshP2: cannot write: " + out_m_file);
    out << "% Auto-generated by MeshP2::write_matlab_field_m\n" << var_prefix << "re = [ ...\n";
    for(const auto& u: U) out << std::real(u) << ";\n";
    out << "];\n\n" << var_prefix << "im = [ ...\n";
    for(const auto& u: U) out << std::imag(u) << ";\n";
    out << "];\n\n" << var_prefix << "mag = sqrt("<<var_prefix<<"re.^2 + "<<var_prefix<<"im.^2);\n";
}

void MeshP2::write_point_cloud_txt(const std::string& out_txt, const std::vector<complexe>& U) const {
    if(U.size()!=nodes.size()) throw std::invalid_argument("write_point_cloud_txt: U size must equal Nbpt.");
    std::ofstream out(out_txt);
    if(!out) throw std::runtime_error("MeshP2: cannot write: " + out_txt);
    out << "# x y ReU ImU\n";
    for(std::size_t i=0;i<nodes.size();++i) out << nodes[i].x << " " << nodes[i].y << " " << std::real(U[i]) << " " << std::imag(U[i]) << "\n";
}

} // namespace usim
