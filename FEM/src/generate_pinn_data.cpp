#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fem.hpp"

using namespace std;
using namespace usim;

namespace {

constexpr int defect_tag = 2;
constexpr int left_tag = 11;
constexpr int right_tag = 12;
constexpr double pollution_eps = 0.01;

struct TagContrast {
    int tag = 0;
    double contrast = 1.0;
};

struct Configuration {
    filesystem::path mesh_file;
    filesystem::path output_dir;
    string defect_name;
    double c0 = 0.0;
    optional<double> legacy_contrast_ratio;
    vector<TagContrast> tag_contrasts;
    optional<filesystem::path> nodal_sound_speed_file;
    vector<double> frequencies;
    vector<int> modes;
    vector<int> incidences{-1};
    optional<size_t> number_of_data_points;
};

struct BoundaryEdge {
    int first = -1;
    int second = -1;
    int midpoint = -1;
};

struct BoundaryPort {
    vector<BoundaryEdge> edges;
    vector<int> node_ids;
    double x = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
};

struct BoundaryEvaluation {
    double x = 0.0;
    double y = 0.0;
    array<int, 3> node_ids{{-1, -1, -1}};
    array<double, 3> weights{{0.0, 0.0, 0.0}};
};

void print_usage(const char* executable) {
    cout
        << "Usage:\n  " << executable
        << " --mesh MESH --defectname NAME --freqs F1,F2 --modes M1,M2"
        << " --outputdir DIR --c0 VALUE MATERIAL [options]\n\n"
        << "Arguments:\n"
        << "  --mesh PATH                  Maillage Gmsh v2 ASCII\n"
        << "  --defectname NAME            Nom utilise dans les fichiers CSV\n"
        << "  --freqs F1,F2                Frequences positives en Hz\n"
        << "  --modes M1,M2                Indices de modes positifs ou nuls\n"
        << "  --outputdir DIR              Dossier de sortie\n"
        << "  --c0 VALUE                   Celerite du milieu sain\n"
        << "  --contrast RATIO             Rapport c_defaut / c0 pour le tag 2 (ancien format)\n"
        << "  --tag-contrasts T:R,T:R      Rapports c_tag / c0 par tag physique de surface\n"
        << "  --nodal-sound-speed PATH     CSV node_id,x,y,c dans l'ordre P2 apres RCM\n"
        << "  --incidence VALUES           -1 pour gauche, 1 pour droite, ou -1,1 (defaut: -1)\n"
        << "  --numberofdatapoints N       N points uniformes par bord, N >= 2\n"
        << "  --help                       Afficher cette aide\n\n"
        << "Sans --numberofdatapoints, tous les degres de liberte P2 des ports sont exportes.\n\n"
        << "Exemple:\n  " << executable
        << " --mesh data/test_us_barhalfup_centree.msh --defectname barhalfup"
        << " --freqs 600,800 --modes 0,1,2 --outputdir ../waveguide_2d/data"
        << " --c0 340 --contrast 0.8 --incidence -1,1 --numberofdatapoints 31\n  " << executable
        << " --mesh data/two_zone.msh --defectname barhalf_sym"
        << " --freqs 600,800 --modes 0,1,2 --outputdir ../waveguide_2d/data"
        << " --c0 340 --tag-contrasts 2:0.8,3:0.9 --numberofdatapoints 31\n";
}

string trim(string value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == string::npos) return {};
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

double parse_double(const string& text, const string& option) {
    const string value = trim(text);
    size_t consumed = 0;
    double result = 0.0;
    try {
        result = stod(value, &consumed);
    } catch (const exception&) {
        throw invalid_argument("Valeur invalide pour " + option + ": " + text);
    }
    if (consumed != value.size() || !isfinite(result)) {
        throw invalid_argument("Valeur invalide pour " + option + ": " + text);
    }
    return result;
}

int parse_int(const string& text, const string& option) {
    const string value = trim(text);
    size_t consumed = 0;
    long long result = 0;
    try {
        result = stoll(value, &consumed);
    } catch (const exception&) {
        throw invalid_argument("Valeur invalide pour " + option + ": " + text);
    }
    if (consumed != value.size() || result < numeric_limits<int>::min() ||
        result > numeric_limits<int>::max()) {
        throw invalid_argument("Valeur invalide pour " + option + ": " + text);
    }
    return static_cast<int>(result);
}

template <typename T, typename Converter>
vector<T> parse_list(const string& text, Converter converter, const string& option) {
    vector<T> values;
    string token;
    const string list_text = trim(text);
    if (list_text.empty()) throw invalid_argument("La liste " + option + " est vide");
    if (list_text.back() == ',') throw invalid_argument("Valeur vide dans " + option);

    stringstream stream(list_text);
    while (getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) throw invalid_argument("Valeur vide dans " + option);
        values.push_back(converter(token));
    }
    if (values.empty()) throw invalid_argument("La liste " + option + " est vide");
    return values;
}

string sanitize_name(string name) {
    for (char& character : name) {
        const bool valid = (character >= 'a' && character <= 'z') ||
                           (character >= 'A' && character <= 'Z') ||
                           (character >= '0' && character <= '9') ||
                           character == '-' || character == '_';
        if (!valid) character = '_';
    }
    if (name.empty()) throw invalid_argument("--defectname ne peut pas etre vide");
    return name;
}

string ratio_label(double ratio) {
    ostringstream stream;
    stream << fixed << setprecision(15) << ratio;
    string token = stream.str();
    while (!token.empty() && token.back() == '0') token.pop_back();
    if (!token.empty() && token.back() == '.') token.pop_back();
    replace(token.begin(), token.end(), '.', 'p');
    return "ratio" + token;
}

vector<TagContrast> parse_tag_contrasts(const string& text, const string& option) {
    vector<TagContrast> contrasts;
    set<int> seen_tags;
    string token;
    stringstream stream(text);
    while (getline(stream, token, ',')) {
        token = trim(token);
        if (token.empty()) throw invalid_argument("Valeur vide dans " + option);

        const size_t separator = token.find(':');
        if (separator == string::npos || separator == 0 ||
            separator == token.size() - 1 || token.find(':', separator + 1) != string::npos) {
            throw invalid_argument("Format attendu pour " + option + ": tag:rapport,tag:rapport");
        }

        const int tag = parse_int(token.substr(0, separator), option);
        const double contrast = parse_double(token.substr(separator + 1), option);
        if (tag <= 0) throw invalid_argument("Les tags de " + option + " doivent etre strictement positifs");
        if (tag == 1) {
            throw invalid_argument("Le tag 1 est reserve au milieu sain et utilise --c0");
        }
        if (!(contrast > 0.0)) {
            throw invalid_argument("Les rapports de " + option + " doivent etre strictement positifs");
        }
        if (!seen_tags.insert(tag).second) {
            throw invalid_argument("Tag duplique dans " + option + ": " + to_string(tag));
        }
        contrasts.push_back({tag, contrast});
    }
    if (contrasts.empty()) throw invalid_argument("La liste " + option + " est vide");
    sort(contrasts.begin(), contrasts.end(), [](const TagContrast& a, const TagContrast& b) {
        return a.tag < b.tag;
    });
    return contrasts;
}

vector<int> parse_incidences(const string& text, const string& option) {
    vector<int> incidences = parse_list<int>(
        text,
        [&](const string& value) { return parse_int(value, option); }, option);
    set<int> seen;
    for (int incidence : incidences) {
        if (incidence != -1 && incidence != 1) {
            throw invalid_argument("--incidence accepte uniquement -1 ou 1");
        }
        if (!seen.insert(incidence).second) {
            throw invalid_argument("Incidence dupliquee dans --incidence: " +
                                   to_string(incidence));
        }
    }
    sort(incidences.begin(), incidences.end());
    return incidences;
}

vector<int> material_tags(const Configuration& config) {
    vector<int> tags;
    tags.reserve(config.tag_contrasts.size());
    for (const auto& tag_contrast : config.tag_contrasts) {
        tags.push_back(tag_contrast.tag);
    }
    return tags;
}

vector<double> load_nodal_sound_speed(const filesystem::path& path,
                                      const MeshP2& mesh) {
    ifstream input(path);
    if (!input) throw runtime_error("Impossible d'ouvrir " + path.string());
    string line;
    if (!getline(input, line) || trim(line) != "node_id,x,y,c") {
        throw runtime_error(
            "Le CSV nodal doit commencer par l'en-tete node_id,x,y,c");
    }

    vector<double> values(mesh.ndof(), numeric_limits<double>::quiet_NaN());
    vector<bool> seen(mesh.ndof(), false);
    size_t row_count = 0;
    while (getline(input, line)) {
        line = trim(line);
        if (line.empty()) continue;
        replace(line.begin(), line.end(), ',', ' ');
        stringstream stream(line);
        int node_id = -1;
        double x = 0.0;
        double y = 0.0;
        double speed = 0.0;
        string extra;
        if (!(stream >> node_id >> x >> y >> speed) || (stream >> extra)) {
            throw runtime_error("Ligne nodale invalide: " + line);
        }
        if (node_id < 0 || static_cast<size_t>(node_id) >= mesh.ndof()) {
            throw runtime_error("node_id nodal hors maillage: " + to_string(node_id));
        }
        if (seen[node_id]) {
            throw runtime_error("node_id nodal duplique: " + to_string(node_id));
        }
        const auto& node = mesh.nodes[node_id];
        const double coordinate_tolerance = 1e-9 * max({1.0, mesh.Lx, mesh.Ly});
        if (abs(x - node.x) > coordinate_tolerance || abs(y - node.y) > coordinate_tolerance) {
            throw runtime_error(
                "Coordonnees incompatibles pour le node_id " + to_string(node_id));
        }
        if (!(speed > 0.0) || !isfinite(speed)) {
            throw runtime_error(
                "Celerite nodale non positive ou non finie au node_id " +
                to_string(node_id));
        }
        values[node_id] = speed;
        seen[node_id] = true;
        ++row_count;
    }
    if (row_count != mesh.ndof()) {
        throw runtime_error(
            "Le CSV nodal contient " + to_string(row_count) + " lignes pour " +
            to_string(mesh.ndof()) + " ddl");
    }
    return values;
}

map<int, double> tag_contrast_map(const Configuration& config) {
    map<int, double> contrasts;
    for (const auto& tag_contrast : config.tag_contrasts) {
        contrasts[tag_contrast.tag] = tag_contrast.contrast;
    }
    return contrasts;
}

string data_suffix(const Configuration& config) {
    if (config.legacy_contrast_ratio.has_value()) {
        return config.defect_name + "_" + ratio_label(*config.legacy_contrast_ratio);
    }
    return config.defect_name;
}

Configuration parse_arguments(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        throw invalid_argument("Arguments manquants");
    }

    Configuration config;
    bool has_mesh = false;
    bool has_defect_name = false;
    bool has_frequencies = false;
    bool has_modes = false;
    bool has_output_dir = false;
    bool has_c0 = false;
    bool has_contrast = false;
    bool has_tag_contrasts = false;
    bool has_nodal_sound_speed = false;

    auto value_after = [&](int& index, const string& option) -> string {
        if (index + 1 >= argc) throw invalid_argument("Valeur manquante apres " + option);
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const string option = argv[i];
        if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (option == "--mesh") {
            config.mesh_file = value_after(i, option);
            has_mesh = true;
        } else if (option == "--defectname") {
            config.defect_name = sanitize_name(value_after(i, option));
            has_defect_name = true;
        } else if (option == "--freqs") {
            config.frequencies = parse_list<double>(
                value_after(i, option),
                [&](const string& value) { return parse_double(value, option); }, option);
            has_frequencies = true;
        } else if (option == "--modes") {
            config.modes = parse_list<int>(
                value_after(i, option),
                [&](const string& value) { return parse_int(value, option); }, option);
            has_modes = true;
        } else if (option == "--outputdir") {
            config.output_dir = value_after(i, option);
            has_output_dir = true;
        } else if (option == "--c0") {
            config.c0 = parse_double(value_after(i, option), option);
            has_c0 = true;
        } else if (option == "--contrast") {
            config.legacy_contrast_ratio = parse_double(value_after(i, option), option);
            has_contrast = true;
        } else if (option == "--tag-contrasts") {
            config.tag_contrasts = parse_tag_contrasts(value_after(i, option), option);
            has_tag_contrasts = true;
        } else if (option == "--nodal-sound-speed") {
            config.nodal_sound_speed_file = value_after(i, option);
            has_nodal_sound_speed = true;
        } else if (option == "--incidence") {
            config.incidences = parse_incidences(value_after(i, option), option);
        } else if (option == "--numberofdatapoints") {
            const int count = parse_int(value_after(i, option), option);
            if (count < 2) throw invalid_argument("--numberofdatapoints doit etre >= 2");
            config.number_of_data_points = static_cast<size_t>(count);
        } else {
            throw invalid_argument("Option inconnue: " + option);
        }
    }

    if (!has_mesh) throw invalid_argument("--mesh est obligatoire");
    if (!has_defect_name) throw invalid_argument("--defectname est obligatoire");
    if (!has_frequencies) throw invalid_argument("--freqs est obligatoire");
    if (!has_modes) throw invalid_argument("--modes est obligatoire");
    if (!has_output_dir) throw invalid_argument("--outputdir est obligatoire");
    if (!has_c0) throw invalid_argument("--c0 est obligatoire");
    const int material_option_count = static_cast<int>(has_contrast) +
                                      static_cast<int>(has_tag_contrasts) +
                                      static_cast<int>(has_nodal_sound_speed);
    if (material_option_count != 1) {
        throw invalid_argument(
            "Fournir exactement un de --contrast ou --tag-contrasts, "
            "ou utiliser --nodal-sound-speed exclusivement");
    }
    if (!(config.c0 > 0.0)) throw invalid_argument("--c0 doit etre strictement positif");
    if (has_contrast && !(*config.legacy_contrast_ratio > 0.0)) {
        throw invalid_argument("--contrast doit etre un rapport strictement positif");
    }
    if (has_contrast) {
        config.tag_contrasts = {{defect_tag, *config.legacy_contrast_ratio}};
    }
    for (double frequency : config.frequencies) {
        if (!(frequency > 0.0)) throw invalid_argument("Toutes les frequences doivent etre positives");
    }
    for (int mode : config.modes) {
        if (mode < 0) throw invalid_argument("Les indices de mode doivent etre positifs ou nuls");
    }

    sort(config.frequencies.begin(), config.frequencies.end());
    config.frequencies.erase(unique(config.frequencies.begin(), config.frequencies.end()),
                             config.frequencies.end());
    sort(config.modes.begin(), config.modes.end());
    config.modes.erase(unique(config.modes.begin(), config.modes.end()), config.modes.end());
    return config;
}

ofstream open_output(const filesystem::path& path) {
    ofstream output(path);
    if (!output) throw runtime_error("Impossible de creer " + path.string());
    output << setprecision(17);
    return output;
}

BoundaryPort collect_boundary_port(const MeshP2& mesh, int boundary_tag, double expected_x) {
    BoundaryPort port;
    port.x = expected_x;
    set<pair<int, int>> visited_edges;
    set<int> unique_nodes;

    const double scale = max({1.0, abs(mesh.Lx), abs(mesh.Ly)});
    const double tolerance = 1e-10 * scale;

    for (const auto& triangle : mesh.triangles) {
        for (int edge = 0; edge < 3; ++edge) {
            if (triangle.edge_ref[edge] != boundary_tag) continue;
            const int first = triangle.node_ids[edge];
            const int second = triangle.node_ids[(edge + 1) % 3];
            const int midpoint = triangle.node_ids[edge + 3];
            const pair<int, int> key = minmax(first, second);
            if (!visited_edges.insert(key).second) continue;

            const auto& p0 = mesh.nodes[first];
            const auto& p1 = mesh.nodes[second];
            const auto& pm = mesh.nodes[midpoint];
            if (abs(p0.x - expected_x) > tolerance || abs(p1.x - expected_x) > tolerance ||
                abs(pm.x - expected_x) > tolerance) {
                throw runtime_error("Le port tague " + to_string(boundary_tag) +
                                    " n'est pas vertical a l'extremite du maillage");
            }
            if (abs(p1.y - p0.y) <= tolerance) {
                throw runtime_error("Une arete du port tague " + to_string(boundary_tag) +
                                    " a une hauteur nulle");
            }
            port.edges.push_back({first, second, midpoint});
            unique_nodes.insert(first);
            unique_nodes.insert(second);
            unique_nodes.insert(midpoint);
        }
    }

    if (port.edges.empty()) {
        throw runtime_error("Aucune arete trouvee pour le tag de port " +
                            to_string(boundary_tag));
    }

    port.node_ids.assign(unique_nodes.begin(), unique_nodes.end());
    sort(port.node_ids.begin(), port.node_ids.end(), [&](int first, int second) {
        const auto& a = mesh.nodes[first];
        const auto& b = mesh.nodes[second];
        if (a.y != b.y) return a.y < b.y;
        return first < second;
    });
    port.ymin = mesh.nodes[port.node_ids.front()].y;
    port.ymax = mesh.nodes[port.node_ids.back()].y;
    return port;
}

vector<BoundaryEvaluation> prepare_boundary_evaluations(
    const MeshP2& mesh, const BoundaryPort& port, optional<size_t> requested_count) {
    vector<BoundaryEvaluation> evaluations;

    if (!requested_count.has_value()) {
        evaluations.reserve(port.node_ids.size());
        for (int node_id : port.node_ids) {
            const auto& node = mesh.nodes[node_id];
            evaluations.push_back({node.x, node.y, {node_id, node_id, node_id}, {1.0, 0.0, 0.0}});
        }
        return evaluations;
    }

    const size_t count = *requested_count;
    evaluations.reserve(count);
    const double scale = max({1.0, abs(mesh.Lx), abs(mesh.Ly)});
    const double tolerance = 1e-10 * scale;

    for (size_t index = 0; index < count; ++index) {
        const double fraction = static_cast<double>(index) / static_cast<double>(count - 1);
        const double y = port.ymin + fraction * (port.ymax - port.ymin);
        bool found = false;

        for (const auto& edge : port.edges) {
            const auto& p0 = mesh.nodes[edge.first];
            const auto& p1 = mesh.nodes[edge.second];
            const double edge_ymin = min(p0.y, p1.y);
            const double edge_ymax = max(p0.y, p1.y);
            if (y < edge_ymin - tolerance || y > edge_ymax + tolerance) continue;

            double t = (y - p0.y) / (p1.y - p0.y);
            t = clamp(t, 0.0, 1.0);
            const double n0 = (1.0 - t) * (1.0 - 2.0 * t);
            const double n1 = t * (2.0 * t - 1.0);
            const double nm = 4.0 * t * (1.0 - t);
            const double x = (1.0 - t) * p0.x + t * p1.x;
            evaluations.push_back({x, y, {edge.first, edge.second, edge.midpoint},
                                   {n0, n1, nm}});
            found = true;
            break;
        }

        if (!found) {
            throw runtime_error("Impossible d'interpoler le point y=" + to_string(y) +
                                " sur le port x=" + to_string(port.x));
        }
    }
    return evaluations;
}

complexe evaluate_boundary(const vector<complexe>& field,
                           const BoundaryEvaluation& evaluation) {
    complexe value = 0.0;
    for (size_t i = 0; i < evaluation.node_ids.size(); ++i) {
        value += evaluation.weights[i] * field[evaluation.node_ids[i]];
    }
    return value;
}

void export_boundary(ofstream& output, const vector<BoundaryEvaluation>& evaluations,
                     const vector<complexe>& field, int incidence, double frequency,
                     double k0, int mode) {
    for (const auto& evaluation : evaluations) {
        const complexe value = evaluate_boundary(field, evaluation);
        output << incidence << ',' << frequency << ',' << k0 << ',' << mode << ','
               << evaluation.x << ',' << evaluation.y << ',' << real(value) << ','
               << imag(value) << '\n';
    }
}

void export_field(ofstream& output, const MeshP2& mesh, const vector<complexe>& field,
                  int incidence, double frequency, double k0, int mode) {
    for (const auto& node : mesh.nodes) {
        output << incidence << ',' << frequency << ',' << k0 << ',' << mode << ','
               << node.id << ',' << node.x << ',' << node.y << ','
               << real(field[node.id]) << ',' << imag(field[node.id]) << '\n';
    }
}

void export_mesh_rcm(const filesystem::path& output_dir, const string& defect_name,
                     const MeshP2& mesh) {
    auto node_output =
        open_output(output_dir / ("mesh_nodes_" + defect_name + ".csv"));
    node_output << "node_id,x,y,ref\n";
    for (const auto& node : mesh.nodes) {
        node_output << node.id << ',' << node.x << ',' << node.y << ','
                    << node.ref << '\n';
    }

    auto triangle_output =
        open_output(output_dir / ("mesh_triangles_" + defect_name + ".csv"));
    triangle_output
        << "triangle_id,n0,n1,n2,n3,n4,n5,ref,is_defect,edge_ref0,edge_ref1,edge_ref2\n";
    for (size_t triangle_id = 0; triangle_id < mesh.triangles.size(); ++triangle_id) {
        const auto& triangle = mesh.triangles[triangle_id];
        triangle_output << triangle_id;
        for (int node_id : triangle.node_ids) {
            triangle_output << ',' << node_id;
        }
        triangle_output << ',' << triangle.ref << ',' << (triangle.is_defect ? 1 : 0)
                        << ',' << triangle.edge_ref[0] << ','
                        << triangle.edge_ref[1] << ',' << triangle.edge_ref[2]
                        << '\n';
    }
}

void export_lower_matrix(const filesystem::path& path, const ProfileMatrix<complexe>& matrix,
                         const vector<size_t>& profile) {
    auto output = open_output(path);
    output << "row,col,value\n";
    for (size_t row = 0; row < profile.size(); ++row) {
        for (size_t column = profile[row]; column <= row; ++column) {
            const complexe value = matrix(static_cast<int>(row), static_cast<int>(column));
            if (abs(value) == 0.0) continue;
            const double real_value = real(value);
            const double imag_tolerance = 1e-13 * max(1.0, abs(real_value));
            if (abs(imag(value)) > imag_tolerance) {
                throw runtime_error("Une matrice geometrique contient une partie imaginaire non nulle");
            }
            output << row << ',' << column << ',' << real_value << '\n';
        }
    }
}

void warn_if_polluted(double frequency, const string& label, double k, double h) {
    const double kh = k * h;
    const double kh_squared = kh * kh;
    const double kh_fourth = kh_squared * kh_squared;
    if (kh_squared + k * kh_fourth < pollution_eps) return;

    cerr << "Attention: pollution numerique possible pour " << label
         << " a f=" << frequency << " Hz: condition (kh)^2+k(kh)^4 < "
         << pollution_eps << " non verifiee, (kh)^2=" << kh_squared
         << ", k(kh)^4=" << k*kh_fourth << ". Simulation poursuivie.\n";
}

void validate_waveguide_geometry(const MeshP2& mesh, const BoundaryPort& left,
                                 const BoundaryPort& right) {
    if (!(mesh.Lx > 0.0 && mesh.Ly > 0.0)) {
        throw runtime_error("Le maillage doit avoir des dimensions strictement positives");
    }
    const double scale = max({1.0, abs(mesh.Lx), abs(mesh.Ly)});
    const double tolerance = 1e-10 * scale;
    if (abs(mesh.xmin + mesh.xmax) > tolerance) {
        throw runtime_error("Le guide doit etre centre avec des ports en x=-L et x=+L");
    }
    if (abs(left.ymin - mesh.ymin) > tolerance || abs(left.ymax - mesh.ymax) > tolerance ||
        abs(right.ymin - mesh.ymin) > tolerance || abs(right.ymax - mesh.ymax) > tolerance) {
        throw runtime_error("Les ports doivent couvrir toute la hauteur du guide");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Configuration config = parse_arguments(argc, argv);
        filesystem::create_directories(config.output_dir);

        MeshP2 mesh;
        mesh.read_msh_v2_ascii(config.mesh_file.string(), material_tags(config));
        Fem::reorder_mesh_rcm(mesh);
        const double h_max = mesh.compute_h_max();
        export_mesh_rcm(config.output_dir, config.defect_name, mesh);
        const optional<vector<double>> nodal_sound_speed =
            config.nodal_sound_speed_file.has_value()
                ? optional<vector<double>>(
                      load_nodal_sound_speed(*config.nodal_sound_speed_file, mesh))
                : nullopt;

        const BoundaryPort left_port = collect_boundary_port(mesh, left_tag, mesh.xmin);
        const BoundaryPort right_port = collect_boundary_port(mesh, right_tag, mesh.xmax);
        validate_waveguide_geometry(mesh, left_port, right_port);

        const auto left_evaluations =
            prepare_boundary_evaluations(mesh, left_port, config.number_of_data_points);
        const auto right_evaluations =
            prepare_boundary_evaluations(mesh, right_port, config.number_of_data_points);

        cout << "Maillage: " << mesh.ndof() << " ddl P2, " << mesh.triangles.size()
             << " triangles, ports " << left_evaluations.size() << "/"
             << right_evaluations.size() << " points, h_max=" << h_max << ".\n";

        const vector<size_t> profile =
            Fem::compute_profile_enhanced(mesh, {left_tag, right_tag});
        ProfileMatrix<complexe> stiffness(profile);
        ProfileMatrix<complexe> mass(profile);
        Fem::A_matrix(mesh, stiffness, 1.0);
        Fem::M_matrix(mesh, mass, 1.0);

        export_lower_matrix(config.output_dir /
                                ("Stiff_matrix_" + config.defect_name + ".csv"),
                            stiffness, profile);
        export_lower_matrix(config.output_dir /
                                ("Mass_matrix_" + config.defect_name + ".csv"),
                            mass, profile);

        const string output_suffix = data_suffix(config);
        auto left_output = open_output(config.output_dir /
                                       ("pinn_boundary_left_" + output_suffix + ".csv"));
        auto right_output = open_output(config.output_dir /
                                        ("pinn_boundary_right_" + output_suffix + ".csv"));
        auto field_output = open_output(config.output_dir /
                                        ("fem_field_" + output_suffix + ".csv"));
        left_output << "incidence,f,k0,mode,x,y,Re_U,Im_U\n";
        right_output << "incidence,f,k0,mode,x,y,Re_U,Im_U\n";
        field_output << "incidence,f,k0,mode,node_id,x,y,Re_U,Im_U\n";

        const int highest_requested_mode = *max_element(config.modes.begin(), config.modes.end());
        const double pi = acos(-1.0);
        const map<int, double> contrasts_by_tag = tag_contrast_map(config);

        for (double frequency : config.frequencies) {
            const double k0 = 2.0 * pi * frequency / config.c0;
            const double omega = 2.0 * pi * frequency;
            const int number_of_dtn_modes =
                max(static_cast<int>(floor(mesh.Ly * k0 / pi)) + 5,
                    highest_requested_mode + 1);

            warn_if_polluted(frequency, "milieu sain", k0, h_max);
            if (nodal_sound_speed.has_value()) {
                const double min_speed =
                    *min_element(nodal_sound_speed->begin(), nodal_sound_speed->end());
                const double material_k = omega / min_speed;
                const double duplicate_tolerance = 1e-12 * max(1.0, abs(k0));
                if (abs(material_k - k0) > duplicate_tolerance) {
                    warn_if_polluted(frequency, "materiau nodal", material_k, h_max);
                }
            } else {
                for (const auto& tag_contrast : config.tag_contrasts) {
                    const double material_k = k0 / tag_contrast.contrast;
                    const double duplicate_tolerance = 1e-12 * max(1.0, abs(k0));
                    if (abs(material_k - k0) > duplicate_tolerance) {
                        warn_if_polluted(
                            frequency,
                            "tag " + to_string(tag_contrast.tag),
                            material_k,
                            h_max);
                    }
                }
            }

            cout << "Resolution f=" << frequency << " Hz, k0=" << k0
                 << ", " << number_of_dtn_modes << " modes DtN...\n";

            ProfileMatrix<complexe> system = stiffness;
            if (nodal_sound_speed.has_value()) {
                Fem::B_matrix_from_nodal_sound_speed(
                    mesh, system, omega, *nodal_sound_speed, -1.0);
            } else if (config.legacy_contrast_ratio.has_value()) {
                const double kd = k0 / *config.legacy_contrast_ratio;
                Fem::B_matrix(mesh, system, k0, kd, -1.0);
            } else {
                Fem::B_matrix_by_tag(mesh, system, k0, contrasts_by_tag, -1.0);
            }

            FullMatrix<complexe> e_left =
                Fem::compute_E(mesh, number_of_dtn_modes, left_tag, k0);
            FullMatrix<complexe> e_right =
                Fem::compute_E(mesh, number_of_dtn_modes, right_tag, k0);
            FullMatrix<complexe> dtn(number_of_dtn_modes, number_of_dtn_modes);
            Fem::compute_D(dtn, number_of_dtn_modes, mesh.Ly, k0);
            Fem::T_matrix(system, e_left, dtn, mesh.Ly, left_tag, -1.0);
            Fem::T_matrix(system, e_right, dtn, mesh.Ly, right_tag, -1.0);
            system.factorize();

            for (int mode : config.modes) {
                const complexe beta = Fem::compute_beta(k0, mesh.Ly, mode);
                if (abs(imag(beta)) > 1e-12) {
                    cerr << "Attention: le mode incident " << mode << " est evanescent a "
                         << frequency << " Hz.\n";
                }

                for (int incidence : config.incidences) {
                    const bool from_left = (incidence == -1);
                    const FullMatrix<complexe>& source_e = from_left ? e_left : e_right;
                    const double source_x = from_left ? mesh.xmin : mesh.xmax;
                    const double phase_sign = from_left ? 1.0 : -1.0;
                    const vector<complexe> source =
                        Fem::assemble_source_vector(
                            mesh, source_e, mode, k0, source_x, phase_sign);
                    vector<complexe> field(mesh.ndof());
                    system.solve(field, source);

                    export_boundary(left_output, left_evaluations, field,
                                    incidence, frequency, k0, mode);
                    export_boundary(right_output, right_evaluations, field,
                                    incidence, frequency, k0, mode);
                    export_field(field_output, mesh, field,
                                 incidence, frequency, k0, mode);
                }
            }
        }

        cout << "Donnees FEM/PINN generees dans " << config.output_dir << '\n';
        return 0;
    } catch (const exception& error) {
        cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
}
