//#define CGAL_EIGEN3_ENABLED

#include "gudhi_tools.h"
#include "phat_tools.h"
#include "utils.h"
#include <gudhi/Persistent_cohomology.h>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <fstream>

//using Field_Zp = Gudhi::persistent_cohomology::Field_Zp;
//using Persistent_cohomology = Gudhi::persistent_cohomology::Persistent_cohomology<gudhi_tools::SimplexTree , Field_Zp>;

void raw_to_persistence_pairs(const std::string &raw_filename, double m, double alpha_value) {

    std::vector<gudhi_tools::Point<_MY_DIM>> points = gudhi_tools::read_points<_MY_DIM>(raw_filename);

    gudhi_tools::AlphaComplex<_MY_DIM> complex(points);
    gudhi_tools::SimplexTree simplex = gudhi_tools::compute_filtration(complex);
    simplex.prune_above_filtration(alpha_value * alpha_value);

    std::vector<double> dtms(points.size());
    for (int i = 0; i < points.size(); i++) {
        dtms[i] = gudhi_tools::dtm_squared<_MY_DIM>(m, points[i], points);
    }

    using VH = gudhi_tools::SimplexTree::Vertex_handle;
    using SH = gudhi_tools::SimplexTree::Simplex_handle;
    for (SH sh : simplex.complex_simplex_range()) {
        double f_val = -1;
        int cnt = 0;
        for (VH vertex : simplex.simplex_vertex_range(sh)) {
            cnt++;
            if (f_val < dtms[vertex]) {
                f_val = dtms[vertex];
            }
        }
        simplex.assign_filtration(sh, f_val);
    }
    simplex.make_filtration_non_decreasing();

    gudhi_tools::save_filtration(complex, simplex, "filtration.txt");
    std::cout << "Filtration computed" << std::endl;

    auto matrix = phat_tools::read_filtration_matrix("filtration.txt");
    std::cout << matrix.get_num_cols() << " simplices read" << std::endl;

    phat_tools::PersistencePairs pairs;
    auto L = phat_tools::compute_persistence_pairs(matrix, simplex, pairs);
    std::cout << pairs.size() << " pairs found" << std::endl;

    phat_tools::write_persistence_pairs(pairs, "pairs.txt");

    std::vector<phat::index> chosen_indices;
    std::vector<phat::index> chosen_killers;
    for (int i = 0; i < pairs.size() && chosen_indices.size() < phat_tools::MAX_REPRESENTATIVE_CNT; i++) {
        if (std::get<0>(pairs[i]) > 0) {
            chosen_indices.push_back(std::get<1>(pairs[i]));
            chosen_killers.push_back(std::get<4>(pairs[i]));
        }
    }
    for (int i = 0; i < chosen_indices.size(); i++) {
        phat_tools::write_representative(L, chosen_indices[i], "repr_" + std::to_string(i) + ".txt", simplex);
        if (chosen_killers[i] >= 0) {
            phat_tools::write_killer(chosen_killers[i], "kill_" + std::to_string(i) + ".txt", simplex);
        }
    }

}

int main(int argc, char **argv) {

    double m = 0.001;

    char *tmp = getCmdOption(argv, argv + argc, "--m");
    if (tmp) {
        m = std::atof(tmp);
    }

    double alpha_value = std::atof(getCmdOption(argv, argv + argc, "--alpha"));

    raw_to_persistence_pairs(std::string(argv[1]), m, alpha_value);
    return 0;
}
