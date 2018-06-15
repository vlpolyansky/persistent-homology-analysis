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

void raw_to_persistence_pairs(const std::string &raw_filename, double m, double percentage, int rep_num) {

    std::vector<gudhi_tools::Point<_MY_DIM>> points = gudhi_tools::read_points<_MY_DIM>(raw_filename);
    std::vector<int> ids = gudhi_tools::filter_by_dtm<_MY_DIM>(points, m, percentage);

    {
        std::ofstream out("filtered_ids.txt");
        for (int i : ids) {
            out << i << std::endl;
        }
        out.close();
    }

    gudhi_tools::AlphaComplex<_MY_DIM> complex(points);
    gudhi_tools::SimplexTree simplex = gudhi_tools::compute_filtration(complex);
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
    for (int i = 0; i < pairs.size() && chosen_indices.size() < rep_num; i++) {
        if (std::get<0>(pairs[i]) > 0) {
            chosen_indices.push_back(std::get<1>(pairs[i]));
            chosen_killers.push_back(std::get<4>(pairs[i]));
        }
    }
    for (int i = 0; i < chosen_indices.size(); i++) {
        phat_tools::write_representative(L, chosen_indices[i], "repr_" + std::to_string(i) + ".txt", simplex, ids);
        if (chosen_killers[i] >= 0) {
            phat_tools::write_killer(chosen_killers[i], "kill_" + std::to_string(i) + ".txt", simplex, ids);
        }
    }

    // GUDHI cohomologies are not used
//    Persistent_cohomology pcoh(simplex);
//    pcoh.init_coefficients(2);
//    pcoh.compute_persistent_cohomology();
//    pcoh.write_output_diagram("diag_" + std::string(argv[1]));

}

int main(int argc, char **argv) {

    double m = 0.001;
    double percentage = 0.95;
    int rep_num = 4;

    char *tmp = getCmdOption(argv, argv + argc, "--m");
    if (tmp) {
        m = std::atof(tmp);
    }

    tmp = getCmdOption(argv, argv + argc, "--percentage");
    if (tmp) {
        percentage = std::atof(tmp);
    }

    tmp = getCmdOption(argv, argv + argc, "--rep_num");
    if (tmp) {
        rep_num = std::atoi(tmp);
    }

    raw_to_persistence_pairs(std::string(argv[1]), m, percentage, rep_num);
    return 0;
}
