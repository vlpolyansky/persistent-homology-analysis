//
// Created by vlad on 2018-01-14.
//

#include <string>
#include <vector>
#include <fstream>
#include <limits>

#include <gudhi/Rips_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/distance_functions.h>

#include "gudhi_tools.h"
#include "phat_tools.h"
#include "utils.h"

using Point = std::vector<double>;
using Rips_complex = Gudhi::rips_complex::Rips_complex<double>;

std::vector<Point> read_points(const std::string &filename, int skip = 1) {
    std::ifstream in(filename);
    std::vector<Point> points;
    int n, d;
    in >> n >> d;
    std::cout << "Reading " << n << " " << d << "-dimensional points" << std::endl;
    Point p(d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            in >> p[j];
        }
        if (i % skip != 0) {
            continue;
        }
        points.push_back(p);
    }
    std::cout << "Done" << std::endl;

    return points;
}

int main(int argc, char **argv) {

    std::string filename(argv[1]);

    int skip = 1;
    char *tmp = getCmdOption(argv, argv + argc, "--skip");
    if (tmp) {
        skip = std::atoi(tmp);
    }

    double percentage = 0.95;
    tmp = getCmdOption(argv, argv + argc, "--percentage");
    if (tmp) {
        percentage = std::atof(tmp);
    }

    int rep_num = 4;
    tmp = getCmdOption(argv, argv + argc, "--rep_num");
    if (tmp) {
        rep_num = std::atoi(tmp);
    }


    std::vector<Point> points = read_points(filename, skip);
    std::vector<int> ids;
    for (int i = 0; i < points.size(); i++) {
        ids.push_back(i);
    }

    double threshold = std::numeric_limits<double>::infinity();
    tmp = getCmdOption(argv, argv + argc, "--threshold");
    if (tmp) {
        threshold = std::atof(tmp);
    }

    Rips_complex rips_complex(points, threshold, Gudhi::Euclidean_distance());

    gudhi_tools::SimplexTree simplex;
    rips_complex.create_complex(simplex, 2);

    std::cout << "Rips complex is of dimension " << simplex.dimension() <<
              " - " << simplex.num_simplices() << " simplices - " <<
              simplex.num_vertices() << " vertices." << std::endl;
//    points.clear();

    gudhi_tools::save_rips_filtration(simplex, "filtration.txt");
    std::cout << "Filtration computed" << std::endl;


    auto matrix = phat_tools::read_filtration_matrix("filtration.txt");
    std::cout << matrix.get_num_cols() << " simplices read" << std::endl;

    phat_tools::PersistencePairs pairs;
    auto L = phat_tools::compute_persistence_pairs(matrix, simplex, pairs);
    std::cout << pairs.size() << " pairs found" << std::endl;
    phat_tools::write_persistence_pairs(pairs, "pairs.txt");

    std::vector<phat::index> chosen_indices;
    for (int i = 0; i < pairs.size() && chosen_indices.size() < rep_num; i++) {
        if (std::get<0>(pairs[i]) == 1) {
            chosen_indices.push_back(std::get<1>(pairs[i]));
        }
    }
    for (int i = 0; i < chosen_indices.size(); i++) {
        phat_tools::write_representative(L, chosen_indices[i], "repr_" + std::to_string(i) + ".txt", simplex, ids);
    }

    return 0;
}