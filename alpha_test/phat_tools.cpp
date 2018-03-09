//
// Created by vlad on 2017-12-22.
//

#include <fstream>

#include "phat_tools.h"
#include <phat/compute_persistence_pairs.h>

phat_tools::BoundaryMatrix phat_tools::read_filtration_matrix(const std::string &filename) {
    std::ifstream in(filename);
    phat_tools::BoundaryMatrix matrix;

    phat::index n;
    int max_dim;
    in >> n >> max_dim;
    matrix.set_num_cols(n);

    for (phat::index i = 0; i < n; i++) {
        int d;
        in >> d;
        matrix.set_dim(i, (phat::dimension) d);

        std::vector<phat::index> column;
        if (d > 0) {
            column.resize((size_t) d + 1);
            for (phat::index j = 0; j <= d; j++) {
                in >> column[j];
            }

            double alpha;
            in >> alpha;
        } else {
            in >> max_dim;
            std::vector<double> coords(max_dim);
            for (int j = 0; j < max_dim; j++) {
                in >> coords[j];
            }
        }
        std::sort(column.begin(), column.end());
        matrix.set_col(i, column);

    }
    std::cout << "Overall, the boundary matrix has " << matrix.get_num_entries() << " entries." << std::endl;

    return matrix;
}


phat_tools::BoundaryMatrix phat_tools::my_standard_reduction::operator()(
        BoundaryMatrix &boundary_matrix) {
    phat_tools::BoundaryMatrix L;
    const phat::index nr_columns = boundary_matrix.get_num_cols();

    L.set_num_cols(boundary_matrix.get_num_cols());
    for (phat::index cur_col = 0; cur_col < nr_columns; cur_col++) {
        L.set_dim(cur_col, boundary_matrix.get_dim(cur_col));
        std::vector<phat::index> col_vec(1, cur_col);
        L.set_col(cur_col, col_vec);
    }

    std::vector<phat::index> lowest_one_lookup(nr_columns, -1);

    for (phat::index cur_col = 0; cur_col < nr_columns; cur_col++) {
        phat::index lowest_one = boundary_matrix.get_max_index(cur_col);
        while (lowest_one != -1 && lowest_one_lookup[lowest_one] != -1) {
            boundary_matrix.add_to(lowest_one_lookup[lowest_one], cur_col);
            L.add_to(lowest_one_lookup[lowest_one], cur_col);
            lowest_one = boundary_matrix.get_max_index(cur_col);
        }
        if (lowest_one != -1) {
            lowest_one_lookup[lowest_one] = cur_col;
        }
        boundary_matrix.finalize(cur_col);
        L.finalize(cur_col);
    }

    return L;
}

void phat_tools::write_persistence_pairs(phat_tools::PersistencePairs &pairs, const std::string &filename) {
    std::ofstream out(filename);

    for (phat::index i = 0; i < pairs.size(); i++) {
        auto p = pairs[i];
        out << std::get<0>(p) << " " << std::get<1>(p) << " " << std::get<2>(p) << " " << std::get<3>(p)
            << std::endl;
    }

    out.close();
}

phat_tools::BoundaryMatrix phat_tools::compute_persistence_pairs(phat_tools::BoundaryMatrix &matrix,
                                                                 gudhi_tools::SimplexTree &simplex,
                                                                 phat_tools::PersistencePairs &pairs) {
    phat_tools::ReductionAlgorithm reduce;
    phat_tools::BoundaryMatrix L = reduce(matrix);

//    {
//        std::ofstream out("L_matrix.txt");
//        for (phat::index idx = 0; idx < matrix.get_num_cols(); idx++) {
//            out << idx;
//            std::vector<phat::index> vec;
//            L.get_col(idx, vec);
//            out << " " << vec.size();
//            for (phat::index idx2 : vec) {
//                out << " " << idx2;
//            }
//            out << std::endl;
//        }
//        out.close();
//    }

    std::map<int, bool> free;

    for (phat::index idx = 0; idx < matrix.get_num_cols(); idx++) {
        free[idx] = true;
        if (!matrix.is_empty(idx)) {
            phat::index birth_idx = matrix.get_max_index(idx);
            phat::index death_idx = idx;
            int dim = simplex.dimension(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) birth_idx));
            double birth = simplex.filtration(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) birth_idx));
            double death = simplex.filtration(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) death_idx));
            if (birth != death) {
                pairs.push_back(std::make_tuple(dim, birth_idx, birth, death, death_idx));
            }
            free[birth_idx] = false;
            free[death_idx] = false;
        }
    }
    double inf = std::numeric_limits<double>::infinity();
    for (std::map<int, bool>::iterator it = free.begin(); it != free.end(); ++it) {
        if (it->second) {
            phat::index birth_idx = it->first;
            int dim = simplex.dimension(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) birth_idx));
            double birth = simplex.filtration(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) birth_idx));

            pairs.push_back(std::make_tuple(dim, birth_idx, birth, inf, -1));
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](phat_tools::Pair &a, phat_tools::Pair &b) {
        double b_a = std::get<2>(a);
        double b_b = std::get<2>(b);
        double d_a = std::get<3>(a);
        double d_b = std::get<3>(b);
        if (std::isinf(d_a) && std::isinf(d_b)) {
            return b_a < b_b;
        } else if (std::isinf(d_a)) {
            return true;
        } else if (std::isinf(d_b)) {
            return false;
        } else {
            return d_a - b_a > d_b - b_b;
        }
    });

    return L;
}

void phat_tools::write_representative(phat_tools::BoundaryMatrix &L, phat::index idx, std::string filename,
                                      gudhi_tools::SimplexTree &simplex) {
    std::vector<phat::index> vec;
    L.get_col(idx, vec);
    std::ofstream out(filename);
    out << idx << std::endl;
    for (phat::index idx2 : vec) {
        auto vertex_range = simplex.simplex_vertex_range(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) idx2));
        for (int vertex : vertex_range) {
            out << vertex << " ";
        }
        out << std::endl;

    }
    out.close();
}

void phat_tools::write_representative(phat_tools::BoundaryMatrix &L, phat::index idx, std::string filename,
                                      gudhi_tools::SimplexTree &simplex, const std::vector<int> &ids) {
    std::vector<phat::index> vec;
    L.get_col(idx, vec);
    std::ofstream out(filename);
    out << idx << std::endl;
    for (phat::index idx2 : vec) {
        auto vertex_range = simplex.simplex_vertex_range(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) idx2));
        for (auto vertex : vertex_range) {
            out << ids[vertex] << " ";
        }
        out << std::endl;

    }
    out.close();
}

void phat_tools::write_killer(phat::index idx, std::string filename,
                                      gudhi_tools::SimplexTree &simplex) {
    std::ofstream out(filename);
    out << idx << std::endl;

    auto vertex_range = simplex.simplex_vertex_range(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) idx));
    for (int vertex : vertex_range) {
        out << vertex << " ";
    }
    out << std::endl;

    out.close();
}

void phat_tools::write_killer(phat::index idx, std::string filename,
                                      gudhi_tools::SimplexTree &simplex, const std::vector<int> &ids) {
    std::ofstream out(filename);
    out << idx << std::endl;

    auto vertex_range = simplex.simplex_vertex_range(simplex.simplex((gudhi_tools::SimplexTree::Simplex_key) idx));
    for (auto vertex : vertex_range) {
        out << ids[vertex] << " ";
    }
    out << std::endl;

    out.close();
}
