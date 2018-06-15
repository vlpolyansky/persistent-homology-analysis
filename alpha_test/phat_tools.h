//
// Created by vlad on 2017-12-22.
//

#ifndef ALPHA_TEST_PHAT_TOOLS_H
#define ALPHA_TEST_PHAT_TOOLS_H

#include <phat/representations/bit_tree_pivot_column.h>
#include <phat/representations/vector_vector.h>
#include <phat/algorithms/standard_reduction.h>
#include <phat/boundary_matrix.h>
#include <phat/persistence_pairs.h>
#include <phat/algorithms/twist_reduction.h>

#include "gudhi_tools.h"

namespace phat_tools {

    using MatrixRepresentation = phat::bit_tree_pivot_column;
    using BoundaryMatrix = phat::boundary_matrix<MatrixRepresentation>;

    class my_standard_reduction {
    public:
        BoundaryMatrix operator()(BoundaryMatrix &boundary_matrix);
    };

    using ReductionAlgorithm = my_standard_reduction;
    using Pair = std::tuple<int, phat::index, double, double, phat::index>;  // dim birth_idx birth death death_idx
    using PersistencePairs = std::vector<Pair>;

    BoundaryMatrix read_filtration_matrix(const std::string &filename);

    /*L*/BoundaryMatrix compute_persistence_pairs(BoundaryMatrix &matrix, gudhi_tools::SimplexTree &simplex,
            /*output*/ PersistencePairs &pairs);

    void write_representative(BoundaryMatrix &L, phat::index idx, std::string filename,
                              gudhi_tools::SimplexTree &simplex, const std::vector<int> &ids);

    void write_representative(BoundaryMatrix &L, phat::index idx, std::string filename,
                              gudhi_tools::SimplexTree &simplex);

    void write_killer(phat::index idx, std::string filename,
                              gudhi_tools::SimplexTree &simplex, const std::vector<int> &ids);

    void write_killer(phat::index idx, std::string filename,
                              gudhi_tools::SimplexTree &simplex);

    void write_persistence_pairs(PersistencePairs &pairs, const std::string &filename);


}

#endif //ALPHA_TEST_PHAT_TOOLS_H
