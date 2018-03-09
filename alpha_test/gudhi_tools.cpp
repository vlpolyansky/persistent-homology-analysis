//
// Created by vlad on 2017-12-30.
//

#include <vector>
#include <fstream>

#include <gudhi/Bottleneck.h>

#include "gudhi_tools.h"

std::vector<gudhi_tools::PersistenceDiagram> gudhi_tools::read_diagram(const std::string &filename) {
    std::vector<gudhi_tools::PersistenceDiagram> diagrams;
    std::ifstream in(filename);

    int dim, id;
    double birth, death;
    while (in >> dim >> id >> birth >> death) {
        while (diagrams.size() <= dim) {
            diagrams.push_back(gudhi_tools::PersistenceDiagram());
        }
        diagrams[dim].push_back(std::make_pair(birth, death));
    }

    return diagrams;

}

double gudhi_tools::bottleneck_distance(std::vector<gudhi_tools::PersistenceDiagram> &a,
                                        std::vector<gudhi_tools::PersistenceDiagram> &b, bool no_zero_dim) {
    assert(a.size() == b.size());
    double res = -1;
    for (int i = no_zero_dim ? 1 : 0; i < a.size(); i++) {
        double val = Gudhi::persistence_diagram::bottleneck_distance(a[i], b[i]);
        if (res < 0 || res < val) {
            res = val;
        }
    }
    return res;
}


void gudhi_tools::reduce_diagram_by_percentage(gudhi_tools::PersistenceDiagram &a, double percentage) {
    // assuming pairs are sorted !
    a.resize(size_t(a.size() * percentage));
}

void gudhi_tools::reduce_diagram_by_min_value(gudhi_tools::PersistenceDiagram &a, double min_value) {
    auto pend = std::remove_if(a.begin(), a.end(), [min_value](const gudhi_tools::PersistencePoint &p) {return p.second - p.first < min_value;});
    a.resize(pend - a.begin());
}

void gudhi_tools::save_rips_filtration(gudhi_tools::SimplexTree &simplex, const std::string &filename) {
    using SH = SimplexTree::Simplex_handle;

    std::ofstream out(filename);
    out << simplex.num_simplices() << " " << simplex.dimension() << std::endl;

    std::vector<SH> range = simplex.filtration_simplex_range();
    for (int i = 0; i < range.size(); i++) {
        SH &sh = range[i];
        sh->second.assign_key((SimplexTree::Simplex_key) i); // GUDHI do not do it for some reason
        SimplexTree::Simplex_key key = simplex.key(sh);

        out << simplex.dimension(sh);
        if (simplex.dimension(sh) == 0) {
            out << " 0" << std::endl;
        } else {
            auto boundary_range = simplex.boundary_simplex_range(sh);
            for (auto b_sh : boundary_range) {
                out << " " << simplex.key(b_sh);
            }
            // note to self: these filtrations are not true alpha complex filtrations
            out << " " << simplex.filtration(sh) << std::endl;
        }
    }

    out.close();
}
