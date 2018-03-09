//
// Created by vlad on 2017-12-22.
//

#ifndef ALPHA_TEST_FILTRATION_H
#define ALPHA_TEST_FILTRATION_H

#include <fstream>

#include <CGAL/Epick_d.h>
#include <gudhi/Alpha_complex.h>
#include <gudhi/Simplex_tree.h>

namespace gudhi_tools {


    template<int D>
    using Kernel = CGAL::Epick_d<CGAL::Dimension_tag<D>>;

    template<int D>
    using AlphaComplex = Gudhi::alpha_complex::Alpha_complex<Kernel<D>>;

    using SimplexTree = Gudhi::Simplex_tree<>;

    template<int D>
    using Point = typename Kernel<D>::Point_d;

    /**
     * P includes x, and x is omitted there
     * @param m
     * @param x
     * @param P
     */
    template<int D>
    double dtm_squared(double m, const Point<D> &x, const std::vector<Point<D>> &P) {
        int k = static_cast<int>(std::ceil(m * P.size()));
        typename Kernel<D>::Squared_distance_d squared_distance;

        std::vector<Point<D>> P2(P);
        std::nth_element(P2.begin(), P2.begin() + k, P2.end(), [&](Point<D> &a, Point<D> &b) {
            return squared_distance(x, a) < squared_distance(x, b);
        });

        double dist = 0;
        for (int i = 1; i <= k; i++) {
            dist += squared_distance(x, P2[i]);
        }

        return dist / k;
    }

    template<int D>
    std::vector<Point<D>> read_points(const std::string &filename) {
        std::vector<Point<D>> points;
        std::ifstream in(filename);

        int n;
        in >> n;
        std::cout << n << " points read" << std::endl;
        for (; n > 0; n--) {
            std::vector<double> coords;
            for (int i = 0; i < D; i++) {
                double t;
                in >> t;
                coords.push_back(t);
            }
            points.push_back(Point<D>(D, coords.begin(), coords.end()));
        }

        return points;
    }

    template<int D>
    std::vector<int> filter_by_dtm(std::vector<Point<D>> &points, double m, double percentage) {
        using WP = std::tuple<int, Point<D>, double>;
        std::vector<WP> weighted_points;
        for (int i = 0; i < points.size(); i++) {
            weighted_points.push_back(
                    std::make_tuple(i, points[i], dtm_squared<D>(m, points[i], points)));
        }
        points.clear();

        std::sort(weighted_points.begin(), weighted_points.end(), [](WP &a, WP &b) {
            return std::get<2>(a) < std::get<2>(b);
        });

        std::vector<int> ids;
        for (int i = 0; i < percentage * weighted_points.size(); i++) {
            ids.push_back(std::get<0>(weighted_points[i]));
            points.push_back(std::get<1>(weighted_points[i]));
        }

        return ids;
    }

    template<int D>
    SimplexTree compute_filtration(AlphaComplex<D> &complex) {
        SimplexTree simplex;
        if (complex.create_complex(simplex, std::numeric_limits<double>::infinity())) {
            std::cout << "Alpha complex is of dimension " << simplex.dimension() <<
                      " - " << simplex.num_simplices() << " simplices - " <<
                      simplex.num_vertices() << " vertices." << std::endl;
        } else {
            throw "no alpha complex";
        }

        return simplex;
    }

    /**
     * Format:
     * <num_simplices> <dimension>
     * <0> <dimension> <coord_1> <coord_2> .. <coord_d>
     * ...
     * <0> <dimension> <coord_1> <coord_2> .. <coord_d>
     * <dim> <boundary_1> <boundary_2> .. <boundary_k> <filtration_value>
     * ...
     * <dim> <boundary_1> <boundary_2> .. <boundary_k> <filtration_value>
     */
    template<int D>
    void save_filtration(AlphaComplex<D> &complex, SimplexTree &simplex, const std::string &filename) {
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
                auto vertex = simplex.simplex_vertex_range(sh).begin();
                out << " " << complex.get_point(*vertex) << std::endl;
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

    void save_rips_filtration(SimplexTree &simplex, const std::string &filename);

    // PERSISTENCE DIAGRAM ALGORITHMS
    using PersistencePoint = std::pair<double, double>;
    using PersistenceDiagram = std::vector<PersistencePoint>;

    std::vector<PersistenceDiagram> read_diagram(const std::string &filename);

    double bottleneck_distance(std::vector<PersistenceDiagram> &a, std::vector<PersistenceDiagram> &b, bool no_zero_dim);

    void reduce_diagram_by_percentage(PersistenceDiagram &a, double percentage);

    void reduce_diagram_by_min_value(PersistenceDiagram &a, double min_value);
}

#endif //ALPHA_TEST_FILTRATION_H
