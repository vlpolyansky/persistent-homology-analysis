//
// Created by vlad on 2017-12-31.
//

#include <fstream>
#include <vector>

#include "gudhi_tools.h"
#include "utils.h"

void save_matrix(const std::string &filename, const std::vector<std::vector<double>> &m) {
    std::ofstream out(filename);
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            out << m[i][j] << " ";
        }
        out << std::endl;
    }
    out.close();
}

void load_matrix(const std::string &filename, std::vector<std::vector<double>> &m) {
    std::ifstream in(filename);
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            in >> m[i][j];
        }
    }
    in.close();
}

int main(int argc, char **argv) {

    std::ifstream in(argv[1]);

    double min_lifespan = -1;
    {
        char *tmp = getCmdOption(argv, argv + argc, "--min_value");
        if (tmp) {
            min_lifespan = std::atof(tmp);
        }
    }


    std::vector<std::vector<gudhi_tools::PersistenceDiagram>> diagrams;

    std::string filename;
    while (std::getline(in, filename)) {
        std::vector<gudhi_tools::PersistenceDiagram> diagram = gudhi_tools::read_diagram(filename);
        if (min_lifespan > 0) {
            for (int i = 0; i < diagram.size(); i++) {
                gudhi_tools::reduce_diagram_by_min_value(diagram[i], min_lifespan);
            }
        }
        diagrams.push_back(diagram);
    }

    size_t n = diagrams.size();

    std::vector<std::vector<double>> dist(n, std::vector<double>(n, -1));
    for (int i = 0; i < n; i++) {
        dist[i][i] = 0;
    }

    {
        char *tmp = getCmdOption(argv, argv + argc, "--dist_matrix");
        if (tmp) {
            load_matrix(tmp, dist);
        }
    }
    bool dump = cmdOptionExists(argv, argv + argc, "--dump");
    bool no_zero_dim = cmdOptionExists(argv, argv + argc, "--no_zero_dim");

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dist[i][j] < 0) {
                std::cout << "Calculating distance: " << i << " " << j << std::endl;
                dist[i][j] = dist[j][i] = gudhi_tools::bottleneck_distance(diagrams[i], diagrams[j], no_zero_dim);
                if (dump) {
                    save_matrix("homology_distances.txt", dist);
                }
            }
        }
    }

    save_matrix("homology_distances.txt", dist);

    return 0;
}