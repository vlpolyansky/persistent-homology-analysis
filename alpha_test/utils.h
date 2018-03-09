//
// Created by vlad on 2018-01-14.
//

#ifndef ALPHA_TEST_UTILS_H
#define ALPHA_TEST_UTILS_H

#include <string>

char* getCmdOption(char **begin, char **end, const std::string &option);

bool cmdOptionExists(char **begin, char **end, const std::string &option);

#endif //ALPHA_TEST_UTILS_H
