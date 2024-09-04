#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

class Representation {
public:
    vector<tuple<int, int, int>> irreps; // Multiplicity, irrep type, even / oddness (0 if even )

    Representation(int rep_id) {
        irreps.emplace_back(1, rep_id, 0);
    }

    Representation(int rep_id, int mult) {
        irreps.emplace_back(mult, rep_id, 0);
    }

    size_t get_rep_length() {
        // TODO: Deal with even / odd here 
        size_t rep_length = 0;
        for (auto& irrep : irreps) {
            rep_length += get<0>(irrep) * (2 * get<1>(irrep) + 1);
        }
        return rep_length;
    }

    Representation(string str_rep) {
        // String type must be of the form 32x1e + 1x2o, etc. 

        str_rep.erase (std::remove (str_rep.begin(), str_rep.end(), ' '), str_rep.end());

        std::string delim1 = "+";
        std::string delim2 = "x";

        auto start = 0U;
        auto end = str_rep.find(delim1);

        // Create a lambda called process_token that will operate on the contents
        //of the while loop
        
        auto process_token = [&](const string& s) {
            int mult, irrep, even;
            size_t pos = s.find(delim2); 

            if (pos != std::string::npos) { 
                std::string part1 = s.substr(0, pos); 
                std::string part2 = s.substr(pos + 1);
                mult = stoi(part1);
                irrep = stoi(part2); 
            }
            else {
                throw std::invalid_argument("Invalid representation string");
            }

            // Get the last character of s. Handle both cases, even or odd, throw error if neither
            if (s.back() == 'e') {
                even = 0;
            } else if (s.back() == 'o') {
                even = 1;
            } else {
                throw std::invalid_argument("Invalid representation string");
            }

            // Emplace back a tuple
            irreps.emplace_back(mult, irrep, even);
        };

        while (end != std::string::npos)
        {
            string s = str_rep.substr(start, end - start);
            process_token(s);
            start = end + delim1.length();
            end = s.find(delim1, start);
        }
        process_token(str_rep.substr(start, end));
    }

    string to_string() {
        std::stringstream ss;
        bool first = true;
        for (auto& irrep : irreps) {
            if(!first) {
                ss << " + ";
            } 
            ss << get<0>(irrep) << "x" << get<1>(irrep) << (get<2>(irrep) == 0 ? "e" : "o"); 
            first = false;
        }
        return ss.str();
    }    
};
 