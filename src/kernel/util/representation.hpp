#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class __attribute__((visibility("default"))) Representation {
public:
    vector<tuple<int, int, int>> irreps; // Multiplicity, irrep type, even / oddness (0 if even )

    Representation(int rep_id) {
        irreps.emplace_back(1, rep_id, 0);
    }

    Representation(int mult, int rep_id) {
        irreps.emplace_back(mult, rep_id, 0);
    }

    size_t get_rep_length() {
        size_t rep_length = 0;
        for (auto& irrep : irreps) {
            rep_length += get<0>(irrep) * (2 * get<1>(irrep) + 1);
        }
        return rep_length;
    }

    vector<int> get_irrep_offsets() {
        vector<int> offsets(irreps.size() + 1, 0);
        int offset = 0;
        offsets.push_back(offset);
        for (int i = 0; i < irreps.size(); i++) {
            auto &irrep = irreps[i];
            offset += get<0>(irrep) * (2 * get<1>(irrep) + 1);
            offsets[i+1] = offset;
        }
        return offsets;
    }

    size_t num_irreps() {
        return irreps.size();
    }

    int mult(int irrep_id) {
        return get<0>(irreps[irrep_id]);
    }

    int type(int irrep_id) {
        return get<1>(irreps[irrep_id]);
    }

    int even(int irrep_id) {
        return get<2>(irreps[irrep_id]);
    }

    Representation() = default; 

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

    /*
    * Given an N x REP_LEN matrix on the CPU, this function transposes
    * the matrix reshaping of the irrep within each row. The transpose 
    * is performed in place.
    *
    * If input is row major: for every submatrix of size MULT x IRREP_LEN,
    * every contiguous block of IRREP_LEN elements forms a row of the submatrix.
    */
    void transpose_irreps_cpu(py::array_t<float> &rep_mat, bool row_major_in);
};

/*
* A RepTriple encapsulates the three-cornered E3NN interaction graph. 
*/
class RepTriple {
public:
    Representation L1;
    Representation L2;
    Representation L3;

    vector<tuple<int, int, int>> interactions_i; 

    RepTriple(Representation &L1_i, Representation &L2_i, Representation &L3_i) :
        L1(L1_i),
        L2(L2_i),
        L3(L3_i) { 

        if(L1.num_irreps() == 1 && L2.num_irreps() == 1 && L3.num_irreps() == 1) {
            interactions_i.emplace_back(0, 0, 0);
        }
    }

    /*
    * Full decomposition up to LMax. 
    */
    RepTriple(Representation &L1_i, Representation &L2_i, int LMax) :
        L1(L1_i),
        L2(L2_i) {

        for(int i = 0; i < L1.num_irreps(); i++) {
            for(int j = 0; j < L2.num_irreps(); j++) {
                int lA = max(L1.type(i), L2.type(j));
                int lB = min(L1.type(i), L2.type(j));

                for(int k = lA - lB; k <= min(lA + lB, LMax); k++) {
                    L3.irreps.emplace_back(L1.mult(i) * L2.mult(j), k, (L1.even(i) + L2.even(j)) % 2);
                    interactions_i.emplace_back(i, j, static_cast<int>(L3.num_irreps()) - 1);
                }
            }
        }
    }

    int num_interactions() {
        return interactions_i.size();
    }

    int num_trainable_weights() {
        // Assumes all "uvu" interactions 
        int total_weights = 0;

        for(auto triple : interactions_i) {
            total_weights += L1.mult(get<0>(triple)) * L2.mult(get<1>(triple));
        }
        return total_weights;
    }

    tuple<int, int, int> interactions(int i) {
        return interactions_i[i]; 
    }

    string to_string() {
        std::stringstream ss;
        ss << "(" << L1.to_string() << ") x ";
        ss << "(" << L2.to_string() << ") -> ";
        ss << "(" << L3.to_string() << ")";
        return ss.str();
    }
};

