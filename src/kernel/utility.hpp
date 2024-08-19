#pragma once

#include <cassert>
#include <fcntl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <bits/stdc++.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <chrono>


#define BIG_CONSTANT(x) (x##LLU)

using namespace std;
namespace py = pybind11;

//using json = nlohmann::json;


#pragma GCC visibility push(hidden)
template<typename T>
class NumpyArray {
public:
    py::buffer_info info;
    T* ptr;

    NumpyArray(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }

    NumpyArray(py::object obj, string attr_name) {
        py::array_t<T> arr_py = obj.attr(attr_name.c_str()).cast<py::array_t<T>>();
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }

    NumpyArray(T* input_ptr) {
        ptr = input_ptr;
    }
};

template<typename T>
class NumpyList {
public:
    vector<py::buffer_info> infos;
    vector<T*> ptrs;
    int length;

    NumpyList(py::list input_list) {
        length = py::len(input_list);
        for(int i = 0; i < length; i++) {
            py::array_t<T> casted = input_list[i].cast<py::array_t<T>>();
            infos.push_back(casted.request());
            ptrs.push_back(static_cast<T*>(infos[i].ptr));
        }
    }

    // Should refactor class name to something 
    // other than NumpyList, since this
    // constructor exists. This constructor 
    // does not perform any data copy 
    NumpyList(vector<T*> input_list) {
        length = input_list.size();
        ptrs = input_list;
    }
};

template<typename IDX_T, typename VAL_T>
class COOSparse {
public:
    vector<IDX_T> rows;
    vector<IDX_T> cols;
    vector<VAL_T> values;

    void print_contents() {
      double normsq = 0.0;
      for(uint64_t i = 0; i < rows.size(); i++) {
        /*cout 
          << rows[i] 
          << " " 
          << cols[i] 
          << " "
          << values[i]
          << endl;*/
        normsq += values[i]; 
      }
      cout << "Norm Squared: " << normsq << endl;
    }

	/*
	 * Computes Y := S^T . X, where S is this
	 * sparse matrix.
	 * 
	 * This is currently a very inefficient single-threaded
	 * version of the code. 
	 */
	void cpu_spmm(double* X, double* Y, int r) {
		IDX_T* row_ptr = rows.data();
		IDX_T* col_ptr = cols.data();
		VAL_T* val_ptr = values.data();

		for(uint64_t i = 0; i < rows.size(); i++) {
			// We perform a transpose here
		    IDX_T row = col_ptr[i];
			IDX_T col = row_ptr[i];
			VAL_T value = val_ptr[i];
			for(int j = 0; j < r; j++) {
				Y[row * r + j] += X[col * r + j] * value;
			}
		}
	}
};

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

//#pragma GCC visibility push(hidden)
template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
public:
    py::buffer_info info;
    unique_ptr<T[]> managed_ptr;
    T* ptr;
    uint64_t dim0;
    uint64_t dim1;

    bool initialized;
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
        :   info(std::move(other.info)), 
            managed_ptr(std::move(other.managed_ptr)),
            ptr(std::move(other.ptr)),
            dim0(other.dim0),
            dim1(other.dim1),
            initialized(other.initialized),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    void steal_resources(Buffer& other) {
        info = std::move(other.info); 
        managed_ptr = std::move(other.managed_ptr);
        ptr = other.ptr;
        dim0 = other.dim0;
        dim1 = other.dim1;
        shape = other.shape;
        initialized = other.initialized;
    }

    Buffer(py::array_t<T> arr_py, bool copy) {
        info = arr_py.request();

        if(info.ndim == 2) {
            dim0 = info.shape[0];
            dim1 = info.shape[1];
        }
        else if(info.ndim == 1) {
            dim0 = info.shape[0];
            dim1 = 1;
        }

        uint64_t buffer_size = 1;
        for(int64_t i = 0; i < info.ndim; i++) {
            shape.push_back(info.shape[i]);
            buffer_size *= info.shape[i];
        }

        if(! copy) {
            ptr = static_cast<T*>(info.ptr);
        }
        else {
            managed_ptr.reset(new T[buffer_size]);
            ptr = managed_ptr.get();
            std::copy(static_cast<T*>(info.ptr), static_cast<T*>(info.ptr) + info.size, ptr);
        }
        initialized = true;
    }

    Buffer(py::array_t<T> arr_py) :
        Buffer(arr_py, false)
    {
        // Default behavior is a thin alias of the C++ array 
    }

    Buffer(initializer_list<uint64_t> args) {
        initialized = false;
        if(args.size() > 0) {
            initialize_to_shape(args);
        }
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;
        initialized = true;
    }

    Buffer() {
        initialized = false;
    }

    void initialize_to_shape(initializer_list<uint64_t> args) {
        if(initialized) {
            throw std::runtime_error("Cannot initialize a buffer twice");
        }
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        managed_ptr.reset(new T[buffer_size]);
        ptr = managed_ptr.get();
        initialized = true;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    void print() {
        cout << "------------------------" << endl;
        if(shape.size() == 1) {
            cout << "[ " << " "; 
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << ptr[i] << " ";
            }
            cout << "]" << endl;
            return;
        }
        else if(shape.size() == 2) {
            for(uint64_t i = 0; i < shape[0]; i++) {
                cout << "[ ";
                for(uint64_t j = 0; j < shape[1]; j++) {
                    cout << ptr[i * shape[1] + j] << " ";
                }
                cout << "]" << endl; 
            }
        }
        else {
            cout << "Cannot print buffer with shape: ";
            for(uint64_t i : shape) {
                cout << i << " ";
            }
            cout << endl;
        }
        cout << "------------------------" << endl;
    }

    ~Buffer() {}
};
