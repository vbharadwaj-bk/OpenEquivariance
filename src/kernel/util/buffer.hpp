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


template<typename T>
class __attribute__((visibility("default"))) Buffer {
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

    uint64_t size() {
        uint64_t buffer_size = 1;
        for(uint64_t i : shape) {
            buffer_size *= i;
        }
        return buffer_size;
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

__attribute__ ((visibility("default")))
void* gpu_alloc (size_t size);

__attribute__ ((visibility("default")))
void gpu_free (void* ptr);

__attribute__ ((visibility("default")))
void copy_host_to_device (void* host, void* device, size_t size);

__attribute__ ((visibility("default")))
void copy_device_to_host (void* host, void* device, size_t size);

template<typename T>
class DeviceBuffer {
public:
    T* ptr;
    uint64_t size;

    DeviceBuffer(uint64_t size) {
        ptr = static_cast<T*>(gpu_alloc(size * sizeof(T)));
        this->size = size;
    }

    DeviceBuffer(py::array_t<T> &host_py) {
        Buffer<T> host(host_py);
        size = host.size();
        ptr = static_cast<T*>(gpu_alloc(size * sizeof(T)));
        copy_from_host_buffer(host);
    }

    ~DeviceBuffer() {
        gpu_free(static_cast<void*>(ptr));
    }

    void copy_from_host_buffer(Buffer<T> &host) {
        copy_host_to_device(static_cast<void*>(host.ptr), 
            static_cast<void*>(ptr), sizeof(T) * size);
    }

    void copy_to_host_buffer(Buffer<T> &host) {
        copy_device_to_host(static_cast<void*>(host.ptr), 
            static_cast<void*>(ptr), sizeof(T) * size);
    }
};
 
class PyDeviceBuffer {
public:
    char* host_ptr;
    char* device_ptr;
    size_t size;

    PyDeviceBuffer(uint64_t size) {
        this->size = size;
        device_ptr = static_cast<char*>(gpu_alloc(size));
        host_ptr = nullptr;
    }

    PyDeviceBuffer(py::buffer host_data) {
        const py::buffer_info &info = host_data.request();
        host_ptr = static_cast<char*>(info.ptr);
        size = 1;
        for(int64_t i = 0; i < info.ndim; i++) {
            size *= info.shape[i];
        }
        size *= info.itemsize;

        device_ptr = static_cast<char*>(gpu_alloc(size));
        copy_host_to_device(host_ptr, device_ptr, size);
    }

    ~PyDeviceBuffer() {
        gpu_free(static_cast<void*>(device_ptr));
    }

    void copy_to_host() {
        copy_device_to_host(host_ptr, device_ptr, size);
    }

    uint64_t data_ptr() {
        return reinterpret_cast<uint64_t>(device_ptr);
    }
};