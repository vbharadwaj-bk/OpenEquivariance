//=========================================================================

/*
* A simple implementation that gets each thread 
* to handle each tensor product based on a coordinate format. 
*/
class __attribute__ ((visibility ("default"))) ThreadTensorProductImpl : public GenericTensorProductImpl {
public:
    DeviceBuffer<uint8_t> coord1; 
    DeviceBuffer<uint8_t> coord2; 
    DeviceBuffer<uint8_t> coord3; 
    DeviceBuffer<float> values;

    ThreadTensorProductImpl(
        RepTriple &reps,
        py::array_t<uint8_t> coord1_py, 
        py::array_t<uint8_t> coord2_py,
        py::array_t<uint8_t> coord3_py,
        py::array_t<float> values_py 
        ) :
        GenericTensorProductImpl(reps),
        coord1(coord1_py),
        coord2(coord2_py),
        coord3(coord3_py),
        values(values_py)
        { 
            if(L1.irreps.size() != 1 || L2.irreps.size() != 1 || L3.irreps.size() != 1) {
                throw std::invalid_argument("ThreadTensorProductImpl only supports single irreps");
            }
        }

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~ThreadTensorProductImpl() = default;
};


//=========================================================================
/*
* A tensor product that executes a dense GEMM after instantiating Kronecker 
* products explicitly using cuBLASLt. 
*/
class __attribute__ ((visibility ("default"))) GemmTensorProductImpl : public GenericTensorProductImpl {
public:
    size_t workspaceSize = 1024 * 1024 * 4;
    DeviceBuffer<char> workspace;

    uint64_t num_products;
    DeviceBuffer<float> cg_coeffs;
    DeviceBuffer<float> kprods;

    cublasLtHandle_t     ltHandle;
    cublasLtMatmulDesc_t operationDesc = NULL; 
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL; 
    cublasLtMatmulPreference_t preference = NULL; 
    cublasLtMatmulHeuristicResult_t heuristicResult {};

    GemmTensorProductImpl(
        RepTriple &reps,
        uint64_t num_products1,
        py::array_t<float> cg_coeffs_py 
        ) :
        GenericTensorProductImpl(reps),
        workspace(workspaceSize),
        num_products(num_products1),
        cg_coeffs(cg_coeffs_py), 
        kprods(num_products * L1.get_rep_length() * L2.get_rep_length())
        { preprocess(); }

    void preprocess();

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~GemmTensorProductImpl(); 
};


//=========================================================================
/*
* A tensor product that uses shuffle primitives. Each tensor product is 
* assigned to a single warp. 
*/
class __attribute__ ((visibility ("default"))) ShuffleTensorProductImpl : public GenericTensorProductImpl {
public:
    int max_lane_length, reduction_depth;

    DeviceBuffer<float> warp_values;
    DeviceBuffer<int> l1_indices;
    DeviceBuffer<int> l2_indices;
    DeviceBuffer<int> red_lanes;

    JITKernel jit;

    ShuffleTensorProductImpl(
        RepTriple &reps,
        py::array_t<float> warp_values_py, 
        py::array_t<int> l1_indices_py, 
        py::array_t<int> l2_indices_py, 
        py::array_t<int> red_lanes_py);

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~ShuffleTensorProductImpl() = default; 
};