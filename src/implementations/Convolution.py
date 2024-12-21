import numpy as np
import numpy.linalg as la
from build.kernel_wrapper import *
from src.benchmark.random_buffer_utils import * 
from src.implementations.TensorProduct import *

from src.benchmark.logging_utils import getLogger, bcolors 
from src.benchmark.correctness_utils import check_similiarity
logger = getLogger()

def flops_data_per_tp(config, direction):
    '''
    Assumes all interactions are "uvu" for now

    Returns (flops_per_tp, data_per_tp, nnz)
    '''
    bytes_per_word = np.dtype(config.irrep_dtype).itemsize 

    assert(not config.shared_weights)
    L1, L2, L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
    ops_per_nz, words_per_tp = None, None
    if direction == "forward":
        ops_per_nz = 3
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel 
    elif direction == "backward":
        ops_per_nz = 9
        words_per_tp = L1.dim + L2.dim + L3.dim + config.weight_numel \
                + L1.dim + L2.dim + config.weight_numel # Output gradients

    ops_per_tp = 0
    nnz = 0
    for (u, v, w, connection_mode, *others) in config.instructions:
        tensor = TensorProduct.load_cg_tensor(L1[u].ir.l, L2[v].ir.l, L3[w].ir.l)
        local_nnz = np.count_nonzero(tensor)
        nnz += local_nnz
        ops_per_tp += ops_per_nz * local_nnz * L1[u].mul * L2[v].mul # Assumes L3.mult(w) = L1.mult(u) * L2.mult(v)

        if connection_mode == "uvu":
            ops_per_tp += L3[w].mul * (2 * L3[w].ir.l + 1) 
        elif connection_mode == "uvw":
            ops_per_tp += L1[u].mul * L2[v].mul * L3[w].ir.dim * L3[w].mul

    return ops_per_tp, words_per_tp * bytes_per_word, nnz

class CoordGraph:
    def __init__(self, coords, rows, cols, name):
        '''
        Because graphs may change constantly, this class is designed
        to be as light as possible. A directed edge from node
        u to v is indicated by the presence of an index i such that
        rows[i] = u, rows[i] = v.
        '''
        assert(len(rows) == len(cols))
        self.nnz = len(rows) # Counts every nonzero in the adjacency matrix 
        self.node_count = coords.shape[0]
        self.rows = rows
        self.cols = cols
        self.coords = coords
        self.name = name

        self.cached_sp_graph = None # Cached scipy sparse matrix 

class Convolution:
    next_conv_id = 0 # Used to assign unique IDs to each conv instance 

    def __init__(self, config, torch_op=False, idx_dtype=np.int32):
        self.config = config 
        self.L1, self.L2, self.L3 = config.irreps_in1, config.irreps_in2, config.irreps_out
        self.internal = None
        self.torch_op = torch_op
        self.idx_dtype = idx_dtype

        self.conv_id = Convolution.next_conv_id
        Convolution.next_conv_id += 1

        if torch_op:
            global torch
            import torch

            self.setup_torch_module()

    @staticmethod
    def name():
        raise NotImplementedError()

    def forward_cpu(self, 
            L1_in, L2_in, weights, L3_out,
            graph, disable_tensor_op=False):

        L1_d, L2_d, weights_d = DeviceBuffer(L1_in), DeviceBuffer(L2_in), DeviceBuffer(weights)
        L3_d = DeviceBuffer(L3_out)

        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)

        self.internal.exec_conv_rawptrs(
            L1_d.data_ptr(),
            L2_d.data_ptr(),
            weights_d.data_ptr(),
            L3_d.data_ptr(),
            rows_d.data_ptr(),
            cols_d.data_ptr(),
            graph.nnz,
            graph.node_count,
            disable_tensor_op)

        L3_d.copy_to_host()

    def backward_cpu(self, 
            L1_in, L1_grad, L2_in, L2_grad, weights, weights_grad, 
            L3_grad, graph, disable_tensor_op=False):
        L1_d = DeviceBuffer(L1_in)
        L2_d = DeviceBuffer(L2_in)
        weights_d = DeviceBuffer(weights)
        L3_d = DeviceBuffer(L3_grad)
        rows_d = DeviceBuffer(graph.rows)
        cols_d = DeviceBuffer(graph.cols)
        
        L1_grad_d = DeviceBuffer(L1_grad)
        L2_grad_d = DeviceBuffer(L2_grad)
        weights_grad_d = DeviceBuffer(weights_grad)

        self.internal.backward_rawptrs(
            L1_d.data_ptr(), L1_grad_d.data_ptr(),
            L2_d.data_ptr(), L2_grad_d.data_ptr(),
            weights_d.data_ptr(), weights_grad_d.data_ptr(),
            L3_d.data_ptr(),
            rows_d.data_ptr(), cols_d.data_ptr(),
            graph.nnz, graph.node_count,
            disable_tensor_op)

        L1_grad_d.copy_to_host()
        L2_grad_d.copy_to_host()
        weights_grad_d.copy_to_host()

        return L1_grad, L2_grad, weights_grad

    def test_correctness_forward(self, graph, thresh, prng_seed, reference_implementation=None):
        L1, L2, L3 = self.L1, self.L2, self.L3

        if reference_implementation is None:
            from src.implementations.E3NNConv import E3NNConv
            reference_implementation = E3NNConv

        result = {
            "thresh": thresh 
        }

        in1, in2, weights, out = get_random_buffers_forward_conv(self.config, 
                graph.node_count, graph.nnz, prng_seed)

        ref_tp = reference_implementation(self.config)
        ref_out = out.copy()
        ref_tp.forward_cpu(
            L1_in=in1.copy(), 
            L2_in=in2.copy(), 
            weights=weights.copy(),
            L3_out=ref_out,
            graph=graph)

        test_out = out.copy()
        self.forward_cpu(
            L1_in=in1.copy(), 
            L2_in=in2.copy(),
            weights=weights.copy(),
            L3_out=test_out,
            graph=graph)

        for name, to_check, ground_truth in [
            ("output", ref_out, test_out)]:
            result[name] = check_similiarity(name, to_check, ground_truth, thresh)

        return result

    def benchmark_forward(self, num_warmup, num_iter, graph, disable_tensor_op, prng_seed=12345):
        direction = "forward"
        disable_tensor_op = False
        L1_in, L2_in, weights, L3_buffer = get_random_buffers_forward_conv(self.config, graph.node_count, graph.nnz, prng_seed)

        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
            torch_L1_in = torch.tensor(L1_in, device='cuda')
            torch_L2_in = torch.tensor(L2_in, device='cuda')
            torch_weights = torch.tensor(weights, device='cuda')
            torch_cols = torch.tensor(graph.cols, device='cuda')
            torch_rows = torch.tensor(graph.rows, device='cuda')

            for i in range(num_warmup): 
                torch_L3_out = self.forward(torch_L1_in, torch_L2_in, torch_weights, torch_cols, torch_rows)

            for i in range(num_iter):
                timer.start()
                torch_L3_out = self.forward(torch_L1_in, torch_L2_in, torch_weights, torch_cols, torch_rows)
                time_millis[i] = timer.stop_clock_get_elapsed()

        elif not self.torch_op:
            L1_d, L2_d, weights_d = DeviceBuffer(L1_in), DeviceBuffer(L2_in), DeviceBuffer(weights)
            L3_d = DeviceBuffer(L3_buffer)
            rows_d = DeviceBuffer(graph.rows)
            cols_d = DeviceBuffer(graph.cols)

            for i in range(num_warmup):
                self.internal.exec_conv_rawptrs(
                    L1_d.data_ptr(), L2_d.data_ptr(), weights_d.data_ptr(), L3_d.data_ptr(),
                    rows_d.data_ptr(), cols_d.data_ptr(), graph.nnz, graph.node_count,
                    disable_tensor_op)

            for i in range(num_iter):
                timer.start()
                self.internal.exec_conv_rawptrs(
                    L1_d.data_ptr(), L2_d.data_ptr(), weights_d.data_ptr(), L3_d.data_ptr(),
                    rows_d.data_ptr(), cols_d.data_ptr(), graph.nnz, graph.node_count,
                    disable_tensor_op)
                time_millis[i] = timer.stop_clock_get_elapsed() 

        ops_per_tp, data_per_tp, _ = flops_data_per_tp(self.config, direction)
        ops_per_tp += self.config.irreps_out.dim # Output accumulation... should check this 

        return self.calculate_bench_stats(direction, ops_per_tp, data_per_tp, 
                time_millis, graph, num_warmup, num_iter, prng_seed)


    def benchmark_backward(self, num_warmup, num_iter, graph, disable_tensor_op, prng_seed=12345):
        direction = "backward"
        disable_tensor_op = False
        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward_conv(self.config, graph.node_count, graph.nnz, prng_seed) 

        time_millis = np.zeros(num_iter, dtype=np.float32)
        timer = GPUTimer()

        if self.torch_op:
            torch_L1_in = torch.tensor(in1, device='cuda', requires_grad=True)
            torch_L2_in = torch.tensor(in2, device='cuda', requires_grad=True) 
            torch_weights = torch.tensor(weights, device='cuda', requires_grad=True) 
            torch_cols = torch.tensor(graph.cols, device='cuda').detach()
            torch_rows = torch.tensor(graph.rows, device='cuda').detach()
            torch_out = self.forward(torch_L1_in, torch_L2_in, torch_weights, torch_cols, torch_rows)
            torch_L3_grad = torch.tensor(out_grad, device='cuda') 

            for i in range(num_warmup): 
                torch_out.backward(torch_L3_grad, retain_graph=True, inputs=[torch_L1_in, torch_L2_in, torch_weights])

            for i in range(num_iter):
                torch_L1_in.grad.zero_()
                torch_L2_in.grad.zero_()
                torch_weights.grad.zero_()

                timer.start()
                torch_out.backward(torch_L3_grad, retain_graph=True, inputs=[torch_L1_in, torch_L2_in, torch_weights])
                time_millis[i] = timer.stop_clock_get_elapsed()

        elif not self.torch_op:
            L1_d = DeviceBuffer(in1)
            L2_d = DeviceBuffer(in2)
            weights_d = DeviceBuffer(weights)
            L3_d = DeviceBuffer(out_grad)
            rows_d = DeviceBuffer(graph.rows)
            cols_d = DeviceBuffer(graph.cols)
            
            L1_grad_d = DeviceBuffer(in1_grad)
            L2_grad_d = DeviceBuffer(in2_grad)
            weights_grad_d = DeviceBuffer(weights_grad)

            for i in range(num_warmup):
                self.internal.backward_rawptrs(
                    L1_d.data_ptr(), L1_grad_d.data_ptr(),
                    L2_d.data_ptr(), L2_grad_d.data_ptr(),
                    weights_d.data_ptr(), weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(), cols_d.data_ptr(),
                    graph.nnz, graph.node_count,
                    disable_tensor_op)

            for i in range(num_iter):
                timer.start()
                self.internal.backward_rawptrs(
                    L1_d.data_ptr(), L1_grad_d.data_ptr(),
                    L2_d.data_ptr(), L2_grad_d.data_ptr(),
                    weights_d.data_ptr(), weights_grad_d.data_ptr(),
                    L3_d.data_ptr(),
                    rows_d.data_ptr(), cols_d.data_ptr(),
                    graph.nnz, graph.node_count,
                    disable_tensor_op)
                time_millis[i] = timer.stop_clock_get_elapsed() 

        ops_per_tp, data_per_tp, _ = flops_data_per_tp(self.config, direction)
        ops_per_tp += self.config.irreps_out.dim

        return self.calculate_bench_stats(direction, ops_per_tp, data_per_tp, 
                time_millis, graph, num_warmup, num_iter, prng_seed)

    def calculate_bench_stats(self, direction, ops_per_tp, data_per_tp, time_millis,
            graph, num_warmup, num_iter, prng_seed): 
        throughputs_gflops = [float(el) for el in graph.nnz * ops_per_tp / (time_millis * 1e6)]
        bandwidth_gbps = [float(el) for el in graph.nnz * data_per_tp / (time_millis * 1e6)]
        time_millis = [float(el) for el in time_millis] 

        result = {
            "direction": "forward",
            "flops_per_tp": ops_per_tp,
            "data_per_tp": data_per_tp,

            "time_millis": list(time_millis),
            "throughputs_gflops": list(throughputs_gflops),
            "bandwidth_gbps": list(bandwidth_gbps),

            "L1": str(self.config.irreps_in1),
            "L2": str(self.config.irreps_in2), 
            "L3": str(self.config.irreps_out),
            "graph_node_count": graph.node_count,
            "graph_adj_nnz": graph.nnz,
            "num_warmup": num_warmup,
            "num_iter": num_iter,
            "prng_seed": prng_seed,
            "time_millis": time_millis,
            "throughputs_gflops": throughputs_gflops,
            "bandwidth_gbps": bandwidth_gbps
        }

        logger.info(f"{bcolors.OKCYAN}Avg. Throughput: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(throughputs_gflops):.2f} ± {np.std(throughputs_gflops):.2f} GFLOPs{bcolors.ENDC}")
        logger.info(f"{bcolors.OKCYAN}Avg. Bandwidth: {bcolors.ENDC} {bcolors.OKGREEN}{np.mean(bandwidth_gbps):.2f} ± {np.std(bandwidth_gbps):.2f} GBPs{bcolors.ENDC}")
        return result

    def test_correctness_backward(self, graph, thresh, prng_seed, reference_implementation=None):
        L1, L2, L3 = self.L1, self.L2, self.L3

        if reference_implementation is None:
            from src.implementations.E3NNConv import E3NNConv
            reference_implementation = E3NNConv

        result = {
            "thresh": thresh 
        }

        in1, in2, out_grad, weights, weights_grad, in1_grad, in2_grad = get_random_buffers_backward_conv(self.config, graph.node_count, graph.nnz, prng_seed) 

        ref_tp = reference_implementation(self.config)

        ref_weights_grad = weights_grad.copy()
        ref_in1_grad = in1_grad.copy()
        ref_in2_grad = in2_grad.copy()

        ref_tp.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=ref_in1_grad,
            L2_in=in2.copy(), 
            L2_grad=ref_in2_grad, 
            L3_grad=out_grad.copy(), 
            weights=weights.copy(), 
            weights_grad=ref_weights_grad,
            graph=graph) 

        # run test version
        test_weights_grad = weights_grad.copy()
        test_in1_grad = in1_grad.copy()
        test_in2_grad = in2_grad.copy()

        self.backward_cpu(
            L1_in=in1.copy(),
            L1_grad=test_in1_grad,
            L2_in=in2.copy(), 
            L2_grad=test_in2_grad, 
            L3_grad=out_grad.copy(), 
            weights=weights.copy(), 
            weights_grad=test_weights_grad,
            graph=graph)

        for name, to_check, ground_truth, threshold in [
                ("weight_grad", test_weights_grad, ref_weights_grad, thresh),
                ("in1_grad", test_in1_grad, ref_in1_grad, thresh),
                ("in2_grad", test_in2_grad, ref_in2_grad, thresh)]:
            result[name] = check_similiarity(name, to_check, ground_truth, threshold)

        return result


    def setup_torch_module(self):
        # ----------------- Forward pass -----------------
        @torch.library.custom_op(f"fast_tp::conv_forward{self.conv_id}", mutates_args=(), device_types="cuda")
        def forward(L1_in : torch.Tensor, L2_in : torch.Tensor, 
                weights : torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
            L1_in_c, L2_in_c, weights_c = L1_in.contiguous(), L2_in.contiguous(), weights.contiguous()
            L3_out = torch.zeros((L1_in_c.shape[0], self.L3.dim ), dtype=L1_in.dtype, device='cuda')

            torch._assert(src.shape[0] == dst.shape[0], "src and dst must have the same number of elements")

            self.internal.exec_conv_rawptrs(
                L1_in_c.data_ptr(),
                L2_in_c.data_ptr(),
                weights_c.data_ptr(),
                L3_out.data_ptr(),
                dst.data_ptr(),
                src.data_ptr(),
                src.shape[0],
                L1_in.shape[0],
                False)

            return L3_out
        
        @forward.register_fake
        def _(L1_in, L2_in, weights, src, dst):
            return L1_in.new_empty(L1_in.shape[0], self.L3.dim)
        
        self.forward = forward
        
        # ---------------- Backward pass -----------------
        @torch.library.custom_op(f"fast_tp::conv_backward{self.conv_id}", mutates_args=(), device_types="cuda")
        def backward_op( L1_in : torch.Tensor, L2_in : torch.Tensor, 
                    weights : torch.Tensor, L3_grad : torch.Tensor,
                    src: torch.Tensor, dst: torch.Tensor) -> typing.List[torch.Tensor]:
            L1_grad = torch.zeros_like(L1_in)
            L2_grad = torch.empty_like(L2_in)
            weights_grad = torch.empty_like(weights)

            self.internal.backward_rawptrs(
                    L1_in.data_ptr(), L1_grad.data_ptr(),
                    L2_in.data_ptr(), L2_grad.data_ptr(),
                    weights.data_ptr(), weights_grad.data_ptr(),
                    L3_grad.data_ptr(),
                    dst.data_ptr(), src.data_ptr(),
                    src.shape[0], L1_in.shape[0],
                    False)
            
            return [L1_grad, L2_grad, weights_grad]
        
        @backward_op.register_fake
        def _(L1_in, L2_in, weights, L3_grad, src, dst):
            return [L1_in.new_empty(*L1_in.shape), L2_in.new_empty(*L2_in.shape), weights.new_empty(*weights.shape)]

        def setup_context(ctx, inputs, output):
            ctx.L1_in, ctx.L2_in, ctx.weights, ctx.src, ctx.dst = inputs
        
        def backward(ctx, grad_output):
            result = backward_op(ctx.L1_in, ctx.L2_in, ctx.weights, grad_output, ctx.src, ctx.dst)
            return result[0], result[1], result[2], None, None

        self.forward.register_autograd(backward, setup_context=setup_context)