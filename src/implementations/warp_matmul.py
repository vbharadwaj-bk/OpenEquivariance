import numpy as np
from build.kernel_wrapper import *
from src.templates.jinja_utils import *

class WarpMatmulTest:
    def __init__(self, M, N, K, dtype_in, dtype_accum, prng_seed=12345):
        self.M, self.N, self.K = M, N, K
        rng = np.random.default_rng(12345)
        self.A = np.array(rng.random((M, K), dtype=np.float32).astype(dtype_in))
        self.B = rng.random((K, N), dtype=np.float32).astype(dtype_in)

        self.A[:] = 1.0
        self.B[:] = 1.0

        self.A[1, 0] = 2.0
        self.B[1, 0] = 2.0
        self.C = (self.A @ self.B).astype(dtype_accum)

    def run(self, kernel):
        cpp_tester = MMTester(kernel)

        A_test = self.A.T.copy()

        a_dev = DeviceBuffer(A_test)
        b_dev = DeviceBuffer(self.B)

        wmm_result = np.zeros((self.M, self.N), dtype=self.C.dtype)
        wmm_dev = DeviceBuffer(wmm_result)
        cpp_tester.execute(a_dev.data_ptr(), b_dev.data_ptr(), wmm_dev.data_ptr())

        wmm_dev.copy_to_host()

        # Try every combo of A.T, B.T, and C.T to see which matmul matches
        # the result of the kernel

        print("#1?", np.allclose(wmm_result, (self.A @ self.B).astype(np.float32)))
        print("#2?", np.allclose(wmm_result, (self.A.T @ self.B).astype(np.float32)))
        print("#3?", np.allclose(wmm_result, (self.A @ self.B.T).astype(np.float32)))
        print("#4?", np.allclose(wmm_result, (self.A.T @ self.B.T).astype(np.float32)))
        print("#5?", np.allclose(wmm_result, (self.A @ self.B).astype(np.float32).T))
        print("#6?", np.allclose(wmm_result, (self.A.T @ self.B).astype(np.float32).T))
        print("#7?", np.allclose(wmm_result, (self.A @ self.B.T).astype(np.float32).T))
        print("#8?", np.allclose(wmm_result, (self.A.T @ self.B.T).astype(np.float32).T))

def test_simple_kernel():
    env = get_jinja_environment()
    template = env.get_template("wmm.cuh")
    env.globals['enumerate'] = enumerate

    M, N, K = 64, 64, 64
    kernel = template.render(M=M, N=N, K=K) 
    test = WarpMatmulTest(M, N, K, np.float16, np.float32)
    test.run(kernel)
