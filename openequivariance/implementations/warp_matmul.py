import numpy as np
from openequivariance.kernel_wrapper import *
from openequivariance.templates.jinja_utils import *

class WarpMatmulTest:
    def __init__(self, M, N, K, dtype, prng_seed=12345):
        self.M, self.N, self.K = M, N, K
        rng = np.random.default_rng(12345)
        self.A = rng.random((M, K), dtype=dtype)
        self.B = rng.random((K, N), dtype=dtype)

        self.A = np.array(self.A, order='F')
        self.C = self.A @ self.B 

    def run(self, kernel):
        cpp_tester = MMTester(kernel)

        a_dev = DeviceBuffer(self.A)
        b_dev = DeviceBuffer(self.B)

        wmm_result = np.zeros((self.M, self.N), dtype=self.C.dtype, order='F')
        wmm_dev = DeviceBuffer(wmm_result)
        cpp_tester.execute(a_dev.data_ptr(), b_dev.data_ptr(), wmm_dev.data_ptr())

        wmm_dev.copy_to_host()

        print("WarpMatmul pass?", np.allclose(wmm_result, self.C))
        print(wmm_result)
        print(self.C)


def test_simple_kernel():
    env = get_jinja_environment()
    template = env.get_template("wmm.cuh")
    env.globals['enumerate'] = enumerate

    M, N, K = 17, 3, 5 
    kernel = template.render(M=M, N=N, K=K, 
            TILES_PER_ROW=2, STOREBACK_RMAJOR=False)
    test = WarpMatmulTest(M, N, K, np.float32)
    test.run(kernel)
