from src.comparison.e3nn import *
from src.torch_modules.fast_tp import test_drive 

if __name__=='__main__':
    test_drive()
    #configs = [
        #("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3), # Last value is Lmax
    #    ("32x1e + 32x0e", "1x0e + 1x1e", 3)
        #("32x5e", "1x5e", "32x3e")
    #]
    #batch_size = 1

    #for config in configs:
    #    compare_output_to_e3nn(config, batch_size)
