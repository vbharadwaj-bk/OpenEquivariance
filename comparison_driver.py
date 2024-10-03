from src.comparison.e3nn import *

if __name__=='__main__':
    configs = [
        ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 3), # Last value is Lmax
        ("32x3e + 32x2e + 32x1e + 32x0e", "1x0e + 1x1e + 1x2e", 4),
        ("32x2e + 32x1e + 32x0e", "1x0e + 1x1e", 3)
    ]
    batch_size = 1

    for config in configs[:1]:
        compare_output_to_e3nn(config, batch_size)
