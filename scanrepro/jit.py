from torch.utils.cpp_extension import load
scancuda = load(
    'scancuda', ['scancuda.cpp', 'scancuda_kernel.cu'], verbose=True)
help(scancuda)
