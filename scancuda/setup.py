from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scancuda',
    ext_modules=[
        CUDAExtension('scancuda', [
            'scancuda.cpp',
            'scancuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
