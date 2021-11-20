from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sph3d_ops',
    ext_modules=[
        CUDAExtension('sph3d_ops_cuda', [
            'sph3d_ops_api.cpp',

            'unpooling/unpool3d.cpp',
            'unpooling/unpool3d_gpu.cu',
            'pooling/pool3d.cpp',
            'pooling/pool3d_gpu.cu',
            'nnquery/nnquery.cpp',
            'nnquery/nnquery_gpu.cu',
            'convolution/conv3d.cpp',
            'convolution/conv3d_gpu.cu',
            'buildkernel/buildkernel.cpp',
            'buildkernel/buildkernel_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
