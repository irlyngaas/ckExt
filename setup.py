import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
from setuptools import setup, find_packages
import subprocess

import sys
import warnings
import os

ROCM_PATH = os.environ["ROCM_PATH"]
CK_PATH = os.environ["CK_PATH"]
CK_EXT_PATH = os.environ["CK_EXT_PATH"]
CK_LIB_PATH = os.environ["CK_LIB_PATH"]
ORT_PATH = os.environ["ORT_PATH"]
ORT_BUILD_PATH = os.environ["ORT_BUILD_PATH"]

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
torch_dir = torch.__path__[0]

context_file = os.path.join(torch_dir, "include", "ATen", "Context.h")
if os.path.exists(context_file):
    lines = open(context_file, 'r').readlines()
    found_Backward_Pass_Guard = False
    found_ROCmBackward_Pass_Guard = False
    for line in lines:
        if "BackwardPassGuard" in line:
            # BackwardPassGuard has been renamed to ROCmBackwardPassGuard
            # https://github.com/pytorch/pytorch/pull/71881/commits/4b82f5a67a35406ffb5691c69e6b4c9086316a43
            if "ROCmBackwardPassGuard" in line:
                found_ROCmBackward_Pass_Guard = True
            else:
                found_Backward_Pass_Guard = True
            break

found_aten_atomic_header = False
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "Atomic.cuh")):
    found_aten_atomic_header = True

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

def check_if_rocm_pytorch():
    is_rocm_pytorch = False
    if TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 5):
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

    return is_rocm_pytorch

IS_ROCM_PYTORCH = check_if_rocm_pytorch()

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

ext_modules = []
extras = {}

if "--self_ck_attn" in sys.argv:
    if "--self_ck_attn" in sys.argv:
        sys.argv.remove("--self_ck_attn")
    hipcc_args_mha = ['-O3',
                      '-std=c++17',
                      '-I'+CK_LIB_PATH+'/include',
                      '-I'+CK_PATH+'/extension/fused_attention',
                      '-I'+ORT_PATH+'/onnxruntime/python/tools/kernel_explorer',
                      '-I'+ORT_PATH+'/onnxruntime',
                      '-I'+ORT_PATH+'/include/onnxruntime',
                      '-I'+ORT_BUILD_PATH,
                      '-I'+ORT_BUILD_PATH+'/_deps/gsl-src/include',
                      '-I'+ORT_BUILD_PATH+'/_deps/abseil_cpp-src',
                      '-I'+ORT_BUILD_PATH+'/_deps/mp11-src/include',
                      '-I'+ORT_BUILD_PATH+'/_deps/google_nsync-src/public',
                      '-I'+ROCM_PATH+'/include/hiprand',
                      '-I'+ROCM_PATH+'/include/rocrand',
                      '-U__HIP_NO_HALF_OPERATORS__',
                      '-U__HIP_NO_HALF_CONVERSIONS__'] + generator_flag
    if found_Backward_Pass_Guard:
        hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=BackwardPassGuard']
    if found_ROCmBackward_Pass_Guard:
        hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=ROCmBackwardPassGuard']

    ext_modules.append(
        CUDAExtension(
            name='self_ck_attn',
            sources=[
                'ckExt/csrc/self_ck_attn/ck_attn_frontend.cu',
                'ckExt/csrc/self_ck_attn/self_ck_attn.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc'),
                                    os.path.join(this_dir, 'ckExt/csrc/self_ck_attn')],
                      extra_compile_args={'cxx': ['-O3',] + generator_flag,
                                          'nvcc': hipcc_args_mha},
                      library_dirs=[CK_EXT_PATH+'/fused_attention', 
                                    CK_LIB_PATH+'/lib', 
                                    ORT_BUILD_PATH],
                      libraries=['fused_attention', 'device_operations']
                      #libraries=['fused_attention', 'device_operations', '_kernel_explorer', 'onnxruntime_providers_rocm']
                      #libraries=['fused_attention', 'device_operations', 'onnxruntime_providers_rocm', 'onnxruntime_pybind11_state']
                      #dlink=True,
                      #dlink_libraries=['fused_attention', 'device_operations']
        )
    )

if "--self_ck_attn_torch" in sys.argv:
    if "--self_ck_attn_torch" in sys.argv:
        sys.argv.remove("--self_ck_attn_torch")
    hipcc_args_mha = ['-O3',
                      '-std=c++17',
                      '-I'+CK_LIB_PATH+'/include',
                      '-I'+CK_PATH+'/extension/fused_attention',
                      '-I'+ROCM_PATH+'/include/hiprand',
                      '-I'+ROCM_PATH+'/include/rocrand',
                      '-U__HIP_NO_HALF_OPERATORS__',
                      '-U__HIP_NO_HALF_CONVERSIONS__'] + generator_flag
    if found_Backward_Pass_Guard:
        hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=BackwardPassGuard']
    if found_ROCmBackward_Pass_Guard:
        hipcc_args_mha = hipcc_args_mha + ['-DBACKWARD_PASS_GUARD'] + ['-DBACKWARD_PASS_GUARD_CLASS=ROCmBackwardPassGuard']

    ext_modules.append(
        CUDAExtension(
            name='self_ck_attn_torch',
            sources=[
                'ckExt/csrc/self_ck_attn_torch/ck_attn_frontend.cpp',
                'ckExt/csrc/self_ck_attn_torch/self_ck_attn.cu',
            ],
            include_dirs=[os.path.join(this_dir, 'csrc'),
                                    os.path.join(this_dir, 'ckExt/csrc/self_ck_attn')],
                      extra_compile_args={'cxx': ['-O3',] + generator_flag,
                                          'nvcc': hipcc_args_mha},
                      library_dirs=[CK_EXT_PATH+'/fused_attention', 
                                    CK_LIB_PATH+'/lib'],
                      libraries=['fused_attention', 'device_operations']
                      #libraries=['fused_attention', 'device_operations', '_kernel_explorer', 'onnxruntime_providers_rocm']
                      #libraries=['fused_attention', 'device_operations', 'onnxruntime_providers_rocm', 'onnxruntime_pybind11_state']
                      #dlink=True,
                      #dlink_libraries=['fused_attention', 'device_operations']
        )
    )

setup(
    name="ckExt",
    version="0.1",
    packages=find_packages(
        exclude=("csrc", "python", "test",)
        #exclude=("build", "csrc", "include", "tests", "dist", "docs", "tests", "examples", "apex.egg-info",)
    ),
    description="Composable Kernel PyTorch Extensions",
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    extras_require=extras,
)
