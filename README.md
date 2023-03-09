# Building ckExt
Need to build dependencies in the following sections and set the following paths
<ul>
<li>CK_PATH: Path to composable_kernel repo</li>
<li>CK_EXT_PATH: Path to composable_kernel extension build dir</li>
<li>CK_LIB_PATH: Path to composable_kernel install dir</li>
<li>ORT_PATH: Path to onnxruntime repo</li>
<li>ORT_BUILD_PATH: Path to onnxruntime build dir</li>
</ul>



self_ck_attn: Uses deviceArray ptr
```
python setup.py install --self_ck_attn
```
self_ck_attn_torch: Uses torchTensor ptr
```
python setup.py install --self_ck_attn_torch
```

self_ck_attn_torchTensor: Uses torchTensor
```
python setup.py install --self_ck_attn_torchTensor
```

self_ck_attn_deviceAray: Uses deviceArray Object
```
python setup.py install --self_ck_attn_deviceArray
```

Any combinations of these is fine


# Building Dependencies
## composable_kernel
```
module load rocm/5.4.0
git clone git@github.com:irlyngaas/composable_kernel.git
cd composable_kernel
export CK_PATH=$PWD
export CMAKE_INSTALL_PREFIX=/PATH/TO/CKLIB
export CK_LIB_PATH=/PATH/TO/CKLIB
mkdir build && cd build
cmake                                                                                             \   
-D CMAKE_PREFIX_PATH=/opt/rocm-5.4.0                                                              \   
-D CMAKE_CXX_COMPILER=/opt/rocm-5.4.0/bin/hipcc                                                   \   
-D CMAKE_CXX_FLAGS="-O3"                                                                          \   
-D CMAKE_BUILD_TYPE=Release              m                                                         \   
-D GPU_TARGETS="gfx908;gfx90a"                                                                    \   
..
make install

```
May take awhile to build if on crusher feel free to use my build at
```
/gpfs/alpine/med106/world-shared/irl1/CKREDO/cklib/usr/local
```

## composable_kernel extensions
```
cd composable_kernel/extension
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/PATH/TO/CKLIB
mkdir build && cd build && export CK_EXT_PATH=$PWD && cmake .. && make -j
```

## onnxruntime
Set up python
```
module load PrgEnv-gnu rocm/5.4.0 cray-python
python3 -m venv onnx_env
source onnx_env/bin/activate
pip3 install --upgrade pip; pip3 install numpy, pip3 install cmake
```
Build onnx
```
git clone git@github.com:irlyngaas/onnxruntime.git
cd onnxruntime
bash build_ke.sh
build_dir="build"
config="Release"

rocm_home="/opt/rocm-5.4.0"

./build.sh --update \
    --build_dir ${build_dir} \
    --config ${config} \
    --cmake_extra_defines \
        CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        onnxruntime_ENABLE_ATEN=OFF \
    --skip_submodule_sync --skip_tests \
    --use_rocm --rocm_home=${rocm_home} --nccl_home=${rocm_home} \
    --build_wheel

cmake --build ${build_dir}/${config} --target kernel_explorer --parallel
export ORT_PATH=$PWD
export ORT_BUILD_PATH=$PWD/build/Release
```

Set environment variables needed to use kernelExplorer through python
```
export KERNEL_EXPLORER_BUILD_DIR=$ORT_BUILD_PATH
export PYTHONPATH=$PYTHONPATH:$ORTPATH/onnxruntime/python/tools/kernel_explorer/kernels
```

Run onnxruntime gemm_softmax_gemm
```
python exp_gemm_softmax_gemm.py
```

## Running Test scripts
Modify ckExt/test/run_rocm_extensions.py:L6 to whichever unit test you want to run, for example
```
test_dirs = ["self_ck_attn"] #Runs only deviceArray w/ pointer Test
test_dirs = ["self_ck_attn", "self_ck_attn_torch"] #Runs 2 unit tests
```

Run unit test on Crusher
```
salloc -A PROJ -t TIME -p batch -N 1
cd ckExt/test/self_ck_attn
srun -n1 python ckExt/test/self_ck_attn/run_rocm_extensions.py
```

Run simple test on Crusher (Mimics exp_gemm_softmax_gemm.py in onnxruntime repo; only exists for options using deviceArray)
```
srun -n1 python ckExt/test/self_ck_attn/simpletest.py
```
