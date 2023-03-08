# Building ckExt
self_ck_attn: Uses deviceArray ptr
```
python setup.py install --self_ck_attn
```
self_ck_attn_torch: Uses torchTensor ptr

self_ck_attn_torchTensor: Uses torchTensor

self_ck_attn_deviceAray: Uses deviceArray Object


# Building Dependencies
## composable_kernel
```
module load rocm/5.4.0
git clone git@github.com:irlyngaas/composable_kernel.git
cd composable_kernel
export CMAKE_INSTALL_PREFIX=/PATH/TO/CKLIB
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
Takes awhile to build if on crusher use my build at
```
/gpfs/alpine/med106/world-shared/irl1/ckIntegration/cklib
```

## composable_kernel extensions
```
cd composable_kernel/extension
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/PATH/TO/CKLIB
mkdir build && cd build && cmake .. && make -j
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
export ORT_PATH=$PWD
export ORT_BUILD_PATH=$PWD/build/Release
```

Set environment variables to use kernelExplorer through python need to set these paths
```
export KERNEL_EXPLORER_BUILD_DIR=$ORT_BUILD_PATH
export PYTHONPATH=$PYTHONPATH:$ORTPATH/onnxruntime/python/tools/kernel_explorer/kernels
```
