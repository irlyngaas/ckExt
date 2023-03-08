# Building ckExt
self_ck_attn: Uses deviceArray
```
python setup.py install --self_ck_attn
```


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
-D CMAKE_BUILD_TYPE=Release                                                                       \   
-D GPU_TARGETS="gfx908;gfx90a"                                                                    \   
..
make install

```

## composable_kernel extensions
```
cd composable_kernel/extension
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/PATH/TO/CKLIB
mkdir build && cd build && cmake .. && make -j
```
## onnxruntime
