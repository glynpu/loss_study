#https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version/46380601#46380601  
cmake ../ \
  -DUSE_CUDA:BOOL=True \
  -DCMAKE_PREFIX_PATH=/ceph-ly/env/kenlm/lib/python3.8/site-packages/torch/share/cmake/
