#/bin/bash
nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
#     -I /home/mathieu/.conda/envs/tensorflow/lib/python2.7/site-packages/tensorflow/include/ \
#     -I /usr/local/cuda-8.0/include \
#     -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
    -I /home/ylao/repo/venvs/semantic3/lib/python3.6/site-packages/tensorflow/include \
    -I /usr/local/cuda/include \
    -lcudart -ltensorflow_framework \
    -L /usr/local/cuda/lib64/ \
    -L /home/ylao/repo/venvs/semantic3/lib/python3.6/site-packages/tensorflow \
    -O2 -D_GLIBCXX_USE_CXX11_ABI=0
