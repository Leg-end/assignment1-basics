# distutils: language = c++

# 从 .pxd 文件导入声明
from cpp_scripts.train_bpe_wrapper cimport train_bpe
from cpp_scripts.train_bpe_wrapper cimport pair_hash, HashMap
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
import time
import psutil
import os
import gc
import sys

cpdef py_train_bpe(int vocab_size,
                   vocabulary_py,
                   wordid_counts_py,
                   wordid_encodings_py,
                   merges_py):

    # 声明 C++ 容器
    cdef unordered_map[int, vector[int]] vocabulary_cpp
    cdef unordered_map[int, long long] wordid_counts_cpp
    cdef unordered_map[int, vector[int]] wordid_encodings_cpp
    cdef vector[pair[vector[int], vector[int]]] merges_cpp

 
    vocabulary_cpp = vocabulary_py

    wordid_counts_cpp = wordid_counts_py

    wordid_encodings_cpp = wordid_encodings_py
    # 调用 C++ 函数
    train_bpe(vocab_size,
              vocabulary_cpp,
              wordid_counts_cpp,
              wordid_encodings_cpp,
              merges_cpp)

    return merges_cpp, vocabulary_cpp