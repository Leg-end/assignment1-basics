# distutils: language = c++

# 导入 C++ 标准库类型
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set


cdef extern from "../lib_train_bpe/include/train_bpe.h" :
    cppclass pair_hash:
        pass

cdef extern from "../lib_train_bpe/include/emhash/hash_table8.hpp" namespace "emhash8":
    cppclass HashMap[K, V, H]:
        #ValueT& operator[](const KeyT& key) noexcept
        V& operator[](const K& key)


cdef extern from "../lib_train_bpe/include/train_bpe.h":
    void train_bpe(int vocab_size,
                   unordered_map[int, vector[int]] & vocabulary,
                   const unordered_map[int, long long] & wordid_counts,
                   unordered_map[int, vector[int]] & wordid_encodings,
                   vector[pair[vector[int], vector[int]]] & merges) except +