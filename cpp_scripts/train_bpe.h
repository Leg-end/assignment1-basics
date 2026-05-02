#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "emhash/hash_table8.hpp"

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

void train_bpe(
    int vocab_size,          
    std::unordered_map<int, std::vector<int>> & vocabulary, 
    const std::unordered_map<int, long long> & wordid_counts, 
    std::unordered_map<int, std::vector<int>> & wordid_encodings, 
    std::vector<std::pair<std::vector<int>, std::vector<int>>> & merges);  