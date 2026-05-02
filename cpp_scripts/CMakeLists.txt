cmake_minimum_required(VERSION 3.20)

project(TRAIN_BPE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(train_bpe SHARED train_bpe.cpp max_heap.cpp)
target_include_directories(train_bpe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(train_bpe PUBLIC MY_LIBRARY_EXPORT)

target_include_directories(train_bpe PUBLIC
    "${PROJECT_SOURCE_DIR}/emhash"
)

install(TARGETS train_bpe
        EXPORT train_bpe_export_targets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES "train_bpe.h" 
        DESTINATION include
)

install(FILES "emhash/hash_table8.hpp" 
        DESTINATION include/emhash
)