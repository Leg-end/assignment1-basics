reference from [BPE](https://github.com/fancyerii/assignment1-basics-bpe)  
also see [BLOG](https://fancyerii.github.io/2025/09/25/bpe-trainer-11/)

# Steps of pack c++ codes as dynamic lib(.so)

- Step 1: place c++ codes(.cpp & .h) under a specific dir, e.g. cpp_scripts
- Step 2: write CMakeLists.txt under cpp_scripts
- Step 3: Encode and install (in project root dir) through following commands:
``` commands
cd cpp_scripts
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=../../${libname}/  -D CMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j8
# 如果是gcc的话可以直接make -j8
cmake --install .  
# 或者make install
```
- Step 4: write .pxd(declare definition of lib) and .pyx(pack function as python-invokable) files and place them under cpp_scripts
- Step 5: write setup to encode .pyx, for example:
``` python code
project_root = os.path.dirname(os.path.abspath(__file__))
ext_modules = [
    Extension(
        name="cpp_scripts.XX",
        sources=["cpp_scripts.XX.pyx"],

        language="c++",
        #extra_compile_args=['-std=c++17', '-O3'],
        extra_compile_args=['-std=c++17'],
        libraries=["${libname}"],

        library_dirs=[f"{project_root}/${libname}/lib"],
        runtime_library_dirs=[f"{project_root}/${libname}/lib"],
        include_dirs=[f"{project_root}/${libname}/include",
                      f"{project_root}/${libname}/include/other libs"],
    )
]

setup(
    packages=['cpp_scripts'],
    name='${libname}',
    ext_modules=cythonize(ext_modules),
)
```
- Step 6: run command: python setup.py build_ext --inplace, then you will see a .so file under cpp_scripts
- Step 7: invoke c++ lib through python code
```python code
from cpp_scripts.XX import YY
# python type to c++ type
results = YY(...)
# c++ type to python type
```

# Notes
## More convienent pack--using pybind11, no need of .pxd and pyx files:
- Step 1: using c++ code to pack lib under support of <pybind11/pybind.h>, e.g.
```python code
#include <pybind11/pybind11.h>
#include "mathlib.h" // 你的 C 头文件

namespace py = pybind11;

// 这是一个可选的辅助函数，用于包装那些有指针参数的 C 函数
// 你也可以在 PYBIND11_MODULE 中直接用 lambda 实现
std::tuple<int, int> get_time_wrapper() {
    int hour, minute;
    get_time(&hour, &minute); // 调用 C 函数
    return std::make_tuple(hour, minute);
}

// 定义一个可以打包 C 结构体的简单 C++ 类或结构体
struct PyPoint {
    int x, y;
    double distance_to(const PyPoint& other) const {
        Point p1{x, y};
        Point p2{other.x, other.y};
        return distance(&p1, &p2);
    }
};

// 将 C 函数和结构体导出到 Python 的模块 'mathlib' 中
PYBIND11_MODULE(mathlib, m) {
    m.doc() = "Example wrapper for a C library";

    // 直接绑定 C 函数
    m.def("add", &add, "A function that adds two numbers");
    m.def("multiply", &multiply, "A function that multiplies two numbers");

    // 使用 lambda 灵活地重新包装函数
    m.def("get_time", []() {
        int hour, minute;
        get_time(&hour, &minute);
        return py::make_tuple(hour, minute);
    });

    // 定义一个 Python 类来封装 C 结构体
    py::class_<PyPoint>(m, "Point")
        .def(py::init<int, int>())
        .def_readwrite("x", &PyPoint::x)
        .def_readwrite("y", &PyPoint::y)
        .def("distance_to", &PyPoint::distance_to);
}
```
- Step 2: write setup file

## cooperate with uv (abandon setup), the whole process will be executed automately by uv
- Step 1: create back-end support c++ encoding
```commands
uv init --lib --build-backend scikit example-ext
cd example-ext
```
- Step 2: add dependencies in pyproject.toml, e.g. add cython or pybind11 in \[build-system\], runtime dependencies in \[project\]
- Step 3: .cpp, CMakeLists, .pxd and .pyx files
- Step 4: exe command ```uv sync ```