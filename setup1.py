from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sysconfig
 
include_path = sysconfig.get_path('include')
internal_include_path = os.path.join(include_path, 'internal')

print(f"{internal_include_path=}")
project_root = os.path.dirname(os.path.abspath(__file__))

ext_modules = [
    Extension(
        name="cpp_scripts.train_bpe_wrapper",
        sources=["cpp_scripts/train_bpe_wrapper.pyx"],

        language="c++",
        #extra_compile_args=['-std=c++17', '-O3'],
        extra_compile_args=['-std=c++17'],
        libraries=["train_bpe"],

        library_dirs=[f"{project_root}/lib_train_bpe/lib"],
        runtime_library_dirs=[f"{project_root}/lib_train_bpe/lib"],
        include_dirs=[f"{project_root}/lib_train_bpe/include",
                      f"{project_root}/lib_train_bpe/include/emhash"],
    )
]

setup(
    packages=['cpp_scripts'],
    name='train_bpe',
    ext_modules=cythonize(ext_modules),
)