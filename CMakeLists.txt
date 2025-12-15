cmake_minimum_required(VERSION 3.14)
project(GremLinEqRe)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 自动寻找 Python 环境
set(Python_EXECUTABLE "/home/ljq/miniconda3/envs/few_env_2025/bin/python")
find_library(GMP_LIB NAMES gmp)
find_library(MPFR_LIB NAMES mpfr)
find_library(FLINT_LIB NAMES flint)
find_library(ARB_LIB arb)
find_library(ACB_LIB acb)




# 2. 自动下载并集成 pybind11 (使用 FetchContent 模式，无需你手动安装)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)
FetchContent_MakeAvailable(pybind11)

# 3. 添加头文件路径
include_directories(include)
include_directories(include ${eigen3_SOURCE_DIR})
# 4. 定义扩展模块 _core
pybind11_add_module(_core
    src/KerrGeo.cpp
    python/bindings.cpp
    src/TeukolskyRadial.cpp
    src/Clebsch.cpp  
    src/SWSH.cpp
    src/TeukolskySource.cpp      
)
target_link_libraries(_core PRIVATE
    ${ARB_LIB}
    ${FLINT_LIB}
    ${MPFR_LIB}
    ${GMP_LIB}
    m # 数学库
)
# 5. 设置输出目录，方便测试脚本直接引用
# 将编译好的库直接放在项目根目录下的 GremLinEqRe 文件夹里，模拟 Python 包结构
set_target_properties(_core PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/GremLinEqRe"
)