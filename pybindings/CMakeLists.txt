set(CMAKE_CXX_STANDARD 20)

pybind11_add_module(index_mystery_py index_bindings.cpp)

#install
# install(TARGETS index_mystery_py DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../python)