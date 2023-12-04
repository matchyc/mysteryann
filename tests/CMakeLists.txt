set(CMAKE_CXX_STANDARD 17)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

target_link_libraries(${PROJECT_NAME}
	gtest
	gtest_main
	# mockcpp
	pthread
	)

	# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable
add_executable(test_util test_util.cpp)
target_link_libraries(test_util ${GTEST_LIBRARIES} gtest_main pthread ${PROJECT_NAME})

add_executable(test_bipartite_index test_bipartite_index.cpp)
target_link_libraries(test_bipartite_index ${GTEST_LIBRARIES} gtest_main pthread ${PROJECT_NAME})

add_executable(test_bipartite_save_load_search test_bipartite_save_load_search.cpp)
target_link_libraries(test_bipartite_save_load_search ${GTEST_LIBRARIES} gtest_main pthread ${PROJECT_NAME})

add_executable(test_build_real_bipartite test_build_real_bipartite.cpp)
target_link_libraries(test_build_real_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME}  Boost::program_options)

add_executable(test_t2i_100 test_t2i_100.cpp)
target_link_libraries(test_t2i_100 ${GTEST_LIBRARIES} pthread ${PROJECT_NAME})

add_executable(test_t2i_1k test_t2i_1k.cpp)
target_link_libraries(test_t2i_1k ${GTEST_LIBRARIES} pthread ${PROJECT_NAME})

add_executable(test_search_real_bipartite test_search_real_bipartite.cpp)
target_link_libraries(test_search_real_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_build_edge_add_after_link test_build_edge_add_after_link.cpp)
target_link_libraries(test_build_edge_add_after_link ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_build_roargraph test_build_roargraph.cpp)
target_link_libraries(test_build_roargraph ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_only_bipartite_build test_only_bipartite_build.cpp)
target_link_libraries(test_only_bipartite_build ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)

add_executable(test_only_search_bipartite test_only_search_bipartite.cpp)
target_link_libraries(test_only_search_bipartite ${GTEST_LIBRARIES} pthread ${PROJECT_NAME} Boost::program_options)
