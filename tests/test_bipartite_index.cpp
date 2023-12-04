#include <gtest/gtest.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "mysteryann/distance.h"
#include "mysteryann/neighbor.h"
#include "mysteryann/parameters.h"
#include "mysteryann/util.h"
#include "index_bipartite.h"

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"

TEST(IndexBipartiteTest, BuildBipartite) {
    // Generate test data
    uint32_t dimension = 8;  // Align dimension to 8
    const size_t n_sq = 1000;
    const size_t n_bp = 1000;
    float *sq_data = new float[n_sq * dimension];
    float *bp_data = new float[n_bp * dimension];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist1(0.5, 1.5);
    std::normal_distribution<float> dist2(0.1, 1.1);

    for (size_t i = 0; i < n_sq; i++) {
        for (size_t j = 0; j < dimension; j++) {
            sq_data[i * dimension + j] = dist1(gen);
        }
    }
    for (size_t i = 0; i < n_bp; i++) {
        for (size_t j = 0; j < dimension; j++) {
            bp_data[i * dimension + j] = dist2(gen);
        }
    }
    float *aligned_sq = mysteryann::data_align(sq_data, n_sq, dimension);
    float *aligned_bp = mysteryann::data_align(bp_data, n_bp, dimension);
    mysteryann::Parameters parameters;
    parameters.Set<uint32_t>("M_sq", 10);
    parameters.Set<uint32_t>("M_bp", 10);
    parameters.Set<uint32_t>("L_pq", 20);
    parameters.Set<uint32_t>("M_pjbp", 20);
    parameters.Set<uint32_t>("L_pjpq", 20);
    parameters.Set<uint32_t>("num_threads", 1);
    std::string bipartite_file = "data/random_bipartite.bin";
    mysteryann::IndexBipartite index(dimension, n_bp, mysteryann::L2, nullptr);
    index.BuildBipartite(n_sq, aligned_sq, n_bp, aligned_bp, parameters);

    EXPECT_EQ(index.GetBipartiteGraph().size(), n_sq + n_bp);
    // Check if bipartite graph's degree meets M_sq and M_bp
    size_t i = 0;
    for (; i < n_bp; i++) {
        size_t degree = index.GetBipartiteGraph()[i].size();
        EXPECT_GE(degree, 1);
        EXPECT_LE(degree, parameters.Get<uint32_t>("M_bp") + 2);
        // GTEST_COUT << "i = " << i << ", degree = " << degree << std::endl;
    }
    for (; i < (n_sq + n_bp); i++) {
        size_t degree = index.GetBipartiteGraph()[i].size();
        EXPECT_GE(degree, 1);
        EXPECT_LE(degree, parameters.Get<uint32_t>("M_sq") + 2);
        // GTEST_COUT << "i = " << i << ", degree = " << degree << std::endl;
    }
    EXPECT_EQ(i, n_sq + n_bp);

    i = 0;
    for (; i < n_bp; i++) {
        auto &nbrs = index.GetBipartiteGraph()[i];
        for (auto &nbr : nbrs) {
            EXPECT_GE(nbr, n_bp);
            EXPECT_LT(nbr, n_sq + n_bp);
        }
    }
    for (; i < (n_sq + n_bp); i++) {
        auto &nbrs = index.GetBipartiteGraph()[i];
        for (auto &nbr : nbrs) {
            EXPECT_GE(nbr, 0);
            EXPECT_LT(nbr, n_bp);
        }
    }
    // check neighbor is unique
    // for (size_t i = 0; i < index.GetBipartiteGraph().size(); i++) {
    //     auto &nbrs = index.GetBipartiteGraph()[i];
    //     std::sort(nbrs.begin(), nbrs.end());
    //     auto last = std::unique(nbrs.begin(), nbrs.end());
    //     EXPECT_EQ(last, nbrs.end());
    // }

    std::string projection_file = "data/random_projection.bin";
    index.Save(bipartite_file.c_str());
    index.SaveProjectionGraph(projection_file.c_str());
    // Clean up
    delete[] aligned_sq;
    delete[] aligned_bp;
}

TEST(IndexBipartiteTest, CheckConnectivity) {
    // Generate test data
    uint32_t dimension = 8;  // Align dimension to 8
    const size_t n_sq = 100;
    const size_t n_bp = 100;
    float *sq_data = new float[n_sq * dimension];
    float *bp_data = new float[n_bp * dimension];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < n_sq; i++) {
        for (size_t j = 0; j < dimension; j++) {
            sq_data[i * dimension + j] = dist(gen);
        }
    }
    for (size_t i = 0; i < n_bp; i++) {
        for (size_t j = 0; j < dimension; j++) {
            bp_data[i * dimension + j] = dist(gen);
        }
    }
    float *aligned_sq = mysteryann::data_align(sq_data, n_sq, dimension);
    float *aligned_bp = mysteryann::data_align(bp_data, n_bp, dimension);
    mysteryann::Parameters parameters;
    parameters.Set<uint32_t>("M_sq", 10);
    parameters.Set<uint32_t>("M_bp", 10);
    parameters.Set<uint32_t>("L_pq", 10);
    parameters.Set<uint32_t>("M_pjbp", 20);
    parameters.Set<uint32_t>("L_pjpq", 20);
    parameters.Set<uint32_t>("num_threads", 1);

    mysteryann::IndexBipartite index(dimension, n_bp, mysteryann::L2, nullptr);
    index.BuildBipartite(n_sq, aligned_sq, n_bp, aligned_bp, parameters);

    // Check if bipartite graph is well-connected or only has one connected component
    std::vector<bool> visited(n_sq + n_bp, false);
    std::queue<size_t> q;
    q.push(n_bp);
    visited[n_bp] = true;
    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        for (auto &v : index.GetBipartiteGraph()[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    bool is_connected = true;
    for (size_t i = 0; i < n_sq + n_bp; i++) {
        if (!visited[i]) {
            is_connected = false;
            break;
        }
    }
    EXPECT_TRUE(is_connected);

    // Clean up
    delete[] aligned_sq;
    delete[] aligned_bp;
}
