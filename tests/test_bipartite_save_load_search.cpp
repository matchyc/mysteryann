// Created by Meng Chen from Fudan University in September 2023. email: mengchen22@m.fudan.edu.cn, mengchen9909@gmail.com
// Copyright reserved.

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mysteryann/distance.h"
#include "mysteryann/neighbor.h"
#include "mysteryann/parameters.h"
#include "mysteryann/util.h"
#include "index_bipartite.h"

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"
using namespace mysteryann;
using std::cout;
using std::endl;

class IndexBipartiteTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // generate data write into testfile
        uint32_t dimension = 32;  // Align dimension to 8
        const uint32_t n_sq = 1000;
        const uint32_t n_bp = 1000;
        const uint32_t n_q = 100;

        float *sq_data = new float[n_sq * dimension];
        float *bp_data = new float[n_bp * dimension];
        float *query_data = new float[n_q * dimension];
        std::random_device rd;
        std::mt19937 gen(rd());
        // std::normal_distribution<float> dist_1(0.1, 1.1);
        // std::normal_distribution<float> dist_2(0.1, 1.1);
        std::uniform_real_distribution<float> dist_1(0.1, 1.1);
        std::uniform_real_distribution<float> dist_2(0.1, 1.1);

        for (size_t i = 0; i < n_bp; i++) {
            for (size_t j = 0; j < dimension; j++) {
                bp_data[i * dimension + j] = dist_1(gen);
            }
        }
        for (size_t i = 0; i < n_sq; i++) {
            for (size_t j = 0; j < dimension; j++) {
                sq_data[i * dimension + j] = dist_2(gen);
            }
        }
        for (size_t i = 0; i < n_q; i++) {
            for (size_t j = 0; j < dimension; j++) {
                query_data[i * dimension + j] = dist_2(gen);
            }
        }

        // compute closest neighbor for query_data
        float *dist = new float[n_q];
        unsigned *idx = new unsigned[n_q];
        for (size_t i = 0; i < n_q; i++) {
            dist[i] = 1000;
            idx[i] = 0;
        }
        for (size_t i = 0; i < n_q; i++) {
            for (size_t j = 0; j < n_bp; j++) {
                float tmp_dist = 0;
                for (size_t k = 0; k < dimension; k++) {
                    tmp_dist -= (query_data[i * dimension + k] * bp_data[j * dimension + k]);
                                // (query_data[i * dimension + k] - bp_data[j * dimension + k]);
                }
                if (tmp_dist < dist[i]) {
                    dist[i] = tmp_dist;
                    idx[i] = j;
                }
            }
        }

        std::ofstream bp_file("data/bp_data.fbin", std::ios::binary);
        bp_file.write((char *)&(n_bp), sizeof(uint32_t));
        bp_file.write((char *)&(dimension), sizeof(uint32_t));
        bp_file.write((char *)bp_data, n_bp * dimension * sizeof(float));
        bp_file.close();
        std::ofstream sq_file("data/sq_data.fbin", std::ios::binary);
        sq_file.write((char *)&(n_sq), sizeof(uint32_t));
        sq_file.write((char *)&(dimension), sizeof(uint32_t));
        sq_file.write((char *)sq_data, n_sq * dimension * sizeof(float));
        sq_file.close();
        std::ofstream q_file("data/query_data.fbin", std::ios::binary);
        q_file.write((char *)&(n_q), sizeof(uint32_t));
        q_file.write((char *)&(dimension), sizeof(uint32_t));
        q_file.write((char *)query_data, n_q * dimension * sizeof(float));
        q_file.close();

        uint32_t gt_dim = 1;
        std::ofstream gt_file("data/query_gt.bin", std::ios::binary);
        gt_file.write((char *)&(n_q), sizeof(uint32_t));
        gt_file.write((char *)&(gt_dim), sizeof(uint32_t));
        gt_file.write((char *)idx, n_q * gt_dim * sizeof(unsigned));
        gt_file.close();

        std::vector<std::vector<uint32_t>> simul_res;
        for (size_t i = 0; i < n_q; i++) {
            simul_res.push_back(std::vector<uint32_t>{idx[i]});
        }
        // GTEST_COUT << "Simulate recall (should be 1.0): "
        //    << ComputeRecall(1, 1, simul_res, idx) << std::endl;
        delete[] sq_data;
        delete[] bp_data;
        delete[] query_data;
        delete[] dist;
        delete[] idx;

        dimension_ = dimension;
        nd_ = n_bp;
        nd_sq_ = n_sq;
        total_pts_ = n_bp + n_sq;
        n_q_ = n_q;
        // Load data
        std::string base_file = "data/bp_data.fbin";
        std::string sampled_query_file = "data/sq_data.fbin";
        std::string query_file = "data/query_data.fbin";
        base_data_ = new float[nd_ * dimension_];
        sampled_query_data_ = new float[nd_sq_ * dimension_];
        query_data_ = new float[n_q_ * dimension_];
        // file type: num_pts, dim, data
        std::ifstream base_in(base_file.c_str(), std::ios::binary | std::ios::in);
        std::ifstream sq_in(sampled_query_file.c_str(), std::ios::binary | std::ios::in);
        std::ifstream query_in(query_file.c_str(), std::ios::binary | std::ios::in);

        if (!base_in.is_open() || !sq_in.is_open() || !query_in.is_open()) {
            std::cout << "Error opening file " << base_file << " or " << sampled_query_file << " or " << query_file
                      << std::endl;
            exit(-1);
        }

        uint32_t base_num, sq_num, query_num, dim;
        base_in.read((char *)&base_num, sizeof(uint32_t));

        sq_in.read((char *)&sq_num, sizeof(uint32_t));

        query_in.read((char *)&query_num, sizeof(uint32_t));

        base_in.read((char *)&dim, sizeof(uint32_t));

        sq_in.read((char *)&dim, sizeof(uint32_t));

        query_in.read((char *)&dim, sizeof(uint32_t));

        EXPECT_TRUE(base_in.good());
        base_in.read((char *)base_data_, base_num * dim * sizeof(float));
        sq_in.read((char *)sampled_query_data_, sq_num * dim * sizeof(float));
        query_in.read((char *)query_data_, query_num * dim * sizeof(float));

        base_in.close();
        sq_in.close();
        query_in.close();

        // LoadVectorData(base_file.c_str(), sampled_query_file.c_str(),
        // query_file.c_str(),
        //                base_data_, sampled_query_data_, query_data_);

        aligned_base_data_ = mysteryann::data_align(base_data_, n_bp, dimension);
        aligned_sampled_query_data_ = mysteryann::data_align(sampled_query_data_, n_sq, dimension);
        aligned_query_data_ = mysteryann::data_align(query_data_, n_q, dimension);
    }

    void TearDown() override {
        std::string base_file = "data/bp_data.fbin";
        std::string sampled_query_file = "data/sq_data.fbin";
        std::string query_file = "data/query_data.fbin";
        std::string gt_file = "data/query_gt.bin";

        // std::remove(base_file.c_str());
        // std::remove(sampled_query_file.c_str());
        // std::remove(query_file.c_str());
        // std::remove(gt_file.c_str());
    }

    // intersection of two sets, get the number of common elements and the ratio as recall
    float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
        uint32_t total_count = 0;
        for (uint32_t i = 0; i < q_num; i++) {
            std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
            std::vector<uint32_t> intersection;
            std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
            std::set_intersection(temp_res.begin(), temp_res.end(), one_gt.begin(), one_gt.end(),
                                  std::back_inserter(intersection));
            total_count += intersection.size();
        }
        return (float)total_count / (k * q_num);
    }

    // void LoadVectorData(const char *base_file, const char *sampled_query_file,
    //                     const char *query_file, float *base_data,
    //                     float *sampled_query_data, float *query_data) {

    // }

    float *base_data_;
    float *query_data_;
    Parameters parameters_;
    float *sampled_query_data_;
    float *aligned_base_data_;
    float *aligned_query_data_;
    float *aligned_sampled_query_data_;
    uint32_t dimension_;
    uint32_t nd_;
    uint32_t nd_sq_;
    uint32_t total_pts_;
    uint32_t n_q_;
};

TEST_F(IndexBipartiteTest, SaveLoadTest) {
    // Save bipartite graph
    std::string bipartite_file = "data/random_bipartite.bin";
    std::string projection_file = "data/random_projection.bin";
    parameters_.Set<uint32_t>("num_threads", 4);
    parameters_.Set<uint32_t>("M_sq", 15);
    parameters_.Set<uint32_t>("M_bp", 15);
    parameters_.Set<uint32_t>("L_pq", 50);
    parameters_.Set<uint32_t>("M_pjbp", 15);
    parameters_.Set<uint32_t>("L_pjpq", 50);;
    omp_set_num_threads(parameters_.Get<uint32_t>("num_threads"));

    mysteryann::IndexBipartite index(dimension_, nd_, mysteryann::L2, nullptr);
    index.BuildBipartite(nd_sq_, aligned_sampled_query_data_, nd_, aligned_base_data_, parameters_);
    index.Save(bipartite_file.c_str());
    index.SaveProjectionGraph(projection_file.c_str());

    // Load bipartite graph
    IndexBipartite index2(dimension_, nd_, mysteryann::L2, nullptr);
    index2.Load(bipartite_file.c_str());

    // Check if bp

    for (size_t i = 0; i < total_pts_; ++i) {
        auto &n1 = index.GetBipartiteGraph()[i];
        auto &n2 = index2.GetBipartiteGraph()[i];
        EXPECT_EQ(n1.size(), n2.size());
        for (size_t j = 0; j < n1.size(); ++j) {
            EXPECT_EQ(n1[j], n2[j]);
        }
    }
}

TEST_F(IndexBipartiteTest, SearchBipartiteTest) {
    // Build index
    // parameters_.Set<uint32_t>("M_sq", 10);
    // parameters_.Set<uint32_t>("M_bp", 10);
    parameters_.Set<uint32_t>("L_pq", 5);
    parameters_.Set<uint32_t>("num_threads", 1);
    std::string base_file = "data/bp_data.fbin";
    std::string sampled_query_file = "data/sq_data.fbin";
    std::string query_file = "data/query_data.fbin";
    std::string gt_file = "data/query_gt.bin";

    uint32_t q_pts, q_dim;
    load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    float *aligned_query_data = data_align(query_data, q_pts, q_dim);

    uint32_t gt_pts, gt_dim;
    uint32_t *gt_ids = nullptr;
    load_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    load_data<uint32_t>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);

    IndexBipartite index(q_dim, nd_ + nd_sq_, mysteryann::INNER_PRODUCT, nullptr);

    index.LoadSearchNeededData(base_file.c_str(), sampled_query_file.c_str());
    std::string bipartite_file = "data/random_bipartite.bin";
    index.Load(bipartite_file.c_str());

    // Search
    uint32_t k = 1;
    std::vector<uint32_t> indices(k);
    uint32_t *res = new uint32_t[q_pts * k];
    // record the search time
    uint32_t *cmps = new uint32_t[q_pts];
    auto start = std::chrono::high_resolution_clock::now();
    // index.dist_cmp_metric.reset();
    // index.memory_access_metric.reset();
    for (size_t i = 0; i < q_pts; ++i) {
        cmps[i] = index.SearchBipartiteGraph(aligned_query_data + i * q_dim, k, i, parameters_, res);
        // EXPECT_EQ(indices.size(), k);
        // for (auto &idx : indices) {
        //     EXPECT_LT(idx, nd_);
        // }
        // res.push_back(indices);
    }
    auto end = std::chrono::high_resolution_clock::now();
    // index.dist_cmp_metric.print(std::string("distance: "));
    // index.memory_access_metric.print(std::string("memory: "));
    // index.block_metric.print(std::string("block: "));
    float avg_cmps = 0;
    for (size_t i = 0; i < q_pts; ++i) {
        avg_cmps += cmps[i];
    }
    avg_cmps /= q_pts;

    GTEST_COUT << "Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
               << std::endl;
    // average latency
    GTEST_COUT << "Average latency: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / q_pts << " ms"
               << std::endl;
    GTEST_COUT << "Testing Recall: " << ComputeRecall(q_pts, k, gt_dim, res, gt_ids) << std::endl;
    GTEST_COUT << "Avg cmps: " << avg_cmps << std::endl;
    // EXPECT_EQ(res.size(), q_pts);
}

TEST_F(IndexBipartiteTest, SearchProjectionTest) {
    // Build index
    // parameters_.Set<uint32_t>("M_sq", 10);
    // parameters_.Set<uint32_t>("M_bp", 10);
    parameters_.Set<uint32_t>("L_pq", 50);
    parameters_.Set<uint32_t>("num_threads", 1);
    std::string base_file = "data/bp_data.fbin";
    std::string sampled_query_file = "data/sq_data.fbin";
    std::string query_file = "data/query_data.fbin";
    std::string gt_file = "data/query_gt.bin";

    uint32_t q_pts, q_dim;
    load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    float *aligned_query_data = data_align(query_data, q_pts, q_dim);

    uint32_t gt_pts, gt_dim;
    uint32_t *gt_ids = nullptr;
    load_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    load_data<uint32_t>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);

    IndexBipartite index(q_dim, nd_ + nd_sq_, mysteryann::INNER_PRODUCT, nullptr);

    index.LoadSearchNeededData(base_file.c_str(), sampled_query_file.c_str());
    std::string bipartite_file = "data/random_bipartite.bin";
    std::string projection_file = "data/random_projection.bin";
    index.Load(bipartite_file.c_str());
    index.LoadProjectionGraph(projection_file.c_str());
    // index.Load
    index.InitVisitedListPool(1);
    // Search
    uint32_t k = 1;
    std::vector<uint32_t> indices(k);
    uint32_t *res = new uint32_t[q_pts * k];
    uint32_t *cmps = new uint32_t[q_pts];
    // record the search time
    auto start = std::chrono::high_resolution_clock::now();
    // index.dist_cmp_metric.reset();
    // index.memory_access_metric.reset();
    for (size_t i = 0; i < q_pts; ++i) {
        // cmps[i] = index.SearchProjectionGraph(aligned_query_data + i * q_dim, k, i, parameters_, res);
        // EXPECT_EQ(indices.size(), k);
        // for (auto &idx : indices) {
        //     EXPECT_LT(idx, nd_);
        // }
        // res.push_back(indices);
    }
    auto end = std::chrono::high_resolution_clock::now();
    // index.dist_cmp_metric.print(std::string("distance: "));
    // index.memory_access_metric.print(std::string("memory: "));
    // index.block_metric.print(std::string("block: "));
    GTEST_COUT << "Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
               << std::endl;
    // average latency
    GTEST_COUT << "Average latency: "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / q_pts << " ms"
               << std::endl;
    GTEST_COUT << "Testing Recall: " << ComputeRecall(q_pts, k, gt_dim, res, gt_ids) << std::endl;
    float avg_cmps = 0;
    for (size_t i = 0; i < q_pts; ++i) {
        avg_cmps += cmps[i];
    }
    avg_cmps /= q_pts;
    GTEST_COUT << "Avg cmps: " << avg_cmps << std::endl;  
    // EXPECT_EQ(res.size(), q_pts);
}