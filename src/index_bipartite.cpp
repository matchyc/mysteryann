#include "index_bipartite.h"

#include <omp.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <ctime>
#include <random>
#include <atomic>
#include <iostream>
#include <thread>

#include "mysteryann/exceptions.h"
#include "mysteryann/parameters.h"

// define likely unlikely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define PROJECTION_SLACK 1.15

namespace mysteryann {

IndexBipartite::IndexBipartite(const size_t dimension, const size_t n, Metric m, Index *initializer)
    : Index(dimension, n, m), initializer_{initializer}, total_pts_const_(n) {
    bipartite_ = true;
    l2_distance_ = new DistanceL2();
    width_ = 1;
    if (m == mysteryann::COSINE) {
        need_normalize = true;
    }
}

IndexBipartite::~IndexBipartite() {
    if (row_ptr_ != nullptr) {
        delete[] row_ptr_;
    }
    if (col_idx_ != nullptr) {
        delete[] col_idx_;
    }
    if (visited_list_pool_ != nullptr) {
        delete visited_list_pool_;
    }
}

void IndexBipartite::FreeBaseData() {
    if (data_bp_ == nullptr) {
        return;
    }

    //delete base data
    free(const_cast<float*>(data_bp_));
}

void IndexBipartite::SaveBaseData(const char * filename) {
    if (data_bp_ == nullptr) {
        return;
    }
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    uint32_t dim = dimension_;
    out.write((char *)&u32_nd_, sizeof(u32_nd_));
    out.write((char *)&dim, sizeof(dim));
    out.write((char *)data_bp_, sizeof(float) * nd_ * dimension_);
    out.close();
    //delete base data
    free(const_cast<float*>(data_bp_));
}

void IndexBipartite::BuildGraphST1(uint32_t n_sq, uint32_t n_bp, uint32_t M_bp, uint32_t L_pq, float *bp_data, const mysteryann::Parameters parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    data_bp_ = bp_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    if (need_normalize) {
        std::cout << "normalizing base data" << std::endl;
        for (size_t i = 0; i < nd_; ++i) {
            float *data = const_cast<float *>(data_bp_);
            normalize(data + i * dimension_, dimension_);
        }
    }
    SimpleNeighbor *simple_graph = nullptr;
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    supply_nbrs_.resize(nd_);
    std::cout << "begin projection graph init" << std::endl;
    projection_graph_.resize(u32_nd_);
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        projection_graph_[i].reserve(M_bp * PROJECTION_SLACK);
    }
    CalculateProjectionep();
    std::cout << "begin link projection" << std::endl;
    LinkProjection(parameters, simple_graph);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;

    has_built = true;
}

void IndexBipartite::AddEdgesInline(const Parameters &parameters) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));

    for (size_t i = 0; i < supply_nbrs_.size(); ++i) {
        // projection_graph_[i].clear();
        // projection_graph_[i] = supply_nbrs_[i];
        // swap to projection_graph_, free supply
        projection_graph_[i].swap(supply_nbrs_[i]);
        projection_graph_[i].reserve(M_pjbp * 2 * PROJECTION_SLACK);
        // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
    }
    std::vector<uint32_t> vis_order;
    std::vector<uint32_t> vis_order_sq;
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        vis_order.push_back(i);
    }
    for (uint32_t i = 0; i < u32_nd_sq_; ++i) {
        vis_order_sq.push_back(i);
    }

    #pragma omp parallel for schedule(static, 100)
    for (uint32_t it_sq = 0; it_sq < u32_nd_sq_; ++it_sq) {
        uint32_t sq = vis_order_sq[it_sq];
        boost::dynamic_bitset<> visited{u32_nd_, false};
        // auto &nn_base = learn_base_knn_[sq];
        // if (nn_base.size() > 100) {
        //     nn_base.resize(100);
        //     nn_base.shrink_to_fit();
        // }
        // uint32_t choose_tgt = 0;
        // for (size_t i = 0; i < 100; ++i) {
        //     if (projection_graph_[nn_base[i]].size() < M_pjbp) {
        //         choose_tgt = nn_base[i];
        //         break;
        //     }
        // }
        uint32_t cur_tgt = sq;
        std::vector<Neighbor> full_retset;
        NeighborPriorityQueue retset(L_pjpq);
        full_retset.reserve(L_pjpq);

        SearchProjectionGraphInternalPJ(retset, data_sq_ + dimension_ * sq, sq + u32_nd_, parameters, visited, full_retset);
        // SearchProjectionGraphInternal(retset, data_sq_ + dimension_ * sq, sq + u32_nd_, parameters, visited, full_retset);
        std::sort(full_retset.begin(), full_retset.end());

        std::vector<uint32_t> pruned_list;
        // pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
        cur_tgt = full_retset[0].id;
        std::remove_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == cur_tgt; });
        // if (projection_graph_[cur_tgt].size() == 0) {
        // } else {
        //     std::remove_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == cur_tgt; });
        //     std::vector<uint32_t> vec_nbrs;
        //     {
        //         LockGuard guard(locks_[cur_tgt]);
        //         vec_nbrs = projection_graph_[cur_tgt];
        //     }
        //     for(auto& nbr: vec_nbrs) {
        //         if (nbr == cur_tgt) {
        //             continue;
        //         }
                
        //         if (std::find_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == nbr; }) == full_retset.end()) {
        //             float dist = distance_->compare(data_sq_ + dimension_ * (size_t)sq, data_bp_ + dimension_ * (size_t)nbr, dimension_);
        //             full_retset.emplace_back(nbr, dist, false);
        //         }
        //     }
        // }
        // for (size_t i = 0; i < full_retset.size(); ++i) {
        //     if (pruned_list.size() >= M_pjbp * PROJECTION_SLACK) {
        //         break;
        //     }
        //     pruned_list.push_back(full_retset[i].id);
        // }
        PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * cur_tgt, cur_tgt, parameters, pruned_list);
        {
            LockGuard guard(locks_[cur_tgt]);
            projection_graph_[cur_tgt] = pruned_list;
        }
        ProjectionAddReverse(cur_tgt, parameters);
        if (sq % 1000 == 0) {
            std::cout << "\r" << (100.0 * sq) / u32_nd_sq_ << "% of projection search bipartite by base completed."
                      << std::flush;
        }
    }

    std::cout << std::endl;
    // std::atomic<uint32_t> degree_cnt(0);
    // std::atomic<uint32_t> zero_cnt(0);
    // std::atomic<uint32_t> less_5_cnt(0);
#pragma omp parallel for schedule(static, 100)
    for (uint32_t i = 0; i < vis_order.size(); ++i) {
        uint32_t node = vis_order[i];
        // if (projection_graph_[node].size() < M_pjbp) {
        //     // std::cout << "Warning: projection graph node " << node << " has less than M_pjbp neighbors." << std::endl;
        //     // degree_cnt.fetch_add(1);
        //     if (projection_graph_[node].size() == 0) {
        //         zero_cnt.fetch_add(1);
        //     }
        //     if (projection_graph_[node].size() < 5) {
        //         less_5_cnt.fetch_add(1);
        //     }
        // }
        ProjectionAddReverse(node, parameters);
    }
    // std::cout << "Warning: " << degree_cnt.load() << " nodes have less than M_pjbp neighbors." << std::endl;
    // std::cout << "Warning: " << zero_cnt.load() << " nodes have no neighbors." << std::endl;
    // std::cout << "Warning: " << less_5_cnt.load() << " nodes have less than 5 neighbors." << std::endl;

#pragma omp parallel for schedule(dynamic, 100)
    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> ok_insert;
        ok_insert.reserve(M_pjbp);
        for (size_t j = 0; j < supply_nbrs_[i].size(); ++j) {
            if (ok_insert.size() >= M_pjbp * 2) {
                break;
            }
            if (std::find(projection_graph_[i].begin(), projection_graph_[i].end(), supply_nbrs_[i][j]) ==
                projection_graph_[i].end()) {
                ok_insert.push_back(supply_nbrs_[i][j]);
            }
        }
        projection_graph_[i].insert(projection_graph_[i].end(), ok_insert.begin(), ok_insert.end());
        // projection_graph_[i] = ok_insert;
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + M_pjbp);
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + projection_graph_[i].size());
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin());
    }
}

void IndexBipartite::BuildGraphOnlyBase(size_t n_bp, const float *bp_data, Parameters &parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    // uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    // uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // aligned and processed memory block for tow datasets
    data_bp_ = bp_data;
    // data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = 0;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    // bp_en_flags_.reserve(u32_nd_);
    // sq_en_flags_.reserve(u32_nd_sq_);
    // bp_en_set_.get_allocator().allocate(200);
    // sq_en_set_.get_allocator().allocate(200);

    SetBipartiteParameters(parameters);
    // InitBipartiteGraph();
    // for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
    //     if (i < nd_) {
    //         bipartite_graph_[i].reserve(((size_t)M_bp) * 1.5);
    //     } else {
    //         bipartite_graph_[i].reserve(((size_t)M_sq) * 1.5);
    //     }
    // }

    // if (need_normalize) {
    //     std::cout << "normalizing base data" << std::endl;
    //     for (size_t i = 0; i < nd_; ++i) {
    //         float *data = const_cast<float *>(data_bp_);
    //         normalize(data + i * dimension_, dimension_);
    //     }
    // }

    SimpleNeighbor *simple_graph = nullptr;
    // LinkBipartite(parameters, simple_graph);
    // this->Load("/ann/cross_domain_index/t2i_1M_M50_30_L100_best_7_7/t2i_1M_bipartite.index");
    // this->Load("/ann/cross_domain_index/t2i_10M_M_50_50_50_L_100_120/t2i_10M_bipartite.index");
    // LinkBipartite(parameters, simple_graph);

    // float bipartite_degree_avg = 0;
    // size_t bipartite_degree_max = 0, bipartite_degree_min = std::numeric_limits<size_t>::max();
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    // std::mt19937 rng(time(nullptr));
    // std::uniform_int_distribution<uint32_t> base_dis(0, u32_nd_ - 1);

    size_t i = 0;
    // for (; i < nd_; i++) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng) + u32_nd_);
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_bp + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_bp + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }
    // for (; i < total_pts_; ++i) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng));
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_sq + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_sq + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }

    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // std::cout << "Build bipartite time: " << diff.count() << std::endl;
    // std::cout << "Bipartite degree avg: " << bipartite_degree_avg / total_pts_ << std::endl;
    // std::cout << "Bipartite degree max: " << bipartite_degree_max << std::endl;
    // std::cout << "Bipartite degree min: " << bipartite_degree_min << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // s = std::chrono::high_resolution_clock::now();

    supply_nbrs_.resize(nd_);
    // for (i = 0; i < nd_; ++i) {
    //     supply_nbrs_[i].reserve(M_pjbp * 1.5);
    // }

    // project bipartite
    BipartiteProjection(parameters);

    CalculateProjectionep();

    assert(projection_ep_ < nd_);
    std::cout << "begin link projection" << std::endl;
    LinkBase(parameters, simple_graph);
    // uint32_t pre_m = M_pjbp;
    // parameters.Set<uint32_t>("M_pjbp", static_cast<uint32_t>(pre_m * 1.5));
    // for (size_t i = 0; i < supply_nbrs_.size(); ++i) {
    //     // projection_graph_[i].clear();
    //     // projection_graph_[i] = supply_nbrs_[i];
    //     // swap to projection_graph_, free supply
    //     // supply_nbrs_[i].shrink_to_fit();
    //     projection_graph_[i].swap(supply_nbrs_[i]);
    //     // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
    // }
    // AddEdgesInlineParts(parameters);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;
    ClearnBuildMemoryRemain4Search();
    has_built = true;
}


void IndexBipartite::AddEdgesInlineParts(const Parameters &parameters) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));

    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        // projection_graph_[i].clear();
        // projection_graph_[i] = supply_nbrs_[i];
        // swap to projection_graph_, free supply
        // projection_graph_[i].swap(supply_nbrs_[i]);
        projection_graph_[i].reserve(M_pjbp * 2 * PROJECTION_SLACK);
        // supply_nbrs_[i].shrink_to_fit();
        // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
    }
    // supply_nbrs_.clear();
    // supply_nbrs_.shrink_to_fit();
    malloc_trim(0);
    std::vector<uint32_t> vis_order;
    std::vector<uint32_t> vis_order_sq;
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        vis_order.push_back(i);
    }
    for (uint32_t i = 0; i < u32_nd_sq_; ++i) {
        vis_order_sq.push_back(i);
    }

    size_t dimm = dimension_;
    data_sq_ = nullptr;
// for each part
    for (uint32_t cur_part = 0; cur_part < train_parts_; ++cur_part) {
        if (data_sq_ != nullptr) {
            free(const_cast<float*>(data_sq_));
        }
        float * data_sq_part = nullptr;
        load_data_with_skip<float>(train_data_file.c_str(), each_part_num_, dimm, data_sq_part, cur_part * each_part_num_);
        data_sq_ = data_sq_part;

#pragma omp parallel for schedule(static, 100)
        for (size_t it_sq = 0; it_sq < nd_sq_; ++it_sq) {
            uint32_t sq = vis_order_sq[it_sq];
            boost::dynamic_bitset<> visited{u32_nd_, false};
            // auto &nn_base = learn_base_knn_[sq];
            // if (nn_base.size() > 100) {
            //     nn_base.resize(100);
            //     nn_base.shrink_to_fit();
            // }
            // uint32_t choose_tgt = 0;
            // for (size_t i = 0; i < 100; ++i) {
            //     if (projection_graph_[nn_base[i]].size() < M_pjbp) {
            //         choose_tgt = nn_base[i];
            //         break;
            //     }
            // }
            uint32_t cur_tgt = sq;
            std::vector<Neighbor> full_retset;
            NeighborPriorityQueue retset(L_pjpq);
            full_retset.reserve(L_pjpq);

            SearchProjectionGraphInternalPJ(retset, data_sq_ + dimension_ * sq, sq + u32_nd_, parameters, visited, full_retset);
            // SearchProjectionGraphInternal(retset, data_sq_ + dimension_ * sq, sq + u32_nd_, parameters, visited, full_retset);
            std::sort(full_retset.begin(), full_retset.end());

            std::vector<uint32_t> pruned_list;
            // pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
            cur_tgt = full_retset[0].id;
            std::remove_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == cur_tgt; });
            // if (projection_graph_[cur_tgt].size() == 0) {
            // } else {
            //     std::remove_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == cur_tgt; });
            //     std::vector<uint32_t> vec_nbrs;
            //     {
            //         LockGuard guard(locks_[cur_tgt]);
            //         vec_nbrs = projection_graph_[cur_tgt];
            //     }
            //     for(auto& nbr: vec_nbrs) {
            //         if (nbr == cur_tgt) {
            //             continue;
            //         }
                    
            //         if (std::find_if(full_retset.begin(), full_retset.end(), [&](Neighbor &x) { return x.id == nbr; }) == full_retset.end()) {
            //             float dist = distance_->compare(data_sq_ + dimension_ * (size_t)sq, data_bp_ + dimension_ * (size_t)nbr, dimension_);
            //             full_retset.emplace_back(nbr, dist, false);
            //         }
            //     }
            // }
            // for (size_t i = 0; i < full_retset.size(); ++i) {
            //     if (pruned_list.size() >= M_pjbp * PROJECTION_SLACK) {
            //         break;
            //     }
            //     pruned_list.push_back(full_retset[i].id);
            // }
            PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * cur_tgt, cur_tgt, parameters, pruned_list);
            {
                LockGuard guard(locks_[cur_tgt]);
                projection_graph_[cur_tgt] = pruned_list;
            }
            ProjectionAddReverse(cur_tgt, parameters);
            if (sq % 1000 == 0) {
                std::cout << "\r" << (100.0 * sq) / u32_nd_sq_ << "% of projection search bipartite by base completed."
                        << std::flush;
            }
        }
    }

    std::cout << std::endl;
    // std::atomic<uint32_t> degree_cnt(0);
    // std::atomic<uint32_t> zero_cnt(0);
    // std::atomic<uint32_t> less_5_cnt(0);
#pragma omp parallel for schedule(static, 100)
    for (uint32_t i = 0; i < vis_order.size(); ++i) {
        uint32_t node = vis_order[i];
        // if (projection_graph_[node].size() < M_pjbp) {
        //     // std::cout << "Warning: projection graph node " << node << " has less than M_pjbp neighbors." << std::endl;
        //     // degree_cnt.fetch_add(1);
        //     if (projection_graph_[node].size() == 0) {
        //         zero_cnt.fetch_add(1);
        //     }
        //     if (projection_graph_[node].size() < 5) {
        //         less_5_cnt.fetch_add(1);
        //     }
        // }
        ProjectionAddReverse(node, parameters);
    }
    // std::cout << "Warning: " << degree_cnt.load() << " nodes have less than M_pjbp neighbors." << std::endl;
    // std::cout << "Warning: " << zero_cnt.load() << " nodes have no neighbors." << std::endl;
    // std::cout << "Warning: " << less_5_cnt.load() << " nodes have less than 5 neighbors." << std::endl;

// #pragma omp parallel for schedule(dynamic, 100)
//     for (size_t i = 0; i < projection_graph_.size(); ++i) {
//         std::vector<uint32_t> ok_insert;
//         ok_insert.reserve(M_pjbp);
//         for (size_t j = 0; j < supply_nbrs_[i].size(); ++j) {
//             if (ok_insert.size() >= M_pjbp * 2) {
//                 break;
//             }
//             if (std::find(projection_graph_[i].begin(), projection_graph_[i].end(), supply_nbrs_[i][j]) ==
//                 projection_graph_[i].end()) {
//                 ok_insert.push_back(supply_nbrs_[i][j]);
//             }
//         }
//         projection_graph_[i].insert(projection_graph_[i].end(), ok_insert.begin(), ok_insert.end());
//         // projection_graph_[i] = ok_insert;
//         // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + M_pjbp);
//         // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + projection_graph_[i].size());
//         // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin());
//     }
}

void IndexBipartite::BuildGraphInline(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data, Parameters &parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // aligned and processed memory block for tow datasets
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    // bp_en_flags_.reserve(u32_nd_);
    // sq_en_flags_.reserve(u32_nd_sq_);
    // bp_en_set_.get_allocator().allocate(200);
    // sq_en_set_.get_allocator().allocate(200);

    SetBipartiteParameters(parameters);
    // InitBipartiteGraph();
    // for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
    //     if (i < nd_) {
    //         bipartite_graph_[i].reserve(((size_t)M_bp) * 1.5);
    //     } else {
    //         bipartite_graph_[i].reserve(((size_t)M_sq) * 1.5);
    //     }
    // }

    if (need_normalize) {
        std::cout << "normalizing base data" << std::endl;
        for (size_t i = 0; i < nd_; ++i) {
            float *data = const_cast<float *>(data_bp_);
            normalize(data + i * dimension_, dimension_);
        }
    }

    SimpleNeighbor *simple_graph = nullptr;
    // LinkBipartite(parameters, simple_graph);
    // this->Load("/ann/cross_domain_index/t2i_1M_M50_30_L100_best_7_7/t2i_1M_bipartite.index");
    // this->Load("/ann/cross_domain_index/t2i_10M_M_50_50_50_L_100_120/t2i_10M_bipartite.index");
    // LinkBipartite(parameters, simple_graph);

    float bipartite_degree_avg = 0;
    size_t bipartite_degree_max = 0, bipartite_degree_min = std::numeric_limits<size_t>::max();
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    // std::mt19937 rng(time(nullptr));
    // std::uniform_int_distribution<uint32_t> base_dis(0, u32_nd_ - 1);

    size_t i = 0;
    // for (; i < nd_; i++) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng) + u32_nd_);
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_bp + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_bp + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }
    // for (; i < total_pts_; ++i) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng));
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_sq + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_sq + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }

    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // std::cout << "Build bipartite time: " << diff.count() << std::endl;
    // std::cout << "Bipartite degree avg: " << bipartite_degree_avg / total_pts_ << std::endl;
    // std::cout << "Bipartite degree max: " << bipartite_degree_max << std::endl;
    // std::cout << "Bipartite degree min: " << bipartite_degree_min << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // s = std::chrono::high_resolution_clock::now();

    supply_nbrs_.resize(nd_);
    // for (i = 0; i < nd_; ++i) {
    //     supply_nbrs_[i].reserve(M_pjbp * 1.5);
    // }

    // project bipartite
    BipartiteProjection(parameters);

    CalculateProjectionep();

    assert(projection_ep_ < nd_);
    std::cout << "begin link projection" << std::endl;
    LinkBase(parameters, simple_graph);
    // uint32_t pre_m = M_pjbp;
    // parameters.Set<uint32_t>("M_pjbp", static_cast<uint32_t>(pre_m * 1.5));
    AddEdgesInline(parameters);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;

    has_built = true;
   
}

void IndexBipartite::ClearnBuildMemoryRemain4Search() {
    // clearn supply nbrs
    for (size_t i = 0; i < supply_nbrs_.size(); ++i) {
        supply_nbrs_[i].clear();
        supply_nbrs_[i].shrink_to_fit();
    }
    supply_nbrs_.clear();
    supply_nbrs_.shrink_to_fit();

    // free data_sq
    if (data_sq_ != nullptr) {
        free(const_cast<float*>(data_sq_));
    }
}

//n_sq: each part
void IndexBipartite::BuildGraphST2(size_t n_bp, const float *bp_data, Parameters &parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    // uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    // uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // aligned and processed memory block for tow datasets
    data_bp_ = bp_data;
    // data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = each_part_num_;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    // bp_en_flags_.reserve(u32_nd_);
    // sq_en_flags_.reserve(u32_nd_sq_);
    // bp_en_set_.get_allocator().allocate(200);
    // sq_en_set_.get_allocator().allocate(200);

    SetBipartiteParameters(parameters);
    // InitBipartiteGraph();
    // for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
    //     if (i < nd_) {
    //         bipartite_graph_[i].reserve(((size_t)M_bp) * 1.5);
    //     } else {
    //         bipartite_graph_[i].reserve(((size_t)M_sq) * 1.5);
    //     }
    // }

    // if (need_normalize) {
    //     std::cout << "normalizing base data" << std::endl;
    //     for (size_t i = 0; i < nd_; ++i) {
    //         float *data = const_cast<float *>(data_bp_);
    //         normalize(data + i * dimension_, dimension_);
    //     }
    // }

    SimpleNeighbor *simple_graph = nullptr;
    // LinkBipartite(parameters, simple_graph);
    // this->Load("/ann/cross_domain_index/t2i_1M_M50_30_L100_best_7_7/t2i_1M_bipartite.index");
    // this->Load("/ann/cross_domain_index/t2i_10M_M_50_50_50_L_100_120/t2i_10M_bipartite.index");
    // LinkBipartite(parameters, simple_graph);

    // float bipartite_degree_avg = 0;
    // size_t bipartite_degree_max = 0, bipartite_degree_min = std::numeric_limits<size_t>::max();
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    // std::mt19937 rng(time(nullptr));
    // std::uniform_int_distribution<uint32_t> base_dis(0, u32_nd_ - 1);

    size_t i = 0;
    // for (; i < nd_; i++) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng) + u32_nd_);
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_bp + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_bp + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }
    // for (; i < total_pts_; ++i) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng));
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_sq + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_sq + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }

    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // std::cout << "Build bipartite time: " << diff.count() << std::endl;
    // std::cout << "Bipartite degree avg: " << bipartite_degree_avg / total_pts_ << std::endl;
    // std::cout << "Bipartite degree max: " << bipartite_degree_max << std::endl;
    // std::cout << "Bipartite degree min: " << bipartite_degree_min << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // s = std::chrono::high_resolution_clock::now();

    // supply_nbrs_.resize(nd_);
    // for (i = 0; i < nd_; ++i) {
    //     supply_nbrs_[i].reserve(M_pjbp * 1.5);
    // }

    // project bipartite
    BipartiteProjection(parameters);

    CalculateProjectionep();

    assert(projection_ep_ < nd_);
    std::cout << "begin link projection" << std::endl;
    LinkBase(parameters, simple_graph);
    // uint32_t pre_m = M_pjbp;
    // parameters.Set<uint32_t>("M_pjbp", static_cast<uint32_t>(pre_m * 1.5));
    AddEdgesInlineParts(parameters);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;

    /***********************/
    // 训练 SQ8Quantizer
    // TrainQuantizer(data_bp_, static_cast<int>(u32_nd_), static_cast<int>(dimension_));
    /***********************/
    ClearnBuildMemoryRemain4Search();
    has_built = true;
}

void IndexBipartite::BuildBipartite(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                                    const Parameters &parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // aligned and processed memory block for tow datasets
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    // bp_en_flags_.reserve(u32_nd_);
    // sq_en_flags_.reserve(u32_nd_sq_);
    // bp_en_set_.get_allocator().allocate(200);
    // sq_en_set_.get_allocator().allocate(200);

    SetBipartiteParameters(parameters);
    // InitBipartiteGraph();
    for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
        if (i < nd_) {
            bipartite_graph_[i].reserve(((size_t)M_bp) * 1.5);
        } else {
            bipartite_graph_[i].reserve(((size_t)M_sq) * 1.5);
        }
    }

    if (need_normalize) {
        std::cout << "normalizing base data" << std::endl;
        for (size_t i = 0; i < nd_; ++i) {
            float *data = const_cast<float *>(data_bp_);
            normalize(data + i * dimension_, dimension_);
        }
    }

    SimpleNeighbor *simple_graph = nullptr;
    // LinkBipartite(parameters, simple_graph);
    // this->Load("/ann/cross_domain_index/t2i_1M_M50_30_L100_best_7_7/t2i_1M_bipartite.index");
    // this->Load("/ann/cross_domain_index/t2i_10M_M_50_50_50_L_100_120/t2i_10M_bipartite.index");
    // LinkBipartite(parameters, simple_graph);

    float bipartite_degree_avg = 0;
    size_t bipartite_degree_max = 0, bipartite_degree_min = std::numeric_limits<size_t>::max();
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    // std::mt19937 rng(time(nullptr));
    // std::uniform_int_distribution<uint32_t> base_dis(0, u32_nd_ - 1);

    size_t i = 0;
    // for (; i < nd_; i++) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng) + u32_nd_);
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_bp + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_bp + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }
    // for (; i < total_pts_; ++i) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng));
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_sq + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_sq + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }

    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // std::cout << "Build bipartite time: " << diff.count() << std::endl;
    // std::cout << "Bipartite degree avg: " << bipartite_degree_avg / total_pts_ << std::endl;
    // std::cout << "Bipartite degree max: " << bipartite_degree_max << std::endl;
    // std::cout << "Bipartite degree min: " << bipartite_degree_min << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // s = std::chrono::high_resolution_clock::now();

    supply_nbrs_.resize(nd_);
    // for (i = 0; i < nd_; ++i) {
    //     supply_nbrs_[i].reserve(M_pjbp * 1.5);
    // }

    // project bipartite
    BipartiteProjection(parameters);

    CalculateProjectionep();

    assert(projection_ep_ < nd_);
    std::cout << "begin link projection" << std::endl;
    LinkProjection(parameters, simple_graph);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;

    has_built = true;
}

void IndexBipartite::BuildEdgeAfterAdd(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                                    const Parameters &parameters) {
    std::cout << "start build bipartite index" << std::endl;
    auto s = std::chrono::high_resolution_clock::now();
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // aligned and processed memory block for tow datasets
    data_bp_ = bp_data;
    data_sq_ = sq_data;
    nd_ = n_bp;
    nd_sq_ = n_sq;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
    locks_ = std::vector<std::mutex>(total_pts_);
    // bp_en_flags_.reserve(u32_nd_);
    // sq_en_flags_.reserve(u32_nd_sq_);
    // bp_en_set_.get_allocator().allocate(200);
    // sq_en_set_.get_allocator().allocate(200);

    SetBipartiteParameters(parameters);
    InitBipartiteGraph();
    for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
        if (i < nd_) {
            bipartite_graph_[i].reserve(((size_t)M_bp) * 1.5);
        } else {
            bipartite_graph_[i].reserve(((size_t)M_sq) * 1.5);
        }
    }

    if (need_normalize) {
        std::cout << "normalizing base data" << std::endl;
        for (size_t i = 0; i < nd_; ++i) {
            float *data = const_cast<float *>(data_bp_);
            normalize(data + i * dimension_, dimension_);
        }
    }

    SimpleNeighbor *simple_graph = nullptr;
    // LinkBipartite(parameters, simple_graph);
    // this->Load("/ann/cross_domain_index/t2i_1M_M50_30_L100_best_7_7/t2i_1M_bipartite.index");
    // this->Load("/ann/cross_domain_index/t2i_10M_M_50_50_50_L_100_120/t2i_10M_bipartite.index");
    // LinkBipartite(parameters, simple_graph);

    float bipartite_degree_avg = 0;
    size_t bipartite_degree_max = 0, bipartite_degree_min = std::numeric_limits<size_t>::max();
    float projection_degree_avg = 0;
    size_t projection_degree_max = 0, projection_degree_min = std::numeric_limits<size_t>::max();

    // std::mt19937 rng(time(nullptr));
    // std::uniform_int_distribution<uint32_t> base_dis(0, u32_nd_ - 1);

    size_t i = 0;
    // for (; i < nd_; i++) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng) + u32_nd_);
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_bp + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_bp + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }
    // for (; i < total_pts_; ++i) {
    //     std::vector<uint32_t> &nbrs = bipartite_graph_[i];
    //     nbrs.push_back(base_dis(rng));
    //     bipartite_degree_avg += static_cast<float>(nbrs.size());
    //     bipartite_degree_max = std::max(bipartite_degree_max, nbrs.size());
    //     bipartite_degree_min = std::min(bipartite_degree_min, nbrs.size());
    //     // if (nbrs.size() > M_sq + 1) {
    //     // std::random_shuffle(nbrs.begin(), nbrs.end());
    //     // nbrs.resize(M_sq + 1);
    //     // nbrs.shrink_to_fit();
    //     // }
    // }

    // auto e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    // std::cout << "Build bipartite time: " << diff.count() << std::endl;
    // std::cout << "Bipartite degree avg: " << bipartite_degree_avg / total_pts_ << std::endl;
    // std::cout << "Bipartite degree max: " << bipartite_degree_max << std::endl;
    // std::cout << "Bipartite degree min: " << bipartite_degree_min << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // s = std::chrono::high_resolution_clock::now();

    supply_nbrs_.resize(nd_);
    // for (i = 0; i < nd_; ++i) {
    //     supply_nbrs_[i].reserve(M_pjbp * 1.5);
    // }

    // project bipartite
    BipartiteProjection(parameters);

    CalculateProjectionep();

    assert(projection_ep_ < nd_);
    std::cout << "begin link projection" << std::endl;
    // LinkProjection(parameters, simple_graph);
    LinkBase(parameters, simple_graph);
    TrainingLink2Projection(parameters, simple_graph);
    std::cout << std::endl;
    // std::cout << "Starting collect points" << std::endl;
    // auto co_s = std::chrono::high_resolution_clock::now();
    // CollectPoints(parameters);
    // auto co_e = std::chrono::high_resolution_clock::now();
    // diff = co_e - co_s;
    // std::cout << "Collect points time: " << diff.count() << std::endl;

    // e = std::chrono::high_resolution_clock::now();
    auto e = std::chrono::high_resolution_clock::now();
    auto diff = e - s;
    std::cout << "Build projection graph time: " << diff.count() / (1000 * 1000 * 1000) << std::endl;

    for (i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> &nbrs = projection_graph_[i];
        projection_degree_avg += static_cast<float>(nbrs.size());
        projection_degree_max = std::max(projection_degree_max, nbrs.size());
        projection_degree_min = std::min(projection_degree_min, nbrs.size());
    }
    std::cout << "total degree: " << projection_degree_avg << std::endl;
    std::cout << "Projection degree avg: " << projection_degree_avg / (float)u32_nd_ << std::endl;
    std::cout << "Projection degree max: " << projection_degree_max << std::endl;
    std::cout << "Projection degree min: " << projection_degree_min << std::endl;

    has_built = true;
}

void IndexBipartite::LinkOneNode(const Parameters &parameters, uint32_t nid, SimpleNeighbor *simple_graph, bool is_base,
                                 boost::dynamic_bitset<> &visited) {
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");

    const float *cur_data = is_base ? data_bp_ : data_sq_;
    uint32_t global_id = is_base ? nid : u32_nd_ + nid;
    // std::vector<Neighbor> retset;
    NeighborPriorityQueue search_queue(L_pq);
    search_queue.reserve(L_pq);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(L_pq * 2);
    if (is_base) {
        SearchBipartitebyBase(cur_data + nid * dimension_, global_id, parameters, simple_graph, search_queue, visited,
                              full_retset);
        std::vector<uint32_t> pruned_list;
        // pruned_list.reserve(M_bp);
        PruneCandidates(full_retset, global_id, parameters, pruned_list, visited);
        if (search_queue.size() <= 0) {
            throw std::runtime_error("search queue is empty");
        }

        {
            LockGuard guard(locks_[global_id]);
            bipartite_graph_[global_id].reserve(M_sq * 1.5);
            bipartite_graph_[global_id] = pruned_list;
        }
        // std::vector<uint32_t> temp_vec_4_sq;
        // temp_vec_4_sq.reserve(pruned_list.size());
        if (sq_en_set_.size() < 200) {
            for (size_t i = 0; i < pruned_list.size(); ++i) {
                if (sq_en_set_.find(pruned_list[i]) == sq_en_set_.end()) {
                    LockGuard guard(sq_set_mutex_);
                    sq_en_set_.insert(pruned_list[i]);
                }
                // if (!sq_en_flags_.test(pruned_list[i])) {
                //     LockGuard guard(sq_set_mutex_);
                //     sq_en_flags_.set(i);
                // }
            }
        }

        if (bp_en_set_.size() < 200) {
            {
                if (bp_en_set_.find(global_id) == bp_en_set_.end()) {
                    LockGuard guard(bp_set_mutex_);
                    bp_en_set_.insert(global_id);
                }
                // bp_en_flags_.set(global_id);
                //     SharedLockGuard guard(bp_set_mutex_);
            }
        }
        AddReverse(search_queue, global_id, pruned_list, parameters, visited);
    } else {
        SearchBipartitebyQuery(cur_data + nid * dimension_, global_id, parameters, simple_graph, search_queue, visited,
                               full_retset);
        std::vector<uint32_t> pruned_list;
        // pruned_list.reserve(M_sq * 1.5);
        PruneCandidates(full_retset, global_id, parameters, pruned_list, visited);

        if (search_queue.size() <= 0) {
            throw std::runtime_error("search queue is empty");
        }

        {
            LockGuard guard(locks_[global_id]);
            bipartite_graph_[global_id].reserve(M_bp * 1.5);
            bipartite_graph_[global_id] = pruned_list;
        }

        if (bp_en_set_.size() < 100) {
            for (size_t i = 0; i < pruned_list.size(); ++i) {
                if (bp_en_set_.find(pruned_list[i]) == bp_en_set_.end()) {
                    LockGuard guard(bp_set_mutex_);
                    bp_en_set_.insert(pruned_list[i]);
                }
                // if (!bp_en_flags_.test(pruned_list[i])) {
                //     LockGuard guard(bp_set_mutex_);
                //     bp_en_flags_.set(i);
                // }
            }
        }

        if (sq_en_set_.size() < 100) {
            {
                if (sq_en_set_.find(global_id) == sq_en_set_.end()) {
                    LockGuard guard(sq_set_mutex_);
                    sq_en_set_.insert(global_id);
                }
                //     SharedLockGuard guard(sq_set_mutex_);
                //     sq_en_set_.insert(global_id);
            }
        }
        // for (size_t i = 0; i < pruned_list.size(); ++i) {
        //     SharedLockGuard guard(bp_set_mutex_);
        //     bp_en_set_.insert(pruned_list[i]);
        // }
        // {
        //     SharedLockGuard guard(sq_set_mutex_);
        //     sq_en_set_.insert(global_id);
        // }
        AddReverse(search_queue, global_id, pruned_list, parameters, visited);
    }
}

void IndexBipartite::LinkBipartite(const Parameters &parameters, SimpleNeighbor *simple_graph) {
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    // insert sampled query nodes and base point nodes alternately
    // first base point nodes
    omp_set_num_threads(static_cast<int>(parameters.Get<uint32_t>("num_threads")));

    std::vector<uint32_t> running_order;
    std::vector<uint32_t> indicate_idx;
    size_t prepare_i = 0;
    for (; prepare_i < total_pts_; prepare_i += 2) {
        indicate_idx.push_back(static_cast<uint32_t>(prepare_i));
    }

    uint32_t i_bp = 0, j_sq = 0;
    while (i_bp + j_sq < u32_total_pts_) {
        if (i_bp < u32_nd_) {
            running_order.push_back(i_bp);
            ++i_bp;
        }
        if (j_sq < u32_nd_sq_) {
            running_order.push_back(u32_nd_ + j_sq);
            ++j_sq;
        }
    }
    int for_n = (int)indicate_idx.size();

#pragma omp parallel for schedule(dynamic, 100)
    for (int iter_loc = 0; iter_loc < for_n; ++iter_loc) {
        boost::dynamic_bitset<> visited(u32_total_pts_);
        visited.reset();
        visited.reserve(u32_total_pts_);
        uint32_t val = running_order[indicate_idx[iter_loc]];
        if (val < u32_nd_) {
            // insert base point node
            LinkOneNode(parameters, val, simple_graph, true, visited);
        } else {
            LinkOneNode(parameters, val - u32_nd_, simple_graph, false, visited);
        }
        if (indicate_idx[iter_loc] + 1 < running_order.size()) {
            val = running_order[indicate_idx[iter_loc] + 1];
        } else {
            std::cout << "haha" << std::endl;
            continue;
        }
        if (val >= u32_nd_) {
            // insert sampled query node
            LinkOneNode(parameters, val - u32_nd_, simple_graph, false, visited);
        } else {
            LinkOneNode(parameters, val, simple_graph, true, visited);
        }
        if (running_order[indicate_idx[iter_loc]] % 1000 == 0) {
            std::cout << "\r" << (100.0 * (val - u32_nd_)) / (u32_nd_sq_) << "% of index build completed."
                      << "val: " << val << std::flush;
        }
    }

    // float max_degree = 0, min_degree = std::numeric_limits<float>::max(), avg_degree = 0;
    // for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
    //     auto &nbrs = bipartite_graph_[i];
    //     max_degree = std::max(max_degree, static_cast<float>(nbrs.size()));
    //     min_degree = std::min(min_degree, static_cast<float>(nbrs.size()));
    //     avg_degree += static_cast<float>(nbrs.size());
    // }

    // std::cout << "after link one node, max degree" << max_degree << std::endl;

    std::cout << std::endl;
#pragma omp parallel for schedule(dynamic, 100)
    for (uint32_t i = 0; i < u32_total_pts_; ++i) {
        boost::dynamic_bitset<> visited(u32_total_pts_);
        NeighborPriorityQueue useless_queue;
        AddReverse(useless_queue, i, bipartite_graph_[i], parameters, visited);
        // std::cout << "\r" << (100.0 * i) / (u32_total_pts_) << "% of adding reverse for bipartite index build
        // completed." << std::flush;
    }

#pragma omp parallel for schedule(dynamic, 100)
    for (size_t i = 0; i < bipartite_graph_.size(); ++i) {
        if (bipartite_graph_[i].size() < M_bp) {
            // std::cout << "haha sb" << std::endl;
            boost::dynamic_bitset<> visited(u32_total_pts_);
            if (i < nd_) {
                LinkOneNode(parameters, i, simple_graph, true, visited);
            } else {
                LinkOneNode(parameters, i - nd_, simple_graph, false, visited);
            }
        }
    }
}

void IndexBipartite::PruneCandidates(std::vector<Neighbor> &search_pool, uint32_t tgt_id, const Parameters &parameters,
                                     std::vector<uint32_t> &pruned_list, boost::dynamic_bitset<> &visited) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");

    std::sort(search_pool.begin(), search_pool.end());
    uint32_t degree_bound = tgt_id < u32_nd_ ? M_bp : M_sq;
    pruned_list.reserve(degree_bound * 1.5);
    auto reachable_flags = visited;
    reachable_flags.reset();
    // tsl::robin_set<uint32_t> reachable_flags;

    // boost::dynamic_bitset<> reachable_flags(u32_total_pts_);  // could be reduced
    for (size_t i = 0; i < search_pool.size(); ++i) {
        if (pruned_list.size() >= (size_t)degree_bound) {
            break;
        }
        auto &node = search_pool[i];
        auto id = node.id;
        if (!reachable_flags.test(id)) {
            reachable_flags.set(id);
            pruned_list.push_back(id);
            // add neighbors' neighbors
            for (auto nbr : bipartite_graph_[id]) {
                for (auto nnbr : bipartite_graph_[nbr]) {
                    reachable_flags.set(nnbr);
                }
            }
        }
    }
    // if not enough
    if (pruned_list.size() < (size_t)degree_bound) {
        // for (size_t i = (search_pool.size() == 0 ? 0 : search_pool.size() - 1); i > 0; --i) {
        for (size_t i = 0; i < search_pool.size(); ++i) {
            if (pruned_list.size() >= (size_t)degree_bound) {
                break;
            }
            if (std::find(pruned_list.begin(), pruned_list.end(), search_pool[i].id) == pruned_list.end()) {
                pruned_list.push_back(search_pool[i].id);
            }
        }
    }
}

void IndexBipartite::AddReverse(NeighborPriorityQueue &search_pool, uint32_t src_node,
                                std::vector<uint32_t> &pruned_list, const Parameters &parameters,
                                boost::dynamic_bitset<> &visited) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    // if true nodes in pruned_list are query, otherwise nodes in pruned_list are base
    // if true, neighbor of pruned_list elements are base, otherwise neighbor of pruned_list elements are query
    bool is_base = src_node < u32_nd_;  // if true tgt is query, otherwise tgt is base

    uint32_t degree_bound = is_base ? M_sq : M_bp;  // note: reverse relationship
    bool need_prune = false;

    for (size_t i = 0; i < pruned_list.size(); ++i) {  // for each tgt
        auto cur_node = pruned_list[i];
        std::vector<uint32_t> copy_vec;
        copy_vec.reserve(degree_bound * 1.5);
        copy_vec = bipartite_graph_[cur_node];
        if (std::find(copy_vec.begin(), copy_vec.end(), src_node) == copy_vec.end()) {
            if (copy_vec.size() < degree_bound) {
                need_prune = false;
                {
                    LockGuard gurad(locks_[cur_node]);
                    bipartite_graph_[cur_node].push_back(src_node);
                }
            } else {
                need_prune = true;
            }
        }

        if (need_prune) {
            // if is_base, copy_vec is base vec
            // else copy_vec is query vec
            const float *cur_data = is_base ? data_bp_ : data_sq_;
            const float *opposite_data = is_base ? data_sq_ : data_bp_;
            // simulate search pool
            // NeighborPriorityQueue simulate_pool;
            std::vector<Neighbor> simulate_pool;
            simulate_pool.reserve(copy_vec.size());
            for (auto id : copy_vec) {
                uint32_t cate_id = is_base ? id : id - u32_nd_;
                uint32_t cate_cur_node = is_base ? cur_node - u32_nd_ : cur_node;
                // mysteryann::prefetch_vector((char *)(cur_data + cate_id * dimension_), dimension_ * sizeof(float));
                float distance = distance_->compare(cur_data + cate_id * dimension_,
                                                    opposite_data + cate_cur_node * dimension_, dimension_);
                // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
                //     dist = -dist;
                // }
                simulate_pool.push_back(Neighbor(id, distance, false));
            }
            std::vector<uint32_t> inside_pruned_list;
            PruneCandidates(simulate_pool, cur_node, parameters, inside_pruned_list, visited);
            copy_vec = inside_pruned_list;
            {
                LockGuard gurad(locks_[cur_node]);
                // bipartite_graph_[cur_node].reserve(degree_bound * 1.5);
                bipartite_graph_[cur_node] = copy_vec;
            }
        }
    }
}

// search by base, return query
void IndexBipartite::SearchBipartitebyBase(const float *query, uint32_t gid, const Parameters &parameters,
                                           SimpleNeighbor *simple_graph, NeighborPriorityQueue &queue,
                                           boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset) {
    uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    // uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_sq_ - 1);
    // srand(time(NULL));
    std::vector<uint32_t> init_ids;
    uint32_t start = dis(gen);
    // rand() % nd_sq_;  // start is sampled query
    init_ids.push_back(u32_nd_ + start);  // push global id
    init_ids.push_back(u32_nd_ + dis(gen));
    // boost::dynamic_bitset<> visited(u32_total_pts_);
    // tsl::robin_map<uint32_t, bool> visited_map;
    // randomly select a start node
    // true random
    if (!sq_en_set_.empty()) {  // if there is sampled query
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, sq_en_set_.size() - 1);
        {
            // std::shared_lock<std::shared_mutex> guard(sq_set_mutex_);
            // LockGuard guard(sq_set_mutex_);
            start = *(std::next(sq_en_set_.begin(), dis(gen)));
        }

        if (start < u32_nd_) {
            std::cout << "Error: start less than sampled query no, exited. start: " << start << " advance: "
                      << "sq_en_set_ size(): " << sq_en_set_.size() << std::endl;
            // print sq_en_set_
            // std::cout << *(std::next(sq_en_set_.begin(), advance)) << std::endl;
            for (auto &i : sq_en_set_) {
                std::cout << i << " ";
            }
            exit(1);
        }
        if (init_ids[0] != start) {
            init_ids.push_back(start);
        }
    }

    while (init_ids.size() < queue.capacity()) {
        init_ids.push_back(u32_nd_ + dis(gen));
    }

    // std::cout << "start: " << start << std::endl;
    // start = start + nd_;
    // queue.reserve(L_pq);
    for (auto id : init_ids) {
        float distance;
        uint32_t cur_local_id = id - u32_nd_;
        // int aa = cur_local_id * dimension_;
        distance = distance_->compare(data_sq_ + cur_local_id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        Neighbor nn = Neighbor(id, distance, false);
        queue.insert(nn);
        visited.set(id);
        full_retset.push_back(nn);
        // visited_map.insert({id, true});
    }

    while (queue.has_unexpanded_node()) {
        auto cur_check_node = queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;

        std::vector<uint32_t> nbr_ids;
        // add neighbors' neighbors

        // SharedLockGuard guard(locks_[cur_id]);
        bool fast_check = false;
        uint32_t first_hop_rank_1 = bipartite_graph_[cur_id].size() == 0 ? rand() % nd_ : bipartite_graph_[cur_id][0];
        if (bipartite_graph_[cur_id].size() < M_sq) {
            fast_check = false;
        }
        float first_hop_min_dist = 1000;
        for (size_t j = 0; j < bipartite_graph_[cur_id].size(); ++j) {
            // neighbors
            uint32_t nbr = bipartite_graph_[cur_id][j];
            if (nbr == gid) {
                continue;
            }
            // neighbors' neighbors are sampled queries

            for (size_t i = 0; i < bipartite_graph_[nbr].size(); ++i) {
                if (visited.test(bipartite_graph_[nbr][i])) {
                    continue;
                }
                // nbr_ids.push_back(bipartite_graph_[nbr][i]);

                uint32_t cate_id = bipartite_graph_[nbr][i] - u32_nd_;
                // mysteryann::prefetch_vector((char *)(data_sq_ + cate_id * dimension_), dimension_ * sizeof(float));

                float distance = distance_->compare(data_sq_ + cate_id * dimension_, query, (unsigned)dimension_);

                if (fast_check) {
                    if (first_hop_min_dist > distance) {
                        first_hop_min_dist = distance;
                        first_hop_rank_1 = nbr;
                    }
                }

                visited.set(bipartite_graph_[nbr][i]);
                Neighbor nn = Neighbor(bipartite_graph_[nbr][i], distance, false);
                queue.insert(nn);
                full_retset.push_back(nn);
                if (fast_check) {
                    break;
                }
            }
        }

        if (fast_check) {
            for (size_t i = 0; i < bipartite_graph_[first_hop_rank_1].size(); ++i) {
                auto nbr = bipartite_graph_[first_hop_rank_1][i];
                if (visited.test(nbr)) {
                    continue;
                }
                // nbr_ids.push_back(bipartite_graph_[nbr][i]);

                uint32_t cate_id = nbr - u32_nd_;
                // mysteryann::prefetch_vector((char *)(data_sq_ + cate_id * dimension_), dimension_ * sizeof(float));

                float distance = distance_->compare(data_sq_ + cate_id * dimension_, query, (unsigned)dimension_);

                visited.set(nbr);
                Neighbor nn = Neighbor(nbr, distance, false);
                queue.insert(nn);
                full_retset.push_back(nn);
            }
        }

        // for (size_t i = 0; i < nbr_ids.size(); ++i) {
        //     uint32_t ns_nbr = nbr_ids[i];
        //     float distance;

        //     uint32_t cate_id = ns_nbr - u32_nd_;
        //     // mysteryann::prefetch_vector((char *)(data_sq_ + cate_id * dimension_), dimension_ * sizeof(float));
        //     distance = distance_->compare(data_sq_ + cate_id * dimension_, query, (unsigned)dimension_);
        //     // if (metric_ == mysteryann::INNER_PRODUCT) {
        //     //     distance = -distance;
        //     // }
        //     Neighbor nn = Neighbor(ns_nbr, distance, false);
        //     queue.insert(nn);
        // }
    }
}

// search by query, return base
void IndexBipartite::SearchBipartitebyQuery(const float *query, uint32_t gid, const Parameters &parameters,
                                            SimpleNeighbor *simple_graph, NeighborPriorityQueue &queue,
                                            boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset) {
    // uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t M_bp = parameters.Get<uint32_t>("M_bp");
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");

    // randomly select a start node
    // srand(time(NULL));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    uint32_t start = dis(gen);
    //  rand() % nd_;  // start is base
    std::vector<uint32_t> init_ids;
    init_ids.push_back(start);
    init_ids.push_back(dis(gen));
    // boost::dynamic_bitset<> visited(u32_total_pts_);

    if (!bp_en_set_.empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        {
            std::uniform_int_distribution<uint32_t> dis(0, bp_en_set_.size() - 1);
            // std::shared_lock<std::shared_mutex> guard(bp_set_mutex_);
            LockGuard guard(bp_set_mutex_);
            start = *(std::next(bp_en_set_.begin(), dis(gen)));
        }
        if (start > u32_nd_) {
            std::cout << "Error: start greater than base no, exited" << std::endl;
            exit(1);
        }
        if (init_ids[0] != start) {
            init_ids.push_back(start);
        }
    }

    while (init_ids.size() < queue.capacity()) {
        init_ids.push_back(dis(gen));
    }

    for (auto &id : init_ids) {
        float distance;

        distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        Neighbor nn = Neighbor(id, distance, false);
        visited.set(id);
        queue.insert(nn);
        full_retset.push_back(nn);
    }

    while (queue.has_unexpanded_node()) {
        auto cur_check_node = queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;

        bool fast_check = false;
        uint32_t first_hop_rank_1 =
            bipartite_graph_[cur_id].size() == 0 ? (rand() % nd_sq_) + nd_ : bipartite_graph_[cur_id][0];
        if (bipartite_graph_[cur_id].size() < (M_bp)) {
            fast_check = false;
        }
        float first_hop_min_dist = 1000;

        // std::vector<uint32_t> nbr_ids;
        // add neighbors' neighbors
        // {
        // SharedLockGuard guard(locks_[cur_id]);
        for (size_t j = 0; j < bipartite_graph_[cur_id].size(); ++j) {
            // for (auto nbr : bipartite_graph_[cur_id]) {  // current check node's
            // neighbors
            uint32_t nbr = bipartite_graph_[cur_id][j];
            if (nbr == gid) {
                continue;
            }
            // neighbors' neighbors are base points

            // SharedLockGuard guard(locks_[nbr]);
            for (size_t i = 0; i < bipartite_graph_[nbr].size(); ++i) {
                // for (auto &ns_nbr : bipartite_graph_[nbr]) {
                uint32_t ns_nbr = bipartite_graph_[nbr][i];
                if (visited.test(ns_nbr)) {
                    continue;
                }
                // nbr_ids.push_back(ns_nbr);
                float distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
                // if (metric_ == mysteryann::INNER_PRODUCT) {
                //     distance = -distance;
                // }
                if (fast_check) {
                    if (first_hop_min_dist > distance) {
                        first_hop_min_dist = distance;
                        first_hop_rank_1 = nbr;
                    }
                }
                Neighbor nn = Neighbor(ns_nbr, distance, false);
                queue.insert(nn);
                visited.set(nbr);
                full_retset.push_back(nn);

                if (fast_check) {
                    break;
                }
            }
        }

        if (fast_check) {
            for (size_t i = 0; i < bipartite_graph_[first_hop_rank_1].size(); ++i) {
                auto nbr = bipartite_graph_[first_hop_rank_1][i];
                if (visited.test(nbr)) {
                    continue;
                }
                // nbr_ids.push_back(bipartite_graph_[nbr][i]);

                // mysteryann::prefetch_vector((char *)(data_sq_ + cate_id * dimension_), dimension_ * sizeof(float));

                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);

                visited.set(nbr);
                Neighbor nn = Neighbor(nbr, distance, false);
                queue.insert(nn);
                full_retset.push_back(nn);
            }
        }
        // }
        // for (size_t i = 0; i < nbr_ids.size(); ++i) {
        //     uint32_t ns_nbr = nbr_ids[i];
        //     float distance;
        //     // mysteryann::prefetch_vector((char *)(data_bp_ + ns_nbr * dimension_), dimension_ * sizeof(float));
        //     distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
        //     // if (metric_ == mysteryann::INNER_PRODUCT) {
        //     //     distance = -distance;
        //     // }
        //     Neighbor nn = Neighbor(ns_nbr, distance, false);
        //     queue.insert(nn);
        // }
    }
}

void IndexBipartite::PruneLocalJoinCandidates(uint32_t node, const Parameters &parameters, uint32_t candi) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");

    NeighborPriorityQueue search_pool;
    search_pool.reserve(M_pjbp + 1);

    for (auto nbr : projection_graph_[node]) {
        if (nbr == node) {
            continue;
        }
        float distance = distance_->compare(data_bp_ + dimension_ * nbr, data_bp_ + dimension_ * node, dimension_);
        search_pool.insert({nbr, distance, false});
    }

    float distance = distance_->compare(data_bp_ + dimension_ * candi, data_bp_ + dimension_ * node, dimension_);
    search_pool.insert({candi, distance, false});

    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 0;
    if (search_pool[start].id == node) {
        start++;
    }
    result.push_back(search_pool[start].id);
    ++start;

    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != node) {
                result.push_back(p.id);
            }
        }
    }

    for (size_t i = 0; i < search_pool.size() && result.size() < M_pjbp; ++i) {
        if (std::find(result.begin(), result.end(), search_pool[i].id) == result.end()) {
            result.push_back(search_pool[i].id);
        }
    }

    {
        LockGuard guard(locks_[node]);
        projection_graph_[node] = result;
    }
    // pruned_list = result;
}

/*
Return the graph G that is the projection of the bipartite graph bipartite_graph_ onto base_points set
*/
void IndexBipartite::BipartiteProjection(const Parameters &parameters) {
    // Return the graph that is the projection of the bipartite graph final_graph_ onto base_points set
    std::cout << "begin projection graph init" << std::endl;
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    projection_graph_.resize(u32_nd_);
    // for (uint32_t i = 0; i < u32_nd_; ++i) {
    //     projection_graph_[i].reserve(M_pjbp * PROJECTION_SLACK);
    // }
    /*
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));

    auto local_join = [&](uint32_t n1, uint32_t n2) {
        if (std::find(projection_graph_[n1].begin(), projection_graph_[n1].end(), n2) == projection_graph_[n1].end()) {
            if (projection_graph_[n1].size() < M_pjbp * 1.3 && n1 != n2) {
                LockGuard guard(locks_[n1]);
                projection_graph_[n1].push_back(n2);
            } else {
                // PruneLocalJoinCandidates(n1, parameters, n2);
            }
        }

        if (std::find(projection_graph_[n2].begin(), projection_graph_[n2].end(), n1) == projection_graph_[n2].end()) {
            if (projection_graph_[n2].size() < M_pjbp * 1.3 && n1 != n2) {
                LockGuard guard(locks_[n2]);
                projection_graph_[n2].push_back(n1);
            } else {
                // PruneLocalJoinCandidates(n2, parameters, n1);
            }
        }
    };

    uint32_t jump_step = 1;
#pragma omp parallel for schedule(dynamic, 100)
    for (uint32_t i = u32_nd_; i < u32_total_pts_; ++i) {  // for each sampled query
        auto &cur_check_neighbors = bipartite_graph_[i];   // neighbors are base points
        for (size_t j = 0; j < cur_check_neighbors.size(); j += jump_step) {
            size_t next = j + 1;
            if (next < cur_check_neighbors.size()) {
                local_join(cur_check_neighbors[j], cur_check_neighbors[next]);
            }
            // uint32_t cnt = 0;
            // for (size_t m = j; m < cur_check_neighbors.size(); ++m) {
            //     local_join(cur_check_neighbors[j], cur_check_neighbors[m]);
            //     ++cnt;
            //     if (cnt >= jump_step) {
            //         break;
            //     }
            // }
        }
        if (i % 1000 == 0) {
            std::cout << "\r" << (100.0 * i) / (u32_nd_sq_) << "% of bipartite projection completed." << std::flush;
        }
    }

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t i = 0; i < u32_nd_; ++i) {
    //         std::vector<uint32_t> my_sqs;
    //         my_sqs.reserve(bipartite_graph_[i].size());

    //         for (auto sqs : bipartite_graph_[i]) {
    //             for (size_t j = 0; j < bipartite_graph_[sqs].size(); ++j) {
    //                 local_join(i, bipartite_graph_[sqs][j]);
    //                 // size_t next = j + 1;
    //                 // if (next < my_sqs.size()) {
    //                 //     local_join(my_sqs[j], my_sqs[next]);
    //                 // }
    //             }
    //         }
    //     }

#pragma omp parallel for schedule(dynamic, 100)
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        if (projection_graph_[i].size() < M_pjbp) {
            // std::cout << "Warning: projection graph node " << i << " has less than M_pjbp neighbors." << std::endl;
            if (bipartite_graph_[i].size() == 0) {
                std::cerr << "Error: projection graph node " << i << " has no neighbors." << std::endl;
                throw std::runtime_error("Error: projection graph node has no neighbors.");
            }
            for (auto &sq_nbrs : bipartite_graph_[i]) {
                auto cur_first = bipartite_graph_[sq_nbrs][0];
                local_join(i, cur_first);
            }
            uint32_t first_sq = bipartite_graph_[i][0];
            auto &first_sq_nbrs = bipartite_graph_[first_sq];
            for (size_t j = 0; j < first_sq_nbrs.size(); ++j) {
                local_join(i, first_sq_nbrs[j]);
            }
        }
    }

    std::cout << std::endl;

    float avg_degree = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        if (projection_graph_[i].size() > max_degree) {
            max_degree = projection_graph_[i].size();
        }
        if (projection_graph_[i].size() < min_degree) {
            min_degree = projection_graph_[i].size();
        }
        avg_degree += projection_graph_[i].size();
    }
    avg_degree /= u32_nd_;
    std::cout << "After local projection, average degree of projection graph: " << avg_degree << std::endl;
    std::cout << "After local projection, max degree of projection graph: " << max_degree << std::endl;
    std::cout << "After local projection, min degree of projection graph: " << min_degree << std::endl;
    */
}

void IndexBipartite::TrainingLink2Projection(const Parameters &parameters, SimpleNeighbor *simple_graph) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");
    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));
    std::vector<uint32_t> vis_order;
    std::vector<uint32_t> vis_order_sq;
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        vis_order.push_back(i);
    }
    for (uint32_t i = 0; i < u32_nd_sq_; ++i) {
        vis_order_sq.push_back(i);
    }

    #pragma omp parallel for schedule(static, 100)
    for (uint32_t it_sq = 0; it_sq < u32_nd_sq_; ++it_sq) {
        uint32_t sq = vis_order_sq[it_sq];
        // boost::dynamic_bitset<> visited{u32_total_pts_, false};
        auto &nn_base = learn_base_knn_[sq];
        if (nn_base.size() > 100) {
            nn_base.resize(100);
            nn_base.shrink_to_fit();
        }
        uint32_t choose_tgt = 0;
        // for (size_t i = 0; i < 100; ++i) {
        //     if (projection_graph_[nn_base[i]].size() < M_pjbp) {
        //         choose_tgt = nn_base[i];
        //         break;
        //     }
        // }
        uint32_t cur_tgt = nn_base[choose_tgt];
        std::vector<Neighbor> full_retset;
        for (size_t i = 0; i < nn_base.size(); ++i) {
            if (nn_base[i] == cur_tgt) {
                continue;
            }
            float distance = distance_->compare(data_bp_ + dimension_ * (uint64_t)nn_base[i], data_bp_ + dimension_ * (uint64_t)cur_tgt,
                                                (unsigned)dimension_);
            full_retset.push_back(Neighbor(nn_base[i], distance, false));
        }
        std::vector<uint32_t> pruned_list;
        pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
        PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * cur_tgt, cur_tgt, parameters, pruned_list);
        {
            LockGuard guard(locks_[cur_tgt]);
            projection_graph_[cur_tgt] = pruned_list;
        }
        ProjectionAddReverse(cur_tgt, parameters);
        if (sq % 1000 == 0) {
            std::cout << "\r" << (100.0 * sq) / u32_nd_sq_ << "% of projection search bipartite by base completed."
                      << std::flush;
        }
    }

    std::cout << std::endl;
    std::atomic<uint32_t> degree_cnt(0);
    std::atomic<uint32_t> zero_cnt(0);
#pragma omp parallel for schedule(static, 100)
    for (uint32_t i = 0; i < vis_order.size(); ++i) {
        uint32_t node = vis_order[i];
        if (projection_graph_[node].size() < M_pjbp) {
            // std::cout << "Warning: projection graph node " << node << " has less than M_pjbp neighbors." << std::endl;
            degree_cnt.fetch_add(1);
            if (projection_graph_[node].size() == 0) {
                zero_cnt.fetch_add(1);
            }
        }
        ProjectionAddReverse(node, parameters);
    }
    std::cout << "Warning: " << degree_cnt.load() << " nodes have less than M_pjbp neighbors." << std::endl;
    std::cout << "Warning: " << zero_cnt.load() << " nodes have no neighbors." << std::endl;

#pragma omp parallel for schedule(dynamic, 100)
    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> ok_insert;
        ok_insert.reserve(M_pjbp);
        for (size_t j = 0; j < supply_nbrs_[i].size(); ++j) {
            if (ok_insert.size() >= M_pjbp * 2) {
                break;
            }
            if (std::find(projection_graph_[i].begin(), projection_graph_[i].end(), supply_nbrs_[i][j]) ==
                projection_graph_[i].end()) {
                ok_insert.push_back(supply_nbrs_[i][j]);
            }
        }
        projection_graph_[i].insert(projection_graph_[i].end(), ok_insert.begin(), ok_insert.end());
        // projection_graph_[i] = ok_insert;
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + M_pjbp);
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + projection_graph_[i].size());
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin());
    }
}


void IndexBipartite::LinkBase(const Parameters &parameters, SimpleNeighbor *simple_graph) {
    uint32_t M_pjbp = static_cast<uint32_t>(parameters.Get<uint32_t>("M_pjbp"));
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");

    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));
    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        // projection_graph_[i].clear();
        // supply_nbrs_[i] = projection_graph_[i];
        // supply_nbrs_[i].reserve(M_pjbp * 2 * PROJECTION_SLACK);
        projection_graph_[i].reserve(M_pjbp * 2 * PROJECTION_SLACK);
        // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
        // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
    }
    std::vector<uint32_t> vis_order;

    for (uint32_t i = 0; i < u32_nd_; ++i) {
        vis_order.push_back(i);
    }


#pragma omp parallel for schedule(dynamic, 2048)
    for (uint32_t i = 0; i < nd_; ++i) {
        size_t node = vis_order[i];
        boost::dynamic_bitset<> visited{u32_nd_, false};
        std::vector<Neighbor> full_retset;
        full_retset.reserve(L_pjpq);
        NeighborPriorityQueue search_pool;
        SearchProjectionGraphInternalPJ(search_pool, data_bp_ + dimension_ * node, node, parameters, visited,
                                      full_retset);
        std::vector<uint32_t> pruned_list;
        // pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
        for (unsigned j = 0; j < full_retset.size(); j++) {
            if (full_retset[j].id == (unsigned)node) {
                full_retset.erase(full_retset.begin() + j);
                j--;
            }
        }
        // PruneProjectionBaseSearchCandidates(full_retset, data_bp_ + dimension_ * node, node, parameters, pruned_list);
        PruneProjectionBaseSearchCandidatesSupply(full_retset, data_bp_ + dimension_ * node, node, parameters, pruned_list);
        {
            LockGuard guard(locks_[node]);

            projection_graph_[node] = pruned_list;
        }
        // SupplyAddReverse(node, parameters);
        ProjectionAddReverse(node, parameters);
        if (node % 1000 == 0) {
            std::cout << "\r" << (100.0 * node) / (u32_nd_) << "% of projection graph base search completed."
                      << std::flush;
        }
    }
    std::cout << "In LinkBase() finish connectivity enhancement" << std::endl;

#pragma omp parallel for schedule(dynamic, 2048)
    for (uint32_t node = 0; node < nd_; ++node) {
        ProjectionAddReverse(node, parameters);
    }
#pragma omp parallel for schedule(dynamic, 2048)
    for (uint32_t i = 0; i < nd_; ++i) {
        size_t node = vis_order[i];
        if (projection_graph_[node].size() > M_pjbp) {
            std::vector<Neighbor> full_retset;
            tsl::robin_set<uint32_t> visited;
            for (size_t j = 0; j < projection_graph_[node].size(); ++j) {
                if (visited.find(projection_graph_[node][j]) != visited.end()) {
                    continue;
                }
                float distance = distance_->compare(data_bp_ + dimension_ * projection_graph_[node][j],
                                                    data_bp_ + dimension_ * node, dimension_);
                visited.insert(projection_graph_[node][j]);
                full_retset.push_back(Neighbor(projection_graph_[node][j], distance, false));
            }
            std::vector<uint32_t> prune_list;
            PruneProjectionBaseSearchCandidatesSupply(full_retset, data_bp_ + dimension_ * node, node, parameters,
                                                prune_list);
            {
                LockGuard guard(locks_[node]);
                projection_graph_[node].clear();
                projection_graph_[node] = prune_list;
            }
        }
    }
    std::cout << "In LinkBase() finish connectivity enhancement degree check" << std::endl;

    // degree stats

    float avg_degree = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();

    for (uint32_t i = 0; i < nd_; ++i) {
        if (projection_graph_[i].size() > max_degree) {
            max_degree = projection_graph_[i].size();
        }
        if (projection_graph_[i].size() < min_degree) {
            min_degree = projection_graph_[i].size();
        }
        avg_degree += projection_graph_[i].size();
    }
    avg_degree /= (float)u32_nd_;
    std::cout << "After Link Base, average degree of projection graph: " << avg_degree << std::endl;
    std::cout << "After Link Base, max degree of projection graph: " << max_degree << std::endl;
    std::cout << "After Link Base, min degree of projection graph: " << min_degree << std::endl;

}

void IndexBipartite::LinkProjection(const Parameters &parameters, SimpleNeighbor *simple_graph) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");

    omp_set_num_threads(parameters.Get<uint32_t>("num_threads"));
    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t sq = 0; sq < nd_sq_; ++sq) {
    //         boost::dynamic_bitset<> visited{u32_total_pts_, false};
    //         NeighborPriorityQueue pj_search_queue(L_pjpq);
    //         pj_search_queue.reserve(L_pjpq);
    //         std::vector<Neighbor> full_retset;
    //         full_retset.reserve(L_pjpq);
    //         SearchBipartitebyQuery(data_sq_ + dimension_ * sq, sq + u32_nd_, parameters, nullptr, pj_search_queue,
    //         visited,
    //                                full_retset);
    //         std::vector<uint32_t> pruned_list;
    //         pruned_list.reserve(M_pjbp * PROJECTION_SLACK);

    //         // std::vector<Neighbor> full_retset;
    //         uint32_t as_source_node =
    //             PruneProjectionBipartiteCandidates(full_retset, data_sq_ + dimension_ * sq, sq, parameters,
    //             pruned_list);
    //         {
    //             LockGuard guard(locks_[as_source_node]);
    //             projection_graph_[as_source_node] = pruned_list;
    //         }
    //         ProjectionAddReverse(as_source_node, parameters);
    //         if (sq % 1000 == 0) {
    //             std::cout << "\r" << (100.0 * sq) / (u32_nd_sq_) << "% of projection index build completed." <<
    //             std::flush;
    //         }
    //     }

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t node = 0; node < nd_; ++node) {
    //         ProjectionAddReverse(node, parameters);
    //     }
    // iterate target is sq, so not all base will be visited/checked.
    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t sq = 0; sq < nd_sq_; ++sq) {
    //         boost::dynamic_bitset<> visited{u32_nd_, false};
    //         visited.reset();
    //         NeighborPriorityQueue search_queue(L_pjpq);
    //         search_queue.reserve(L_pjpq);
    //         std::vector<Neighbor> full_retset;
    //         SearchProjectionbyQuery(data_sq_ + dimension_ * sq, parameters, search_queue, visited, full_retset);

    //         std::vector<uint32_t> pruned_list;
    //         pruned_list.reserve(M_pjbp * PROJECTION_SLACK);

    //         uint32_t as_source_node =
    //             PruneProjectionCandidates(full_retset, data_sq_ + dimension_ * sq, sq, parameters, pruned_list);
    //         {
    //             LockGuard guard(locks_[as_source_node]);
    //             projection_graph_[as_source_node] = pruned_list;
    //         }
    //         ProjectionAddReverse(as_source_node, parameters);
    //         if (sq % 1000 == 0) {
    //             std::cout << "\r" << (100.0 * sq) / (u32_nd_sq_) << "% of projection index build completed." <<
    //             std::flush;
    //         }
    //     }
    //     // note: connectivity of projection graph is an unsolved and unverified problem.
    //     // here: add reverse. 2023.6.29

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t node = 0; node < nd_; ++node) {
    //         ProjectionAddReverse(node, parameters);
    //     }

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t bp = 0; bp < nd_; ++bp) {
    //         boost::dynamic_bitset<> visited{u32_total_pts_, false};
    //         NeighborPriorityQueue pj_search_queue(50);
    //         pj_search_queue.reserve(50);
    //         std::vector<Neighbor> full_retset;
    //         full_retset.reserve(L_pjpq);
    //         SearchBipartitebyBase(data_bp_ + dimension_ * bp, bp, parameters, nullptr, pj_search_queue, visited,
    //                               full_retset);

    //         std::vector<uint32_t> pruned_list;
    //         pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
    //         // full_retset.push_back(Neighbor);

    //         // std::vector<Neighbor> full_retset;

    //         PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * bp, bp, parameters, pruned_list);
    //         {
    //             LockGuard guard(locks_[bp]);
    //             projection_graph_[bp] = pruned_list;
    //         }
    //         ProjectionAddReverse(bp, parameters);
    //         if (bp % 1000 == 0) {
    //             std::cout << "\r" << (100.0 * bp) / (u32_nd_) << "% of projection search bipartite by base
    //             completed."
    //                       << std::flush;
    //         }
    //     }

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t node = 0; node < nd_; ++node) {
    //         ProjectionAddReverse(node, parameters);
    //     }
    std::vector<uint32_t> vis_order;
    std::vector<uint32_t> vis_order_sq;
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        vis_order.push_back(i);
    }
    for (uint32_t i = 0; i < u32_nd_sq_; ++i) {
        vis_order_sq.push_back(i);
    }
// #pragma omp parallel for schedule(static, 100)
//     for (uint32_t bp = 0; bp < nd_; ++bp) {
//         bp = vis_order[bp];
//         // boost::dynamic_bitset<> visited{u32_total_pts_, false};
//         auto &nn_learn = base_learn_knn_[bp];
//         std::vector<Neighbor> full_retset;
//         for (size_t i = 0; i < nn_learn.size(); ++i) {
//             full_retset.push_back(Neighbor(nn_learn[i], 0, false));
//         }
//         std::vector<uint32_t> pruned_list;
//         pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
//         PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * bp, bp, parameters, pruned_list);
//         {
//             LockGuard guard(locks_[bp]);
//             projection_graph_[bp] = pruned_list;
//         }
//         ProjectionAddReverse(bp, parameters);
//         if (bp % 1000 == 0) {
//             std::cout << "\r" << (100.0 * bp) / (u32_nd_) << "% of projection search bipartite by base completed."
//                       << std::flush;
//         }
//     }
#pragma omp parallel for schedule(static, 100)
    for (uint32_t it_sq = 0; it_sq < u32_nd_sq_; ++it_sq) {
        uint32_t sq = vis_order_sq[it_sq];
        // boost::dynamic_bitset<> visited{u32_total_pts_, false};
        auto &nn_base = learn_base_knn_[sq];
        if (nn_base.size() > 100) {
            nn_base.resize(100);
            nn_base.shrink_to_fit();
        }
        uint32_t choose_tgt = 0;
        // for (size_t i = 0; i < 100; ++i) {
        //     if (projection_graph_[nn_base[i]].size() < M_pjbp) {
        //         choose_tgt = nn_base[i];
        //         break;
        //     }
        // }
        uint32_t cur_tgt = nn_base[choose_tgt];
        std::vector<Neighbor> full_retset;
        for (size_t i = 0; i < nn_base.size(); ++i) {
            if (nn_base[i] == cur_tgt) {
                continue;
            }
            float distance = distance_->compare(data_bp_ + dimension_ * (uint64_t)nn_base[i], data_bp_ + dimension_ * (uint64_t)cur_tgt,
                                                (unsigned)dimension_);
            full_retset.push_back(Neighbor(nn_base[i], distance, false));
        }
        std::vector<uint32_t> pruned_list;
        pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
        PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * cur_tgt, cur_tgt, parameters, pruned_list);
        {
            LockGuard guard(locks_[cur_tgt]);
            projection_graph_[cur_tgt] = pruned_list;
        }
        ProjectionAddReverse(cur_tgt, parameters);
        if (sq % 1000 == 0) {
            std::cout << "\r" << (100.0 * sq) / u32_nd_sq_ << "% of projection search bipartite by base completed."
                      << std::flush;
        }
    }

    std::cout << std::endl;
    std::atomic<uint32_t> degree_cnt(0);
    std::atomic<uint32_t> zero_cnt(0);
#pragma omp parallel for schedule(static, 100)
    for (uint32_t i = 0; i < vis_order.size(); ++i) {
        uint32_t node = vis_order[i];
        if (projection_graph_[node].size() < M_pjbp) {
            // std::cout << "Warning: projection graph node " << node << " has less than M_pjbp neighbors." << std::endl;
            degree_cnt.fetch_add(1);
            if (projection_graph_[node].size() == 0) {
                zero_cnt.fetch_add(1);
            }
        }
        ProjectionAddReverse(node, parameters);
    }
    std::cout << "Warning: " << degree_cnt.load() << " nodes have less than M_pjbp neighbors." << std::endl;
    std::cout << "Warning: " << zero_cnt.load() << " nodes have no neighbors." << std::endl;

#pragma omp parallel for schedule(static, 2048)
    for (uint32_t i = 0; i < vis_order.size(); ++i) {
        size_t node = (size_t)vis_order[i];
        if (projection_graph_[node].size() > M_pjbp) {
            std::vector<Neighbor> full_retset;
            tsl::robin_set<uint32_t> visited;
            for (size_t j = 0; j < projection_graph_[node].size(); ++j) {
                if (visited.find(projection_graph_[node][j]) != visited.end()) {
                    continue;
                }
                float distance = distance_->compare(data_bp_ + dimension_ * (size_t)projection_graph_[node][j],
                                                    data_bp_ + dimension_ * (size_t)node, dimension_);
                visited.insert(projection_graph_[node][j]);
                full_retset.push_back(Neighbor(projection_graph_[node][j], distance, false));
            }
            for (unsigned j = 0; j < full_retset.size(); j++) {
                if (full_retset[j].id == (unsigned)node) {
                    full_retset.erase(full_retset.begin() + j);
                    j--;
                }
            }
            std::vector<uint32_t> prune_list;
            PruneBiSearchBaseGetBase(full_retset, data_bp_ + dimension_ * (size_t)node, node, parameters, prune_list);
            {
                LockGuard guard(locks_[node]);
                projection_graph_[node].clear();
                projection_graph_[node] = prune_list;
            }
        }
    }

    // stats projection graph degree
    float avg_degree = 0;
    uint64_t total_degree = 0;
    uint32_t max_degree = 0;
    uint32_t min_degree = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        if (projection_graph_[i].size() > max_degree) {
            max_degree = projection_graph_[i].size();
        }
        if (projection_graph_[i].size() < min_degree) {
            min_degree = projection_graph_[i].size();
        }
        avg_degree += static_cast<float>(projection_graph_[i].size());
        total_degree += projection_graph_[i].size();
    }
    std::cout << "total degree: " << total_degree << std::endl;
    avg_degree /= (float)u32_nd_;
    std::cout << "After projection, average degree of projection graph: " << avg_degree << std::endl;
    std::cout << "After projection, max degree of projection graph: " << max_degree << std::endl;
    std::cout << "After projection, min degree of projection graph: " << min_degree << std::endl;

    std::cout << std::endl;

    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        // projection_graph_[i].clear();
        supply_nbrs_[i] = projection_graph_[i];
        supply_nbrs_[i].reserve(M_pjbp * 2 * PROJECTION_SLACK);
        // supply_nbrs_[i].reserve(M_pjbp * PROJECTION_SLACK);
    }

#pragma omp parallel for schedule(dynamic, 2048)
    for (uint32_t i = 0; i < nd_; ++i) {
        size_t node = vis_order[i];
        boost::dynamic_bitset<> visited{u32_nd_, false};
        std::vector<Neighbor> full_retset;
        full_retset.reserve(L_pjpq);
        NeighborPriorityQueue search_pool;
        SearchProjectionGraphInternal(search_pool, data_bp_ + dimension_ * node, node, parameters, visited,
                                      full_retset);
        std::vector<uint32_t> pruned_list;
        pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
        for (unsigned j = 0; j < full_retset.size(); j++) {
            if (full_retset[j].id == (unsigned)node) {
                full_retset.erase(full_retset.begin() + j);
                j--;
            }
        }
        PruneProjectionBaseSearchCandidates(full_retset, data_bp_ + dimension_ * node, node, parameters, pruned_list);
        {
            LockGuard guard(locks_[node]);

            supply_nbrs_[node] = pruned_list;
        }
        SupplyAddReverse(node, parameters);
        if (node % 1000 == 0) {
            std::cout << "\r" << (100.0 * node) / (u32_nd_) << "% of projection graph base search completed."
                      << std::flush;
        }
    }
    std::cout << "finish connectivity enhancement" << std::endl;

// #pragma omp parallel for schedule(dynamic, 2048)
//     for (uint32_t node = 0; node < nd_; ++node) {
//         SupplyAddReverse(node, parameters);
//     }
#pragma omp parallel for schedule(dynamic, 2048)
    for (uint32_t i = 0; i < nd_; ++i) {
        size_t node = vis_order[i];
        if (supply_nbrs_[node].size() > M_pjbp) {
            std::vector<Neighbor> full_retset;
            tsl::robin_set<uint32_t> visited;
            for (size_t j = 0; j < supply_nbrs_[node].size(); ++j) {
                if (visited.find(supply_nbrs_[node][j]) != visited.end()) {
                    continue;
                }
                float distance = distance_->compare(data_bp_ + dimension_ * supply_nbrs_[node][j],
                                                    data_bp_ + dimension_ * node, dimension_);
                visited.insert(supply_nbrs_[node][j]);
                full_retset.push_back(Neighbor(supply_nbrs_[node][j], distance, false));
            }
            std::vector<uint32_t> prune_list;
            PruneProjectionBaseSearchCandidates(full_retset, data_bp_ + dimension_ * node, node, parameters,
                                                prune_list);
            {
                LockGuard guard(locks_[node]);
                supply_nbrs_[node].clear();
                supply_nbrs_[node] = prune_list;
            }
        }
    }
    std::cout << "finish connectivity enhancement degree check" << std::endl;

#pragma omp parallel for schedule(dynamic, 100)
    for (size_t i = 0; i < projection_graph_.size(); ++i) {
        std::vector<uint32_t> ok_insert;
        ok_insert.reserve(M_pjbp);
        for (size_t j = 0; j < supply_nbrs_[i].size(); ++j) {
            if (ok_insert.size() >= M_pjbp * 2) {
                break;
            }
            if (std::find(projection_graph_[i].begin(), projection_graph_[i].end(), supply_nbrs_[i][j]) ==
                projection_graph_[i].end()) {
                ok_insert.push_back(supply_nbrs_[i][j]);
            }
        }
        projection_graph_[i].insert(projection_graph_[i].end(), ok_insert.begin(), ok_insert.end());
        // projection_graph_[i] = ok_insert;
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + M_pjbp);
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin() + projection_graph_[i].size());
        // std::copy(ok_insert.begin(), ok_insert.end(), projection_graph_[i].begin());
    }

    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (uint32_t node = 0; node < nd_; ++node) {
    //         ProjectionAddReverse(node, parameters);
    //     }
}

void IndexBipartite::SearchProjectionGraphInternalPJ(NeighborPriorityQueue &search_queue, const float *query,
                                                   uint32_t tgt, const Parameters &parameters,
                                                   boost::dynamic_bitset<> &visited,
                                                   std::vector<Neighbor> &full_retset) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pjpq");

    search_queue.reserve(L_pq);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    uint32_t start = projection_ep_;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(start);
    // for (uint32_t i = 0; i < 5; ++i) {
    //     init_ids.push_back(dis(gen));
    // }
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // std::bitset<bsize> visited;
    // block_metric.record();
    for (auto &id : init_ids) {
        float distance;
        // dist_cmp_metric.reset();
        distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        visited.set(id);
        // memory_access_metric.record();
    }
    // uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        full_retset.push_back(cur_check_node);
        // visited.set(cur_id);

        // memory_access_metric.record();

        // get neighbors' neighbors, first
        for (auto nbr : projection_graph_[cur_id]) {  // current check node's neighbors
        // for (auto nbr : supply_nbrs_[cur_id]) {  // current check node's neighbors

            // memory_access_metric.reset();
            if (visited.test(nbr) || nbr == tgt) {
                // if (visited.test(nbr)) {
                continue;
            }
            // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
            visited.set(nbr);
            // memory_access_metric.record();
            float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);
            // dist_cmp_metric.reset();
            // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
            //     distance = -distance;
            // }

            // dist_cmp_metric.record();
            // memory_access_metric.reset();
            // ++cmps;
            Neighbor nn = Neighbor(nbr, distance, false);
            search_queue.insert(nn);
            // full_retset.push_back(nn);
            // memory_access_metric.record();
        }
    }
}

void IndexBipartite::SearchProjectionGraphInternal(NeighborPriorityQueue &search_queue, const float *query,
                                                   uint32_t tgt, const Parameters &parameters,
                                                   boost::dynamic_bitset<> &visited,
                                                   std::vector<Neighbor> &full_retset) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pjpq");

    search_queue.reserve(L_pq);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    uint32_t start = projection_ep_;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(start);
    // for (uint32_t i = 0; i < 5; ++i) {
    //     init_ids.push_back(dis(gen));
    // }
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // std::bitset<bsize> visited;
    // block_metric.record();
    for (auto &id : init_ids) {
        float distance;
        // dist_cmp_metric.reset();
        distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        visited.set(id);
        // memory_access_metric.record();
    }
    // uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        full_retset.push_back(cur_check_node);
        // visited.set(cur_id);

        // memory_access_metric.record();

        // get neighbors' neighbors, first
        // for (auto nbr : projection_graph_[cur_id]) {  // current check node's neighbors
        for (auto nbr : supply_nbrs_[cur_id]) {  // current check node's neighbors

            // memory_access_metric.reset();
            if (visited.test(nbr) || nbr == tgt) {
                // if (visited.test(nbr)) {
                continue;
            }
            // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
            visited.set(nbr);
            // memory_access_metric.record();
            float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)dimension_);
            // dist_cmp_metric.reset();
            // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
            //     distance = -distance;
            // }

            // dist_cmp_metric.record();
            // memory_access_metric.reset();
            // ++cmps;
            Neighbor nn = Neighbor(nbr, distance, false);
            search_queue.insert(nn);
            // full_retset.push_back(nn);
            // memory_access_metric.record();
        }
    }
}

void IndexBipartite::SupplyAddReverse(uint32_t src_node, const Parameters &parameters) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp") * 2;
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    std::vector<uint32_t> &nbrs = supply_nbrs_[src_node];

    for (size_t i = 0; i < nbrs.size(); ++i) {
        auto des = nbrs[i];
        auto &des_nbrs = supply_nbrs_[des];
        bool need_prune = false;
        {
            LockGuard guard(locks_[des]);
            if (std::find(des_nbrs.begin(), des_nbrs.end(), src_node) == des_nbrs.end()) {
                if (des_nbrs.size() < M_pjbp) {
                    des_nbrs.push_back(src_node);
                } else {
                    need_prune = true;
                }
            } else {
                continue;
            }
        }
        if (need_prune) {
            std::vector<uint32_t> copy_vec;
            copy_vec.reserve(M_pjbp * PROJECTION_SLACK);
            for (size_t j = 0; j < des_nbrs.size(); ++j) {
                copy_vec.push_back(des_nbrs[j]);
            }
            copy_vec.push_back(src_node);
            // std::vector<uint32_t> pruned_list;
            // pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
            PruneProjectionInternalReverseCandidates(des, parameters, copy_vec);
            {
                LockGuard guard(locks_[des]);
                supply_nbrs_[des] = copy_vec;
            }
        }
    }
}

void IndexBipartite::ProjectionAddReverse(uint32_t src_node, const Parameters &parameters) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    std::vector<uint32_t> &nbrs = projection_graph_[src_node];
    float avg_copy = 0;
    
    for (size_t i = 0; i < nbrs.size(); ++i) {
        auto des = nbrs[i];
        auto &des_nbrs = projection_graph_[des];
        bool need_prune = false;
        {
            LockGuard guard(locks_[des]);
            if (std::find(des_nbrs.begin(), des_nbrs.end(), src_node) == des_nbrs.end()) {
                if (des_nbrs.size() < M_pjbp) {
                    des_nbrs.push_back(src_node);
                } else {
                    need_prune = true;
                }
            } else {
                continue;
            }
        }

        if (need_prune) {
            std::vector<uint32_t> copy_vec;
            copy_vec.reserve(M_pjbp * PROJECTION_SLACK);
            {
                LockGuard guard(locks_[des]);
                for (size_t j = 0; j < des_nbrs.size(); ++j) {
                    copy_vec.push_back(des_nbrs[j]);
                }
            }
            copy_vec.push_back(src_node);
            // std::vector<uint32_t> pruned_list;
            // pruned_list.reserve(M_pjbp * PROJECTION_SLACK);
            PruneProjectionReverseCandidates(des, parameters, copy_vec);
            {
                LockGuard guard(locks_[des]);
                projection_graph_[des] = copy_vec;
            }
        }
    }
}

void IndexBipartite::PruneProjectionInternalReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                                              std::vector<uint32_t> &pruned_list) {
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // NeighborPriorityQueue prune_queue(pruned_list.size());
    // prune_queue.reserve(pruned_list.size());
    // for (size_t i = 0; i < pruned_list.size(); ++i) {
    //     float distance = distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ * pruned_list[i],
    //                                         (unsigned)dimension_);
    //     // float distance = l2_distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ *
    //     // pruned_list[i],
    //     //                                        (unsigned)dimension_);
    //     // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
    //     //     distance = -distance;
    //     // }
    //     prune_queue.insert({pruned_list[i], distance, false});
    // }
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp") * 2;
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    std::vector<Neighbor> prune_queue(pruned_list.size());
    prune_queue.reserve(pruned_list.size());
    for (size_t i = 0; i < pruned_list.size(); ++i) {
        float distance = distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ * pruned_list[i],
                                            (unsigned)dimension_);
        // float distance = l2_distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ *
        // pruned_list[i],
        //                                        (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        Neighbor nn = Neighbor(pruned_list[i], distance, false);
        if (std::find(prune_queue.begin(), prune_queue.end(), nn) == prune_queue.end()) {
            prune_queue.push_back(nn);
        }
        // prune_queue.push_back({pruned_list[i], distance, false});
    }
    std::sort(prune_queue.begin(), prune_queue.end());
    // std::random_shuffle(pruned_list.begin(), pruned_list.end());
    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    // std::vector<float> occlude_factor;
    // uint32_t degree = M_pjbp;
    // auto &pool = prune_queue;
    // // occlude_list can be called with the same scratch more than once by
    // // search_for_point_and_add_link through inter_insert.
    // occlude_factor.clear();
    // // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    // occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    // float cur_alpha = 1;
    // float alpha = 1.2;
    // // used for MIPS, where we store a value of eps in cur_alpha to
    // // denote pruned out entries which we can skip in later rounds.
    // while (cur_alpha <= alpha && result.size() < degree) {
    //     float eps = cur_alpha + 0.01f;

    //     for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
    //         if (occlude_factor[iter - pool.begin()] > cur_alpha) {
    //             continue;
    //         }
    //         // Set the entry to float::max so that is not considered again
    //         occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
    //         // Add the entry to the result if its not been deleted, and doesn't add a self loop

    //         if (iter->id != src_node) {
    //             result.push_back(iter->id);
    //         }

    //         // Update occlude factor for points from iter+1 to pool.end()
    //         for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
    //             auto t = iter2 - pool.begin();
    //             if (occlude_factor[t] > alpha) continue;
    //             float djk = distance_->compare(data_bp_ + dimension_ * (size_t)iter2->id,
    //                                            data_bp_ + dimension_ * (size_t)iter->id, (unsigned)dimension_);

    //             occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
    //                        : std::max(occlude_factor[t], iter2->distance / djk);
    //             // if ((-djk) > cur_alpha * (-iter2->distance)) {
    //             //     occlude_factor[t] = std::max(occlude_factor[t], eps);
    //             // }
    //         }
    //     }
    //     cur_alpha *= 1.2;
    // }
    uint32_t start = 0;
    if (prune_queue[start].id == src_node) {
        start++;
    }
    result.push_back(prune_queue[start].id);
    while (result.size() < M_pjbp && (++start) < prune_queue.size()) {
        auto &p = prune_queue[start];
        bool occlude = false;
        for (size_t i = 0; i < result.size(); ++i) {
            if (p.id == result[i]) {
                occlude = true;
                break;
            }
            // mysteryann::prefetch_vector((char *)(data_bp_ + p.id * dimension_), dimension_ * sizeof(float));
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
                                           (unsigned)dimension_);
            // float djk = l2_distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
            //                                   (unsigned)dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != src_node) {
                result.push_back(p.id);
            }
        }
    }
    start = 0;
    while (result.size() < M_pjbp && (++start) < prune_queue.size()) {
        auto &p = prune_queue[start];
        bool occlude = false;
        for (size_t i = 0; i < result.size(); ++i) {
            if (p.id == result[i]) {
                occlude = true;
                break;
            }
            // mysteryann::prefetch_vector((char *)(data_bp_ + p.id * dimension_), dimension_ * sizeof(float));
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
                                           (unsigned)dimension_);
            // float djk = l2_distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
            //                                   (unsigned)dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != src_node) {
                if (std::find(result.begin(), result.end(), p.id) == result.end()) {
                    result.push_back(p.id);
                }
            }
        }
    }

    // for (size_t i = 0; i < pruned_list.size() && result.size() < M_pjbp; ++i) {
    //     if (std::find(result.begin(), result.end(), pruned_list[i]) == result.end()) {
    //         result.push_back(pruned_list[i]);
    //     }
    // }

    pruned_list = result;
}

void IndexBipartite::PruneProjectionReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                                      std::vector<uint32_t> &pruned_list) {
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    // NeighborPriorityQueue prune_queue(pruned_list.size());
    // prune_queue.reserve(pruned_list.size());
    // for (size_t i = 0; i < pruned_list.size(); ++i) {
    //     float distance = distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ * pruned_list[i],
    //                                         (unsigned)dimension_);
    //     // float distance = l2_distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ *
    //     // pruned_list[i],
    //     //                                        (unsigned)dimension_);
    //     // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
    //     //     distance = -distance;
    //     // }
    //     prune_queue.insert({pruned_list[i], distance, false});
    // }
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    std::vector<Neighbor> prune_queue;
    prune_queue.reserve(pruned_list.size());
    for (size_t i = 0; i < pruned_list.size(); ++i) {
        float distance = distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ * pruned_list[i],
                                            (unsigned)dimension_);
        // float distance = l2_distance_->compare(data_bp_ + dimension_ * src_node, data_bp_ + dimension_ *
        // pruned_list[i],
        //                                        (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        Neighbor nn = Neighbor(pruned_list[i], distance, false);
        if (std::find(prune_queue.begin(), prune_queue.end(), nn) == prune_queue.end()) {
            prune_queue.push_back(nn);
        }
        // prune_queue.push_back({pruned_list[i], distance, false});
    }
    std::sort(prune_queue.begin(), prune_queue.end());
    // std::random_shuffle(pruned_list.begin(), pruned_list.end());
    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    // std::vector<float> occlude_factor;
    // uint32_t degree = M_pjbp;
    // auto &pool = prune_queue;
    // // occlude_list can be called with the same scratch more than once by
    // // search_for_point_and_add_link through inter_insert.
    // occlude_factor.clear();
    // // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    // occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    // float cur_alpha = 1;
    // float alpha = 1.0;
    // // used for MIPS, where we store a value of eps in cur_alpha to
    // // denote pruned out entries which we can skip in later rounds.
    // float eps = cur_alpha + 0.01f;

    // for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
    //     if (occlude_factor[iter - pool.begin()] > cur_alpha) {
    //         continue;
    //     }
    //     // Set the entry to float::max so that is not considered again
    //     occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
    //     // Add the entry to the result if its not been deleted, and doesn't add a self loop

    //     if (iter->id != src_node) {
    //         result.push_back(iter->id);
    //     }

    //     // Update occlude factor for points from iter+1 to pool.end()
    //     for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
    //         auto t = iter2 - pool.begin();
    //         if (occlude_factor[t] > alpha) continue;
    //         float djk = distance_->compare(data_bp_ + dimension_ * (size_t)iter2->id,
    //                                        data_bp_ + dimension_ * (size_t)iter->id, (unsigned)dimension_);

    //         if ((-djk) > cur_alpha * (-iter2->distance)) {
    //             occlude_factor[t] = std::max(occlude_factor[t], eps);
    //         }
    //     }
    // }
    uint32_t start = 0;
    if (prune_queue[start].id == src_node) {
        start++;
    }
    result.push_back(prune_queue[start].id);
    while (result.size() < M_pjbp && (++start) < prune_queue.size()) {
        auto &p = prune_queue[start];
        bool occlude = false;
        for (size_t i = 0; i < result.size(); ++i) {
            if (p.id == result[i]) {
                occlude = true;
                break;
            }
            // mysteryann::prefetch_vector((char *)(data_bp_ + p.id * dimension_), dimension_ * sizeof(float));
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
                                           (unsigned)dimension_);
            // float djk = l2_distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
            //                                   (unsigned)dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            if (djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != src_node) {
                result.push_back(p.id);
            }
        }
    }

    start = 0;
    while (result.size() < M_pjbp && (++start) < prune_queue.size()) {
        auto &p = prune_queue[start];
        bool occlude = false;
        for (size_t i = 0; i < result.size(); ++i) {
            if (p.id == result[i]) {
                occlude = true;
                break;
            }
            // mysteryann::prefetch_vector((char *)(data_bp_ + p.id * dimension_), dimension_ * sizeof(float));
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
                                           (unsigned)dimension_);
            // float djk = l2_distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[i],
            //                                   (unsigned)dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != src_node) {
                if (std::find(result.begin(), result.end(), p.id) == result.end()) {
                    result.push_back(p.id);
                }
            }
        }
    }

    for (size_t i = 0; i < pruned_list.size() && result.size() < M_pjbp; ++i) {
        if (std::find(result.begin(), result.end(), pruned_list[i]) == result.end()) {
            result.push_back(pruned_list[i]);
        }
    }
    // for (size_t i = pruned_list.size() - 1; i > 0 && result.size() < M_pjbp * 1.5; --i) {
    //     if (std::find(result.begin(), result.end(), pruned_list[i]) == result.end()) {
    //         if (pruned_list[i] != src_node) {
    //             result.push_back(pruned_list[i]);
    //         }
    //     }
    // }

    pruned_list = result;
}

void IndexBipartite::PruneBiSearchBaseGetBase(std::vector<Neighbor> &search_pool, const float *query, uint32_t tgt_base,
                                              const Parameters &parameters, std::vector<uint32_t> &pruned_list) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t L_pjpq = parameters.Get<uint32_t>("L_pjpq");

    // std::sort(search_pool.begin(), search_pool.end());

    std::vector<Neighbor>& base_pool = search_pool;
    // for (auto &q_node : search_pool) {
    //     if (base_pool.size() >= L_pjpq) {
    //         break;
    //     }
    //     // for (auto &b_node : bipartite_graph_[q_node.id]) {
    //     //     if (std::find(base_id.begin(), base_id.end(), b_node) == base_id.end()) {
    //     //         float distance = distance_->compare(data_bp_ + dimension_ * b_node, query, (unsigned)dimension_);
    //     //         base_pool.push_back({b_node, distance, false});
    //     //         base_id.push_back(b_node);
    //     //     }
    //     // }
    //     // uint32_t cnt = 0;
    //     for (auto &b_node : learn_base_knn_[q_node.id]) {
    //         if (std::find(base_id.begin(), base_id.end(), b_node) == base_id.end()) {
    //             if (b_node == tgt_base) {
    //                 continue;
    //             }
    //             float distance = distance_->compare(data_bp_ + dimension_ * (uint64_t)b_node, query, (unsigned)dimension_);
    //             base_pool.push_back({b_node, distance, false});
    //             base_id.push_back(b_node);
    //             // ++cnt;
    //         }
    //         // if (cnt > 100) {
    //         //     break;
    //         // }
    //     }
    // }
    

    // for (auto &b_node : search_pool) {
    //     if (std::find(base_id.begin(), base_id.end(), b_node.id) == base_id.end()) {
    //         if (b_node.id == tgt_base) {
    //             continue;
    //         }
    //         // base_pool.push_back(b_node);
    //         base_id.push_back(b_node.id);
    //         // ++cnt;
    //     }
    //     // if (cnt > 100) {
    //     //     break;
    //     // }
    // }

    std::sort(base_pool.begin(), base_pool.end());
    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 0;
    result.push_back(base_pool[start].id);

    while (result.size() < M_pjbp && (++start) < base_pool.size()) {
        Neighbor &p = base_pool[start];
        bool occlude = false;
        // float dik =
        // distance_->compare(data_bp_ + dimension_ * p.id, data_sq_ + dimension_ * search_pool[0].id, dimension_);
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != tgt_base) {
                result.push_back(p.id);
            }
        }
    }

    start = 0;
    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        if (std::find(result.begin(), result.end(), p.id) != result.end()) {
            continue;
        }
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != tgt_base) {
                // if (std::find(src_nbrs.begin(), src_nbrs.end(), p.id) == src_nbrs.end()) {
                if (std::find(result.begin(), result.end(), p.id) == result.end()) {
                    result.push_back(p.id);
                }
                // result.push_back(p.id);
                // }
            }
        }
    }

    for (size_t i = 1; i < base_pool.size() && result.size() < M_pjbp; ++i) {
        if (std::find(result.begin(), result.end(), base_pool[i].id) == result.end()) {
            if (base_pool[i].id != tgt_base) {
                result.push_back(base_pool[i].id);
            }
        }
    }
    // for (size_t i = base_pool.size() - 1; i > 0 && result.size() < M_pjbp * 1.5; --i) {
    //     if (std::find(result.begin(), result.end(), base_pool[i].id) == result.end()) {
    //         if (base_pool[i].id != tgt_base) {
    //             result.push_back(base_pool[i].id);
    //         }
    //     }
    // }
    pruned_list = result;
}

uint32_t IndexBipartite::PruneProjectionBipartiteCandidates(std::vector<Neighbor> &search_pool, const float *query,
                                                            uint32_t qid, const Parameters &parameters,
                                                            std::vector<uint32_t> &pruned_list) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");

    uint32_t exp_candidate_len =
        search_pool.size() + bipartite_graph_[u32_nd_ + qid].size() + projection_graph_[qid].size();
    // search_pool.size() + projection_graph_[qid].size();
    std::sort(search_pool.begin(), search_pool.end());
    std::vector<uint32_t> candidate_only_id;
    candidate_only_id.reserve(exp_candidate_len);
    for (size_t i = 0; i < search_pool.size(); ++i) {
        candidate_only_id.push_back(search_pool[i].id);
    }

    // for (size_t i = 0; i < bipartite_graph_[u32_nd_ + qid].size(); ++i) {
    //     candidate_only_id.push_back(bipartite_graph_[u32_nd_ + qid][i]);
    // }

    // for (size_t i = 0; i < projection_graph_[qid].size(); ++i) {
    //     candidate_only_id.push_back(projection_graph_[qid][i]);
    // }

    uint32_t src_node = candidate_only_id[0];
    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 1;  // begin with 1
    result.push_back(candidate_only_id[start]);

    while (result.size() < M_pjbp && (++start) < candidate_only_id.size()) {
        auto &p = candidate_only_id[start];
        float dik = distance_->compare(data_bp_ + dimension_ * p, data_bp_ + dimension_ * src_node, dimension_);
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p, data_bp_ + dimension_ * result[t], dimension_);
            // float djk = l2_distance_->compare(data_bp_ + dimension_ * p, data_bp_ + dimension_ * result[t],
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = l2_distance_->compare(data_bp_ + dimension_ * p, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (djk < dik) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p != src_node) {
                result.push_back(p);
            }
        }
    }

    // begin with 1, loc 0 as src node.
    for (size_t i = 1; i < candidate_only_id.size() && result.size() < M_pjbp; ++i) {
        if (std::find(result.begin(), result.end(), candidate_only_id[i]) == result.end()) {
            if (candidate_only_id[i] != src_node) {
                result.push_back(candidate_only_id[i]);
            }
        }
    }

    pruned_list = result;
    return src_node;
}

uint32_t IndexBipartite::PruneProjectionCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                                   const Parameters &parameters, std::vector<uint32_t> &pruned_list) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t exp_candidate_len = search_pool.size() + bipartite_graph_[u32_nd_ + qid].size();
    std::vector<uint32_t> candidate_only_id;

    std::sort(search_pool.begin(), search_pool.end());
    for (size_t i = 0; i < search_pool.size(); ++i) {
        candidate_only_id.push_back(search_pool[i].id);
    }

    // for (size_t i = 0; i < bipartite_graph_[u32_nd_ + qid].size(); ++i) {
    //     candidate_only_id.push_back(bipartite_graph_[u32_nd_ + qid][i]);
    // }

    NeighborPriorityQueue pruned_pool(candidate_only_id.size());
    pruned_pool.reserve(candidate_only_id.size());

    for (size_t i = 0; i < search_pool.size(); ++i) {
        pruned_pool.insert(search_pool[i]);
    }

    for (size_t i = search_pool.size(); i < candidate_only_id.size(); ++i) {
        // float distance = distance_->compare(data_bp_ + dimension_ * candidate_only_id[i], query, dimension_);

        // don't need distance between sampled query and base here
        // float distance = 0.1;
        pruned_pool.insert({candidate_only_id[i], 0.1, false});
    }

    uint32_t src_node = pruned_pool[0].id;  // 1-nn as the src node
    std::vector<uint32_t> result;
    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 1;  // begin with 1
    result.push_back(pruned_pool[start].id);

    while (result.size() < M_pjbp && (++start) < pruned_pool.size()) {
        Neighbor &p = pruned_pool[start];
        bool occlude = false;
        float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node, dimension_);
        for (size_t t = 0; t < result.size(); ++t) {
            // if (p.id == result[t]) {
            //     occlude = true;
            //     break;
            // }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (djk < dik) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != src_node) {
                result.push_back(p.id);
            }
        }
    }

    // begin with 1, loc 0 as src node.
    for (size_t i = 1; i < candidate_only_id.size() && result.size() < M_pjbp; ++i) {
        if (std::find(result.begin(), result.end(), candidate_only_id[i]) == result.end()) {
            if (candidate_only_id[i] != src_node) {
                result.push_back(candidate_only_id[i]);
            }
        }
    }

    pruned_list = result;
    return src_node;
}

void IndexBipartite::PruneProjectionBaseSearchCandidates(std::vector<Neighbor> &search_pool, const float *query,
                                                         uint32_t qid, const Parameters &parameters,
                                                         std::vector<uint32_t> &pruned_list) {
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp") * 2;
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t degree = M_pjbp;
    std::vector<uint32_t> result;
    std::sort(search_pool.begin(), search_pool.end());
    // auto &pool = search_pool;

    // std::vector<float> occlude_factor;
    // // occlude_list can be called with the same scratch more than once by
    // // search_for_point_and_add_link through inter_insert.
    // occlude_factor.clear();
    // // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    // occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    // float cur_alpha = 1;
    // float alpha = 1.2;
    // // used for MIPS, where we store a value of eps in cur_alpha to
    // // denote pruned out entries which we can skip in later rounds.
    // while (cur_alpha <= alpha && result.size() < degree) {
    //     float eps = cur_alpha + 0.01f;

    //     for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
    //         if (occlude_factor[iter - pool.begin()] > cur_alpha) {
    //             continue;
    //         }
    //         // Set the entry to float::max so that is not considered again
    //         occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
    //         // Add the entry to the result if its not been deleted, and doesn't add a self loop

    //         if (iter->id != qid) {
    //             result.push_back(iter->id);
    //         }

    //         // Update occlude factor for points from iter+1 to pool.end()
    //         for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
    //             auto t = iter2 - pool.begin();
    //             if (occlude_factor[t] > alpha) continue;
    //             float djk = distance_->compare(data_bp_ + dimension_ * (size_t)iter2->id,
    //                                            data_bp_ + dimension_ * (size_t)iter->id, (unsigned)dimension_);
    //             occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
    //                        : std::max(occlude_factor[t], iter2->distance / djk);
    //             // if ((-djk) > cur_alpha * (-iter2->distance)) {
    //             //     occlude_factor[t] = std::max(occlude_factor[t], eps);
    //             // }
    //         }
    //     }
    //     cur_alpha *= 1.2;
    // }

    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 0;

    if (search_pool[start].id == qid) {
        start++;
    }
    auto &src_nbrs = projection_graph_[qid];
    while (std::find(src_nbrs.begin(), src_nbrs.end(), search_pool[start].id) != src_nbrs.end()) {
        ++start;
    }
    result.push_back(search_pool[start].id);
    // ++start;
    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != qid) {
                // if (std::find(src_nbrs.begin(), src_nbrs.end(), p.id) == src_nbrs.end()) {
                result.push_back(p.id);
                // }
            }
        }
    }
    start = 0;
    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != qid) {
                // if (std::find(src_nbrs.begin(), src_nbrs.end(), p.id) == src_nbrs.end()) {
                if (std::find(result.begin(), result.end(), p.id) == result.end()) {
                    result.push_back(p.id);
                }
                // result.push_back(p.id);
                // }
            }
        }
    }

    // for (size_t i = 0; i < search_pool.size() && result.size() < M_pjbp; ++i) {
    //     if (std::find(result.begin(), result.end(), search_pool[i].id) == result.end()) {
    //         if (std::find(src_nbrs.begin(), src_nbrs.end(), search_pool[i].id) == src_nbrs.end()) {
    //             result.push_back(search_pool[i].id);
    //         }
    //         // result.push_back(search_pool[i].id);
    //     }
    // }

    pruned_list = result;
}

void IndexBipartite::PruneProjectionBaseSearchCandidatesSupply(std::vector<Neighbor> &search_pool, const float *query,
                                                         uint32_t qid, const Parameters &parameters,
                                                         std::vector<uint32_t> &pruned_list) {
    // uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp") * 2;
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");
    uint32_t degree = M_pjbp;
    std::vector<uint32_t> result;
    std::sort(search_pool.begin(), search_pool.end());
    // auto &pool = search_pool;

    // std::vector<float> occlude_factor;
    // // occlude_list can be called with the same scratch more than once by
    // // search_for_point_and_add_link through inter_insert.
    // occlude_factor.clear();
    // // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    // occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    // float cur_alpha = 1;
    // float alpha = 1.2;
    // // used for MIPS, where we store a value of eps in cur_alpha to
    // // denote pruned out entries which we can skip in later rounds.
    // while (cur_alpha <= alpha && result.size() < degree) {
    //     float eps = cur_alpha + 0.01f;

    //     for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
    //         if (occlude_factor[iter - pool.begin()] > cur_alpha) {
    //             continue;
    //         }
    //         // Set the entry to float::max so that is not considered again
    //         occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
    //         // Add the entry to the result if its not been deleted, and doesn't add a self loop

    //         if (iter->id != qid) {
    //             result.push_back(iter->id);
    //         }

    //         // Update occlude factor for points from iter+1 to pool.end()
    //         for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
    //             auto t = iter2 - pool.begin();
    //             if (occlude_factor[t] > alpha) continue;
    //             float djk = distance_->compare(data_bp_ + dimension_ * (size_t)iter2->id,
    //                                            data_bp_ + dimension_ * (size_t)iter->id, (unsigned)dimension_);
    //             occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
    //                        : std::max(occlude_factor[t], iter2->distance / djk);
    //             // if ((-djk) > cur_alpha * (-iter2->distance)) {
    //             //     occlude_factor[t] = std::max(occlude_factor[t], eps);
    //             // }
    //         }
    //     }
    //     cur_alpha *= 1.2;
    // }

    result.reserve(M_pjbp * PROJECTION_SLACK);
    uint32_t start = 0;

    if (search_pool[start].id == qid) {
        start++;
    }
    // auto &src_nbrs = supply_nbrs_[qid];
    // while (std::find(src_nbrs.begin(), src_nbrs.end(), search_pool[start].id) != src_nbrs.end()) {
    //     ++start;
    // }
    result.push_back(search_pool[start].id);
    // ++start;
    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t]) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != qid) {
                // if (std::find(src_nbrs.begin(), src_nbrs.end(), p.id) == src_nbrs.end()) {
                result.push_back(p.id);
                // }
            }
        }
    }
    start = 0;
    while (result.size() < M_pjbp && (++start) < search_pool.size()) {
        Neighbor &p = search_pool[start];
        bool occlude = false;
        for (size_t t = 0; t < result.size(); ++t) {
            if (p.id == result[t] || (std::find(result.begin(), result.end(), p.id) != result.end())) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * result[t], dimension_);
            // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     djk = -djk;
            // }
            // float dik = distance_->compare(data_bp_ + dimension_ * p.id, data_bp_ + dimension_ * src_node,
            // dimension_); if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
            //     dik = -dik;
            // }
            if (1.0 * djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) {
            if (p.id != qid) {
                // if (std::find(src_nbrs.begin(), src_nbrs.end(), p.id) == src_nbrs.end()) {
                if (std::find(result.begin(), result.end(), p.id) == result.end()) {
                    result.push_back(p.id);
                }
                // result.push_back(p.id);
                // }
            }
        }
    }

    // for (size_t i = 0; i < search_pool.size() && result.size() < M_pjbp; ++i) {
    //     if (std::find(result.begin(), result.end(), search_pool[i].id) == result.end()) {
    //         if (std::find(src_nbrs.begin(), src_nbrs.end(), search_pool[i].id) == src_nbrs.end()) {
    //             result.push_back(search_pool[i].id);
    //         }
    //         // result.push_back(search_pool[i].id);
    //     }
    // }

    pruned_list = result;
}

void IndexBipartite::SearchProjectionbyQuery(const float *query, const Parameters &parameters,
                                             NeighborPriorityQueue &search_pool, boost::dynamic_bitset<> &visited,
                                             std::vector<Neighbor> &full_retset) {
    uint32_t M_pjbp = parameters.Get<uint32_t>("M_pjbp");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);

    std::vector<uint32_t> init_ids(search_pool.capacity() + 1);

    // boost::dynamic_bitset<> visited(u32_nd_);

    // todo : use bit set choose start. 2023.7.2: 19:30
    // forget the todo, what should I do...? nevermind...
    init_ids[0] = projection_ep_;
    // init_ids[0] = dis(gen);
    for (size_t i = 1; i < (init_ids.size() - 1) && (i - 1) < projection_graph_[projection_ep_].size(); ++i) {
        init_ids[i] = projection_graph_[projection_ep_][i - 1];
        // init_ids[i] = dis(gen);
        visited.set(init_ids[i]);
    }
    init_ids.push_back(dis(gen));
    while (init_ids.size() < search_pool.capacity()) {
        uint32_t rand_id = dis(gen);
        if (!visited.test(rand_id)) {
            init_ids.push_back(rand_id);
            visited.set(rand_id);
        }
    }

    for (size_t i = 0; i < init_ids.size(); ++i) {
        uint32_t id = init_ids[i];
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        Neighbor nn(id, distance, false);
        search_pool.insert(nn);
        full_retset.push_back(nn);
    }

    while (search_pool.has_unexpanded_node()) {
        auto cur_check = search_pool.closest_unexpanded();
        auto cur_id = cur_check.id;

        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j) {
            uint32_t id = projection_graph_[cur_id][j];
            if (!visited.test(id)) {
                float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
                // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
                //     distance = -distance;
                // }
                Neighbor nn(id, distance, false);
                search_pool.insert(nn);
                visited.set(id);
                full_retset.push_back(nn);
            }
        }
    }
}

void IndexBipartite::CalculateProjectionep() {
    float *center = new float[dimension_]();
    memset(center, 0, sizeof(float) * dimension_);
    // calculate centroid in base point
    for (size_t i = 0; i < nd_; ++i) {
        for (size_t d = 0; d < dimension_; ++d) {
            center[d] += data_bp_[i * dimension_ + d];
        }
    }

    for (size_t d = 0; d < dimension_; ++d) {
        center[d] /= (float)nd_;
    }

    float *distances = new float[nd_]();
    memset(distances, 0, sizeof(float) * nd_);
#pragma omp parallel for
    for (size_t i = 0; i < nd_; ++i) {
        const float *cur_data = data_bp_ + i * dimension_;
        float diff = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            diff += ((center[j] - cur_data[j]) * (center[j] - cur_data[j]));
        }
        distances[i] = diff;
        // distances[i] = distance_->compare(center, data_bp_ + i * dimension_, dimension_);
    }

    uint32_t closest = 0;
    for (size_t i = 1; i < nd_; ++i) {
        if (distances[i] < distances[closest]) {
            closest = static_cast<uint32_t>(i);
        }
    }
    projection_ep_ = closest;
    // projection_ep_ = min_idx;
    std::cout << "projection ep: " << projection_ep_ << std::endl;
    delete[] center;
    delete[] distances;
}

void IndexBipartite::Build(size_t n, const float *data, const Parameters &parameters){};

void IndexBipartite::Save(const char *filename) {
    // write graph
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    uint32_t npts = static_cast<uint32_t>(total_pts_);
    out.write((char *)&npts, sizeof(npts));
    for (uint32_t i = 0; i < total_pts_; i++) {
        uint32_t nbr_size = static_cast<uint32_t>(bipartite_graph_[i].size());
        out.write((char *)&nbr_size, sizeof(nbr_size));
        out.write((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    out.close();
}

void IndexBipartite::Load(const char *filename) {
    // load graph to bipartite_graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&npts, sizeof(npts));
    bipartite_graph_.resize(npts);
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        bipartite_graph_[i].resize(nbr_size);
        in.read((char *)bipartite_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    in.close();
}

void IndexBipartite::LoadNsgGraph(const char *filename) {
    // load graph to projection graph
    std::ifstream in(filename, std::ios::binary);
    uint32_t width = 0;
    in.read((char *)&width, sizeof(width));
    uint32_t npts = 1000000;
    in.read((char *)&projection_ep_, sizeof(uint32_t));
    std::cout << "Projection graph, "
              << "ep: " << projection_ep_ << std::endl;
    // in.read((char *)&npts, sizeof(npts));
    projection_graph_.resize(npts);
    float out_degree = 0.0;
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        out_degree += static_cast<float>(nbr_size);
        projection_graph_[i].resize(nbr_size);
        in.read((char *)projection_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    std::cout << "Projection graph, "
              << "avg_degree: " << out_degree / npts << std::endl;
    in.close();
}

void IndexBipartite::LoadProjectionGraph(const char *filename) {
    // load graph to projection graph
    std::cout << "loading graph from " << filename << std::endl;
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    in.read((char *)&projection_ep_, sizeof(uint32_t));
    std::cout << "Projection graph, "
              << "ep: " << projection_ep_ << std::endl;
    in.read((char *)&npts, sizeof(npts));
    projection_graph_.resize(npts);
    float out_degree = 0.0;
    for (uint32_t i = 0; i < npts; i++) {
        uint32_t nbr_size;
        in.read((char *)&nbr_size, sizeof(nbr_size));
        out_degree += static_cast<float>(nbr_size);
        projection_graph_[i].resize(nbr_size);
        in.read((char *)projection_graph_[i].data(), nbr_size * sizeof(uint32_t));
    }
    std::cout << "Projection graph, "
              << "avg_degree: " << out_degree / npts << std::endl;
    in.close();
}
// void IndexBipartite::LoadProjectionGraph(const char *filename) {
//     // load graph to projection graph
//     std::ifstream in(filename, std::ios::binary);
//     uint32_t npts = 1000000;
//     in.seekg(8 + 4, std::ios::beg);
//     in.read((char *)&projection_ep_, sizeof(uint32_t));
//     std::cout << "Projection graph, " << std::string(filename) << " ep: " << projection_ep_ << std::endl;
//     // in.read((char *)&npts, sizeof(npts));
//     in.seekg(8, std::ios::cur);
//     projection_graph_.resize(npts);
//     size_t out_degree = 0.0;
//     for (uint32_t i = 0; i < npts; i++) {
//         uint32_t nbr_size;
//         in.read((char *)&nbr_size, sizeof(nbr_size));
//         out_degree += nbr_size;
//         projection_graph_[i].resize(nbr_size);
//         in.read((char *)projection_graph_[i].data(), nbr_size * sizeof(uint32_t));
//     }
//     std::cout << "Projection graph, "
//               << "avg_degree: " << out_degree / (float)npts << " total edges: " << out_degree << std::endl;
//     in.close();
// }

uint32_t IndexBipartite::SearchBipartiteGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                              unsigned *indices) {
    // uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    search_queue.reserve(L_pq);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    uint32_t start = dis(gen);  // start is base
    std::vector<uint32_t> init_ids;
    init_ids.push_back(start);
    // block_metric.reset();
    boost::dynamic_bitset<> visited{total_pts_, 0};
    // std::bitset<bsize> visited;
    // block_metric.record();
    for (auto &id : init_ids) {
        float distance;
        // dist_cmp_metric.reset();
        distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        visited.set(id);
        // memory_access_metric.record();
    }

    uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);

        // memory_access_metric.record();

        uint32_t first_hop_rank_1 = bipartite_graph_[cur_id][0];
        float first_hop_min_dist = 1000;
        // get neighbors' neighbors, first
        for (auto nbr : bipartite_graph_[cur_id]) {  // current check node's neighbors

            for (auto ns_nbr : bipartite_graph_[nbr]) {  // neighbors' neighbors
                // memory_access_metric.reset();
                if (visited.test(ns_nbr)) {
                    continue;
                }
                visited.set(ns_nbr);
                // memory_access_metric.record();
                float distance;
                // dist_cmp_metric.reset();
                distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }
                if (distance < first_hop_min_dist) {
                    // std::cout << "update" << std::endl;
                    first_hop_min_dist = distance;
                    first_hop_rank_1 = nbr;
                }
                // dist_cmp_metric.record();
                // memory_access_metric.reset();
                ++cmps;
                Neighbor nn = Neighbor(ns_nbr, distance, false);
                search_queue.insert(nn);
                // memory_access_metric.record();
                break;  // break
            }
        }

        for (auto &ns_nbr : bipartite_graph_[first_hop_rank_1]) {
            if (visited.test(ns_nbr)) {
                continue;
            }
            visited.set(ns_nbr);
            // memory_access_metric.record();
            float distance;
            // dist_cmp_metric.reset();
            distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
            // if (metric_ == mysteryann::INNER_PRODUCT) {
            //     distance = -distance;
            // }
            // dist_cmp_metric.record();
            // memory_access_metric.reset();
            ++cmps;
            Neighbor nn = Neighbor(ns_nbr, distance, false);
            search_queue.insert(nn);
            // memory_access_metric.record();
        }
    }

    if (search_queue.size() < k) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }
    // std::cout << "cmps: " << cmps << std::endl;
    for (size_t i = 0; i < k; ++i) {
        indices[qid * k + i] = search_queue[i].id;
    }
    return cmps;
}
void IndexBipartite::Search(const float *query, const float *x, size_t k, const Parameters &parameters,
                            unsigned *indices, float* res_dists) {
    // uint32_t M_sq = parameters.Get<uint32_t>("M_sq");
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    search_queue.reserve(L_pq);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    uint32_t start = dis(gen);  // start is base
    std::vector<uint32_t> init_ids;
    init_ids.push_back(start);
    // block_metric.reset();
    boost::dynamic_bitset<> visited{total_pts_, 0};
    // std::bitset<bsize> visited;
    // block_metric.record();
    for (auto &id : init_ids) {
        float distance;
        // dist_cmp_metric.reset();
        distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        visited.set(id);
        // memory_access_metric.record();
    }

    uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);

        // memory_access_metric.record();

        uint32_t first_hop_rank_1 = bipartite_graph_[cur_id][0];
        float first_hop_min_dist = 1000;
        // get neighbors' neighbors, first
        for (auto &nbr : bipartite_graph_[cur_id]) {  // current check node's neighbors

            for (auto &ns_nbr : bipartite_graph_[nbr]) {  // neighbors' neighbors
                // memory_access_metric.reset();
                if (visited.test(ns_nbr)) {
                    continue;
                }
                visited.set(ns_nbr);
                // memory_access_metric.record();
                float distance;
                // dist_cmp_metric.reset();
                distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
                // if (metric_ == mysteryann::INNER_PRODUCT) {
                //     distance = -distance;
                // }
                if (first_hop_min_dist > distance) {
                    // std::cout << "update" << std::endl;
                    first_hop_min_dist = distance;
                    first_hop_rank_1 = nbr;
                }
                // dist_cmp_metric.record();
                // memory_access_metric.reset();
                ++cmps;
                Neighbor nn = Neighbor(ns_nbr, distance, false);
                search_queue.insert(nn);
                // memory_access_metric.record();
                // break;  // break
            }
        }

        // for (auto &ns_nbr : bipartite_graph_[first_hop_rank_1]) {
        //     if (visited.test(ns_nbr)) {
        //         continue;
        //     }
        //     visited.set(ns_nbr);
        //     memory_access_metric.record();
        //     float distance;
        //     dist_cmp_metric.reset();
        //     distance = distance_->compare(data_bp_ + ns_nbr * dimension_, query, (unsigned)dimension_);
        //     if (metric_ == mysteryann::INNER_PRODUCT) {
        //         distance = -distance;
        //     }
        //     dist_cmp_metric.record();
        //     memory_access_metric.reset();
        //     ++cmps;
        //     Neighbor nn = Neighbor(ns_nbr, distance, false);
        //     search_queue.insert(nn);
        //     memory_access_metric.record();
        // }
    }

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }
    // std::cout << "cmps: " << cmps << std::endl;
    for (size_t i = 0; i < k; ++i) {
        indices[i] = search_queue[i].id;
    }
}

void IndexBipartite::TrainQuantizer(const float *data_quant, int n_quant, int dim_qunt)
{
    std::cout << "TrainQuantizer 维度:" << dim_qunt << std::endl;
    quant = new glass::SQ8Quantizer<glass::Metric::IP>(dim_qunt);
    quant->train(data_quant, n_quant);
    quant->save_code(quant_file_path);
    if (metric_ == mysteryann::L2) {
        printf("for small L2 dataet, no learn no optimize");
        this->po = 4;
        this->pl = 5;
        FILE *F = fopen(prefetch_file_path.c_str(), "wb");
        fwrite(&this->po, sizeof(int), 1, F);
        fwrite(&this->pl, sizeof(int), 1, F);
        fclose(F);
        delete quant;
        quant = nullptr;
        return;
    }
    OptimizePrefetch(125, 8);
    delete quant;
    quant = nullptr;
}

int  IndexBipartite::read_fvecs(std::string path_file, int &N, int &Dim, std::vector<float> &optimize_queries, int kOptimizePoints)
{
    FILE *F;
    F = fopen(path_file.c_str(), "rb");
    if (F == NULL)
    {
        printf("Dataset not found\n");
        exit(0);
    }
    int xxx = fread(&N, sizeof(int), 1, F);
    xxx = fread(&Dim, sizeof(int), 1, F);
    int sample_points_num = std::min(kOptimizePoints, N);

    optimize_queries.resize(sample_points_num * Dim);
    for (int i = 0; i < sample_points_num; i++)
    {
        float *vf = new float[Dim];
        xxx = fread(vf, sizeof(float), Dim, F);
        memcpy(optimize_queries.data() + i * Dim, vf, Dim * sizeof(float)); // 用完记得释放 optimize_queries 空间
        delete[] vf;
    }
    fclose(F);
    return sample_points_num;
}

inline constexpr size_t upper_div(size_t x, size_t y)
{
    return (x + y - 1) / y;
}

void IndexBipartite::LoadPrefetch(){
    FILE * F = nullptr;
    std::cout << "Loading... Prefetch: po = " << po << ", pl = " << pl << std::endl;
	F = fopen(prefetch_file_path.c_str(), "rb");
	if(F != nullptr){
        fread(&po, sizeof(int), 1, F);
        fread(&pl, sizeof(int), 1, F);
        fclose(F);
	}
    std::cout << "Loaded Prefetch: po = " << po << ", pl = " << pl << std::endl;
}

void IndexBipartite::OptimizePrefetch(uint32_t L_pq = 125, int num_threads = 8)
{
    if(visited_list_pool_ == nullptr) {
        printf("visited_list_pool_没有初始化, Optimize 未生效,请在 setThreads 后调用 Optimize\n");
        return;
    }

    printf("=============Start optimization=============\n");
    // Optimization parameters
    constexpr static int kOptimizePoints = 2000;
    constexpr static int kTryPos = 10;
    constexpr static int kTryPls = 5;
    constexpr static int kTryK = 10;

    if (num_threads == 0)
    {
        num_threads = std::thread::hardware_concurrency();
    }

    // 求最大出度
    int graph_K = 0;
    // for (size_t i = 0; i < projection_graph_.size(); ++i)
    // {
    //     if (graph_K < projection_graph_[i].size())
    //     {
    //         graph_K = projection_graph_[i].size();
    //         }
    // }

    FindMaxOutDegree(row_ptr_, col_idx_, graph_K);

    std::cout << "最大出度：" << graph_K << std::endl;

    // 读 query 集
    std::vector<float> optimize_queries;
    int N = 0, Dim = 0, sample_points_num = 0;
    sample_points_num = read_fvecs(query_file_path, N, Dim, optimize_queries, kOptimizePoints);
    std::cout << "N:D = " << N << ":" << Dim << std::endl;
    std::cout << "sample_points_num =" << sample_points_num << std::endl;
    std::cout << "L_pq =" << L_pq << std::endl;

    std::vector<int> try_pos(std::min(kTryPos, graph_K));
    std::vector<int> try_pls(std::min(kTryPls, (int)upper_div(quant->code_size, 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);

    // warmup
    {                          
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t i = 0; i < sample_points_num; ++i)
        {
            size_t qid = i;
            // SearchGraph_SQ(optimize_queries.data() + i * Dim, kTryK, qid, L_pq, nullptr);
            SearchReorderGraph4Tune(optimize_queries.data() + i * Dim, kTryK, qid, L_pq, nullptr);
        }
    }
    printf("=============Done warmup=============\n");
    float min_ela = std::numeric_limits<float>::max();
    int best_po = 0, best_pl = 0;
    for (auto try_po : try_pos)
    {
        for (auto try_pl : try_pls)
        {
            this->po = try_po;
            this->pl = try_pl;
            auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
            for (size_t i = 0; i < sample_points_num; ++i)
            {
                 size_t qid = i;
                // SearchGraph_SQ(optimize_queries.data() + i * Dim, kTryK, qid, L_pq, nullptr);
                SearchReorderGraph4Tune(optimize_queries.data() + i * Dim, kTryK, qid, L_pq, nullptr);
            }

            auto ed = std::chrono::high_resolution_clock::now();
            auto ela = std::chrono::duration<double>(ed - st).count();
            if (ela < min_ela)
            {
                min_ela = ela;
                best_po = try_po;
                best_pl = try_pl;
            }
        }
    }
    this->po = 1;
    this->pl = 1;
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < sample_points_num; ++i)
    {
        size_t qid = i;
        SearchReorderGraph4Tune(optimize_queries.data() + i * Dim, kTryK, qid, L_pq, nullptr);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    float baseline_ela = std::chrono::duration<double>(ed - st).count();
    printf("settint best po = %d, best pl = %d\n"
           "gaining %.2f%% performance improvement\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
    this->po = best_po;
    this->pl = best_pl;

    // 保存参数
    FILE *F = fopen(prefetch_file_path.c_str(), "wb");
    fwrite(&this->po, sizeof(int), 1, F);
    fwrite(&this->pl, sizeof(int), 1, F);
    fclose(F);
}

void IndexBipartite::SearchGraph_SQ(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices = nullptr)
{
    auto computer = quant->get_computer(query);// 

    NeighborPriorityQueue search_queue(L_pq);
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);

    // visited_list_pool_ 必须在 setThread 后使用
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids)
    {
        float distance = computer(id); // 量化距离计算
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
    }
    while (search_queue.has_unexpanded_node())
    {
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;

        uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        // 预取
        for (int j = 0; j < po; ++j) {
            int to = (int)*(cur_nbrs + j);
            computer.prefetch(to, pl);
        }
        
        size_t size = projection_graph_[cur_id].size();//
        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j)
        {   
            // current check node's neighbors
            uint32_t nbr = *(cur_nbrs + j);
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            // 预取
            if (j + po < size) {
                int to = (int)*(cur_nbrs + j + po);
                computer.prefetch(to, pl);
            }
            
            if (visited_array[nbr] != visited_array_tag)
            {
                visited_array[nbr] = visited_array_tag;
                float distance = computer(nbr); // 量化距离计算
                search_queue.insert({nbr, distance, false});
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k))
    {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    if(indices != nullptr)
    {
        for (size_t i = 0; i < k; ++i)
        {
            indices[i] = search_queue[i].id;
        }
    }
}

void IndexBipartite::SearchGraph(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices) {
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);
    prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    // uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // get neighbors' neighbors, first
        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j) {  // current check node's neighbors
            uint32_t nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            // }
            // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), _MM_HINT_T0);
            if (visited_array[nbr] != visited_array_tag) {
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                visited_array[nbr] = visited_array_tag;
                // memory_access_metric.record();
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, (unsigned)search_dim_);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                // ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        // indices[qid * k + i] = search_queue[i].id;
        indices[i] = search_queue[i].id;
        // res_dists[i] = search_queue[i].distance;
    }
    // return cmps;
}

void IndexBipartite::SearchReorderGraph4Tune(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices = nullptr) {
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    auto computer = quant->get_computer(query);// 
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(new_order_[projection_ep_]);
    prefetch_vector((char *)(data_bp_ + new_order_[projection_ep_] * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        // float distance = distance_->compare(data_bp_ + id * dimension_, query, search_dim_);
        float distance = computer(id);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    // uint32_t cmps = 0;
    uint32_t *cur_nbrs = nullptr;
    uint32_t neighbor_size = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        // uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        csr_get_neighbors(cur_id, cur_nbrs, neighbor_size);
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // 预取
        for (int j = 0; j < po; ++j) {
            int to = (int)*(cur_nbrs + j);
            computer.prefetch(to, pl);
        }

        // get neighbors' neighbors, first
        for (size_t j = 0; j < neighbor_size; ++j) {  // current check node's neighbors
            uint32_t& nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            // }
            // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), _MM_HINT_T0);
            if (j + po < neighbor_size) {
                int to = (int)*(cur_nbrs + j + po);
                computer.prefetch(to, pl);
            }
            if (visited_array[nbr] != visited_array_tag) {
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                visited_array[nbr] = visited_array_tag;
                // memory_access_metric.record();
                float distance = computer(nbr); // 量化距离计算
                // float distance = distance_->compare(data_bp_ + nbr * dimension_, query, search_dim_);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                // ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    if (indices) {
        for (size_t i = 0; i < k; ++i) {
            // indices[qid * k + i] = search_queue[i].id;
            indices[i] = Porigin_[search_queue[i].id];
            // res_dists[i] = search_queue[i].distance;
        }
    }
    // return cmps;
}

void IndexBipartite::SearchReorderGraph(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices) {
    // uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(new_order_[projection_ep_]);
    prefetch_vector((char *)(data_bp_ + new_order_[projection_ep_] * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        float distance = distance_->compare(data_bp_ + id * dimension_, query, search_dim_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    // uint32_t cmps = 0;
    uint32_t *cur_nbrs = nullptr;
    uint32_t neighbor_size = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        // uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        csr_get_neighbors(cur_id, cur_nbrs, neighbor_size);
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // get neighbors' neighbors, first
        for (size_t j = 0; j < neighbor_size; ++j) {  // current check node's neighbors
            uint32_t& nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
            // }
            // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), _MM_HINT_T0);
            if (visited_array[nbr] != visited_array_tag) {
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                visited_array[nbr] = visited_array_tag;
                // memory_access_metric.record();
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, search_dim_);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                // ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        // indices[qid * k + i] = search_queue[i].id;
        indices[i] = Porigin_[search_queue[i].id];
        // res_dists[i] = search_queue[i].distance;
    }
    // return cmps;
}

uint32_t IndexBipartite::SearchProjectionGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                               unsigned *indices, std::vector<float>& res_dists) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);
    prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    uint32_t here_dim = (uint32_t)dimension_;
    uint32_t cmps = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // get neighbors' neighbors, first
        for (size_t j = 0; j < projection_graph_[cur_id].size(); ++j) {  // current check node's neighbors
            // _emm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            uint32_t& nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);

            if (visited_array[nbr] != visited_array_tag) {
            // if (visited[nbr] == 0) {
                // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                // prefetch_vector((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), dimension_);
                visited_array[nbr] = visited_array_tag;
                // visited[nbr] = 1;
                // memory_access_metric.record();
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, here_dim);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        // indices[qid * k + i] = search_queue[i].id;
        indices[i] = search_queue[i].id;
        res_dists[i] = search_queue[i].distance;
    }
    return cmps;
}

uint32_t IndexBipartite::SearchProjectionCSRwithOrder(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                               unsigned *indices, std::vector<float>& res_dists) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(new_order_[projection_ep_]);
    // prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    uint32_t here_dim = (uint32_t)dimension_;
    uint32_t cmps = 0;
    uint32_t *cur_nbrs = nullptr;
    uint32_t neighbor_size = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        // uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        csr_get_neighbors(cur_id, cur_nbrs, neighbor_size);
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // get neighbors' neighbors, first
        for (size_t j = 0; j < neighbor_size; ++j) {  // current check node's neighbors
            // _emm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            uint32_t& nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);

            if (visited_array[nbr] != visited_array_tag) {
            // if (visited[nbr] == 0) {
                // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                // prefetch_vector((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), dimension_);
                visited_array[nbr] = visited_array_tag;
                // visited[nbr] = 1;
                // memory_access_metric.record();
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, here_dim);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        // indices[qid * k + i] = search_queue[i].id;
        indices[i] = Porigin_[search_queue[i].id];
        res_dists[i] = search_queue[i].distance;
    }
    return cmps;
}

uint32_t IndexBipartite::SearchProjectionCSR(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                               unsigned *indices, std::vector<float>& res_dists) {
    uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
    NeighborPriorityQueue search_queue(L_pq);
    // search_queue.reserve(L_pq);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> dis(0, u32_nd_ - 1);
    // uint32_t start = dis(gen);  // start is base
    // uint32_t start = projection_ep_;
    // projection_ep_ = start;
    std::vector<uint32_t> init_ids;
    init_ids.push_back(projection_ep_);
    // prefetch_vector((char *)(data_bp_ + projection_ep_ * dimension_), dimension_);
    // init_ids.push_back(projection_ep_);
    // _mm_prefetch((char *)data_bp_ + projection_ep_ * dimension_, _MM_HINT_T0);
    // init_ids.push_back(dis(gen));
    // block_metric.reset();
    // boost::dynamic_bitset<> visited{u32_nd_, 0};
    // tsl::robin_set<uint32_t> visited(5000);
    // std::bitset<bsize> visited;
    // block_metric.record();
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    for (auto &id : init_ids) {

        // dist_cmp_metric.reset();
        float distance = distance_->compare(data_bp_ + id * dimension_, query, (unsigned)dimension_);
        // if (metric_ == mysteryann::Metric::INNER_PRODUCT) {
        //     distance = -distance;
        // }
        // dist_cmp_metric.record();

        // memory_access_metric.reset();
        Neighbor nn = Neighbor(id, distance, false);
        search_queue.insert(nn);
        // visited_array[id] = visited_array_tag;
        // visited.set(id);
        // visited.insert(id);
        // memory_access_metric.record();
    }
    uint32_t here_dim = (uint32_t)dimension_;
    uint32_t cmps = 0;
    uint32_t *cur_nbrs = nullptr;
    uint32_t neighbor_size = 0;
    while (search_queue.has_unexpanded_node()) {
        // memory_access_metric.reset();
        auto cur_check_node = search_queue.closest_unexpanded();
        auto cur_id = cur_check_node.id;
        // visited.set(cur_id);
        // uint32_t *cur_nbrs = projection_graph_[cur_id].data();
        csr_get_neighbors(cur_id, cur_nbrs, neighbor_size);
        // memory_access_metric.record();
        // _mm_prefetch((char *)(visited_array + *(cur_nbrs + 1)), _MM_HINT_T0);
        // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs) * dimension_), _MM_HINT_T0);

        // get neighbors' neighbors, first
        for (size_t j = 0; j < neighbor_size; ++j) {  // current check node's neighbors
            // _emm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            uint32_t& nbr = *(cur_nbrs + j);
            // memory_access_metric.reset();
            // if (visited.find(nbr) != visited.end()) {
            // _mm_prefetch((char *)(visited_array + *(cur_nbrs + j)), _MM_HINT_T0);
            // if (j + 1 <= projection_graph_[cur_id].size()) {
            _mm_prefetch((char *)(visited_array + *(cur_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);

            if (visited_array[nbr] != visited_array_tag) {
            // if (visited[nbr] == 0) {
                // _mm_prefetch((char *)(data_bp_ + *(cur_nbrs + j + 1) * dimension_), _MM_HINT_T0);
                // if (visited.test(nbr)) {
                //     continue;
                // }
                // prefetch_vector((char *)(data_bp_ + nbr * dimension_), dimension_);
                // visited.insert(nbr);
                // visited.set(nbr);
                // prefetch_vector((char *)(data_bp_ + *(cur_nbrs + j) * dimension_), dimension_);
                visited_array[nbr] = visited_array_tag;
                // visited[nbr] = 1;
                // memory_access_metric.record();
                float distance = distance_->compare(data_bp_ + nbr * dimension_, query, here_dim);
                // _mm_prefetch((char *) data_bp_ + )
                // dist_cmp_metric.reset();
                // if (likely(metric_ == mysteryann::INNER_PRODUCT)) {
                //     distance = -distance;
                // }

                // dist_cmp_metric.record();
                // memory_access_metric.reset();

                ++cmps;
                search_queue.insert({nbr, distance, false});
                // if(search_queue.insert({nbr, distance, false})) {
                //     _mm_prefetch((char *)projection_graph_[nbr].data(), _MM_HINT_T2);
                // }
                // memory_access_metric.record();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (unlikely(search_queue.size() < k)) {
        std::stringstream ss;
        ss << "not enough results: " << search_queue.size() << ", expected: " << k;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < k; ++i) {
        // indices[qid * k + i] = search_queue[i].id;
        indices[i] = search_queue[i].id;
        res_dists[i] = search_queue[i].distance;
    }
    return cmps;
}

// uint32_t IndexBipartite::SearchProjectionGraph(const float *query, size_t k, size_t &qid, const Parameters
// &parameters,
//                                                unsigned *indices) {
//     uint32_t L_pq = parameters.Get<uint32_t>("L_pq");
//     NeighborPriorityQueue search_queue(L_pq);
//     search_queue.reserve(L_pq);
//     tsl::robin_set<unsigned> inserted_into_pool_rs;
//     boost::dynamic_bitset<> inserted_into_pool_bs;
//     std::vector<unsigned> id_scratch;
//     std::vector<float> dist_scratch;
//     bool fast_iterate = true;

//     if (fast_iterate) {
//         inserted_into_pool_bs.resize(1000000);
//     }
//     auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const unsigned id) {
//         return fast_iterate ? inserted_into_pool_bs[id] == 0
//                             : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
//     };

//     std::vector<uint32_t> init_ids;
//     init_ids.push_back(projection_ep_);
//     for (auto id : init_ids) {
//         if (is_not_visited(id)) {
//             if (fast_iterate) {
//                 inserted_into_pool_bs[id] = 1;
//             } else {
//                 inserted_into_pool_rs.insert(id);
//             }

//             float distance;

//             distance = distance_->compare(data_bp_ + dimension_ * (size_t)id, query, (unsigned)dimension_);
//             Neighbor nn = Neighbor(id, distance, false);
//             search_queue.insert(nn);
//         }
//     }
//     uint32_t hops = 0;
//     uint32_t cmps = 0;

//     while (search_queue.has_unexpanded_node()) {
//         auto nbr = search_queue.closest_unexpanded();
//         auto n = nbr.id;
//         // Add node to expanded nodes to create pool for prune later
//         // Find which of the nodes in des have not been visited before
//         id_scratch.clear();
//         dist_scratch.clear();
//         {
//             for (auto id : projection_graph_[n]) {
//                 if (is_not_visited(id)) {
//                     id_scratch.push_back(id);
//                 }
//             }
//         }

//         // Mark nodes visited
//         for (auto id : id_scratch) {
//             if (fast_iterate) {
//                 inserted_into_pool_bs[id] = 1;
//             } else {
//                 inserted_into_pool_rs.insert(id);
//             }
//         }

//         assert(dist_scratch.size() == 0);
//         for (size_t m = 0; m < id_scratch.size(); ++m) {
//             unsigned id = id_scratch[m];

//             if (m + 1 < id_scratch.size()) {
//                 auto nextn = id_scratch[m + 1];
//                 // prefetch_vector(
//                 //     (const char *) _data + _aligned_dim * (size_t) nextn,
//                 //     sizeof(T) * _aligned_dim);
//             }

//             dist_scratch.push_back(distance_->compare(query, data_bp_ + dimension_ * (size_t)id,
//             (unsigned)dimension_));
//         }

//         cmps += id_scratch.size();

//         // Insert <id, dist> pairs into the pool of candidates
//         for (size_t m = 0; m < id_scratch.size(); ++m) {
//             search_queue.insert(Neighbor(id_scratch[m], dist_scratch[m], false));
//         }
//     }

//     if (search_queue.size() < k) {
//         std::stringstream ss;
//         ss << "not enough results: " << search_queue.size() << ", expected: " << k;
//         throw std::runtime_error(ss.str());
//     }

//     for (size_t i = 0; i < k; ++i) {
//         indices[qid * k + i] = search_queue[i].id;
//     }
//     return cmps;
// }

void IndexBipartite::findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameters) {
    unsigned id = nd_;
    for (unsigned i = 0; i < nd_; i++) {
        if (flag[i] == false) {
            id = i;
            break;
        }
    }

    if (id == nd_) return;  // No Unlinked Node

    std::vector<Neighbor> tmp, pool;
    NeighborPriorityQueue temp_queue;
    SearchProjectionGraphInternal(temp_queue, data_bp_ + dimension_ * id, id, parameters, flag, pool);
    // get_neighbors(data_ + dimension_ * id, parameters, tmp, pool);
    std::sort(pool.begin(), pool.end());

    unsigned found = 0;
    for (unsigned i = 0; i < pool.size(); i++) {
        if (flag[pool[i].id]) {
            // std::cout << pool[i].id << '\n';
            root = pool[i].id;
            found = 1;
            break;
        }
    }
    if (found == 0) {
        while (true) {
            unsigned rid = rand() % nd_;
            if (flag[rid]) {
                root = rid;
                break;
            }
        }
    }
    projection_graph_[root].push_back(id);
}

void IndexBipartite::dfs(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
    unsigned tmp = root;
    std::stack<unsigned> s;
    s.push(root);
    if (!flag[root]) cnt++;
    flag[root] = true;
    while (!s.empty()) {
        unsigned next = nd_ + 1;
        for (unsigned i = 0; i < projection_graph_[tmp].size(); i++) {
            if (flag.test(projection_graph_[tmp][i]) == false) {
                next = projection_graph_[tmp][i];
                break;
            }
        }
        // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
        if (next == (nd_ + 1)) {
            s.pop();
            if (s.empty()) break;
            tmp = s.top();
            continue;
        }
        tmp = next;
        flag[tmp] = true;
        s.push(tmp);
        cnt++;
    }
}

void IndexBipartite::CollectPoints(const Parameters &parameters) {
    unsigned root = projection_ep_;
    boost::dynamic_bitset<> flags{nd_, 0};
    unsigned unlinked_cnt = 0;
    while (unlinked_cnt < nd_) {
        dfs(flags, root, unlinked_cnt);
        // std::cout << unlinked_cnt << '\n';
        if (unlinked_cnt >= nd_) break;
        findroot(flags, root, parameters);
        // std::cout << "new root"
        //           << ":" << root << '\n';
    }
    // for (size_t i = 0; i < nd_; ++i) {
    //     if (final_graph_[i].size() > width_) {
    //         width_ = final_graph_[i].size();
    //     }
    // }
}

void IndexBipartite::SaveProjectionGraph(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    out.write((char *)&projection_ep_, sizeof(uint32_t));
    out.write((char *)&u32_nd_, sizeof(uint32_t));
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        uint32_t nbr_size = projection_graph_[i].size();
        out.write((char *)&nbr_size, sizeof(uint32_t));
        out.write((char *)projection_graph_[i].data(), sizeof(uint32_t) * nbr_size);
    }
    out.close();
    // free projection_graph_
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        projection_graph_[i].clear();
        projection_graph_.shrink_to_fit();
    }
    projection_graph_.clear();
    projection_graph_.shrink_to_fit();
}

void IndexBipartite::SaveReorder(std::string& filename) {
    std::string order_file = filename + ".order";
    std::string original_order_file = filename + ".original_order";
    std::ofstream out(order_file, std::ios::out);
    std::ofstream out_original(original_order_file, std::ios::out);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    if (!out_original.is_open()) {
        throw std::runtime_error("cannot open file");
    }
    for (uint32_t i = 0; i < u32_nd_; ++i) {
        out << new_order_[i] << std::endl;
        out_original << Porigin_[i] << std::endl;
    }
    out.close();
    out_original.close();
}

void IndexBipartite::LoadReorder(std::string& order_file, std::string& original_order_file) {
    std::cout << "order file: " << order_file << std::endl;
    std::cout << "original order file" << original_order_file << std::endl;
    std::ifstream reorder_in;
    reorder_in.open(order_file, std::ios::in);
    if (!reorder_in.is_open()) {
        throw std::runtime_error("order file open failed");
    }
    new_order_.clear();
    uint32_t temp;
    while (reorder_in >> temp) {
        new_order_.push_back(temp);
    }
    reorder_in.close();
    std::cout << "load new_order_ size: " << new_order_.size() << std::endl;
    std::ifstream reorder_in_origin;
    reorder_in_origin.open(original_order_file, std::ios::in);
    if (!reorder_in_origin.is_open()) {
        throw std::runtime_error("order original file open failed");
    }
    Porigin_.clear();
    while (reorder_in_origin >> temp) {
        Porigin_.push_back(temp);
    }
    reorder_in_origin.close();
    std::cout << "load Porigin_ size: " << Porigin_.size() << std::endl;
}

// gt file: base in query
void IndexBipartite::LoadLearnBaseKNN(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    uint32_t k_dim;
    in.read((char *)&npts, sizeof(npts));
    in.read((char *)&(k_dim), sizeof(k_dim));
    std::cout << "learn base knn npts: " << npts << ", k_dim: " << k_dim << std::endl;

    learn_base_knn_.resize(npts);
    for (uint32_t i = 0; i < npts; i++) {
        learn_base_knn_[i].resize(k_dim);
        in.read((char *)learn_base_knn_[i].data(), sizeof(uint32_t) * k_dim);
    }
    if (learn_base_knn_.back().size() != k_dim) {
        throw std::runtime_error("learn base knn file error");
    }
    in.close();
}

void IndexBipartite::LoadBaseLearnKNN(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    uint32_t npts;
    uint32_t k_dim;
    in.read((char *)&npts, sizeof(npts));
    in.read((char *)&(k_dim), sizeof(k_dim));
    std::cout << "base learn knn npts: " << npts << ", k_dim: " << k_dim << std::endl;

    base_learn_knn_.resize(npts);
    for (uint32_t i = 0; i < npts; i++) {
        base_learn_knn_[i].resize(k_dim);
        in.read((char *)base_learn_knn_[i].data(), sizeof(uint32_t) * k_dim);
    }
    if (base_learn_knn_.back().size() != k_dim) {
        throw std::runtime_error("base learn knn file error");
    }
    in.close();
}

// todo, if use projection search, only need base point data
void IndexBipartite::LoadVectorData(const char *base_file, const char *sampled_query_file) {
    uint32_t base_num, sq_num, base_dim, q_dim;

    load_meta<float>(base_file, base_num, base_dim);
    load_meta<float>(sampled_query_file, sq_num, q_dim);
    if (base_dim != q_dim) {
        throw std::runtime_error("base and query dimension mismatch");
    }
    float *base_data = nullptr;
    float *sampled_query_data = nullptr;
    load_data_search<float>(base_file, base_num, base_dim, base_data);
    // load_data<float>(sampled_query_file, sq_num, q_dim, sampled_query_data);

    if (need_normalize) {
        std::cout << "Normalizing base data" << std::endl;
        for (size_t i = 0; i < base_num; ++i) {
            normalize<float>(base_data + i * (uint64_t)base_dim, (uint64_t)base_dim);
        }
    }

    data_bp_ = base_data;
    // data_bp_ = data_align(base_data, base_num, base_dim);
    // data_sq_ = data_align(sampled_query_data, sq_num, q_dim);

    nd_ = base_num;
    nd_sq_ = sq_num;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
}

// todo, if use projection search, only need base point data
void IndexBipartite::LoadVectorDataReorder(const char *base_file, const char *sampled_query_file) {
    uint32_t base_num, sq_num, base_dim, q_dim;

    load_meta<float>(base_file, base_num, base_dim);
    load_meta<float>(sampled_query_file, sq_num, q_dim);
    if (base_dim != q_dim) {
        throw std::runtime_error("base and query dimension mismatch");
    }
    float *base_data = nullptr;
    float *sampled_query_data = nullptr;
    load_data_search_with_order<float>(base_file, base_num, base_dim, base_data, new_order_);
    // load_data<float>(sampled_query_file, sq_num, q_dim, sampled_query_data);

    if (need_normalize) {
        std::cout << "Normalizing base data" << std::endl;
        for (size_t i = 0; i < base_num; ++i) {
            normalize<float>(base_data + i * (uint64_t)base_dim, (uint64_t)base_dim);
        }
    }

    data_bp_ = base_data;
    // data_bp_ = data_align(base_data, base_num, base_dim);
    // data_sq_ = data_align(sampled_query_data, sq_num, q_dim);

    nd_ = base_num;
    nd_sq_ = sq_num;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
}

void IndexBipartite::LoadIndexDataReorder(const char *base_file) {
    uint32_t base_num, base_dim;

    load_meta<float>(base_file, base_num, base_dim);
    // if (base_dim != q_dim) {
    //     throw std::runtime_error("base and query dimension mismatch");
    // }
    float *base_data = nullptr;
    load_data_build_with_order<float>(base_file, base_num, base_dim, base_data, new_order_);
    // load_data<float>(sampled_query_file, sq_num, q_dim, sampled_query_data);

    data_bp_ = base_data;
    // data_bp_ = data_align(base_data, base_num, base_dim);
    // data_sq_ = data_align(sampled_query_data, sq_num, q_dim);

    nd_ = base_num;
    nd_sq_ = 0;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);
}

// only for search
void IndexBipartite::LoadIndexData(const char *base_file) {
    uint32_t base_num, base_dim;
    load_meta<float>(base_file, base_num, base_dim);

    float * base_data = nullptr;
    load_data_search<float>(base_file, base_num, base_dim, base_data);
    data_bp_ = base_data;
    // data_bp_ = data_align(base_data, base_num, base_dim);

    nd_ = base_num;
    nd_sq_ = 0;
    total_pts_ = nd_ + nd_sq_;
    u32_nd_ = static_cast<uint32_t>(nd_);
    u32_nd_sq_ = static_cast<uint32_t>(nd_sq_);
    u32_total_pts_ = static_cast<uint32_t>(total_pts_);

    /********测试********/
    // TrainQuantizer(data_bp_, nd_, base_dim);
    // LoadQuantizer(base_dim);
}

void IndexBipartite::LoadQuantizer(int dim_qunt){
    std::cout << "LoadQuantizer 维度:" << dim_qunt << std::endl;
    quant = new glass::SQ8Quantizer<glass::Metric::IP>(dim_qunt);
    quant->load_code(quant_file_path);
    std::cout << "Load prefetch param" << std::endl;
    LoadPrefetch();
}

void IndexBipartite::gorder(int w){
    u32_nd_ = nd_;

    // Find entry node for gorder
    internal_heap_loc_t seed_node = projection_ep_;
    // for (int l = entry_node_level; l > 0; l--){
    //     seed_node = *(highwayNodeChild(seed_node));
    // }
    std::cout << "In gorder, vertices: "  << u32_nd_ << std::endl;
    // Create table of in-degrees
    std::unordered_map<internal_heap_loc_t,std::vector<internal_heap_loc_t> > indegree_table;
    for (internal_heap_loc_t node = 0; node < u32_nd_; node++){
        // internal_heap_loc_t* node_edges = dataNodeLinks(node);
        internal_heap_loc_t* node_edges = projection_graph_[node].data();
        for (int m = 0; m < projection_graph_[node].size(); m++){
            if (node_edges[m] != node){
                // this was a bug (i think):
                // indegree_table[node].push_back(node_edges[m]);
                indegree_table[node_edges[m]].push_back(node);
            }
        }
    }

    std::cout << "starting order" << std::endl;
    // do the actual gorder
    GorderPriorityQueue<internal_heap_loc_t> Q(u32_nd_);
    std::vector<internal_heap_loc_t> P(u32_nd_, 0);
    Q.increment(seed_node);
    P[0] = Q.pop();

    // for i = 1 to N:
    for (int i = 1; i < u32_nd_; i++){
        internal_heap_loc_t v_e = P[i-1];
        // ve = P[i-1] # the newest node in window
        // for each node u in out-edges of ve:
        internal_heap_loc_t* v_e_edges = projection_graph_[v_e].data();
        for (int m = 0; m < projection_graph_[v_e].size(); m++){
            if (v_e_edges[m] != v_e){
                // if u in Q, increment priority of u
                Q.increment(v_e_edges[m]);
            }
        }

        // for each node u in in-edges of ve: 
        for (internal_heap_loc_t& u : indegree_table[v_e]){
            // if u in Q, increment priority of u
            Q.increment(u);
            // for each node v in out-edges of u:
            internal_heap_loc_t* u_edges = projection_graph_[u].data();
            for (int m = 0; m < projection_graph_[u].size(); m++){
                if (u_edges[m] != u){
                    // if v in Q, increment priority of v
                    Q.increment(u_edges[m]);
                }
            }
        }

        if (i > w+1){
            internal_heap_loc_t v_b = P[i-w-1];
            // for each node u in out-edges of vb:
            internal_heap_loc_t* v_b_edges = projection_graph_[v_b].data();
            for (int m = 0; m < projection_graph_[v_b].size(); m++){
                if (v_b_edges[m] != v_b){
                    // if u in Q, decrement priority of u
                    Q.decrement(v_b_edges[m]);
                }
            }
            // for each node u in in-edges of vb:
            for (internal_heap_loc_t& u : indegree_table[v_b]){
                // if u in Q, increment priority of u
                Q.increment(u); // this could be a bug?
                // for each node v in out-edges of u:
                internal_heap_loc_t* u_edges = projection_graph_[u].data();
                for (int m = 0; m < projection_graph_[u].size(); m++){
                    if (u_edges[m] != u){
                        // if v in Q, decrement priority of v
                        Q.decrement(u_edges[m]);
                    }
                }
            }
        }
        P[i] = Q.pop();
        // add progress, no change line, flush in same line
        if (i % 10000 == 0){
            std::cout << "\r" << i << "/" << u32_nd_ << std::flush;
        }
    }

    // for(int n = 0; n < cur_num_data_nodes; n++){
    //     // void* node_data = dataNodeData(data_node_ID);
    //     internal_heap_loc_t* node_links = dataNodeLinks(n);
    //     label_t* node_label = dataNodeLabel(n);
    //     std::cout<<"DataNode "<<*node_label<<"@{"<<n<<"}->("<<P[n]<<"): ";
    //     std::cout<<"[";
    //     for (int i = 0; i < M; i++){
    //         if (node_links[i] == n) continue;
    //         label_t* neighbor_label = dataNodeLabel(node_links[i]);

    //         std::cout<<*neighbor_label<< "@{"<<node_links[i]<<"}->("<<P[node_links[i]]<<") ";
    //     }
    //     std::cout<<"]"<<std::endl;
    // }
    // now we have a mapping P[i] -> node whose new label is i
    // std::vector<internal_heap_loc_t> Pinv(u32_nd_, 0);
    new_order_.resize(u32_nd_);
    Porigin_.resize(u32_nd_);
    for (int n = 0; n < u32_nd_; n++){
        new_order_[P[n]] = n;
        Porigin_[n] = P[n];
    }
    // now we have a mapping Pinv[i] -> new label of node i

    // permut to origin



    // return Pinv;
}

// void relabel(const std::vector<internal_heap_loc_t>& P){

//     // 1: rewire all the highway nodes
//     // std::vector<internal_heap_loc_t> highway_nodes;
//     // highway_nodes.push_back(entry_node);
//     // for (int l = entry_node_level; l>0; l--){
//     //     connectedHighwayComponent(highway_nodes);
//     //     // do the reconnections:
//     //     for (int i = 0; i < highway_nodes.size(); i++){
//     //         // WARN: potential bug: if data layout changes, we have to change this part
//     //         // update the data location for this highway node, to point to the new data location
//     //         internal_heap_loc_t data_node_ID = *highwayNode(highway_nodes[i]);
//     //         internal_heap_loc_t highway_node_ID = highway_nodes[i];
//     //         *highwayNode(highway_node_ID) = P[data_node_ID];
//     //         if (l == 1){
//     //             // then update the child location too, since child is in data-space
//     //             internal_heap_loc_t child_node = *( highwayNodeChild(highway_node_ID) );
//     //             *( highwayNodeChild(highway_node_ID) ) = P[child_node];
//     //         }
//     //         // drop down a level
//     //         highway_nodes[i] = *highwayNodeChild(highway_nodes[i]);
//     //     }
//     // }

//     // 2: re-layout the datanodes
//     // 2a: allocate a buffer for the new data node space
//     char* new_partition = new char[data_node_size*max_num_data_nodes];
//     char* old_partition = data_partition;

//     // 2b: iterate through the existing datanodes, relabeling and remapping everything
//     for (int old_node = 0; old_node < cur_num_data_nodes; old_node++){
//         // load pointers for the old data (access functions like dataNodeData use an offset from
//         // "data_partition" so we assign data_partition = old_partition to load from old)
//         data_partition = old_partition;
//         void* old_node_data_p = dataNodeData(old_node);
//         internal_heap_loc_t* old_node_links = dataNodeLinks(old_node);
//         label_t* old_node_label = dataNodeLabel(old_node);

//         data_partition = new_partition;
//         // copy the data from the old to the new partition
//         void* new_node_data_p = dataNodeData(P[old_node]);
//         std::memcpy(new_node_data_p, old_node_data_p, data_size);

//         // copy the links from the old to the new partition, remapping links as we go
//         internal_heap_loc_t* new_node_links = dataNodeLinks(P[old_node]);
//         for (int i = 0; i < M; i++){
//             new_node_links[i] = P[old_node_links[i]];
//         }
//         // copy over the label
//         label_t* new_node_label = dataNodeLabel(P[old_node]);
//         *(new_node_label) = *(old_node_label);
//     }

//     // copy old address into new one
//     std::memcpy(old_partition, new_partition, data_node_size*max_num_data_nodes);

//     // reassign data_partition to be correct
//     data_partition = old_partition;

//     // free the temporary partition
//     delete[] new_partition;
// }

void IndexBipartite::ConvertAdjList2CSR(uint32_t*& row_ptr, uint32_t*& col_idx) {
        const uint32_t num_vertices = projection_graph_.size();
        uint32_t num_edges = 0;
        for (auto& neighbors : projection_graph_) {
            num_edges += neighbors.size();
        }

        row_ptr = new uint32_t[num_vertices + 1];
        col_idx = new uint32_t[num_edges];

        row_ptr[0] = 0;
        for (uint32_t i = 0; i < num_vertices; ++i) {
            row_ptr[i + 1] = row_ptr[i] + projection_graph_[i].size();
        }

        uint32_t edge_idx = 0;
        for (uint32_t i = 0; i < num_vertices; ++i) {
            for (auto& neighbor : projection_graph_[i]) {
            col_idx[edge_idx] = neighbor;
            ++edge_idx;
            }
        }
        //clean memory in adj_list
        // for (auto& neighbors : projection_graph_) {
        //     neighbors.clear();
        //     neighbors.shrink_to_fit();
        // }
        // projection_graph_.clear();
        // projection_graph_.shrink_to_fit();
        std::cout << "Convert projection graph to CSR format" << std::endl;
}

void IndexBipartite::ConvertAdjList2CSR(uint32_t*& row_ptr, uint32_t*& col_idx, std::vector<uint32_t>& P) {
        const uint32_t num_vertices = reordered_graph_.size();
        uint32_t num_edges = 0;
        for (auto& neighbors : reordered_graph_) {
            num_edges += neighbors.size();
        }

        row_ptr = new uint32_t[num_vertices + 1];
        col_idx = new uint32_t[num_edges];

        row_ptr[0] = 0;
        for (uint32_t i = 0; i < num_vertices; ++i) {
            row_ptr[i + 1] = row_ptr[i] + reordered_graph_[i].size();
        }

        uint32_t edge_idx = 0;
        for (uint32_t i = 0; i < num_vertices; ++i) {
            for (auto& neighbor : reordered_graph_[i]) {
            col_idx[edge_idx] = neighbor;
            ++edge_idx;
            }
        }
        //clean memory in adj_list
        for (auto& neighbors : reordered_graph_) {
            neighbors.clear();
            neighbors.shrink_to_fit();
        }
        reordered_graph_.clear();
        reordered_graph_.shrink_to_fit();
        malloc_trim(0);
        std::cout << "Convert projection graph to CSR format" << std::endl;
}

void IndexBipartite::ConvertAdjList2CSR(const std::vector<uint32_t>& P, uint32_t*& row_ptr, uint32_t*& col_idx) {
  const uint32_t num_vertices = P.size();
  uint32_t num_edges = 0;
  for (auto& neighbors : projection_graph_) {
    num_edges += neighbors.size();
  }

  row_ptr = new uint32_t[num_vertices + 1];
  col_idx = new uint32_t[num_edges];

  // Compute the new row_ptr array
  for (uint32_t i = 0; i < num_vertices; ++i) {
    row_ptr[P[i]] = i == 0 ? 0 : row_ptr[P[i - 1] + 1];
    for (auto& neighbor : projection_graph_[i]) {
      if (P[neighbor] < P[i + 1]) {
        ++row_ptr[P[i]];
      }
    }
  }
  for (uint32_t i = 1; i < num_vertices; ++i) {
    row_ptr[P[i]] += row_ptr[P[i - 1] + 1];
  }
  row_ptr[num_vertices] = num_edges;

  // Compute the new col_idx array
  for (uint32_t i = 0; i < num_vertices; ++i) {
    for (auto& neighbor : projection_graph_[i]) {
      if (P[neighbor] < P[i + 1]) {
        col_idx[row_ptr[P[i]]] = P[neighbor];
        ++row_ptr[P[i]];
      }
    }
  }

  // Reset the row_ptr array
  for (uint32_t i = num_vertices; i > 0; --i) {
    row_ptr[P[i - 1] + 1] = row_ptr[i];
  }
  row_ptr[P[0]] = 0;

    for (auto& neighbors : projection_graph_) {
        neighbors.clear();
        neighbors.shrink_to_fit();
    }
    projection_graph_.clear();
    projection_graph_.shrink_to_fit();

  std::cout << "Converted adjacency list to CSR format with permutation P" << std::endl;
}

void IndexBipartite::ReorderAdjList(const std::vector<uint32_t>& P) {
  const uint32_t num_vertices = P.size();
  std::cout << "Premutation size: " << num_vertices << std::endl;
  reordered_graph_.resize(num_vertices);

  // Populate the new adjacency list
  for (uint32_t i = 0; i < num_vertices; ++i) {
    for (auto& neighbor : projection_graph_[i]) {
      reordered_graph_[P[i]].push_back(P[neighbor]);
    }
  }
    for (auto& neighbors : projection_graph_) {
        neighbors.clear();
        neighbors.shrink_to_fit();
    }
    projection_graph_.clear();
    projection_graph_.shrink_to_fit();
  std::cout << "Reordered adjacency list with permutation P" << std::endl;
}

void IndexBipartite::FindMaxOutDegree(uint32_t* row_ptr, uint32_t* col_idx, int& max_out_degree) {
  max_out_degree = 0;
//   const uint32_t num_vertices = row_ptr.size() - 1;
  const uint32_t num_vertices = u32_nd_;
  for (uint32_t i = 0; i < num_vertices; ++i) {
    const uint32_t out_degree = row_ptr[i + 1] - row_ptr[i];
    if (out_degree > max_out_degree) {
      max_out_degree = out_degree;
    }
  }
}

}  // namespace mysteryann

