// index bindings of bipartite_index for python

#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>
#include <unistd.h>

#include "index_bipartite.h"
#include "mysteryann/distance.h"
#include "mysteryann/neighbor.h"
#include "mysteryann/parameters.h"
#include "mysteryann/util.h"

namespace py = pybind11;

template <typename T>

class IndexMystery {
   public:
    IndexMystery(const size_t dimension, const size_t n, mysteryann::Metric m) {
        // index_ = new mysteryann::IndexBipartite(dimension, n, m, nullptr);
        // init_ = true;
    }

    ~IndexMystery() { delete index_; }
    void Search(py::array_t<float, py::array::c_style | py::array::forcecast> &q_data, size_t k, uint32_t L_pq,
                py::array_t<uint32_t, py::array::c_style> &res_id, size_t q_num, uint32_t num_threads, bool using_sq = true) {
        // size_t dim = index_->GetDimension();
        auto items = q_data.unchecked();
        auto res = res_id.mutable_unchecked();
        if (using_sq) {
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t qid = 0; qid < q_num; qid++) {
                const float *query = &items(qid, 0);
                // index_->SearchGraph_SQ(query, k, qid, L_pq, &res(qid, 0));
                index_->SearchReorderGraph4Tune(query, k, qid, L_pq, &res(qid, 0));
            }
        } else {
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t qid = 0; qid < q_num; qid++) {
                const float *query = &items(qid, 0);
                // index_->SearchGraph(query, k, qid, L_pq, &res(qid, 0));
                index_->SearchReorderGraph(query, k, qid, L_pq, &res(qid, 0));
            }
        }
    }
    //     void Search(py::object q_data, size_t k, uint32_t L_pq, py::array_t<float>& res_id, size_t q_num, uint32_t
    //     num_threads) {
    //         // std::cout << "here111" << std::endl;
    //         size_t dim = index_->GetDimension();
    //         py::array_t <T, py::array::c_style | py::array::forcecast > items(q_data);
    //         // py::array_t <T, py::array::c_style | py::array::forcecast > res(res_id);

    // #pragma omp parallel for schedule (dynamic, 1)
    //         for (size_t qid = 0; qid < q_num; qid++) {
    //             float* query = (float*)items.data(qid);
    //             // std::cout << "QID:" << qid << std::endl;
    //             index_->SearchGraph(query , k, qid, L_pq, res_id[qid].data());
    //             // index_->SearchGraph(query , k, qid, L_pq, (uint32_t*)res.data(qid));
    //         }
    //     }

    void BuildST1(uint32_t num_sq, uint32_t num_base, uint32_t M_bp, uint32_t L_pq, uint32_t num_threads,
                  std::string learn_base_nn, std::string base_file) {
        uint32_t base_num, base_dim;
        mysteryann::load_meta<float>(base_file.c_str(), base_num, base_dim);
        float *data_bp = nullptr;
        mysteryann::load_data<float>(base_file.c_str(), base_num, base_dim, data_bp);
        float *aligned_data_bp = nullptr;
        aligned_data_bp = mysteryann::data_align(data_bp, base_num, base_dim);
        omp_set_num_threads(num_threads);
        index_->LoadLearnBaseKNN(base_file.c_str());
        mysteryann::Parameters parameters;
        parameters.Set<uint32_t>("M_bp", M_bp);
        parameters.Set<uint32_t>("L_pq", L_pq);
        parameters.Set<uint32_t>("M_pjbp", M_bp);
        parameters.Set<uint32_t>("L_pjpq", L_pq);
        parameters.Set<uint32_t>("num_threads", num_threads);
        index_->BuildGraphST1(num_sq, num_base, M_bp, L_pq, aligned_data_bp, parameters);
    }


    void Load(const char *graph_file, const char *data_file, mysteryann::Metric m) {
        uint32_t base_num, base_dim;
        mysteryann::load_meta<float>(data_file, base_num, base_dim);
        uint32_t origin_dim = base_dim;
        if (m == mysteryann::INNER_PRODUCT) {
            uint32_t search_align = DATA_ALIGN_FACTOR * 2;
            base_dim = (base_dim + search_align - 1) / search_align * search_align;
        }
        index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
        index_->search_dim_ = origin_dim;
        index_->LoadProjectionGraph(graph_file);
        index_->LoadIndexData(data_file);

        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }

    void LoadwithReorder(const char *graph_file, const char *data_file, mysteryann::Metric m) {
        uint32_t base_num, base_dim;
        mysteryann::load_meta<float>(data_file, base_num, base_dim);
        uint32_t origin_dim = base_dim;
        if (m == mysteryann::INNER_PRODUCT) {
            uint32_t search_align = DATA_ALIGN_FACTOR;
            base_dim = (base_dim + search_align - 1) / search_align * search_align;
        }
        index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
        index_->search_dim_ = origin_dim;
        index_->LoadProjectionGraph(graph_file);
        std::string order_file_name = std::string(graph_file) + ".order";
        std::string original_order_file_name = std::string(graph_file) + ".original_order";
        index_->LoadReorder(order_file_name, original_order_file_name);
        index_->ReorderAdjList(index_->new_order_);
        index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
        index_->LoadIndexDataReorder(data_file);
    }

    void LoadwithReorderwithSQ(const char *graph_file, const char *data_file, mysteryann::Metric m) {
        uint32_t base_num, base_dim;
        mysteryann::load_meta<float>(data_file, base_num, base_dim);
        uint32_t origin_dim = base_dim;
        if (m == mysteryann::INNER_PRODUCT) {
            uint32_t search_align = DATA_ALIGN_FACTOR;
            base_dim = (base_dim + search_align - 1) / search_align * search_align;
        }
        index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
        index_->search_dim_ = origin_dim;
        index_->LoadProjectionGraph(graph_file);
        index_->prefetch_file_path = std::string(graph_file) + ".prefetch";
        index_->quant_file_path = std::string(graph_file) + ".quant";
        std::string order_file_name = std::string(graph_file) + ".order";
        std::string original_order_file_name = std::string(graph_file) + ".original_order";
        std::cout << "loading order..." << std::endl;
        index_->LoadReorder(order_file_name, original_order_file_name);
        std::cout << "reorder adjlist..." << std::endl;
        index_->ReorderAdjList(index_->new_order_);
        std::cout << "convert adjlist 2 csr..." << std::endl;
        index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
        // std::cout << "load data in order 2 csr..." << std::endl;
        // index_->LoadIndexDataReorder(data_file);
        // load for test
        // index_->InitVisitedListPool(10);
        // std::cout << "train sq..." << std::endl;
        // index_->TrainQuantizer(index_->get_base_ptr(), base_num, base_dim);
        
        // std::cout << "train sq... done" << std::endl;
        std::cout << "loading sq..." << std::endl;
        index_->LoadQuantizer(base_dim);
        std::cout << "loaded sq" << std::endl;
    }


    void setThreads(uint32_t num_threads) {
        omp_set_num_threads(num_threads);
        index_->InitVisitedListPool(2 * num_threads);

        /********测试********/
        // 目前是在 setThreads 时优化参数
        // 必须 InitVisitedListPool 后调用
        // index_->OptimizePrefetch(); 
        // index_->LoadPrefetch();
    }

   private:
    mysteryann::IndexBipartite *index_;
    // bool init_ = false;
};
void Save(std::string filename, mysteryann::IndexBipartite *index_) {
    index_->SaveProjectionGraph(filename.c_str());
    index_->SaveReorder(filename);
}

void BuildST2(uint32_t M_pjbp, uint32_t L_pjpq, uint32_t num_threads, std::string base_file, mysteryann::Metric m,
                std::string traning_set_file, uint32_t each_train_num, uint32_t plan_train_num,
                std::string index_save_path) {
    // if (!init_) {
    //     std::cout << "Index not initialized" << std::endl;
    //     return;
    // }
    if (traning_set_file.size() == 0) {
        std::cout << "small dataset" << std::endl;
        uint32_t base_num, base_dim;

        mysteryann::load_meta<float>(base_file.c_str(), base_num, base_dim);
        mysteryann::IndexBipartite * index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
        float *data_bp = nullptr;
        mysteryann::load_data_build<float>(base_file.c_str(), base_num, base_dim, data_bp);
        omp_set_num_threads(num_threads);
        mysteryann::Parameters parameters;
        parameters.Set<uint32_t>("M_pjbp", M_pjbp);
        parameters.Set<uint32_t>("L_pjpq", L_pjpq);
        parameters.Set<uint32_t>("num_threads", num_threads);
        index_->BuildGraphOnlyBase(base_num, data_bp, parameters);
        // index_->LoadProjectionGraph(index_save_path.c_str());
        std::cout << "Saving data file ..." << std::endl;
        std::string data_file_name = index_save_path + ".data";
        index_->SaveBaseData(data_file_name.c_str());
        std::cout << "Data file saved" << std::endl;
        std::cout << "roding ..." << std::endl;
        index_->gorder(index_->gorder_w);
        std::cout << "roding done" << std::endl;
        std::cout << "saving graph and order" << std::endl;
        Save(index_save_path, index_);
        std::cout << "graph and order saved" << std::endl; 
        delete index_;
        malloc_trim(0);
        std::cout << "create search context for train sq" << std::endl;
        mysteryann::load_meta<float>(data_file_name.c_str(), base_num, base_dim);
        uint32_t origin_dim = base_dim;
        // if (m == mysteryann::INNER_PRODUCT) {
            uint32_t search_align = DATA_ALIGN_FACTOR;
            base_dim = (base_dim + search_align - 1) / search_align * search_align;
        // }
        index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
        index_->search_dim_ = origin_dim;
        index_->LoadProjectionGraph(index_save_path.c_str());
        std::string order_file_name = std::string(index_save_path) + ".order";
        std::string original_order_file_name = std::string(index_save_path) + ".original_order";
        std::cout << "load order..." << std::endl;
        index_->LoadReorder(order_file_name, original_order_file_name);
        std::cout << "reorder adjlist..." << std::endl;
        index_->ReorderAdjList(index_->new_order_);
        std::cout << "convert to csr..." << std::endl;
        index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
        std::cout << "load base data in order..." << std::endl;
        index_->LoadIndexDataReorder(data_file_name.c_str());

        std::cout << "train sq" << std::endl;
        index_->InitVisitedListPool(num_threads);    
        index_->query_file_path = traning_set_file;
        index_->prefetch_file_path = index_save_path + ".prefetch";
        index_->quant_file_path = index_save_path + ".quant";
        index_->TrainQuantizer(index_->get_base_ptr(), base_num, base_dim);
        //use search 
        // SearchReorderGraph()
        index_->FreeBaseData();
        delete index_;
        return;
    }
    uint32_t base_num, base_dim, sq_num, sq_dim;
    mysteryann::load_meta<float>(base_file.c_str(), base_num, base_dim);
    mysteryann::load_meta<float>(traning_set_file.c_str(), sq_num, sq_dim);
    if (sq_num < plan_train_num) {
        std::cout << "FAIL: sampled query num is less than plan train num" << std::endl;
        return;
    }
    mysteryann::IndexBipartite * index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);

    index_->train_parts_ = static_cast<size_t>(plan_train_num / each_train_num);
    index_->each_part_num_ = each_train_num * 1000000;
    index_->plan_train_num_ = plan_train_num * 1000000;
    index_->train_data_file = traning_set_file;
    std::cout << "train_parts: " << index_->train_parts_ << std::endl;
    float *data_bp = nullptr;
    mysteryann::load_data_build<float>(base_file.c_str(), base_num, base_dim, data_bp);
    omp_set_num_threads(num_threads);
    // index_->LoadLearnBaseKNN(base_file.c_str());
    mysteryann::Parameters parameters;

    parameters.Set<uint32_t>("M_pjbp", M_pjbp);
    parameters.Set<uint32_t>("L_pjpq", L_pjpq);
    parameters.Set<uint32_t>("num_threads", num_threads);
    index_->BuildGraphST2(base_num, data_bp, parameters);
    // index_->LoadProjectionGraph(index_save_path.c_str());
    std::cout << "Saving data file ..." << std::endl;
    std::string data_file_name = index_save_path + ".data";
    index_->SaveBaseData(data_file_name.c_str());
    std::cout << "Data file saved" << std::endl;
    std::cout << "roding ..." << std::endl;
    index_->gorder(index_->gorder_w);
    std::cout << "roding done" << std::endl;
    std::cout << "saving graph and order" << std::endl;
    Save(index_save_path, index_);
    std::cout << "graph and order saved" << std::endl; 
    delete index_;
    malloc_trim(0);
    std::cout << "create search context for train sq" << std::endl;
    mysteryann::load_meta<float>(data_file_name.c_str(), base_num, base_dim);
    uint32_t origin_dim = base_dim;
    if (m == mysteryann::INNER_PRODUCT) {
        uint32_t search_align = DATA_ALIGN_FACTOR;
        base_dim = (base_dim + search_align - 1) / search_align * search_align;
    }
    index_ = new mysteryann::IndexBipartite(base_dim, base_num, m, nullptr);
    index_->search_dim_ = origin_dim;
    index_->LoadProjectionGraph(index_save_path.c_str());
    std::string order_file_name = std::string(index_save_path) + ".order";
    std::string original_order_file_name = std::string(index_save_path) + ".original_order";
    std::cout << "load order..." << std::endl;
    index_->LoadReorder(order_file_name, original_order_file_name);
    std::cout << "reorder adjlist..." << std::endl;
    index_->ReorderAdjList(index_->new_order_);
    std::cout << "convert to csr..." << std::endl;
    index_->ConvertAdjList2CSR(index_->row_ptr_, index_->col_idx_, index_->new_order_);
    std::cout << "load base data in order..." << std::endl;
    index_->LoadIndexDataReorder(data_file_name.c_str());

    std::cout << "train sq" << std::endl;
    index_->InitVisitedListPool(num_threads);    
    index_->query_file_path = traning_set_file;
    index_->prefetch_file_path = index_save_path + ".prefetch";
    index_->quant_file_path = index_save_path + ".quant";
    index_->TrainQuantizer(index_->get_base_ptr(), base_num, base_dim);
    //use search 
    // SearchReorderGraph()
    index_->FreeBaseData();
    delete index_;

}

// write a load function
IndexMystery<float> *load(const char *graph_file, const char *data_file, mysteryann::Metric m) {
    uint32_t base_num, base_dim;
    mysteryann::load_meta<float>(data_file, base_num, base_dim);
    IndexMystery<float> *mystery_index = new IndexMystery<float>(base_dim, base_num, m);
    // mystery_index->Load(graph_file, data_file, m);
    // mystery_index->LoadwithReorder(graph_file, data_file, m);
    mystery_index->LoadwithReorderwithSQ(graph_file, data_file, m);
    // index->LoadProjectionGraph(graph_file);
    // index->LoadIndexData(data_file);
    return mystery_index;
}

PYBIND11_MODULE(index_mystery, m) {
    m.doc() = "pybind11 index_mystery plugin";  // optional module docstring
    // enumerate...
    py::enum_<mysteryann::Metric>(m, "Metric")
        .value("L2", mysteryann::Metric::L2)
        .value("IP", mysteryann::Metric::INNER_PRODUCT)
        .value("COSINE", mysteryann::Metric::COSINE)
        .value("IP_BUILD", mysteryann::Metric::IP_BUILD)
        .export_values();
    py::class_<IndexMystery<float>>(m, "IndexMystery")
        .def(py::init<const size_t, const size_t, mysteryann::Metric>())
        .def("search", &IndexMystery<float>::Search)
        .def("load", &IndexMystery<float>::Load)
        .def("setThreads", &IndexMystery<float>::setThreads);
    m.def("load", &load)
    .def("buildST2", &BuildST2);
}
