#include <boost/container/set.hpp>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "mysteryann/index.h"
#include "mysteryann/neighbor.h"
#include "mysteryann/parameters.h"
#include "mysteryann/util.h"
#include "mysteryann/GorderPriorityQueue.h"
#include "visited_list_pool.h"
#include "mysteryann/distance.h"


#include "SQ/quant/quant.hpp"
#include "SQ/common.hpp"

namespace mysteryann {
typedef unsigned int internal_heap_loc_t;
using LockGuard = std::lock_guard<std::mutex>;
using SharedLockGuard = std::lock_guard<std::shared_mutex>;

class IndexBipartite : public Index {
    typedef std::vector<std::vector<uint32_t>> CompactGraph;

   public:
    explicit IndexBipartite(const size_t dimension, const size_t n, Metric m, Index *initializer);
    virtual ~IndexBipartite();
    virtual void Save(const char *filename) override;
    virtual void Load(const char *filename) override;
    virtual void Search(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices,
                        float *res_dists) override;
    virtual void BuildBipartite(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                                const Parameters &parameters) override;

    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

    inline void SetBipartiteParameters(const Parameters &parameters) {}
    uint32_t SearchBipartiteGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                  unsigned *indices);
    void LinkBipartite(const Parameters &parameters, SimpleNeighbor *simple_graph);
    void LinkOneNode(const Parameters &parameters, uint32_t nid, SimpleNeighbor *simple_graph, bool is_base,
                     boost::dynamic_bitset<> &visited);

    void BuildEdgeAfterAdd(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data,
                           const Parameters &parameters);

    void SearchBipartitebyBase(const float *query, uint32_t nid, const Parameters &parameters,
                               SimpleNeighbor *simple_graph, NeighborPriorityQueue &search_pool,
                               boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    void SearchBipartitebyQuery(const float *query, uint32_t nid, const Parameters &parameters,
                                SimpleNeighbor *simple_graph, NeighborPriorityQueue &search_pool,
                                boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    void LoadVectorData(const char *base_file, const char *sampled_query_file);

    void LoadVectorDataReorder(const char *base_file, const char *sampled_query_file);

    void BuildGraphST1(uint32_t n_sq, uint32_t n_bp, uint32_t M_bp, uint32_t L_pq, float *bp_data, const mysteryann::Parameters parameters);

    CompactGraph &GetBipartiteGraph() { return bipartite_graph_; }
    inline void InitBipartiteGraph() { bipartite_graph_.resize(total_pts_); }

    inline void LoadSearchNeededData(const char *base_file, const char *sampled_query_file) {
        LoadVectorData(base_file, sampled_query_file);
    }

    inline void LoadSearchNeededDataReorder(const char *base_file, const char *sampled_query_file) {
        LoadVectorDataReorder(base_file, sampled_query_file);
    }

    void LoadIndexData(const char *base_file);

    /***********************/
    // 预取参数(把 setThread 测试删掉)
    int po = 5; // 预取邻居数
    int pl = 4; // 预取 cache line 数
    // std::string query_file_path = "data/indices/ood/mysteryann/Text2Image1B-10000000/M_bp35_L_pq500_NoT5_ord/learn.5M.fbin";
    std::string query_file_path;
    std::string prefetch_file_path;
    void LoadPrefetch();
    void OptimizePrefetch(uint32_t L_pq, int num_threads);
	int read_fvecs(std::string path_file, int &N, int &Dim, std::vector<float> &optimize_queries, int kOptimizePoints);

    // 标量量化(把 LoadIndexData 测试删掉)
    std::string quant_file_path;
	glass::SQ8Quantizer<glass::Metric::IP> *quant = nullptr;
    void LoadQuantizer(int dim_qunt); // dim_qunt 在 Train、Load、query 中保存一致
    void TrainQuantizer(const float *data_quant, int n_quant, int dim_qunt);
	void SearchGraph_SQ(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices);
    /***********************/

    void SearchGraph(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices);

    void PruneCandidates(std::vector<Neighbor> &search_pool, uint32_t tgt_id, const Parameters &parameters,
                         std::vector<uint32_t> &pruned_list, boost::dynamic_bitset<> &visited);
    void AddReverse(NeighborPriorityQueue &search_pool, uint32_t src_node, std::vector<uint32_t> &pruned_list,
                    const Parameters &parameters, boost::dynamic_bitset<> &visited);

    void BipartiteProjection(const Parameters &parameters);

    void CalculateProjectionep();

    void LinkProjection(const Parameters &parameters, SimpleNeighbor *simple_graph);

    void LinkBase(const Parameters &parameters, SimpleNeighbor *simple_graph);

    void AddEdgesInline(const Parameters &parameters);

    void AddEdgesInlineParts(const Parameters &parameters);

    void BuildGraphInline(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data, Parameters &parameters);

    void BuildGraphST2(size_t n_bp, const float *bp_data, Parameters &parameters);

    void ClearnBuildMemoryRemain4Search();

    void SaveBaseData(const char *filename);

    void TrainingLink2Projection(const Parameters &parameters, SimpleNeighbor *simple_graph);

    void SearchProjectionbyQuery(const float *query, const Parameters &parameters, NeighborPriorityQueue &search_pool,
                                 boost::dynamic_bitset<> &visited, std::vector<Neighbor> &full_retset);

    uint32_t PruneProjectionCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                       const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void PruneProjectionBaseSearchCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                             const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void PruneProjectionBaseSearchCandidatesSupply(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                             const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void ProjectionAddReverse(uint32_t src_node, const Parameters &parameters);

    void SupplyAddReverse(uint32_t src_node, const Parameters &parameters);

    void PruneProjectionReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                          std::vector<uint32_t> &pruned_list);

    void PruneProjectionInternalReverseCandidates(uint32_t src_node, const Parameters &parameters,
                                                  std::vector<uint32_t> &pruned_list);

    uint32_t SearchProjectionGraph(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                   unsigned *indices, std::vector<float> &res_dists);

    uint32_t SearchProjectionCSR(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                   unsigned *indices, std::vector<float> &res_dists);
    uint32_t SearchProjectionCSRwithOrder(const float *query, size_t k, size_t &qid, const Parameters &parameters,
                                   unsigned *indices, std::vector<float> &res_dists);

    void SaveProjectionGraph(const char *filename);

    void LoadProjectionGraph(const char *filename);

    void LoadNsgGraph(const char *filename);

    void LoadLearnBaseKNN(const char *filename);

    void LoadBaseLearnKNN(const char *filename);

    inline std::vector<std::vector<uint32_t>> &GetProjectionGraph() { return projection_graph_; }

    uint32_t PruneProjectionBipartiteCandidates(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                                const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void SearchProjectionGraphInternal(NeighborPriorityQueue &search_queue, const float *query, uint32_t tgt,
                                       const Parameters &parameters, boost::dynamic_bitset<> &visited,
                                       std::vector<Neighbor> &full_retset);

    void SearchProjectionGraphInternalPJ(NeighborPriorityQueue &search_queue, const float *query, uint32_t tgt,
                                       const Parameters &parameters, boost::dynamic_bitset<> &visited,
                                       std::vector<Neighbor> &full_retset);

    void PruneBiSearchBaseGetBase(std::vector<Neighbor> &search_pool, const float *query, uint32_t qid,
                                  const Parameters &parameters, std::vector<uint32_t> &pruned_list);

    void PruneLocalJoinCandidates(uint32_t node, const Parameters &parameters, uint32_t candi);

    void CollectPoints(const Parameters &parameters);

    void dfs(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);

    void InitVisitedListPool(uint32_t num_threads) { 
        if (visited_list_pool_ == nullptr) {
            visited_list_pool_ = new VisitedListPool(num_threads, nd_); 
        }
    };

    void SaveReorder(std::string& filename);

    void LoadReorder(std::string& order_file, std::string& original_order_file);

    void LoadIndexDataReorder(const char *base_file);

    void SearchReorderGraph(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices);

    void gorder(int w);

    void FreeBaseData();

    const float *get_base_ptr() {
        return data_bp_;
    }

    void SearchReorderGraph4Tune(const float *query, size_t k, size_t &qid, const uint32_t L_pq, unsigned *indices);

    void FindMaxOutDegree(uint32_t* row_ptr, uint32_t* col_idx, int& max_out_degree);


    void ConvertAdjList2CSR(uint32_t*& row_ptr, uint32_t*& col_idx);
    void ConvertAdjList2CSR(uint32_t*& row_ptr, uint32_t*& col_idx, std::vector<uint32_t>& P);

    //inline get neighbors from CSR
    inline void csr_get_neighbors(uint32_t node_id, uint32_t*& neighbors, uint32_t& neighbors_size){
        neighbors_size = row_ptr_[node_id + 1] - row_ptr_[node_id];
        neighbors = col_idx_ + row_ptr_[node_id];
        // for (size_t i = 0; i < neighbors_size; i++) {
        // _mm_prefetch((char*)(neighbors), _MM_HINT_T0);
        // }
    }

    void ConvertAdjList2CSR(const std::vector<uint32_t>& P, uint32_t*& row_ptr, uint32_t*& col_idx);

    void ReorderAdjList(const std::vector<uint32_t>& P);

    void BuildGraphOnlyBase(size_t n_bp, const float *bp_data, Parameters &parameters);

    // void AddBasePoint();
    // void AddSampledQuery();
    // void SearchWithOptGraph(const float *query, size_t K, const Parameters &parameters,
    //                         unsigned *indices);

    Index *initializer_;
    TimeMetric dist_cmp_metric;
    TimeMetric memory_access_metric;
    TimeMetric block_metric;
    VisitedListPool *visited_list_pool_{nullptr};
    bool need_normalize = false;
    std::string train_data_file;
    size_t train_parts_;
    size_t each_part_num_;
    size_t plan_train_num_;
    uint32_t search_dim_;
    std::vector<uint32_t> new_order_;
    std::vector<uint32_t> Porigin_;
    uint32_t gorder_w = 5;

    // CSR
    uint32_t *row_ptr_ = nullptr;
    uint32_t *col_idx_ = nullptr;
    
    

   protected:
    std::vector<std::vector<uint32_t>> bipartite_graph_;
    std::vector<std::vector<uint32_t>> final_graph_;
    std::vector<std::vector<uint32_t>> projection_graph_;
    std::vector<std::vector<uint32_t>> supply_nbrs_;
    std::vector<std::vector<uint32_t>> learn_base_knn_;
    std::vector<std::vector<uint32_t>> base_learn_knn_;
    std::vector<std::vector<uint32_t>> reordered_graph_;

   private:
    const size_t total_pts_const_;
    size_t total_pts_;
    Distance *l2_distance_;
    // boost::dynamic_bitset<> sq_en_flags_;
    // boost::dynamic_bitset<> bp_en_flags_;
    uint32_t width_;
    std::set<uint32_t> sq_en_set_;
    std::set<uint32_t> bp_en_set_;
    std::mutex sq_set_mutex_;
    std::mutex bp_set_mutex_;
    std::vector<std::mutex> locks_;
    uint32_t u32_nd_;
    uint32_t u32_nd_sq_;
    uint32_t u32_total_pts_;
    uint32_t projection_ep_;
};
}  // namespace mysteryann