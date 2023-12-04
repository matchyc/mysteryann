
#include <mysteryann/index_nsg.h>
#include <mysteryann/util.h>
#include <omp.h>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned>>& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned)results[i].size();
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, std::vector<std::vector<uint32_t>> res, uint32_t* gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res[i]);
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }
        // std::set_intersection(temp_res.begin(), temp_res.end(), one_gt.begin(), one_gt.end(),
        //   std::back_inserter(intersection));

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return (float)total_count / (k * q_num);
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim, std::vector<std::vector<float>>& res_dists, uint32_t k, mysteryann::Metric metric) {
    double total_err = 0;
    uint32_t q_num = res_dists.size();

    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i]);
        if (metric == mysteryann::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == mysteryann::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * ( 1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (uint32_t j = 0; j < k; j++) {
            err += std::fabs(double(temp_res[j] - one_gt[j])) / double(one_gt[j]);
        }
        err = err / (double)k;
        total_err += err;
    }
    return total_err / (double)q_num;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cout << argv[0] << " data_file query_file nsg_path search_L search_K result_path eval_path dist_fn" << std::endl;
        exit(-1);
    }
    float* data_load = NULL;
    unsigned points_num, dim;
    // load_data(argv[1], data_load, points_num, dim);
    mysteryann::load_meta<float>(argv[1], points_num, dim);
    mysteryann::load_data<float>(argv[1], points_num, dim, data_load);

    data_load = mysteryann::data_align(data_load, points_num, dim);  // one must
                                                                   // align the
                                                                   // data before
                                                                   // build

    float* query_load = NULL;
    unsigned query_num, query_dim;
    mysteryann::load_meta<float>(argv[2], query_num, query_dim);
    mysteryann::load_data<float>(argv[2], query_num, query_dim, query_load);
    query_load = mysteryann::data_align(query_load, query_num, query_dim);

    // load gt
    uint32_t gt_pts, gt_dim;
    uint32_t* gt_ids = nullptr;
    float * gt_dists = nullptr;
    mysteryann::load_gt_meta<uint32_t>(argv[3], gt_pts, gt_dim);
    // mysteryann::load_gt_data<uint32_t>(argv[3], gt_pts, gt_dim, gt_ids);  
    mysteryann::load_gt_data_with_dist<uint32_t, float>(argv[3], gt_pts, gt_dim, gt_ids, gt_dists);

    // load_data(argv[2], query_load, query_num, query_dim);
    assert(dim == query_dim);

    // unsigned L = (unsigned)atoi(argv[5]);
    unsigned K = (unsigned)atoi(argv[6]);
    std::cout << "K: " << K << "\n";

    std::string evaluation_save_path = argc < 8 ? "" : std::string(argv[8]);
    std::cout << "evaluation_save_path: " << evaluation_save_path << "\n";

    std::string dist_fn = argc < 9 ? "L2" : std::string(argv[9]);

    mysteryann::Metric metric = mysteryann::L2;
    if (dist_fn == "l2") {
        metric = mysteryann::L2;
        std::cout << "L2" << std::endl;
    } else if (dist_fn == "ip") {
        metric = mysteryann::INNER_PRODUCT;
        std::cout << "IP" << std::endl;
    } else if (dist_fn == "cosine") {
        metric = mysteryann::COSINE;
    } else {
        std::cout << "Unknown distance function: " << dist_fn << "\n";
        exit(-1);
    }

    if (metric == mysteryann::COSINE) {
        for (size_t i = 0; i < points_num; ++i) {
            mysteryann::normalize<float>(data_load + i * dim, dim);
        }
        for (size_t i = 0; i < query_num; ++i) {
            mysteryann::normalize<float>(query_load + i * query_dim, query_dim);
        }
    }

    std::vector<uint32_t> L_vec_10 = 
    {
       10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,550,600,700,800,900,1100,1200,1300,1400,1500,1600,1700,1900,2000
    };
    std::vector<uint32_t> L_vec_100 = {
    100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,350,400,450,500,550,600,650,700,750,800,900,1000,1100,1200,1400,1500,1600,1700,1800
    };
    std::vector<uint32_t>& L_vec = L_vec_10;
    if (K == 100) {
        L_vec = L_vec_100;
    }
    // data_load = mysteryann::data_align(data_load, points_num, dim);//one must
    // align the data before build query_load = mysteryann::data_align(query_load,
    // query_num, query_dim);
    mysteryann::IndexNSG index(dim, points_num, metric, nullptr);
    std::cout << "loading index from " << argv[4] << "\n";
    index.Load(argv[4]);

    std::ofstream evaluation_ofs;
    if (!evaluation_save_path.empty()) {
        evaluation_ofs.open(evaluation_save_path, std::ios::out);
    }
    omp_set_num_threads(16);
    for (auto L : L_vec) {
        if (L < K) {
            std::cout << "search_L " << L << " cannot be smaller than search_K!" << std::endl;
            exit(-1);
        }
        mysteryann::Parameters paras;
        paras.Set<unsigned>("L_search", L);
        paras.Set<unsigned>("P_search", L);

        std::vector<std::vector<unsigned>> res;
        std::vector<std::vector<float>> res_dists;
        res.resize(query_num);
        res_dists.resize(query_num);
        for (unsigned i = 0; i < query_num; i++) {
            res[i].resize(K);
            res_dists[i].resize(K);
        }
        std::vector<uint32_t> cmps(query_num, 0);
        std::vector<uint32_t> rec_hops(query_num, 0);
        double total_latency = 0;
        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < query_num; i++) {
            // std::vector<unsigned> tmp(K);
            // auto a = std::chrono::high_resolution_clock::now();
            // std::cout << "here" << std::endl;
            // cmps[i] = index.search(query_load + i * dim, data_load, K, paras, res[i].data(), res_dists[i].data());
            cmps[i] = index.search(query_load + i * dim, data_load, K, paras, res[i].data(), res_dists[i].data(), rec_hops[i]);
            // auto b = std::chrono::high_resolution_clock::now();
            // total_latency += std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            // res.push_back(tmp);
        }
        auto e = std::chrono::high_resolution_clock::now();
        // std::cout << "Query num: " << query_num << " Avg cmps: " << (float)index.total_cmps / (query_num * 1.0)
        //           << std::endl;  // output # of total
                                 // distance computations
                                 // (for debugging
        // get duration milliseconds
        double avg_cmps = 0.0;
        for (auto c : cmps) {
            avg_cmps += c;
        }
        avg_cmps = avg_cmps / (double)query_num;
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        total_latency = (double)diff;
        // std::cout << "compute recall@" << K << " and rderr@" << K << "\n";
        double rderr = ComputeRderr(gt_dists, gt_dim, res_dists, K, metric);
        float recall = ComputeRecall(query_num, K, gt_dim, res, gt_ids);
        float avg_hops = 0.0;
        for (auto h : rec_hops) {
            avg_hops += h;
        }
        avg_hops = avg_hops / (float)query_num;

        std::cout << "search time: " << diff << "\tAvg latency: " << (double)(total_latency) / (double)(query_num)
                  << "ms" << "\tcmps: " << avg_cmps
                  << "\tQSP: " << (float)query_num / (diff / 1000.0) << "\trecall@" << K << " "<< recall << "\trderr" << rderr << "\thops " << avg_hops << "\n";
        if (evaluation_ofs.is_open()) {
            evaluation_ofs << L << "," << (float)query_num / (diff / 1000.0) << "," << avg_cmps
                        << "," << (double)(total_latency) / (double)(query_num) << "," << recall << "," << rderr << "," << avg_hops << std::endl;
        }
        index.total_cmps = 0;
        std::cout << "Recall: " << recall << std::endl;
        // save_result(argv[6], res);
    }
    if (!evaluation_save_path.empty()) {
        evaluation_ofs.close();
    }

    return 0;
}
