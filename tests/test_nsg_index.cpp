

#include <mysteryann/index_nsg.h>
#include <mysteryann/util.h>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << argv[0] << " data_file nn_graph_path L R C save_graph_file"
              << std::endl;
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

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);
  std::string dist_metric(argv[7]);
  mysteryann::Metric metric;
  if (dist_metric == "l2") {
    metric = mysteryann::L2;
  } else if (dist_metric == "cosine") {
    metric = mysteryann::L2;
    for (size_t i = 0; i < points_num; ++i) {
      mysteryann::normalize<float>(data_load, dim);
    }
  } else if (dist_metric == "ip") {
    metric = mysteryann::INNER_PRODUCT;
  } else {
    std::cerr << "Unknon distance" << std::endl;
    exit(1);
  }
  data_load = mysteryann::data_align(data_load, points_num, dim);//one must
  // align the data before build
  mysteryann::IndexNSG index(dim, points_num, metric, nullptr);

  auto s = std::chrono::high_resolution_clock::now();
  mysteryann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  std::cout << nn_graph_path << std::endl;
  std::cout << "begin build\n";
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  index.Save(argv[6]);

  return 0;
}
