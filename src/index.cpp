//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <mysteryann/index.h>
namespace mysteryann {
Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
    : dimension_(dimension), nd_(n), has_built(false) {
    metric_ = metric;
    switch (metric_) {
        case mysteryann::L2:
            distance_ = new DistanceL2();
            std::cout << "Inside using L2 distance." << std::endl;
            break;
        case mysteryann::COSINE:
        case mysteryann::INNER_PRODUCT:
            distance_ = new DistanceInnerProduct();
            std::cout << "Inside using IP distance." << std::endl;
            break;
        default:
            distance_ = new DistanceL2();
            std::cout << "Using L2 distance." << std::endl;
            break;
    }
}
Index::~Index() {}
}  // namespace mysteryann
