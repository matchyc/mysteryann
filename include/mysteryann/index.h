//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef mysteryann_INDEX_H
#define mysteryann_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"

namespace mysteryann {

class Index {
 public:
  explicit Index(const size_t dimension, const size_t n, Metric metric);


  virtual ~Index();

  virtual void Build(size_t n, const float *data, const Parameters &parameters) = 0;

  virtual void BuildBipartite(size_t n_sq, const float *sq_data, size_t n_bp, const float *bp_data, const Parameters &parameters) = 0;
  // virtual void Search(
  //     const float *query,
  //     const float *x,
  //     size_t k,
  //     const Parameters &parameters,
  //     unsigned *indices);
  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices, float* res_dists) = 0;

  virtual void Save(const char *filename) = 0;

  virtual void Load(const char *filename) = 0;

  inline bool HasBuilt() const { return has_built; }

  inline size_t GetDimension() const { return dimension_; };

  inline size_t GetSizeOfDataset() const { return nd_; }

  inline const float *GetDataset() const { return data_; }

  inline const float *GetSampledQuerySet() const {return data_sq_; }

  inline const float *GetBasePointSet() const {return data_bp_; }
  
 protected:
  const size_t dimension_;
  const float *data_;
  const float *data_sq_ = nullptr;
  const float *data_bp_ = nullptr;
  size_t nd_;
  size_t nd_sq_;
  bool has_built;
  bool bipartite_ = false;
  Distance* distance_;
  Metric metric_;
};

}

#endif //mysteryann_INDEX_H
