// BEGIN: 5a8b7c3d3e2f
#include <gtest/gtest.h>
#include <mysteryann/util.h>

// TEST(UtilTest, GenRandomTest) {
//   std::mt19937 rng(0);
//   unsigned addr[5];
//   unsigned size = 5;
//   unsigned N = 10;
//   mysteryann::GenRandom(rng, addr, size, N);
//   for (unsigned i = 0; i < size; ++i) {
//     EXPECT_GE(addr[i], 0);
//     EXPECT_LT(addr[i], N - size);
//   }
// }

// TEST(UtilTest, DataAlignTest) {
//   unsigned point_num = 3;
//   unsigned dim = 3;
//   float *data_ori = new float[point_num * dim];
//   for (unsigned i = 0; i < point_num * dim; i++) {
//     data_ori[i] = i;
//   }
//   float *data_new = mysteryann::data_align(data_ori, point_num, dim);
//   for (unsigned i = 0; i < point_num; i++) {
//     for (unsigned j = 0; j < dim; j++) {
//       EXPECT_EQ(data_new[i * dim + j], i * dim + j);
//     }
//     for (unsigned j = dim; j < dim + 4; j++) {
//       EXPECT_EQ(data_new[i * dim + j], 0);
//     }
//   }
//   delete[] data_new;
// }

TEST(UtilTest, LoadMetaTest) {
  unsigned points_num = 3;
  unsigned dim = 3;
  const char *filename = "test.bin";
  std::ofstream out(filename, std::ios::binary);
  out.write((char *)&points_num, 4);
  out.write((char *)&dim, 4);
  for (unsigned i = 0; i < points_num; i++) {
    for (unsigned j = 0; j < dim; j++) {
      float val = i * dim + j;
      out.write((char *)&val, sizeof(float));
    }
  }
  out.close();
  mysteryann::load_meta<float>(filename, points_num, dim);
  EXPECT_EQ(points_num, 3);
  EXPECT_EQ(dim, 3);
  std::remove(filename);
}

TEST(UtilTest, LoadDataTest) {
  unsigned points_num = 3;
  unsigned dim = 3;
  const char *filename = "test.bin";
  std::ofstream out(filename, std::ios::binary);
  out.write((char *)&points_num, 4);
  out.write((char *)&dim, 4);
  for (unsigned i = 0; i < points_num; i++) {
    for (unsigned j = 0; j < dim; j++) {
      float val = i * dim + j;
      out.write((char *)&val, sizeof(float));
    }
  }
  out.close();
  float *data;
  mysteryann::load_data<float>(filename, points_num, dim, data);
  for (unsigned i = 0; i < points_num; i++) {
    for (unsigned j = 0; j < dim; j++) {
      EXPECT_EQ(data[i * dim + j], i * dim + j);
    }
  }
  delete[] data;
  std::remove(filename);
}
// END: 5a8b7c3d3e2f