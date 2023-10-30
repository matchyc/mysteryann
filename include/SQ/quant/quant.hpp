#pragma once // 保证头文件只被编译一次

#include <string>
#include <unordered_map>

#include "../quant/fp32_quant.hpp"
#include "../quant/sq4_quant.hpp"
#include "../quant/sq8_quant.hpp"

namespace glass {

enum class QuantizerType { FP32, SQ8, SQ4 };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ4;
  return 42;
}();

} // namespace glass
