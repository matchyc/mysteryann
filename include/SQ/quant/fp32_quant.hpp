#pragma once

#include "../common.hpp"
#include "../memory.hpp"
#include "../simd/distance.hpp"

namespace glass {

    /**
     * FP32 量化：使用原始数据（32位 float point 类型）
     */
    template<Metric metric, int DIM = 0>
    struct FP32Quantizer {
        using data_type = float;
        constexpr static int kAlign = 16; // 维数按 kAlign 对齐
        int d, d_align; // 原始数据维数和对齐后的维数
        int64_t code_size; // 量化编码后的单个向量大小（字节数），FP32 中 code_size 等于 d_align * 4
        char *codes = nullptr; // 量化编码后的向量数组指针

        FP32Quantizer() = default;

        explicit FP32Quantizer(int dim) : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

        ~FP32Quantizer() { free(codes); }

        void train(const float *data, int64_t n) {
            codes = (char *) alloc2M(n * code_size);
            for (int64_t i = 0; i < n; ++i) { // 对每个数据点进行标量量化
                encode(data + i * d, get_data(i));
            }
        }

        // 由于 FP32 量化使用原始数据，因此 encode 直接进行拷贝
        void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

        char *get_data(int u) const { return codes + u * code_size; }

        template<typename Pool>
        void reorder(const Pool &pool, const float *, int *dst, int k) const {
            for (int i = 0; i < k; ++i) {
                dst[i] = pool.id(i);
            }
        }

        template<int DALIGN = do_align(DIM, kAlign)>
        struct Computer {
            using dist_type = float;
            constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
            const FP32Quantizer &quant;
            float *q = nullptr;

            Computer(const FP32Quantizer &quant, const float *query)
                    : quant(quant), q((float *) alloc64B(quant.d_align * 4)) {
                std::memcpy(q, query, quant.d * 4);
            }

            ~Computer() { free(q); }

            dist_type operator()(int u) const {
                return dist_func(q, (data_type *) quant.get_data(u), quant.d);
            }

            void prefetch(int u, int lines) const {
                mem_prefetch(quant.get_data(u), lines);
            }
        };

        auto get_computer(const float *query) const {
            return Computer<0>(*this, query);
        }
    };

} // namespace glass
