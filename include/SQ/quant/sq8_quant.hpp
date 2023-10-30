#pragma once

#include "../common.hpp"
#include "../memory.hpp"
#include "../neighbor.hpp"
#include "../quant/fp32_quant.hpp"
#include "../simd/distance.hpp"

#include <cmath>
#include <vector>
#include <iostream>
/**
 * SQ8：对向量做标量量化，浮点数表示转为uint8型表示，4字节->1字节
 */
namespace glass {

    template<Metric metric, int DIM = 0>
    struct SQ8Quantizer {
        using data_type = uint8_t;
        constexpr static int kAlign = 16; // 按 kAlign 个维度进行对齐（对齐后的维度是 kAlign 的整数个），kAlign = 16，一个数据占 4 字节，因此是 64 字节对齐
        int N;
        int d, d_align;
        int64_t code_size;  // 一个向量量化后的字节数，SQ8 中 code_size = d_align
        char *codes = nullptr; // 量化后的数据
        std::vector<float> mx, mi, dif;

        SQ8Quantizer() = default;

        explicit SQ8Quantizer(int dim)
                : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
                  mx(d_align, -HUGE_VALF), mi(d_align, HUGE_VALF), dif(d_align) {}

        ~SQ8Quantizer() { free(codes); }

        void train(const float *data, int n) {
            this->N = n;
	        std::cout<< "归一化" << std::endl;
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < d; ++j) {
                    mx[j] = std::max(mx[j], data[i * d + j]); // 维度 j 的最大值
                    mi[j] = std::min(mi[j], data[i * d + j]); // 维度 j 的最小值
                }
            }
            for (int64_t j = 0; j < d; ++j) {
                dif[j] = mx[j] - mi[j]; // 维度 j 的最大差值
            }
            for (int64_t j = d; j < d_align; ++j) {
                dif[j] = mx[j] = mi[j] = 0; // 将由于对齐增加的维度的最大差值设为 0
            }
            codes = (char *) alloc2M((size_t) n * code_size);
            for (int i = 0; i < n; ++i) {
                encode(data + i * d, get_data(i));
            }
        }

        char *get_data(int u) const { return codes + u * code_size; }

        /**
         * 量化编码，float 压缩为 uint8，量后后空间开销更小，距离计算时 scale 回原浮点空间
         */
        void encode(const float *from, char *to) const {
            for (int j = 0; j < d; ++j) {
                float x = (from[j] - mi[j]) / dif[j]; // 归一化，x 取值范围为 0 到 1
                if (x < 0) {
                    x = 0.0;
                }
                if (x > 1.0) {
                    x = 1.0;
                }
                uint8_t y = x * 255; // scale x 为 y， 取值范围为 0 到 255，即 uint8_t 取值范围
                to[j] = y;
            }
        }

        void save_code(std::string quant_file)
        {
            printf("save quant\n");
            FILE *F = fopen(quant_file.c_str(), "wb");
            if (F == nullptr) {
                printf("can't open quant save file\n");
                return;
            }
            //  code_size 是一个对齐后向量的字节数
            fwrite(&N, sizeof(int), 1, F);
            fwrite(codes, sizeof(char), (size_t)N * code_size, F);
            fwrite(mx.data(), sizeof(float), (size_t)d_align, F);
            fwrite(mi.data(), sizeof(float), (size_t)d_align, F);
            fwrite(dif.data(), sizeof(float), (size_t)d_align, F);
            fclose(F);
            printf("save quant finish.\n");
        }

        bool load_code(std::string quant_file)
        {
            printf("load quant\n");
            if(codes != nullptr) {
                printf("无需重复加载\n");
                return true;
            }
            FILE * F = nullptr;
	        F = fopen(quant_file.c_str(), "rb");
	        if(F == nullptr){
		        printf("codes not found\n");
		        return false;
	        }
            
            fread(&N, sizeof(int), 1, F);

            codes = (char *)alloc2M((size_t)N * code_size);
            fread(codes, sizeof(char), (size_t)N * code_size, F);

            float *buffer = new float[d_align];

		    fread(buffer, sizeof(float), d_align, F);
            mx = std::vector<float>(buffer, buffer + d_align);

            fread(buffer, sizeof(float), d_align, F);
            mi = std::vector<float>(buffer, buffer + d_align);

            fread(buffer, sizeof(float), d_align, F);
            dif = std::vector<float>(buffer, buffer + d_align);

            fclose(F);
            delete[] buffer;
            return true;
        }

        template<typename Pool>
        void reorder(const Pool &pool, const float * /**q*/, int *dst, int k) const {
            for (int i = 0; i < k; ++i) {
                dst[i] = pool.id(i);
            }
        }

        template<int DALIGN = do_align(DIM, kAlign)>
        struct Computer {
            using dist_type = float;
            constexpr static auto dist_func =
                    metric == Metric::L2 ? L2SqrSQ8_ext : IPSQ8_ext;
            const SQ8Quantizer &quant;
            float *q;
            const float *mi, *dif;

            Computer(const SQ8Quantizer &quant, const float *query)
                    : quant(quant), q((float *) alloc64B(quant.d_align * 4)),
                      mi(quant.mi.data()), dif(quant.dif.data()) {
                std::memcpy(q, query, quant.d * 4);
            }

            ~Computer() { free(q); }

            dist_type operator()(int u) const {
                return dist_func(q, (data_type *) quant.get_data(u), quant.d_align, mi,
                                 dif);
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
