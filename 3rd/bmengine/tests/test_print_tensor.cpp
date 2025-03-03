#include "bmengine/core/core.h"
#include <thread>
#include <iostream>

int main() {
    bmengine::core::Engine engine({
        { 3, 5ll * 1024 * 1024 * 1024 },
    });
    auto ctx = engine.create_context({ 0 });
    bmengine::core::WithDevice device(ctx, 0);

    {
        auto inp = ctx.tensor({ 4, 16, 16 }, bmengine::core::DataType::kHalf);
        uint16_t buffer[4][16][16];
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    buffer[k][i][j] = 15360 + (k * 16 + i) * 16 + j;
                }
            }
        }
        inp.from_buffer(buffer);
        std::cout << inp << std::endl;
    }

    {
        auto inp = ctx.tensor({ 4, 16, 16 }, bmengine::core::DataType::kFloat);
        float buffer[4][16][16];
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    buffer[k][i][j] = ((k * 16 + i) * 16 + j) * 0.01;
                }
            }
        }
        inp.from_buffer(buffer);
        std::cout << inp << std::endl;
    }
    return 0;
}