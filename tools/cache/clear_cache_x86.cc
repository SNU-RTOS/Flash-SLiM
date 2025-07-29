#include <immintrin.h>  // clflush 포함
#include <iostream>

constexpr size_t CACHE_SIZE = 8 * 1024 * 1024;
char buffer[CACHE_SIZE];

void flush_cache_clflush() {
    for (size_t i = 0; i < CACHE_SIZE; i += 64) {
        _mm_clflush(&buffer[i]);
    }
}

int main() {
    std::cout << "Flushing CPU cache using clflush..." << std::endl;
    flush_cache_clflush();
    std::cout << "Cache flushed!" << std::endl;
    return 0;
}
