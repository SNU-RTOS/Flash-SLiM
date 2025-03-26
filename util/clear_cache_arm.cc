#include <iostream>

void flush_cache_arm() {
    char buffer[4 * 1024 * 1024]; // 4MB 크기의 메모리 블록
    __builtin___clear_cache(buffer, buffer + sizeof(buffer));
}

int main() {
    std::cout << "Flushing CPU cache on ARM..." << std::endl;
    flush_cache_arm();
    std::cout << "Cache flushed!" << std::endl;
    return 0;
}
