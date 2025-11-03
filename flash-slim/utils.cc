/* Copyright 2025 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "utils.h"


namespace flash_slim::util
{
    // Function to detect which cores the process is actually running on
    void detect_active_cores(std::vector<int> &cores)
    {
        // Try to read process affinity mask
        cpu_set_t mask;
        CPU_ZERO(&mask);

        if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        {
            for (int i = 0; i < CPU_SETSIZE; i++)
            {
                if (CPU_ISSET(i, &mask))
                {
                    cores.push_back(i);
                }
            }
        }

        std::cout << "[INFO] Process is running on cores: ";
        for (int core : cores)
        {
            std::cout << core << " ";
        }
        std::cout << std::endl;
    }

    void set_cores_for_inference_and_io(const std::vector<int> &active_cores, std::vector<int> &cores_to_use_inference, std::vector<int> &cores_to_use_io, int requested_threads)
    {
        const size_t n_cores = active_cores.size();
        if (requested_threads > 0 && n_cores > 0)
        {
            const size_t take = std::min(static_cast<size_t>(requested_threads), n_cores);
            // inference: take the last `take` cores from active_cores (preserve order)
            if (take > 0)
            {
                cores_to_use_inference.assign(active_cores.end() - take, active_cores.end());
            }
            // I/O: the remaining prefix
            const size_t keep = (n_cores > take) ? (n_cores - take) : 0;
            if (keep > 0)
            {
                cores_to_use_io.assign(active_cores.begin(), active_cores.begin() + keep);
            }
        }

        std::cout << "[INFO] Cores to use for inference thread: ";
        std::sort(cores_to_use_inference.begin(), cores_to_use_inference.end());
        for (const auto &core : cores_to_use_inference)
            std::cout << core << " ";
        std::cout << std::endl;

        std::cout << "[INFO] Cores to use for I/O thread: ";
        std::sort(cores_to_use_io.begin(), cores_to_use_io.end());
        for (const auto &core : cores_to_use_io)
            std::cout << core << " ";
        std::cout << std::endl;
    }

    void set_affinity_to_cores(const std::vector<int> &cores)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int core : cores)
        {
            CPU_SET(core, &cpuset);
        }
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
        {
            perror("Failed to set affinity");
        }
    }

    void run_thread_with_affinity_and_join(const std::function<void()> &fn,
                                           const std::vector<int> &cores)
    {
#if defined(__linux__)
        std::thread t([fn, &cores]()
                      {
            if (!cores.empty())
            {
                set_affinity_to_cores(cores);
            }
            fn(); });
        t.join();
#else
        // Fallback on non-Linux: just run synchronously on the calling thread.
        fn();
#endif
    }

    // --------------------------------------------------------------------------
    // Utility to get current page cache size from /proc/meminfo (Linux only)
    // --------------------------------------------------------------------------
    void print_current_page_cache_kb()
    {
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open())
        {
            std::cerr << "Failed to open /proc/meminfo\n";
            return;
        }

        std::string key, unit;
        size_t value = 0;
        bool found = false;

        while (meminfo >> key >> value >> unit)
        {
            if (key == "Cached:")
            {
                found = true;
                break;
            }
        }

        if (found)
        {
            std::cout << "[INFO] Current Page Cache: " << value << " kB" << std::endl;
        }
        else
        {
            std::cout << "[INFO] Current Page Cache: unknown" << std::endl;
        }
    }

    int drop_page_cache()
    {
        // 1) sync filesystem buffers
        ::sync();

        // 2) drop_caches=3 (pagecache + dentries + inodes)
        const char *path = "/proc/sys/vm/drop_caches";
        int fd = ::open(path, O_WRONLY);
        if (fd < 0)
        {
            std::cerr << "[ERR] open(" << path << ") failed: "
                      << std::strerror(errno)
                      << " (errno=" << errno << ")\n";
            return 1;
        }

        const char *val = "3\n";
        ssize_t n = ::write(fd, val, std::strlen(val));
        int saved = errno;
        ::close(fd);

        if (n < 0)
        {
            std::cerr << "[ERR] write(" << path << ") failed: "
                      << std::strerror(saved)
                      << " (errno=" << saved << ")\n";
            return 2;
        }

        return 0;
    }
} // namespace flash_slim::util