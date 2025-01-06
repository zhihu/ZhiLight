#include "bmengine/core/exception.h"
#include "bmengine/logger/kernel_time_trace.hpp"
#include "bmengine/logger/std_log_op.hpp"
#include "private/allocator.h"
#include "private/guard.h"
#include <vector>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>

static int _get_tid() {
    return syscall(SYS_gettid);
}

inline std::uintptr_t convert_uint(void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}
inline void* convert_voidp(std::uintptr_t ptr) {
    return reinterpret_cast<void*>(ptr);
}

inline size_t round_up(size_t num, size_t multiple) {
    return (num + multiple - 1) / multiple * multiple;
}

namespace bmengine {

namespace core {

Memory MemoryAllocator::new_mem(int pos, void* ptr, size_t size) {
    // int pid = _get_tid();
    auto deleter = [=](void* ptr) {
        // BM_ASSERT_EQ(pid, _get_tid(), "");
        this->free(ptr, size);
    };
    Memory ret = std::make_shared<Memory_>(ptr, virtual_dev_id, size, deleter);
    mems.insert(mems.begin() + pos, std::weak_ptr<Memory_>(ret));
    used += size;
    peak = std::max(used, peak);
    return ret;
}

void MemoryAllocator::memory_move(void* dst, void* src, size_t nbytes) {
    std::uintptr_t ptr_dst = convert_uint(dst);
    std::uintptr_t ptr_src = convert_uint(src);
    if (ptr_dst > ptr_src)
        throw std::logic_error("memory move dst > src");
    auto d2d = cudaMemcpyDeviceToDevice;
    bool overlap = ptr_dst + nbytes > ptr_src;
    auto gap_size = ptr_src - ptr_dst;
    if (overlap && gap_size >= mem_reserve / 2) {
        // std::cout << "Handle overlap1\n";
        for (size_t i = 0; i < (nbytes + gap_size - 1) / gap_size; ++i) {
            size_t piece = std::min(nbytes - i * gap_size, gap_size);
            auto src1 = (char*)src + i * gap_size;
            auto dst1 = (char*)dst + i * gap_size;
            BM_CUDART_ASSERT(cudaMemcpyAsync(dst1, src1, piece, d2d, stream));
        }
    } else if (overlap) {
        // std::cout << "Handle overlap2\n";
        for (size_t i = 0; i < (nbytes + mem_reserve - 1) / mem_reserve; ++i) {
            size_t piece = std::min(nbytes - i * mem_reserve, mem_reserve);
            auto src1 = (char*)src + i * mem_reserve;
            auto dst1 = (char*)dst + i * mem_reserve;
            BM_CUDART_ASSERT(cudaMemcpyAsync(move_buf, src1, piece, d2d, stream));
            BM_CUDART_ASSERT(cudaMemcpyAsync(dst1, move_buf, piece, d2d, stream));
        }
    } else {
        BM_CUDART_ASSERT(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    }
}

void* MemoryAllocator::defragmentation() {
    // bmengine::print_demangled_trace(15);
    DeviceGuard guard(dev_id);

    BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    logger::createStartEvent(true, &start, &stop, stream);
    auto last_ptr = convert_uint(base_ptr);
    for (size_t i = 0; i < mems.size(); ++i) {
        Memory mem = mems[i].lock();
        if (mem == nullptr) {
            // tensor is freed in another thread
            BM_ASSERT(mem != nullptr, "Memory was released unexpectedly");
            continue;
        }
        BM_ASSERT(mem != nullptr, "Memory was released unexpectedly");
        auto ptr = convert_uint(mem->ptr);
        if (last_ptr < ptr) {
            memory_move(convert_voidp(last_ptr), mem->ptr, mem->num_bytes);
            mem->ptr = convert_voidp(last_ptr);
        }
        last_ptr += mem->num_bytes;
    }

    float elapsed_ms = logger::destroyDiffEvent(start, stop, stream);
    size_t freed = convert_uint(end_ptr) - last_ptr;
    if (std::getenv("BM_DEBUG_LEVEL") != nullptr) {
        std::cout << "defragmentation: used=" << used / 1024 / 1024
            << "MB, freed=" << freed / 1024 / 1024
            << "MB, cost=" << elapsed_ms << "ms" << std::endl;
        std::cout << std::hex << "base_ptr=" << convert_uint(org_base_ptr)
            << ", end_ptr=" << convert_uint(end_ptr) << std::dec << std::endl;
    }

    return convert_voidp(last_ptr);
}

MemoryAllocator::MemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : dev_id(dev_id), virtual_dev_id(virtual_dev_id), memory_limit(memory_limit), stream(stream) {

    char* print_env = std::getenv("BM_PRINT_MEM_ALLOC_TIME");
    if (print_env != nullptr) {
        print_alloc_time_step = std::atoi(print_env);
    }

    cudaDeviceProp prop;
    BM_CUDART_ASSERT(cudaGetDeviceProperties(&prop, dev_id));

    {
        DeviceGuard guard(dev_id);
        BM_CUDART_ASSERT(cudaMalloc(&move_buf, mem_reserve));
        if (memory_limit > (20L << 30)) {
            memory_limit -= mem_reserve;
        }
        BM_CUDART_ASSERT(cudaMalloc(&org_base_ptr, memory_limit));
        base_ptr = org_base_ptr;
        end_ptr = (char*)base_ptr + memory_limit;
        // std::cout << "BeginPtr:" << base_ptr << ", EndPtr:" << end_ptr << "\n";move_buf
    }
}

MemoryAllocator::MemoryAllocator(MemoryAllocator& p, size_t child_size, cudaStream_t stream)
    : dev_id(p.dev_id), virtual_dev_id(p.virtual_dev_id), memory_limit(child_size), stream(stream) {
    parent = &p;
    // BM_ASSERT(p.mems.empty(), "parent allocator is busy.");
    BM_ASSERT_LE((char*)p.base_ptr, (char*)p.end_ptr - child_size, "Not enough memory");
    move_buf = (char*)p.end_ptr - mem_reserve;
    memory_limit -= mem_reserve;
    end_ptr = move_buf;
    base_ptr = (char*)p.end_ptr - child_size;
    p.end_ptr = base_ptr;
    p.memory_limit -= child_size;
}

MemoryAllocator::~MemoryAllocator() {
    if (parent) {
        if (parent->end_ptr != base_ptr)
            std::cerr << "Free children not match parent end_ptr\n";
        parent->end_ptr = (char*) base_ptr + memory_limit + mem_reserve;
        parent->memory_limit += memory_limit + mem_reserve;
        return;
    }
    try {
        DeviceGuard guard(dev_id);
        BM_CUDART_ASSERT(cudaFree(org_base_ptr));
        BM_CUDART_ASSERT(cudaFree(move_buf));
    } catch (BMEngineException e) { std::cerr << e.what() << std::endl; }
}

void MemoryAllocator::freeze_model_memory() {
    if (mems.empty()) {
        return;
    }
    // BM_ASSERT(frozen_mems.empty(), "freeze_model_memory should be called only once");
    std::lock_guard<std::mutex> lock(mutex);
    // move base_ptr to free memory after model tensors
    base_ptr = defragmentation();
    // move mems(model tensors) to frozen_mems;
    // so they will not be searched again in the following alloc() calls.
    frozen_mems.insert(frozen_mems.end(), mems.begin(), mems.end());
    mems.clear();
}

Memory MemoryAllocator::alloc(size_t num_bytes, size_t round_up_bytes) {
    BM_ASSERT(num_bytes > 0, "num_bytes must be greater than 0");
    BM_ASSERT(round_up_bytes % 512 == 0, "round_up_bytes must be multiple of 512");

    static long count = 0;
    long start = print_alloc_time_step ? logger::get_time_us() : 0;

    num_bytes = round_up(num_bytes, round_up_bytes);

    std::lock_guard<std::mutex> lock(mutex);
    BM_ASSERT_LE( num_bytes, memory_limit - used,
              logger::str_cat("Exceeded memory_limit:", memory_limit / 1024 / 1024, "MB"));

    auto last_ptr = convert_uint(base_ptr);
    for (size_t i = 0; i < mems.size(); ++i) {
        Memory mem = mems[i].lock();
        if (mem == nullptr) {
            // tensor freed in another thread
            // BM_ASSERT(mem != nullptr, "Memory was released unexpectedly");
            continue;
        }
        if (last_ptr + num_bytes <= convert_uint(mem->ptr)) {
            if (print_alloc_time_step && count++ % print_alloc_time_step == 0) {
                long time = logger::get_time_us() - start;
                std::cout << "Alloc1 take " << time << "us, mems=" << mems.size() << "\n";
            }
            return new_mem(i, convert_voidp(last_ptr), num_bytes);
        }
        last_ptr = convert_uint(mem->ptr) + mem->num_bytes;
    }

    // at the end
    if (last_ptr + num_bytes <= convert_uint(end_ptr)) {
        if (print_alloc_time_step && count++ % print_alloc_time_step == 0) {
            long time = logger::get_time_us() - start;
            std::cout << "Alloc2 take " << time << "us, mems=" << mems.size() << "\n";
        }
        return new_mem(mems.size(), convert_voidp(last_ptr), num_bytes);
    }

    if (!allow_gc_) {
        throw std::runtime_error("GC is disabled");
    }

    void* ptr = defragmentation();
    BM_ASSERT(convert_uint(ptr) + num_bytes <= convert_uint(end_ptr),
              logger::str_cat("Exceeded memory_limit:", memory_limit / 1024 / 1024, "MB"));
    return new_mem(mems.size(), ptr, num_bytes);
}

void MemoryAllocator::free(void* ptr, size_t size) {
    // std::cout << "Free " << (size / 1000000) << "MB\n";
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t i = 0; i < mems.size(); ++i) {
        Memory mem = mems[i].lock();
        if (mem == nullptr || mem->ptr == ptr) {
            mems.erase(mems.begin() + i);
            used -= size;
            return;
        }
    }
    for (int i = int(frozen_mems.size()) - 1; i >= 0; --i) {
        Memory mem = frozen_mems[i].lock();
        if (mem == nullptr || mem->ptr == ptr) {
            frozen_mems.erase(frozen_mems.begin() + i);
            used -= size;
            return;
        }
    }
    BM_EXCEPTION("Memory was not allocated by this allocator");
}

size_t MemoryAllocator::used_memory() const {
    return used;
}

size_t MemoryAllocator::peak_memory() const {
    return peak;
}

DirectMemoryAllocator::DirectMemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : MemoryAllocator(dev_id, virtual_dev_id, 1024, stream) {
    this->memory_limit = memory_limit;
    std::cout << "Use DirectMemoryAllocator" << std::endl;
}

Memory DirectMemoryAllocator::alloc(size_t size, size_t round_up_bytes) {
    if (round_up_bytes > 4096) {
        size_t old_size = size;
        size = round_up(size, round_up_bytes);
        // std::cout << "round up from " << old_size << " to " << size << ", round_up_bytes=" <<
        // round_up_bytes << std::endl;
    }
    void* ptr;
    {
        DeviceGuard guard(dev_id);
        BM_CUDART_ASSERT(cudaMalloc(&ptr, size));
    }
    // std::cout << "dev:" << dev_id << " alloc " << ptr << ", size=" << size << std::endl;
    used += size;
    peak = std::max(used, peak);
    return std::make_shared<Memory_>(
        ptr, virtual_dev_id, size, [this, size](void* ptr) { this->free(ptr, size); });
}

void DirectMemoryAllocator::free(void* ptr, size_t size) {
    bool free_early = std::getenv("DIRECT_ALLOC_FREE_EARLY") != nullptr;
    if (free_early) {
        DeviceGuard guard(dev_id);
        BM_CUDART_ASSERT(cudaDeviceSynchronize());
        BM_CUDART_ASSERT(cudaFree(ptr));
    } else {
        frees.push_back(ptr);
    }
    used -= size;
}

void DirectMemoryAllocator::free_session() {
    DeviceGuard guard(dev_id);
    for (void* ptr : frees) {
        std::cerr << "dev:" << dev_id << " free " << ptr << std::endl;
        BM_CUDART_ASSERT(cudaFree(ptr));
    }
    frees.clear();
}

} // namespace core

} // namespace bmengine
