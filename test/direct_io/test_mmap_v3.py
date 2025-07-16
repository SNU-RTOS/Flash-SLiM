#!/usr/bin/env python3
import os
import mmap
import time
import subprocess
import os
import time
import ctypes

FILE_PATH = "test_4GB.bin"
FILE_SIZE = 1 * 1024 * 1024 * 1024  # 4GB
BLOCK_SIZE = 4096
ALIGNMENT  = 4096

def drop_caches():
    """
    Linux에서 페이지 캐시를 비우는 함수.
    관리자 권한(또는 sudo) 필요.
    """
    try:
        print("drop cache")
        subprocess.run(["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] drop_caches 실패: {e}")
        print("sudo 권한이 없거나, Linux 환경이 아닐 수 있습니다.")

def create_4gb_file(path, size=FILE_SIZE):
    """
    4GB 짜리 파일 생성 (간단히 truncate).
    """
    print(f"Creating file {path} ({size} bytes)...")
    with open(path, "wb") as f:
        f.truncate(size)
    print("File creation complete.\n")

def get_current_memory_usage_mb():
    """
    현재 프로세스가 사용하는 물리 메모리(RSS)를 MB 단위로 반환.
    /proc/self/status에서 VmRSS 항목을 파싱 (Linux 전용).
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # 예: "VmRSS:    123456 kB"
                    parts = line.split()
                    if len(parts) >= 3:
                        rss_kb = float(parts[1])  # 단위: kB
                        return rss_kb / 1024.0    # MB로 변환
    except FileNotFoundError:
        pass
    return 0.0

def measure_mmap_io(path):
    """
    mmap 기반으로 파일을 전부 읽은 뒤 걸리는 시간을 측정.
    (mmap의 경우, 메모리에 전부 저장할 필요 없이
     매핑된 영역에서 직접 읽습니다.
     필요한 경우 별도 버퍼에 복사해도 되지만, 여기서는 기존 로직 유지)
    """
    start = time.time()
    
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        total_read = 0
        while total_read < FILE_SIZE:
            _ = mm[total_read:total_read + BLOCK_SIZE]
            total_read += BLOCK_SIZE
            if(total_read % (1024 * 1024 * 1024) == 0):
                print(f"[mmap] Current RSS usage: {get_current_memory_usage_mb():.2f} MB")
        end = time.time()  # 시간 측정 위치를 mmap 해제 직전으로 이동
        mm.close()
    
    duration = end - start
    return duration

def measure_read_io(path):
    """
    일반적인 read() 기반 I/O로 파일을 전부 읽은 뒤,
    읽어들인 데이터를 메모리(크기가 FILE_SIZE인 bytearray)에 저장.
    """
    start = time.time()
    
    # 4GB 전체를 담을 수 있는 버퍼 (매우 커질 수 있음)
    big_buffer = bytearray(FILE_SIZE)
    offset = 0
    
    with open(path, "rb") as f:
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk:
                break
            # 읽은 데이터를 big_buffer에 저장
            big_buffer[offset:offset+len(chunk)] = chunk
            offset += len(chunk)
    
    end = time.time()
    duration = end - start
    print(f"[read()] Current RSS usage: {get_current_memory_usage_mb():.2f} MB")
    return duration

def measure_direct_io(path):
    """
    Direct I/O (O_DIRECT) 방식으로 파일을 전부 읽은 뒤,
    읽어들인 데이터를 메모리에 저장.
    """
    # Direct I/O로 파일 열기
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    
    # Direct I/O는 버퍼가 파일시스템 블록 크기에 정렬되어 있어야 함
    buf = bytearray(BLOCK_SIZE)
    buf_view = memoryview(buf)
    
    # 4GB 저장용 버퍼
    big_buffer = bytearray(FILE_SIZE)
    offset = 0
    
    start = time.time()
    while True:
        # os.readv()의 반환값은 '읽은 총 바이트 수(int)'
        bytes_read = os.readv(fd, [buf_view])
        if bytes_read == 0:
            break
        # 읽은 데이터를 big_buffer에 저장
        big_buffer[offset:offset+bytes_read] = buf[:bytes_read]
        offset += bytes_read
    
    end = time.time()
    
    os.close(fd)
    
    duration = end - start
    print(f"[O_DIRECT] Current RSS usage: {get_current_memory_usage_mb():.2f} MB")
    return duration

def measure_direct_io_aligned(path):
    """
    Direct I/O (O_DIRECT)로 파일 전체를 읽고, 페이지 정렬된 버퍼에 저장한 뒤
    최종적으로 큰 버퍼에 복사하는 예시 코드 (posix_memalign 사용).
    """
    # (1) Direct I/O로 파일 열기
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)

    # (2) posix_memalign으로 페이지 크기에 맞춰 버퍼 할당
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    posix_memalign = libc.posix_memalign
    posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    posix_memalign.restype = ctypes.c_int

    # aligned 메모리를 받을 포인터
    aligned_ptr = ctypes.c_void_p(None)
    ret = posix_memalign(ctypes.byref(aligned_ptr), ALIGNMENT, BLOCK_SIZE)
    if ret != 0:
        raise OSError(ret, "posix_memalign failed")

    # Python 객체에 매핑
    AlignedArray = (ctypes.c_char * BLOCK_SIZE)
    aligned_buf = AlignedArray.from_address(aligned_ptr.value)
    buf_view = memoryview(aligned_buf)

    # 최종적으로 모든 데이터를 저장할 큰 버퍼 (bytearray)
    big_buffer = bytearray(FILE_SIZE)
    offset = 0
    
    start = time.time()
    
    try:
        while True:
            # os.readv() 사용
            bytes_read = os.readv(fd, [buf_view])
            if bytes_read == 0:
                break
            # aligned_buf → big_buffer로 복사
            big_buffer[offset : offset + bytes_read] = aligned_buf[:bytes_read]
            offset += bytes_read
    finally:
        os.close(fd)
        libc.free(aligned_ptr)  # posix_memalign으로 할당한 메모리는 수동 해제
    
    end = time.time()
    
    duration = end - start
    print(f"[O_DIRECT aligned] Read done in {duration:.2f} seconds.")
    print(f"[O_DIRECT aligned] Current RSS usage: {get_current_memory_usage_mb():.2f} MB")
    return duration

def run_io_test(test_name, func, drop_cache_first=True):
    """
    주어진 I/O 테스트 함수(func)를 호출하고,
    걸린 시간을 출력하는 헬퍼 함수.
    """
    print(f"\n=== {test_name} ===")
    
    if drop_cache_first:
        drop_caches()
        time.sleep(10)  # 캐시가 완전히 비워지도록 잠깐 대기
    
    elapsed = func(FILE_PATH)
    print(f"{test_name} time: {elapsed:.2f} sec")

def main():
    # 1. 4GB 파일 존재 검사 후 필요시 생성
    if not os.path.exists(FILE_PATH):
        create_4gb_file(FILE_PATH, FILE_SIZE)
    else:
        print("4GB 테스트 파일이 이미 존재합니다.\n")

    # 2. 테스트 시나리오 실행
    #    - (with cache) 테스트는 drop_caches=False 로 호출
    run_io_test("mmap I/O test (cold)", measure_mmap_io, drop_cache_first=True)
    run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=True)
    run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=True)
    # run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=False)
    # run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=False)
    # run_io_test("mmap I/O test (with cache)", measure_mmap_io, drop_cache_first=False)
    
    
    
    # run_io_test("read() I/O test (cold)", measure_read_io, drop_cache_first=True)
    # run_io_test("read() I/O test (with cache)", measure_read_io, drop_cache_first=False)
    
    run_io_test("Direct I/O test (cold)", measure_direct_io_aligned, drop_cache_first=True)
    # run_io_test("Direct I/O test (with cache)", measure_direct_io, drop_cache_first=False)
    # run_io_test("Direct I/O test (with cache)", measure_direct_io, drop_cache_first=False)
    # run_io_test("Direct I/O test (with cache)", measure_direct_io, drop_cache_first=False)
    

if __name__ == "__main__":
    main()
