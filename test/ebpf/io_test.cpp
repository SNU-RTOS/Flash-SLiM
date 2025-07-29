#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/sdt.h>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <liburing.h>

#define BUF_SIZE 4 * 1024 * 1024 // 1MB buffer size
#define QUEUE_DEPTH 8

void dummy_processing()
{
    int cnt = 0;
    for (volatile int j = 0; j < 10; ++j)
    {
        for (volatile int i = 0; i < 100000000; ++i)
        {
            cnt += 1;
        }
        cnt = 0;
    }
}

void run_mmap(int fd, off_t file_size)
{
    DTRACE_PROBE(mytest, logic_start);

    char *map = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }

    DTRACE_PROBE(mytest, io_start);

    volatile char dummy;
    for (off_t i = 0; i < file_size; i += sysconf(_SC_PAGE_SIZE))
    {
        dummy = map[i];
    }

    DTRACE_PROBE(mytest, io_end);

    dummy_processing();
    munmap(map, file_size);

    DTRACE_PROBE(mytest, logic_end);
}

void run_read(int fd, off_t file_size)
{
    DTRACE_PROBE(mytest, logic_start);

    char buf[BUF_SIZE];
    off_t read_bytes = 0;

    DTRACE_PROBE(mytest, io_start);

    while (read_bytes < file_size)
    {
        ssize_t ret = read(fd, buf, BUF_SIZE);
        if (ret < 0)
        {
            perror("read");
            exit(1);
        }
        if (ret == 0) break; // EOF
        read_bytes += ret;
    }

    DTRACE_PROBE(mytest, io_end);

    dummy_processing();

    DTRACE_PROBE(mytest, logic_end);
}

void run_io_uring(int fd, off_t file_size)
{
    DTRACE_PROBE(mytest, logic_start);

    struct io_uring ring;
    if (io_uring_queue_init(QUEUE_DEPTH, &ring, 0) < 0)
    {
        perror("io_uring_queue_init");
        exit(1);
    }

    char buf[BUF_SIZE];
    off_t offset = 0;

    DTRACE_PROBE(mytest, io_start);

    while (offset < file_size)
    {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (!sqe)
        {
            fprintf(stderr, "io_uring_get_sqe failed\n");
            exit(1);
        }
        io_uring_prep_read(sqe, fd, buf, BUF_SIZE, offset);

        if (io_uring_submit(&ring) < 0)
        {
            perror("io_uring_submit");
            exit(1);
        }

        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring, &cqe) < 0)
        {
            perror("io_uring_wait_cqe");
            exit(1);
        }

        if (cqe->res < 0)
        {
            fprintf(stderr, "io_uring read failed: %s\n", strerror(-cqe->res));
            exit(1);
        }

        offset += cqe->res;
        io_uring_cqe_seen(&ring, cqe);
    }

    DTRACE_PROBE(mytest, io_end);

    io_uring_queue_exit(&ring);

    dummy_processing();

    DTRACE_PROBE(mytest, logic_end);
}

void run_direct_io(int fd, off_t file_size)
{
    DTRACE_PROBE(mytest, logic_start);

    void *buf;
    if (posix_memalign(&buf, sysconf(_SC_PAGESIZE), BUF_SIZE) != 0)
    {
        perror("posix_memalign failed");
        exit(1);
    }

    off_t read_bytes = 0;

    DTRACE_PROBE(mytest, io_start);

    while (read_bytes < file_size)
    {
        ssize_t ret = read(fd, buf, BUF_SIZE);
        if (ret < 0)
        {
            perror("read (O_DIRECT)");
            free(buf);
            exit(1);
        }
        if (ret == 0) break; // EOF
        read_bytes += ret;
    }

    DTRACE_PROBE(mytest, io_end);

    dummy_processing();

    free(buf);

    DTRACE_PROBE(mytest, logic_end);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <mmap|read|io_uring|direct_io>\n", argv[0]);
        exit(1);
    }

    bool direct_io = strcmp(argv[1], "direct_io") == 0;
    int open_flags = O_RDONLY;
    if (direct_io)
    {
        open_flags |= O_DIRECT;
    }

    int fd = open("dummy4g.bin", open_flags);
    if (fd < 0)
    {
        perror("open");
        exit(1);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        perror("fstat");
        close(fd);
        exit(1);
    }
    off_t file_size = sb.st_size;

    auto start_time = std::chrono::steady_clock::now();

    if (strcmp(argv[1], "mmap") == 0)
    {
        run_mmap(fd, file_size);
    }
    else if (strcmp(argv[1], "read") == 0)
    {
        run_read(fd, file_size);
    }
    else if (strcmp(argv[1], "io_uring") == 0)
    {
        run_io_uring(fd, file_size);
    }
    else if (strcmp(argv[1], "direct_io") == 0)
    {
        run_direct_io(fd, file_size);
    }
    else
    {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        close(fd);
        exit(1);
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    printf("[CODE] scope duration (wall-clock): %ld ms\n", duration_ms);

    close(fd);
    return 0;
}
