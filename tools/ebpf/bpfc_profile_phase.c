#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/blkdev.h>

// ── VM_FAULT_* 플래그 (커널 버전에 따라 상수값 다를 수 있음) ──
#define VM_FAULT_MAJOR 4
#define VM_FAULT_NOPAGE 256
#define VM_FAULT_LOCKED 512
#define VM_FAULT_RETRY 1024
// NOTE: These VM_FAULT_* constants may vary across kernel versions.
//       Verify these against your target kernel headers if you see
//       unexpected classification of page-fault types.

/* ------------------------------------------------------------------
 * BPF type & map declarations
 * - This section groups all C struct/type definitions first, then the
 *   perf outputs and BPF maps. The comments explain key types and map
 *   key/value semantics so new readers can follow what is collected.
 * ------------------------------------------------------------------ */

/* --- Event type sent to userspace via perf buffer ---
 * event_t: simple notification for phase START/END
 *  - ts   : timestamp in ns (monotonic)
 *  - pid  : tgid (process id) packed as u32 in userspace
 *  - kind : 0=start, 1=end
 *  - phase: null-terminated phase name (user-provided)
 *
 * NOTE: `phase` is limited to 64 bytes; long strings will be truncated
 *       when read into this struct (userspace should handle truncation).
 */
struct event_t
{
    u64 ts;         // ns, monotonic
    u32 pid;        // tgid
    int kind;       // 0 = start, 1 = end
    char phase[64]; // update at end or check
};

/* --- Page-fault accounting (per TID entry; summed to TGID in Python) ---
 * pf_stat_t accumulates counts and elapsed ns for different PF types.
 * Note: keys in the pfstat map are u64 pid_tgid (upper 32 bits = tgid,
 * lower 32 bits = tid). The reader filters by tgid when summing.
 */
struct pf_stat_t
{
    u64 start_ns;        // PF entry ts (0 means inactive)
    u64 major_cnt;       // major faults count
    u64 minor_cnt;       // minor faults (no page) count
    u64 minor_retry_cnt; // minor-retry faults count
    u64 major_ns;        // accumulated major handling time (ns)
    u64 minor_ns;        // accumulated minor (NOPAGE) handling time (ns)
    u64 minor_retry_ns;  // accumulated minor (RETRY) handling time (ns)
    u64 readahead_cnt;   // readahead invocation count
};

/* --- Read/Write syscall timing (per TID entry) ---
 * - start_ns is used to record sys_enter timestamp; exit adds delta.
 * - map key is u64 pid_tgid (tid-scoped) to avoid races between threads.
 */
struct rw_stat_t
{
    u64 read_cnt;
    u64 read_ns;
    u64 read_bytes;

    u64 write_cnt;
    u64 write_ns;
    u64 write_bytes;

    u64 start_ns; // sys_enter timestamp storage
};

/* --- Block I/O helper structs ---
 * block_start_t : stored at issue time keyed by bio (dev,sector)
 *  - pid_tgid records the originating pid/tid so completion can be
 *    attributed back to the issuing thread.
 * block_stat_t  : accumulator per pid_tgid for completed requests
 * block_io_interval_t : optional interval payload (perf) used for
 *    more advanced post-processing (interval-level events).
 */

struct block_start_t
{
    u64 start_ns;
    u64 pid_tgid; // pid/tid captured at start time (u64: [63:32]=tgid)
    char op;      // 'R' or 'W' (from rwbs[0])
    u64 bytes;    // request size captured at start
};

struct block_stat_t
{
    u64 time_ns;     // total elapsed between issue and complete
    u64 count;       // number of completed requests
    u64 read_bytes;  // bytes completed for reads
    u64 write_bytes; // bytes completed for writes
};

struct bio_key_t
{
    u64 dev;
    u64 sector;
};

struct block_io_interval_t
{
    u64 pid_tgid; // pid/tid for the interval
    u64 start_ns; // interval start (ns)
    u64 end_ns;   // interval end (ns)
    u64 bytes;    // bytes processed in interval
    char op;      // 'R' or 'W'
};

struct sched_stat_t
{
    u64 runtime_ns;
    u64 wait_ns;
};

/* --- Perf output endpoints (userspace consumers) --- */
BPF_PERF_OUTPUT(events); // phase start/end events -> Python handler
/* PERF OUTPUT declared for detailed intervals. Currently optional: the
 * BPF code in this file does not submit `intervals` in all code paths.
 * Keep this endpoint if you later want per-IO interval payloads to be
 * submitted to userspace; otherwise it can be removed.
 */
BPF_PERF_OUTPUT(intervals); // optional detailed I/O intervals -> Python

/* --- BPF Maps ---
 * Naming & key conventions:
 *  - phase_ts_pid : key=u32 (tgid) -> value=u64 start timestamp
 *  - pfstat, rwstat, blockstat, schedstat : key=u64 (pid_tgid)
 *      where upper 32 bits = tgid, lower 32 bits = tid. Python code
 *      filters by tgid and sums per-thread entries into a per-process
 *      aggregate for reporting.
 *
 * Python usage hint: when iterating map keys from userspace you can
 * extract tgid with: `tgid = (key_u64 >> 32) & 0xffffffff` before
 * deciding whether to include the entry in a per-process sum.
 */
/* --- Maps: purpose + key conventions (detailed) ---
 *
 * phase_ts_pid (u32 -> u64)
 *  - Tracks whether a phase gate is active for a given process (tgid).
 *  - Key: u32 tgid (process). Value: start timestamp (ns) when logic_start
 *    was observed. Cleared at logic_end. Used as a fast membership test
 *    inside many probes to avoid unnecessary work when no phase is active.
 */
BPF_HASH(phase_ts_pid, u32, u64);

/*
 * pfstat (u64 pid_tgid -> pf_stat_t)
 *  - Per-thread (tid-scoped) page-fault accounting entries are stored here.
 *  - Key: u64 pid_tgid (upper 32 bits = tgid, lower 32 bits = tid).
 *  - Rationale: kprobe/kretprobe handlers execute on the faulting thread,
 *    so keeping per-tid entries avoids races across threads. Python later
 *    reads and reduces per-tid entries into a per-tgid summary for output.
 */
BPF_HASH(pfstat, u64, struct pf_stat_t);

/*
 * rwstat (u64 pid_tgid -> rw_stat_t)
 *  - Tracks syscall enter/exit timing for read/write per-thread.
 *  - start_ns is set on sys_enter and used on exit to compute elapsed ns.
 */
BPF_HASH(rwstat, u64, struct rw_stat_t);

/*
 * readahead_count (u32 tgid -> u64)
 *  - Simple per-process counter incremented from the ondemand_readahead kprobe.
 *  - Keyed by tgid because readahead is attributed to the process scope.
 *
 * DECLARED BUT UNUSED: this map exists for historical reasons. The current
 * implementation increments readahead counters inside `pfstat` entries
 * (pf_stat_t.readahead_cnt) and userspace reads that field. Consider
 * removing this separate map or using it consistently.
 */
BPF_HASH(readahead_count, u32, u64);

/*
 * block_start : keyed by bio (dev,sector) -> block_start_t
 *  - At BIO issue time we capture a small blob (start_ns, pid_tgid, bytes, op)
 *    indexed by the bio key. On completion we lookup using the same key.
 *  - Note: bio keys can collide or be reused by the kernel in some cases;
 *    code defensively checks lookups and deletes entries after completion.
 *  - NOTE: `bio_key_t` layout and alignment must match the kernel tracepoint
 *    usage; key reuse/collisions are mitigated by defensive lookup logic.
 */
BPF_HASH(block_start, struct bio_key_t, struct block_start_t);

/*
 * blockstat (u64 pid_tgid -> block_stat_t)
 *  - Accumulates completed block I/O statistics per issuing pid/tid.
 *  - Key: u64 pid_tgid so that completions are attributed back to the
 *    issuing thread. Python will combine per-tid entries into per-tgid
 *    aggregates when generating the report.
 */
BPF_HASH(blockstat, u64, struct block_stat_t);

/*
 * schedstat (u64 pid_tgid -> sched_stat_t)
 *  - Runtime/wait counters are accumulated via sched tracepoints and stored
 *  - per pid_tgid (thread-scoped) for the same reasons as other per-thread
 *    counters. Summation into a per-process view is done in userspace.
 */
BPF_HASH(schedstat, u64, struct sched_stat_t);

/* 02 BPF Helpers */
static __always_inline u32 get_tgid(void)
{
    // Return the current process id (tgid). Use this in probes where we
    // only need process-level membership (phase gate checks etc.).
    return (u32)(bpf_get_current_pid_tgid() >> 32);
}
static __always_inline u64 get_pid_tgid(void)
{
    // Return the raw 64-bit pid_tgid as produced by bpf_get_current_pid_tgid().
    // Layout: upper 32 bits = tgid, lower 32 bits = tid. Many map keys use
    // this packed value so consumers can recover both process and thread ids.
    return bpf_get_current_pid_tgid(); // [63:32]=tgid, [31:0]=tid
}

/* BPF Functions */
/* ---------- USDT: logic_start / logic_end ---------- */
int trace_logic_start(struct pt_regs *ctx)
{
    u64 now = bpf_ktime_get_ns();
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);

    // Phase 게이트 ON
    phase_ts_pid.update(&tgid, &now);


    // 이벤트 알림
    struct event_t e = {};
    e.ts = now;
    e.pid = tgid;
    e.kind = 0;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

int trace_logic_end(struct pt_regs *ctx)
{
    u64 now = bpf_ktime_get_ns();
    u32 tgid = get_tgid();

    // Phase 게이트 OFF
    phase_ts_pid.delete(&tgid);

    // 이벤트 알림(+phase 문자열)
    struct event_t e = {};
    e.ts = now;
    e.pid = tgid;
    e.kind = 1;
    u64 addr = 0;
    bpf_usdt_readarg(1, ctx, &addr);
    bpf_probe_read_user_str(e.phase, sizeof(e.phase), (void *)addr);
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

/* ---------- tracepoint: syscalls (read/write) ---------- */
/* Replaced function-style tracepoint handlers with TRACEPOINT_PROBE to
 * ensure the tracepoint `args` struct is available in BPF context.
 */
TRACEPOINT_PROBE(syscalls, sys_enter_read)
{
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid))
        return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_read)
{
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0)
        return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->read_cnt++;
    s->read_ns += delta;

    long ret = args->ret; // bytes read or error(<0)
    if (ret > 0)
        s->read_bytes += (u64)ret;

    s->start_ns = 0;
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_write)
{
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid))
        return 0;

    struct rw_stat_t zero = {};
    struct rw_stat_t *s = rwstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_write)
{
    u64 pid_tgid = get_pid_tgid();
    struct rw_stat_t *s = rwstat.lookup(&pid_tgid);
    if (!s || s->start_ns == 0)
        return 0;

    u64 delta = bpf_ktime_get_ns() - s->start_ns;
    s->write_cnt++;
    s->write_ns += delta;

    long ret = args->ret; // bytes written or error(<0)
    if (ret > 0)
        s->write_bytes += (u64)ret;

    s->start_ns = 0;
    return 0;
}

/* ---------- kprobe: handle_mm_fault ---------- */
int kprobe__handle_mm_fault(struct pt_regs *ctx)
{
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid))
        return 0;

    struct pf_stat_t zero = {};
    struct pf_stat_t *s = pfstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;

    s->start_ns = bpf_ktime_get_ns();
    return 0;
}

/* ---------- kretprobe: handle_mm_fault ---------- */
int kretprobe__handle_mm_fault(struct pt_regs *ctx)
{
    u64 pid_tgid = get_pid_tgid();
    struct pf_stat_t *s = pfstat.lookup(&pid_tgid);
    if (!s)
        return 0;

    u64 st = s->start_ns;
    if (st == 0)
        return 0;

    u64 delta = bpf_ktime_get_ns() - st;
    s->start_ns = 0;

    long retval = PT_REGS_RC(ctx);

    if ((retval & VM_FAULT_MAJOR) != 0)
    {
        s->major_cnt++;
        s->major_ns += delta;
    }
    else if ((retval & VM_FAULT_RETRY) != 0)
    { // Retry 전용
        s->minor_retry_cnt++;
        s->minor_retry_ns += delta;
    }
    else
    {
        // 커널에 'minor' 비트는 없음 → 'major 아님 && 에러 아님'을 minor로 취급
        s->minor_cnt++;
        s->minor_ns += delta;
    }

    return 0;
}

/* ---------- kprobe: ondemand_readahead ---------- */
int kprobe__ondemand_readahead(struct pt_regs *ctx)
{
    u64 pid_tgid = get_pid_tgid();
    u32 tgid = (u32)(pid_tgid >> 32);
    if (!phase_ts_pid.lookup(&tgid))
        return 0;

    struct pf_stat_t zero = {};
    struct pf_stat_t *s = pfstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;

    s->readahead_cnt++;
    return 0;
}

/* ---------- tracepoint: block_io_start (start) / block_io_done (done) ---------- */
TRACEPOINT_PROBE(block, block_io_start)
{
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid))
        return 0;

    struct block_start_t st = {};
    st.start_ns = bpf_ktime_get_ns();
    u64 pid_tgid = get_pid_tgid();
    st.pid_tgid = pid_tgid;
    st.bytes = args->bytes;

    char c = 0;
    bpf_probe_read_kernel(&c, sizeof(c), &args->rwbs[0]);
    if (c == 'W')
        st.op = 'W';
    else
        st.op = 'R';

    struct bio_key_t key = {.dev = args->dev, .sector = args->sector};
    block_start.update(&key, &st);

    return 0;
}

TRACEPOINT_PROBE(block, block_io_done)
{
    struct bio_key_t key = {.dev = args->dev, .sector = args->sector};
    struct block_start_t *st = block_start.lookup(&key);
    if (!st)
        return 0;

    u64 now = bpf_ktime_get_ns();
    u64 delta = now - st->start_ns;

    struct block_stat_t zero = {};
    struct block_stat_t *bs = blockstat.lookup_or_try_init(&st->pid_tgid, &zero);
    if (bs)
    {
        bs->time_ns += delta;
        bs->count += 1;
        if (st->op == 'W')
            bs->write_bytes += st->bytes;
        else
            bs->read_bytes += st->bytes;
    }

    struct block_io_interval_t iv = {};
    iv.pid_tgid = st->pid_tgid;
    iv.start_ns = st->start_ns;
    iv.end_ns = now;
    iv.bytes = st->bytes;
    iv.op = st->op;
    intervals.perf_submit(args, &iv, sizeof(iv));

    block_start.delete(&key);

    return 0;
}

/* ---------- sched tracepoints: cpu runtime / cpu wait ---------- */
TRACEPOINT_PROBE(sched, sched_stat_runtime)
{
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid))
        return 0;
    u64 pid_tgid = get_pid_tgid();
    struct sched_stat_t zero = {};
    struct sched_stat_t *s = schedstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;
    s->runtime_ns += args->runtime;
    return 0;
}

TRACEPOINT_PROBE(sched, sched_stat_wait)
{
    u32 tgid = get_tgid();
    if (!phase_ts_pid.lookup(&tgid))
        return 0;
    u64 pid_tgid = get_pid_tgid();
    struct sched_stat_t zero = {};
    struct sched_stat_t *s = schedstat.lookup_or_try_init(&pid_tgid, &zero);
    if (!s)
        return 0;
    s->wait_ns += args->delay;
    return 0;
}
