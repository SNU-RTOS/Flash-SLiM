#!/usr/bin/env python3
"""
USDT 인자 디버깅용 간단한 스크립트
"""
from bcc import BPF, USDT
import sys

# BPF 프로그램 - 최소한의 디버그용
BPF_PROGRAM = """
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 ts;
    u32 pid;
    u32 kind;
    u64 arg1;
    u64 arg2;
    u64 arg3;
    char phase[64];
};

BPF_PERF_OUTPUT(events);

int trace_ops_start(struct pt_regs *ctx) {
    struct event_t e = {};
    e.ts = bpf_ktime_get_ns();
    e.pid = bpf_get_current_pid_tgid() >> 32;
    e.kind = 0;
    
    bpf_trace_printk("ops_start triggered\\n");
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}

int trace_ops_check(struct pt_regs *ctx) {
    struct event_t e = {};
    e.ts = bpf_ktime_get_ns();
    e.pid = bpf_get_current_pid_tgid() >> 32;
    e.kind = 1;
    
    // USDT 인자 읽기
    bpf_usdt_readarg(1, ctx, &e.arg1);
    bpf_usdt_readarg(2, ctx, &e.arg2);
    bpf_usdt_readarg(3, ctx, &e.arg3);
    
    // 커널 로그에 출력
    bpf_trace_printk("ops_check: arg1=%llu, arg2=%llu, arg3=%llx\\n", 
                    e.arg1, e.arg2, e.arg3);
    
    // 문자열 읽기
    if (e.arg3 != 0) {
        bpf_probe_read_user_str(e.phase, sizeof(e.phase), (void*)e.arg3);
        bpf_trace_printk("String: %.20s\\n", e.phase);
    }
    
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}
"""

def print_event(cpu, data, size):
    event = b["events"].event(data)
    if event.kind == 0:
        print(f"[{event.ts}] START: pid={event.pid}")
    else:
        phase_str = event.phase.decode('utf-8', 'replace').rstrip('\x00')
        print(f"[{event.ts}] END: pid={event.pid}, arg1={event.arg1}, arg2={event.arg2}, arg3={event.arg3:x}, phase='{phase_str}'")

if __name__ == "__main__":
    binary_path = "bin/text_generator_main"
    
    # USDT 설정
    usdt = USDT(path=binary_path)
    usdt.enable_probe_or_bail("text_gen:ops_start", "trace_ops_start")
    usdt.enable_probe_or_bail("text_gen:ops_check", "trace_ops_check")
    
    # BPF 로드
    b = BPF(text=BPF_PROGRAM, usdt_contexts=[usdt])
    b["events"].open_perf_buffer(print_event)
    
    print("Tracing USDT probes... Ctrl-C to stop.")
    print("Check kernel logs with: sudo cat /sys/kernel/debug/tracing/trace_pipe")
    
    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nStopping...")
