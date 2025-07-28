# 📓 DEVELOPMENT_LOG.md

- Project: On-Device LLM Weight Streaming
- Scope: Profiling Infrastructure, LiteRT Runtime, Multi-Level Logging
- Duration: 2025.06.28 \~ 2025.07.28
- Author: Geonha Park

---

## 📆 Phase 1 — Refactored Baseline LLM Inference Codes with LiteRT

**⏱️ Period**: 2025.06.28 \~ 2025.07.10  
**🌟 Goal**: Establish minimal LLM inference pipeline on LiteRT runtime (CPU-only)

- ✅ Migrated inference pipeline from TensorFlow Lite to LiteRT
- ✅ Removed QNN delegate and simplified to CPU execution
- ✅ Integrated tokenizer, prompt handler, and JSON-based output
- ✅ Added initial latency measurement hooks for end-to-end text generation
- ✅ Modularized profiler and metrics logging infrastructure

📁 **Key Commits**

- `f8762fa3` – Verify complete TensorFlow Lite to LiteRT migration
- `5e44c557` – Add latency metrics tracking
- `3e78241a` – Refactor profiler namespace and cleanup

---

## 📆 Phase 1.5 — Experimental Testbed for Tracing Features

**⏱️ Period**: 2025.07.01 \~ 2025.07.15  
**🌟 Goal**: Safely prototype tracing and profiling features before integrating with main LLM runtime

- ✅ Conducted unit testing on [`minimal-litert`](https://github.com/SNU-RTOS/minimal-litert) project to isolate USDT/eBPF instrumentation
- ✅ Validated USDT probe PoC and CSV-based profiling outputs
- ✅ Verified XNNPACK integration and latency logging in isolated minimal runtime
- ✅ Refactored CSV processing, reporting utilities, and environment configuration

📁 **Key Commits (minimal-litert)**

- `0983e5e3`, `8c51f218`, `23152b23` – Build system migration, profiling script support
- `c153be04`, `fc43f4fb`, `46933685` – XNNPACK integration, bpftrace support, profiling orchestration

---

## 📆 Phase 2 — eBPF-based System-Level I/O Tracing

**⏱️ Period**: 2025.07.11 \~ 2025.07.17  
**🌟 Goal**: Add support for system-level performance profiling with eBPF

- ✅ Introduced bpftrace-based scripts for I/O, page fault, and `io_uring` tracking
- ✅ Moved to Bazel-based build system for reproducibility and modularity
- ✅ Integrated stage-level eBPF hooks to capture profiling scopes
- ✅ Connected USDT (User Statically Defined Tracing) for lightweight runtime probing

📁 **Key Commits**

- `f21def60` – Clean up QNN delegate and finalize CPU backend
- `2657e630` – Add I/O performance testing scripts with bpftrace
- `9b44af88` – Add stage-specific eBPF performance tracing

---

## 📆 Phase 2.5 — Qualcomm Linux System Customization

**⏱️ Period**: 2025.07.14 \~ 2025.07.21  
**🌟 Goal**: Deploy traceable runtime on actual target (QCS6490 SoC)

- ✅ Custom kernel build with eBPF and USDT support for target platform
- ✅ Enabled Clang/LLVM-based toolchain and verified compatibility with bpftrace
- ✅ Customized rootfs with debugfs mount, persistent log storage, and systemd integration
- ✅ Verified full USDT + perf_event + eBPF log collection on-device

📁 **Outputs**

- `custom-linux-image.tar.gz`
- `qti-profiling.target` systemd unit
- Shell scripts for deployment, verification, and automation

---

## 📆 Phase 3 — Stage-Level Custom Profiler Integration

**⏱️ Period**: 2025.07.18 \~ 2025.07.23  
**🌟 Goal**: Instrument and extract stage-wise runtime latency via custom USDT

- ✅ Implemented custom profiler using `BeginStage`, `EndStage` scoped USDT
- ✅ Integrated `getrusage()` to measure per-stage CPU usage
- ✅ Designed `custom_profiler.log` format for stage timeline parsing
- ✅ Analyzed and matched TFLite subgraph structure for consistency
- ✅ Prepared groundwork for multi-source log merging

📁 **Key Commits**

- `053ca5c4` – Start integrating stage-level custom profiler
- `f40d02a6` – Complete logger and profiling format definition
- `6e7ad3be` – USDT + Rusage stage profiler finalized

---

## 📆 Phase 4 — Delegate Operator-Level Profiling & Merging

**⏱️ Period**: 2025.07.24 \~ 2025.07.28  
**🌟 Goal**: Extend profiling to delegate-internal operators (XNNPACK) and begin timeline merging

- ✅ Analyzed TFLite Profiler internals and operator-subgraph-delegate mappings
- ✅ Injected USDT probes into XNNPACK delegate to capture per-op execution time
- ✅ Enabled full delegate-observable profiling at operator-level
- ✅ Drafted `merge_profilers.py` to align stage ↔ subgraph ↔ operator timelines
- ✅ Designed `chunk_metadata.json` for token-boundary level analysis

📁 **Key Commits**

- `994206e6`, `ddf2030c` – Successfully capture XNNPACK op latency with USDT
- `dev (uncommitted)` – In progress: profiler merging + timeline visualization

---

## ✅ Summary of Achievements

| Layer             | Implementation Status                       |
| ----------------- | ------------------------------------------- |
| Inference         | ✔ LiteRT-based LLM runtime (CPU)            |
| Stage Profiler    | ✔ USDT + Rusage-based custom profiler       |
| Operator Profiler | ✔ XNNPACK-level latency logging             |
| I/O Tracing       | ✔ eBPF-based I/O tracing and analysis       |
| Merging Tool      | ⭕ In development (`merge_profilers.py`)    |
| Visualization     | ⭕ In design (`plot_profiling_timeline.py`) |
| System Support    | ✔ Qualcomm-targeted custom Linux + tracing  |

---

## 📌 Next Steps (Planned)

- [ ] Finalize profiler merging and chunk-level alignment
- [ ] Implement timeline visualizer and interactive HTML report generator
- [ ] Run controlled experiments to analyze compute–I/O overlap
- [ ] Integrate chunk-aware memory prefetcher evaluation
- [ ] Optimize token-stage-chunk mapping for adaptive prefetch

---
