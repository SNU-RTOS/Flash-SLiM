#!/usr/bin/env python3
# power_logger.py
#
# Board-level power sampler for Qualcomm-based boards.
# - Samples power from:
#     1) /sys/class/power_supply/qcom-battmgr-bat (voltage_now[µV], current_now[µA])
#     2) fallback: /sys/class/hwmon/<X> (in0_input[mV], curr1_input[mA])
# - Logs CSV: t_mono_ns,power_W,src,valid
# - Prints summary on exit (SIGINT) or after --duration
#
# Notes:
#   - "valid=1" rows only are used for integration and stats.
#   - batt path expects Discharging -> current_now negative; we flip sign so P_draw>0.

import argparse, csv, math, os, signal, sys, time, statistics
from typing import Tuple, Optional

DEFAULT_BATT = "/sys/class/power_supply/qcom-battmgr-bat"
# Example hwmon: "/sys/class/hwmon/hwmon33"
DEFAULT_HWMON = None


# ======================================
def read_int(path: str) -> int:
    with open(path, "r") as f:
        return int(f.read().strip())


def detect_hwmon_path(user_hwmon: Optional[str]) -> Optional[str]:
    if user_hwmon:
        return user_hwmon if os.path.isdir(user_hwmon) else None
    # Auto-detect: prioritize battmgr_bat related hwmon by name/file existence among batt/usb/wls/ucsi
    for d in sorted(os.listdir("/sys/class/hwmon")):
        p = os.path.join("/sys/class/hwmon", d)
        try:
            name = open(os.path.join(p, "name")).read().strip()
        except Exception:
            continue
        if name in (
            "qcom_battmgr_bat",
            "qcom_battmgr_usb",
            "qcom_battmgr_wls",
            "ucsi_source_psy_pmic_glink.ucsi.01",
        ):
            # Both in0_input and curr1_input must exist to use
            if os.path.exists(os.path.join(p, "in0_input")) and os.path.exists(
                os.path.join(p, "curr1_input")
            ):
                return p
    # Fallback: any hwmon with in0_input and curr1_input
    for d in sorted(os.listdir("/sys/class/hwmon")):
        p = os.path.join("/sys/class/hwmon", d)
        if os.path.exists(os.path.join(p, "in0_input")) and os.path.exists(
            os.path.join(p, "curr1_input")
        ):
            return p
    return None


def read_power_w(batt_base: str, hwmon_base: Optional[str]) -> Tuple[float, str, int]:
    """
    Returns (power_W, src, valid)
      src: "batt" | "hwmon" | "none"
      valid: 1 if numeric, 0 if not
    """
    # 1) Battery manager path (µV, µA) -> W
    try:
        v_uv = read_int(os.path.join(batt_base, "voltage_now"))  # µV
        i_ua = read_int(
            os.path.join(batt_base, "current_now")
        )  # µA (Discharging = negative)
        # Flip sign to make power consumption positive
        return (-(v_uv * i_ua) / 1e12, "batt", 1)
    except Exception:
        pass

    # 2) Hwmon fallback (mV, mA) -> W
    if hwmon_base:
        try:
            v_mv = read_int(os.path.join(hwmon_base, "in0_input"))  # mV
            i_ma = read_int(os.path.join(hwmon_base, "curr1_input"))  # mA
            return (-(v_mv / 1000.0 * i_ma / 1000.0), "hwmon", 1)
        except Exception:
            pass

    return (math.nan, "none", 0)


# ======================================
class CSVAnalyzer:
    """Unified CSV analysis for both power summary and timing statistics."""

    def __init__(self, csv_path: str, target_dt: float):
        self.csv_path = csv_path
        self._data: Optional[dict] = None
        self._parsed = False
        self.target_dt = target_dt

    def _parse_csv(self):
        """Parse CSV once and cache the results."""
        if self._parsed:
            return

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(self.csv_path)

        times_all = []
        times_valid = []
        power_data = []  # (timestamp, power, valid) tuples

        with open(self.csv_path, "r") as f:
            r = csv.reader(f)
            header = next(r, None)
            # Detect column unit: support 'power_mW' or 'power_W' in header
            power_is_mw = False
            if header and len(header) > 1:
                h = header[1].lower()
                if "mw" in h:
                    power_is_mw = True

            for row in r:
                if not row:
                    continue
                try:
                    t = int(row[0])
                    p_raw = float(row[1])
                    # Normalize to mW for internal processing: if header is W, convert W->mW
                    if power_is_mw:
                        p = p_raw
                    else:
                        # assume W if header not explicitly mW
                        p = p_raw * 1000.0 if not math.isnan(p_raw) else p_raw
                    valid = int(row[3])
                except Exception:
                    continue

                times_all.append(t)
                if valid:
                    times_valid.append(t)
                    power_data.append((t, p, valid))

        self._data = {
            "times_all": times_all,
            "times_valid": times_valid,
            "power_data": power_data,
        }
        self._parsed = True

    def compute_power_summary(self):
        """Compute power summary from parsed CSV data."""
        self._parse_csv()
        if self._data is None:
            print("[Warn] No valid data found")
            return {
                "duration_s": 0.0,
                "energy_mj": 0.0,
                "avg_mw": 0.0,
                "peak_mw": 0.0,
                "valid_samples": 0,
            }

        power_data = self._data["power_data"]

        if not power_data:
            print("[Warn] No valid power_data found")
            return {
                "duration_s": 0.0,
                "energy_mj": 0.0,
                "avg_mw": 0.0,
                "peak_mw": 0.0,
                "valid_samples": 0,
            }

        first_ns = power_data[0][0]
        last_ns = power_data[-1][0]
        # energy in millijoules (mW * s = mJ)
        energy_mj = 0.0
        peak_mw = 0.0
        valid_samples = len(power_data)

        if first_ns == last_ns:
            return {
                "duration_s": 0.0,
                "energy_mj": 0.0,
                "avg_mw": 0.0,
                "peak_mw": peak_mw,
                "valid_samples": valid_samples,
            }

        prev_t, prev_p = None, None
        for t, p, valid in power_data:
            if not valid:
                continue
            if p > peak_mw:
                peak_mw = p
            if prev_t is not None:
                dt_s = (t - prev_t) * 1e-9
                if dt_s > 0:
                    if (dt_s - self.target_dt * 1.05) > 0:
                        print(
                            f"[Warn] Long interval detected, interval error is over than 5%"
                        )
                        print(
                            f"       Expected {self.target_dt * 1e3:.3f} ms, but got {dt_s * 1e3:.3f} ms"
                        )
                        print(
                            f"       Gap: {dt_s * 1e3 - self.target_dt * 1e3:.3f} ms"
                        )
                    # integrate using mW values -> result in mW*s == mJ
                    energy_mj += 0.5 * (prev_p + p) * dt_s
                else:
                    print("[Warn] Invalid interval detected, interval is < 0")
            prev_t, prev_p = t, p
        dur = (last_ns - first_ns) * 1e-9
        # avg in mW
        avg_mw = energy_mj / dur if dur > 0 else 0.0

        return {
            "duration_s": dur,
            "energy_mj": energy_mj,
            "avg_mw": avg_mw,
            "peak_mw": peak_mw,
            "valid_samples": valid_samples,
        }

    def analyze_timing(self, target_dt: Optional[float] = None):
        """Analyze timing statistics from parsed CSV data."""
        self._parse_csv()
        if self._data is None:
            print("\n===== Log Timing Analysis =====")
            print(" No valid data found")
            return

        times_valid = self._data["times_valid"]

        def sec_intervals(ts):
            if len(ts) < 2:
                return []
            return [(ts[i + 1] - ts[i]) / 1e9 for i in range(len(ts) - 1)]

        def stats(values):
            if not values:
                return None
            vals = sorted(values)
            n = len(vals)
            mean = statistics.mean(vals)
            stdev = statistics.pstdev(vals) if n > 1 else 0.0
            p50 = vals[int(n * 0.50)]
            p90 = vals[int(n * 0.90)]
            p99 = vals[int(n * 0.99)] if n > 100 else vals[-1]
            return {
                "count": n,
                "mean": mean,
                "stdev": stdev,
                "min": vals[0],
                "max": vals[-1],
                "p50": p50,
                "p90": p90,
                "p99": p99,
            }

        intervals_valid = sec_intervals(times_valid)

        print("\n===== Log Timing Analysis =====")
        print(f" Valid samples     : {len(times_valid):8d} (total)")
        if target_dt:
            print(
                f" Target interval   : {target_dt*1000:8.3f} (ms) (-> {1.0/target_dt:.2f} Hz)"
            )

        def print_stats(name, ivals):
            st = stats(ivals)
            if st is None:
                print(f" {name}: no data")
                return
            mean_ms = st["mean"] * 1000.0
            stdev_ms = st["stdev"] * 1000.0

            print(f"\n-- {name} --")
            print(f" intervals         : {st['count']:8d} (inter-sample)")
            print(f" mean ± std        : {mean_ms:8.3f} (ms) ± {stdev_ms:8.3f} (ms)")
            if target_dt:
                err = (mean_ms / 1000.0 - target_dt) / target_dt * 100.0
                print(f" mean error        : {err:8.4f} %")

        print_stats("Timing Intervals", intervals_valid)


def print_power_summary(summary_dict: dict, csv_path: str, from_csv: bool = False):
    """Print power summary in a consistent format."""
    title = "Power Summary (from CSV)" if from_csv else "Power Summary"
    print(f"\n===== {title} =====")
    print(f" Samples (valid)   : {summary_dict['valid_samples']}")
    print(f" Duration          : {summary_dict['duration_s']:.3f} (sec) ")
    # energy is in mJ, power in mW
    print(f" Total Energy      : {summary_dict.get('energy_mj', 0.0):.3f} (mJ)")
    print(f" Avg Power         : {summary_dict.get('avg_mw', 0.0):.3f} (mW)")
    print(f" Peak Power        : {summary_dict.get('peak_mw', 0.0):.3f} (mW)")
    print(f" CSV               : {csv_path}")


def analyze_and_report_csv(csv_path: str, target_hz: float):
    """Common analysis and reporting logic for both logger and parser modes."""
    target_dt = 1.0 / max(target_hz, 0.1)

    try:
        analyzer = CSVAnalyzer(csv_path, target_dt)

        # Compute and display power summary
        power_summary = analyzer.compute_power_summary()
        print_power_summary(power_summary, csv_path, from_csv=True)

        # Perform timing analysis
        analyzer.analyze_timing(target_dt=target_dt)

    except Exception as e:
        print(f"[WARN] CSV analysis failed: {e}", file=sys.stderr)


def print_logger_metadata(args, hwmon_base: Optional[str] = None):
    print("\n===== Logger Metadata =====")
    print(f" Sample Rate (Hz) : {args.hz}")
    print(
        f" Duration (s)     : {args.duration if args.duration else 'Infinite (stopped by user)'}"
    )
    print(f" Output CSV       : {args.csv}")
    print(f" Battery Path     : {args.batt}")
    # Prefer detected hwmon_base for clarity, fall back to user arg or Auto-detected
    if hwmon_base:
        print(f" Hwmon Path       : {hwmon_base}")
    else:
        print(f" Hwmon Path       : {args.hwmon if args.hwmon else 'Auto-detected'}")
    print(f" Busy Fraction    : {args.busy_fraction}")


def run_logger_mode(args):
    """Execute power logging mode - samples power and writes to CSV."""
    print("[INFO] Running in LOGGER mode")

    # Detect hwmon path early so metadata can show actual path
    hwmon_base = detect_hwmon_path(args.hwmon)
    if args.hwmon and hwmon_base is None:
        print(f"[WARN] hwmon path not usable: {args.hwmon}", file=sys.stderr)
    elif hwmon_base:
        try:
            name = open(os.path.join(hwmon_base, "name")).read().strip()
        except Exception:
            name = "(unknown)"
        # print info here but metadata will also show hwmon_base
        print(f"[INFO] hwmon fallback path: {hwmon_base} (name={name})")

    # Print metadata before logging
    print_logger_metadata(args, hwmon_base=hwmon_base)

    # Wait Until User Input
    # try:
    #     input("\n[INFO] Press Enter to start logging..\n")
    # except KeyboardInterrupt:
    #     print("\n[INFO] Logging stopped by user.")
    #     sys.exit(0)
    
    print("\n[INFO] Logging started...")

    # Validate hardware paths
    if not os.path.isdir(args.batt):
        print(f"[WARN] battery path not present: {args.batt}", file=sys.stderr)

    # (hwmon_base already detected above)

    # Calculate timing parameters
    dt = 1.0 / max(args.hz, 0.1)
    busy_frac = max(0.0, min(args.busy_fraction, 0.9))
    busy_wait_threshold = busy_frac * dt

    # Setup signal handling
    stop = False

    def on_sigint(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    # Prepare CSV output and run sampling loop
    t_end = time.monotonic() + args.duration if args.duration else None
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.writer(f)
        # 기록 단위를 mW로 변경해서 해상도 향상
        wr.writerow(["t_mono_ns", "power_mW", "src", "valid"])
        next_t = time.monotonic()

        while not stop:
            t_ns = time.monotonic_ns()
            pw, src, valid = read_power_w(args.batt, hwmon_base)
            # CSV에는 mW 단위로 저장 (파서에서 mW->W로 복원)
            pw_mw = pw * 1000.0 if not math.isnan(pw) else pw
            wr.writerow([t_ns, f"{pw_mw:.3f}" if not math.isnan(pw_mw) else "nan", src, int(valid)])

            if t_end and time.monotonic() >= t_end:
                break

            # Hybrid timing: sleep most of the time, then busy-wait for precision
            next_t += dt
            sleep_for = next_t - time.monotonic()
            if sleep_for > busy_wait_threshold:
                time.sleep(sleep_for - busy_wait_threshold)
            while time.monotonic() < next_t:
                pass

    # Run CSV analysis and reporting after logging
    analyze_and_report_csv(args.csv, args.hz)


def run_parser_mode(args):
    """Execute parser mode - analyze existing CSV file only."""
    print("[INFO] Running in PARSER mode")

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    # Run CSV analysis and reporting
    analyze_and_report_csv(args.csv, args.hz)


# ======================================
def main():
    """Main entry point - mode selection via command line arguments."""
    ap = argparse.ArgumentParser(
        description="Board power sampler (battmgr + hwmon fallback)"
    )

    # Required mode selection
    ap.add_argument(
        "--mode",
        required=True,
        choices=["logger", "parser"],
        help="Operation mode: 'logger' for power sampling, 'parser' for CSV analysis",
    )

    # Required CSV file
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV file path (output for logger mode, input for parser mode)",
    )

    # Logger mode specific arguments
    logger_group = ap.add_argument_group("logger mode options")
    logger_group.add_argument(
        "--hz",
        type=float,
        default=20.0,
        help="sample rate in Hz (default: %(default)s)",
    )
    logger_group.add_argument(
        "--duration",
        type=float,
        default=None,
        help="sampling duration in seconds (default: infinite until Ctrl+C)",
    )
    logger_group.add_argument(
        "--batt",
        type=str,
        default=DEFAULT_BATT,
        help="battery base path (default: %(default)s)",
    )
    logger_group.add_argument(
        "--hwmon",
        type=str,
        default=DEFAULT_HWMON,
        help="hwmon base path (default: auto-detect)",
    )
    logger_group.add_argument(
        "--busy-fraction",
        type=float,
        default=0.05,
        help="fraction of dt to busy-wait for timing precision (default: %(default)s)",
    )

    # Parser mode specific arguments
    parser_group = ap.add_argument_group("parser mode options")
    parser_group.add_argument(
        "--target-hz",
        type=float,
        default=20.0,
        help="expected sample rate for timing analysis (default: %(default)s)",
    )

    args = ap.parse_args()

    # Set check_only flag based on mode
    args.check_only = args.mode == "parser"

    # For parser mode, map target_hz to hz for compatibility
    if args.mode == "parser":
        args.hz = args.target_hz

    # Mode selection and execution
    if args.mode == "parser":
        run_parser_mode(args)
    else:
        run_logger_mode(args)


if __name__ == "__main__":
    main()
