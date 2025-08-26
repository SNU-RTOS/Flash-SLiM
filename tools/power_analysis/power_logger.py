#!/usr/bin/env python3
# power_logger.py
#
# Board-level power sampler for Qualcomm-based boards.
# - Samples power from:
#     1) /sys/class/power_supply/qcom-battmgr-bat (voltage_now[µV], current_now[µA])
#     2) fallback: /sys/class/hwmon/<X> (in0_input[mV], curr1_input[mA])
# - Logs CSV: t_mono_ns,power_W,src,valid
# - Online integrates Energy[J] via trapezoidal rule (valid samples only)
# - Prints summary on exit (SIGINT) or after --duration
#
# Usage examples:
#   python3 power_logger.py --hz 20 --out power_log.csv
#   python3 power_logger.py --duration 60
#   python3 power_logger.py --hwmon /sys/class/hwmon/hwmon33
#
# Notes:
#   - "valid=1" rows only are used for integration and stats.
#   - batt path expects Discharging -> current_now negative; we flip sign so P_draw>0.

import argparse, csv, math, os, signal, sys, time
from typing import Tuple, Optional

DEFAULT_BATT = "/sys/class/power_supply/qcom-battmgr-bat"
# hwmon 예: "/sys/class/hwmon/hwmon33"
DEFAULT_HWMON = None

def read_int(path: str) -> int:
    with open(path, "r") as f:
        return int(f.read().strip())

def detect_hwmon_path(user_hwmon: Optional[str]) -> Optional[str]:
    if user_hwmon:
        return user_hwmon if os.path.isdir(user_hwmon) else None
    # 자동 탐색: 이름/파일 존재로 batt/usb/wls/ucsi 중에서 battmgr_bat 쪽 hwmon을 우선
    for d in sorted(os.listdir("/sys/class/hwmon")):
        p = os.path.join("/sys/class/hwmon", d)
        try:
            name = open(os.path.join(p, "name")).read().strip()
        except Exception:
            continue
        if name in ("qcom_battmgr_bat", "qcom_battmgr_usb", "qcom_battmgr_wls", "ucsi_source_psy_pmic_glink.ucsi.01"):
            # in0_input/curr1_input 둘 다 있어야 사용
            if os.path.exists(os.path.join(p, "in0_input")) and os.path.exists(os.path.join(p, "curr1_input")):
                return p
    # fallback: 아무 hwmon이나 in0_input+curr1_input 있는 놈
    for d in sorted(os.listdir("/sys/class/hwmon")):
        p = os.path.join("/sys/class/hwmon", d)
        if os.path.exists(os.path.join(p, "in0_input")) and os.path.exists(os.path.join(p, "curr1_input")):
            return p
    return None

def read_power_w(batt_base: str, hwmon_base: Optional[str]) -> Tuple[float, str, int]:
    """
    Returns (power_W, src, valid)
      src: "batt" | "hwmon" | "none"
      valid: 1 if numeric, 0 if not
    """
    # 1) battery manager path (µV, µA) -> W
    try:
        v_uv = read_int(os.path.join(batt_base, "voltage_now"))   # µV
        i_ua = read_int(os.path.join(batt_base, "current_now"))   # µA (Discharging = negative)
        # 소비전력(+)이 되도록 부호 반전
        return (-(v_uv * i_ua) / 1e12, "batt", 1)
    except Exception:
        pass

    # 2) hwmon fallback (mV, mA) -> W
    if hwmon_base:
        try:
            v_mv = read_int(os.path.join(hwmon_base, "in0_input"))     # mV
            i_ma = read_int(os.path.join(hwmon_base, "curr1_input"))   # mA
            return (-(v_mv/1000.0 * i_ma/1000.0), "hwmon", 1)
        except Exception:
            pass

    return (math.nan, "none", 0)

class OnlineIntegrator:
    """Trapezoidal integration with validity handling."""
    def __init__(self) -> None:
        self.have_prev = False
        self.t0_ns = None  # first valid timestamp
        self.tprev_ns = 0
        self.pprev = 0.0
        self.energy_j = 0.0
        self.valid_samples = 0
        self.peak_w = 0.0
        self.last_valid_ns = None

    def add(self, t_ns: int, p_w: float, valid: int):
        if valid:
            self.valid_samples += 1
            if self.t0_ns is None:
                self.t0_ns = t_ns
            self.last_valid_ns = t_ns
            if p_w > self.peak_w:
                self.peak_w = p_w

            if self.have_prev:
                # 이전 샘플도 유효해야 적분
                dt_s = (t_ns - self.tprev_ns) * 1e-9
                if dt_s > 0:
                    self.energy_j += 0.5 * (self.pprev + p_w) * dt_s

            self.pprev = p_w
            self.tprev_ns = t_ns
            self.have_prev = True
        else:
            # invalid sample breaks trapezoid continuity
            self.have_prev = False

    def summary(self):
        if self.t0_ns is None or self.last_valid_ns is None or self.last_valid_ns == self.t0_ns:
            return {
                "duration_s": 0.0, "energy_j": 0.0, "avg_w": 0.0, "peak_w": 0.0,
                "valid_samples": self.valid_samples
            }
        dur = (self.last_valid_ns - self.t0_ns) * 1e-9
        avg = self.energy_j / dur if dur > 0 else 0.0
        return {
            "duration_s": dur,
            "energy_j": self.energy_j,
            "avg_w": avg,
            "peak_w": self.peak_w,
            "valid_samples": self.valid_samples
        }

def main():
    ap = argparse.ArgumentParser(description="Board power sampler (battmgr + hwmon fallback)")
    ap.add_argument("--batt", type=str, default=DEFAULT_BATT, help="battery base path (default: %(default)s)")
    ap.add_argument("--hwmon", type=str, default=DEFAULT_HWMON, help="hwmon base path (default: auto-detect)")
    ap.add_argument("--hz", type=float, default=20.0, help="sample rate (Hz)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to run (default: until Ctrl+C)")
    ap.add_argument("--out", type=str, default="power_log.csv", help="CSV output")
    args = ap.parse_args()

    if not os.path.isdir(args.batt):
        print(f"[WARN] battery path not present: {args.batt}", file=sys.stderr)

    hwmon_base = detect_hwmon_path(args.hwmon)
    if args.hwmon and hwmon_base is None:
        print(f"[WARN] hwmon path not usable: {args.hwmon}", file=sys.stderr)
    elif hwmon_base:
        try:
            name = open(os.path.join(hwmon_base, "name")).read().strip()
        except Exception:
            name = "(unknown)"
        print(f"[INFO] hwmon fallback: {hwmon_base} (name={name})")

    dt = 1.0 / max(args.hz, 0.1)
    stop = False

    def on_sigint(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    integ = OnlineIntegrator()

    # Run loop
    t_end = time.monotonic() + args.duration if args.duration else None
    # CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_mono_ns", "power_W", "src", "valid"])
        next_t = time.monotonic()
        while not stop:
            t_ns = time.monotonic_ns()
            pw, src, valid = read_power_w(args.batt, hwmon_base)
            wr.writerow([t_ns, f"{pw:.6f}", src, int(valid)])
            integ.add(t_ns, pw, int(valid))

            if t_end and time.monotonic() >= t_end:
                break
            next_t += dt
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    s = integ.summary()
    print("\n===== Power Summary =====")
    print(f" Samples(valid) : {s['valid_samples']}")
    print(f" Duration(s)    : {s['duration_s']:.3f}")
    print(f" Energy(J)      : {s['energy_j']:.3f}")
    print(f" Avg Power(W)   : {s['avg_w']:.3f}")
    print(f" Peak Power(W)  : {s['peak_w']:.3f}")
    print(f" CSV            : {args.out}")

if __name__ == "__main__":
    main()
