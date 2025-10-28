import os
from typing import Callable

def print_report(PROFILE_PHASE_REPORT_PATH: str, _print_report: Callable[[], None]) -> None:
    import sys
    original_stdout = sys.stdout
    with open(PROFILE_PHASE_REPORT_PATH, 'w') as f:
        sys.stdout = f
        _print_report()
        sys.stdout = original_stdout
    print(f"Profile report saved to {PROFILE_PHASE_REPORT_PATH}")