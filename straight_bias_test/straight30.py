"""One straight-line run at equal 30% PWM for the selected hold time."""
import argparse
import signal

from straight_test import trial


def interrupt(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4, help="30%% hold time, 1 to 18 seconds")
    parser.add_argument("--drive", action="store_true", help="enable motor outputs")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 18:
        parser.error("--seconds must be between 1 and 18")
    for sig in {signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)}:
        signal.signal(sig, interrupt)
    ok = trial("straight30", args.drive, duty_limit=0.30, hold_s=args.seconds, right_sign=1)
    raise SystemExit(0 if ok else 1)
