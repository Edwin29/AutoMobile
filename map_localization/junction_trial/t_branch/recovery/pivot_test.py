"""One time-limited right pivot. Camera provides the measured angle afterwards."""
import argparse
import signal
from straight_test import trial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", action="store_true")
    args = parser.parse_args()

    def stop(*_):
        raise KeyboardInterrupt

    for sig in {signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)}:
        signal.signal(sig, stop)
    ok = trial("right_pivot", args.drive, duty_limit=.26, hold_s=3.0, right_sign=-1)
    raise SystemExit(0 if ok else 1)
