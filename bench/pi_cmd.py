"""Send orchestrator commands to a running bench.pi_run (same machine), and show its events.

    python -m bench.pi_cmd stop
    python -m bench.pi_cmd resume
    python -m bench.pi_cmd right            # queue a direction for the next junction
    python -m bench.pi_cmd stop3            # stop, wait 3 s, resume (stop sign)
    python -m bench.pi_cmd -i               # interactive: s=stop r=resume L/R/S=direction 3=stop3 q=quit run
    python -m bench.pi_cmd --listen         # print events (junction decisions, acks)
"""
import argparse
import json
import socket
import sys
import time


def send(port, msg):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))
    s.close()


def command(port, word):
    word = word.strip().lower()
    if word in ("left", "right", "straight"):
        send(port, {"cmd": "direction", "move": word})
    elif word in ("stop", "resume", "quit"):
        send(port, {"cmd": word})
    elif word == "stop3":
        send(port, {"cmd": "stop"})
        time.sleep(3.0)
        send(port, {"cmd": "resume"})
    else:
        return False
    print(time.strftime("%H:%M:%S"), "sent", word, flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("word", nargs="?")
    ap.add_argument("-i", action="store_true", help="interactive keys")
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--cmd-port", type=int, default=8765)
    ap.add_argument("--event-port", type=int, default=8766)
    a = ap.parse_args()
    if a.listen:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", a.event_port))
        while True:
            raw, _ = s.recvfrom(65536)
            print(time.strftime("%H:%M:%S"), raw.decode(), flush=True)
    if a.i:
        keys = {"s": "stop", "r": "resume", "l": "left", "R": "right", "S": "straight", "3": "stop3", "q": "quit"}
        print("s=stop r=resume l=left R=right S=straight 3=stop+3s+resume q=quit run   (Enter after each key)")
        for line in sys.stdin:
            k = line.strip()
            w = keys.get(k)
            if w and command(a.cmd_port, w) and w == "quit":
                break
        return
    if not a.word or not command(a.cmd_port, a.word):
        ap.error("word: left / right / straight / stop / resume / stop3 / quit")


if __name__ == "__main__":
    main()
