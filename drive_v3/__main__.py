"""CLI entry point. Every bundled command is offline and imports no GPIO."""
import argparse
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["observe","simulate","shadow","build-map","matrix"])
    args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    if args.command=="observe":
        from .replay import main as entry
    elif args.command=="simulate":
        from .simulation import main as entry
    elif args.command=="shadow":
        from .shadow import main as entry
    elif args.command=="matrix":
        from .validate_matrix import main as entry
    else:
        from .build_map import build
        q=argparse.ArgumentParser();q.add_argument("--output",required=True)
        return build(q.parse_args().output)
    return entry()


if __name__=="__main__":main()
