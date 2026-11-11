#!/bin/bash
# Pi: run bench.pi_run with the hybrid_mission venv (scipy) and system camera/GPIO packages.
#   ./run_mission.sh                 dry (camera + decisions, motors untouched)
#   ./run_mission.sh --drive         motors on
#   ./run_mission.sh --help
cd "$(dirname "$0")"
exec .venv/bin/python -m bench.pi_run "$@"
