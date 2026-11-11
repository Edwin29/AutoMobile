#!/bin/bash
# Pi: send an orchestrator command to a running run_mission.sh   (./send.sh stop | resume | right | left | straight | stop3 | quit | -i | --listen)
cd "$(dirname "$0")"
exec .venv/bin/python -m bench.pi_cmd "$@"
