"""Isolated baseline entry point; existing controller code remains unchanged."""
import argparse
from pathlib import Path
import json
from bench.controllers import HybridCNN
from bench.run import run, vehicle_from_args

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--cnn-mixer', choices=['v4'], default='v4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--vehicles', default='field_estimate,weak_left_turn,sluggish,unequal_sides')
    ap.add_argument('--directions', default='cw,ccw')
    args=ap.parse_args()
    HybridCNN.CNN_MIXER=args.cnn_mixer
    rows=[]
    for vehicle in args.vehicles.split(','):
        for direction in args.directions.split(','):
            dst=Path(args.out)/(vehicle+'_'+direction)
            if (dst/'summary.json').exists():
                r=json.loads((dst/'summary.json').read_text())
            else:
                r=run('hyb2',vehicle_from_args(vehicle,None),['realcam'],direction,'own',41,200,dst)
            r['vehicle_name']=vehicle
            rows.append(r)
            print(vehicle,direction,r['verified_lap'],r['min_margin_cm'],flush=True)
            (Path(args.out)/'matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
