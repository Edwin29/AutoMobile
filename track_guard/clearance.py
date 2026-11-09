"""Static straight-road envelope only; does not establish dynamic safety."""
import json
import math
from pathlib import Path


def projected_width(width, length, yaw_degrees):
    a=math.radians(yaw_degrees)
    return width*abs(math.cos(a))+length*abs(math.sin(a))


def first_contact_angle(width,length,road_width):
    # First intersection starting from straight ahead; excludes later rotations.
    if road_width<width:
        return None
    peak=math.hypot(width,length)
    if road_width>=peak:
        return 90.0
    return math.degrees(math.asin(road_width/peak)-math.atan2(width,length))


def main():
    root=Path(__file__).parent
    config=json.loads((root/'config.real.json').read_text(encoding='utf-8'))
    w,l=config['vehicle_width_including_wheels_m'],config['vehicle_length_m']
    rows=[]
    for name,road in config['road_widths_m'].items():
        if road is None:continue
        rows.append({'segment':name,'road_width_cm':road*100,
                     'one_side_clearance_cm':{str(a):round((road-projected_width(w,l,a))*50,3) for a in (0,3,5,10)},
                     'first_boundary_contact_yaw_deg':round(first_contact_angle(w,l,road),3)})
    result={'vehicle_cm':{'width':w*100,'length':l*100},
            'assumptions':'Centered rigid rectangular footprint on an infinite straight road; zero localization error, zero margins. Contact angle is NOT an operational limit.',
            'rows':rows}
    (root/'output').mkdir(exist_ok=True)
    (root/'output/clearance.real.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
