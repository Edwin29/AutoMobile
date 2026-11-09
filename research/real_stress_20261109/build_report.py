"""Collect completed runs, provenance, a comparison contact sheet and a local video."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from experiment import HERE, ROOT, CORE, OUT, save

runs=[]
for f in sorted(OUT.glob('*.json')):
    z=json.loads(f.read_text(encoding='utf8'))
    if isinstance(z,dict) and 'scenario' in z and 'name' in z:runs.append(z)
save(OUT/'simulation_summary.json',runs)
fields=['name','reason','end_s','simulated_laps','min_true_margin_cm',
        'max_position_error_cm','max_yaw_error_deg','final_longitudinal_error_cm']
with (OUT/'simulation_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(runs)

files=list(HERE.glob('*.py'))+[HERE/'photo_plus5mm_rim.npy']
files += [CORE/x for x in ['controller.py','feedback_driver.py','yaw_gain.py','track.npz',
                          'validate_sim.py','calibrated_lap.py','paint_yaw.csv','second_corner_yaw.json',
                          'calibrated_runs/cw_calibration.json']]
files += [ROOT/'map_localization/c_drive_test/simulate_c.py',
          ROOT/'junction_logic/visibility_review/source/sim_drive.py',
          ROOT/'junction_logic/visibility_review/source/calib_checkerboard.npz',
          ROOT/'junction_logic/visibility_review/source/ground.npz',
          ROOT/'junction_logic/review_video/lane_pipeline.py']
files += list((CORE/'field_runs').glob('*_edges/[0-9][0-9].npz'))
files += list((CORE/'field_runs').glob('*_edges/index.csv'))
files += list(OUT.glob('*.csv'))
save(HERE/'provenance.json',dict(
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    opencv=cv2.__version__,numpy=np.__version__,
    hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    truth_usage='rendering, fault injection and grading only; no truth feedback to controller',
    controller='Feedback + FixedGain, V135, photo_plus5mm; translation prediction gain=1',
    excluded='Pi runtime latency/watchdog, rolling shutter, sensor noise, tire torque, exact 3-D occluders',
))

table=['| 조건 | 반시계 | 시계 |','|---|---|---|']
labels={'clean':'간섭 없음','response_only':'4초 구동 응답 변화','exposure_66ms':'66.5ms 노출 / 4초',
        'glare':'반사 / 4초','shadow':'그림자 / 4초','seam':'1cm 이음새 모사',
        'object':'물체 모사 / 4초','combined':'구동+노출+반사+이음새+물체',
        'stall_combined':'2초 좌측 지연 모사+노출+반사','forward_persistent':'지속적인 전진 1.27배',
        'prior_error':'추정 자세 주입: 옆 2cm / 앞 6cm / -5도',
        'prior_lateral':'추정 자세 주입: 옆 3cm / +6도'}
for scenario,label in labels.items():
    cells=[]
    for direction in ('ccw','cw'):
        candidates=[r for r in runs if r['scenario']==scenario and r['direction']==direction]
        if not candidates:cells.append('미실행');continue
        r=candidates[0]
        reason={'lap complete':'완주','unmatched road edges':'정합 실패','footprint limit':'여유거리 정지',
                'route discontinuity':'진행량 불연속 정지'}.get(r['reason'],r['reason'])
        cells.append(f"{reason} ({r['end_s']:.2f}s, 최대 위치오차 {r['max_position_error_cm']:.2f}cm)")
    table.append('| '+label+' | '+' | '.join(cells)+' |')
(HERE/'RESULTS.md').write_text('# 실행 결과\n\n'+f'완료된 폐루프 시험 {len(runs)}개. 결정론적 시나리오 시험이며 확률적 완주율 추정이 아니다.\n\n'+
 '\n'.join(table)+'\n\n자세 주입은 추정기 진단용 인공 오류다. 나머지 시험은 영상 또는 차량 운동만 바꾸고 제어기는 유지했다. '
 '20초부터 4초간 적용하며, 이음새는 계속 남는다. 지속 속도 오차만 2초부터 계속 적용했다. '
 '각 방향은 같은 출발점에서 출발하므로 20초의 위치·코너 단계가 다르다. 방향 자체의 우열을 입증하는 비교는 아니다.\n',encoding='utf8')

names=['ccw_clean_20','ccw_forward_persistent_2','cw_glare_20',
       'cw_seam_20','ccw_prior_lateral_20','ccw_combined_20']
tiles=[]
for name in names:
    f=OUT/(name+'_last.jpg')
    if not f.exists():continue
    tile=np.full((292,400,3),24,np.uint8)
    tile[:256]=cv2.resize(cv2.imread(str(f)),(400,256))
    cv2.putText(tile,name,(5,278),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1,cv2.LINE_AA)
    tiles.append(tile)
if len(tiles)==6:
    cv2.imwrite(str(OUT/'comparison.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))

# These videos are excerpts around injection, then the final recorded state.
# Each panel retains its source timestamp; the ending frame is held after STOP.
paths=[OUT/(n+'_review.mp4') for n in names]
if all(p.exists() for p in paths):
    caps=[cv2.VideoCapture(str(p)) for p in paths]
    length=max(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    writer=cv2.VideoWriter(str(OUT/'comparison_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(2250,900))
    last=[None]*len(caps)
    for i in range(length):
        tiles=[]
        for j,c in enumerate(caps):
            ok,frame=c.read()
            if ok:last[j]=frame
            tiles.append(cv2.resize(last[j],(750,450)))
        writer.write(np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
    writer.release()
    for c in caps:c.release()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(OUT/'comparison_raw.mp4'),
                    '-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',
                    str(OUT/'comparison.mp4')],check=True)
print(f'{len(runs)} completed simulations; reports generated',flush=True)
