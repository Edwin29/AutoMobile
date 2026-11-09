"""Recompute comparison metrics from the saved physical poses and commands."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
cases=("normal","delay","delay_slide","weak_clear","weak_blind")
modes=("output","joint","joint_edges")
scores=[]
for case in cases:
    baseline=None
    for mode in modes:
        stem=f"{case}_{mode}"
        info=json.loads((ROOT/f"{stem}.json").read_text())
        rows=np.genfromtxt(ROOT/f"{stem}.csv",names=True,delimiter=",")
        with (ROOT/f"{stem}_gains.csv").open() as stream:
            gains=list(csv.DictReader(stream))
        assert len(rows)==len(gains) and np.isclose(rows["margin"].min(),info["min_cm"])
        assert not abs(rows["L"][-1])+abs(rows["R"][-1])
        spin=(rows["L"]*rows["R"])<0
        assert np.abs(rows["L"][spin]).max()<=.260001
        assert np.abs(rows["R"][spin]).max()<=.260001
        if info["outcome"]=="pass":
            assert info["crossed_frames"]==0 and info["truth_exit_progress_cm"]>39
            assert abs(info["truth_exit_offset_cm"])<2 and info["heading_error_deg"]<5
        motion=np.array([float(g["true_heading_deg"]) for g in gains])
        active=(np.arange(len(rows))+1>=info["first_spin"]) & (
            np.arange(len(rows))+1<=info["phase_ticks"].get("exit",len(rows)))
        assert np.isclose(max(0.,motion[active].max()-90.),info["peak_pivot_overshoot_deg"])
        hidden=np.array([g["blind"]=="1" for g in gains])
        assert hidden.sum()==(9 if case=="weak_blind" else 0)
        if hidden.any():
            assert np.isclose(rows["angle_error"][hidden][-1],info["at_last_blind_deg"])
            assert not rows["matched"][hidden].any()
        if baseline is None:baseline=rows
        identical=np.array_equal(rows,baseline)
        if mode=="joint" and info["paint_events"]==0:assert identical
        scores.append(dict(case=case,mode=mode,result=info["outcome"],
                           minimum_cm=info["min_cm"],overshoot_deg=info["peak_pivot_overshoot_deg"],
                           last_blind_error_deg=info["at_last_blind_deg"],seconds=info["elapsed_s"],
                           paint_frames=info["eligible_frames_with_cross"],
                           prediction_updates=(len(info.get("edge_events",[])) if mode=="joint_edges"
                                               else info["paint_events"] if mode=="joint" else 0),
                           output_updates=info["output_events"],final_prediction=info["final_prediction"],
                           final_output=info["final_output_multiplier"],identical_to_A=identical))
(ROOT/"summary.json").write_text(json.dumps(scores,indent=2),encoding="utf8")
assumptions=dict(
    simulation_only=True,frames_per_second=15,inside_road_cm=18.5,
    user_reported_inside_road_cm=18.0,vehicle_width_cm=16.,vehicle_length_cm=24.5,
    pivot_reference_behind_front_cm=14.5,
    comparison="A output learner, B also existing crosswalk gain tracker, C also learns prediction from accepted boundary-corrected pose intervals",
    approach_and_exit="Shared pivot route and controls; adaptation enabled during pivot only",
    limits="0.26 opposite wheel command cap, 0.6 duty/s slew, 0.25cm predicted margin stop, up to 3 missing frames if prior margin >=1cm",
    recovery="Zero output on prolonged lost edges, three consecutive stationary map-layout confirmations before resuming",
    delay="For 2.5s after first opposed wheel command, actual yaw response is 60% of nominal. This does not model individual wheel dynamics.",
    slide="Delay-slide also moves the real body by [1.3cm right,2.4cm forward] per 52 degrees during the delay only. This is a sensitivity assumption from one photo, not a calibrated transient.",
    weak="Actual right response 0.6317018569 times 0.85, versus prior 0.6217314378",
    blind="Whole camera white 9 frames from 4.4s after pivot actuation starts, same timing policy for each arm",
    estimator="C updates use camera-corrected pose, which includes prediction. They are not an independent pose sensor.",
    scope="One deterministic simulated run per condition and controller. No inertia, motor electrical model, per-wheel friction, or observation-noise sweep."
)
(ROOT/"conditions.json").write_text(json.dumps(assumptions,indent=2),encoding="utf8")
print(json.dumps(scores,indent=2))
