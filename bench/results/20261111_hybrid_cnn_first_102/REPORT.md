
## I_hybrid

- cnnA5b (own timing): laps 25/34, clean 23/34, min margin median +0.84 worst -12.09 cm, outside avg 1.6 s, held 0, departed 9
- hyb2 (own timing): laps 28/34, clean 28/34, min margin median +1.81 worst -12.06 cm, outside avg 2.8 s, held 0, departed 6
- v4 (own timing): laps 26/34, clean 22/34, min margin median +1.14 worst -12.08 cm, outside avg 2.2 s, held 4, departed 4

| case | direction | cnnA5b (own) | hyb2 (own) | v4 (own) |
|---|---|---|---|---|
| field_estimate - | ccw | PASS +1.8 / 0s | PASS +2.2 / 0s | PASS +1.3 / 0s |
| field_estimate - | cw | PASS +1.5 / 0s | PASS +2.1 / 0s | PASS +1.6 / 0s |
| unequal_sides - | ccw | PASS +1.6 / 0s | PASS +2.1 / 0s | PASS -0.0 / 1s |
| unequal_sides - | cw | PASS +0.9 / 0s | PASS +1.6 / 0s | PASS +1.4 / 0s |
| weak_left_turn - | ccw | PASS +1.2 / 0s | PASS +1.8 / 0s | PASS +1.4 / 0s |
| weak_left_turn - | cw | PASS +2.1 / 0s | PASS +2.4 / 0s | PASS +1.6 / 0s |
| high_start_duty - | ccw | PASS +1.8 / 0s | PASS +1.8 / 0s | PASS -1.3 / 4s |
| high_start_duty - | cw | PASS +2.1 / 0s | PASS +2.0 / 0s | PASS -2.4 / 7s |
| field_estimate glare | ccw | PASS +1.8 / 0s | PASS +2.1 / 0s | PASS +1.5 / 0s |
| field_estimate glare | cw | PASS +1.5 / 0s | PASS +2.0 / 0s | PASS +1.6 / 0s |
| field_estimate occlusion | ccw | PASS +1.8 / 0s | PASS +2.2 / 0s | PASS +1.1 / 0s |
| field_estimate occlusion | cw | PASS +1.5 / 0s | PASS +2.1 / 0s | PASS +1.0 / 0s |
| field_estimate dropout | ccw | PASS +1.8 / 0s | PASS +2.2 / 0s | PASS +1.2 / 0s |
| field_estimate dropout | cw | PASS +1.5 / 0s | PASS +2.1 / 0s | PASS +1.6 / 0s |
| field_estimate whiteout_midturn | ccw | severe_departure -12.1 / 2s | PASS +1.9 / 0s | PASS +1.5 / 0s |
| field_estimate whiteout_midturn | cw | severe_departure -12.1 / 2s | PASS +1.2 / 0s | PASS +0.8 / 0s |
| field_estimate blur_warm+glare+low_power+occlusion (mild) | ccw | PASS +1.9 / 0s | PASS +1.8 / 0s | PASS -2.8 / 14s |
| field_estimate blur_warm+glare+low_power+occlusion (mild) | cw | PASS +2.2 / 0s | PASS +2.1 / 0s | PASS -0.2 / 5s |
| field_estimate realcam | ccw | PASS +0.4 / 0s | PASS +1.5 / 0s | PASS +1.3 / 0s |
| field_estimate realcam | cw | PASS +0.5 / 0s | PASS +1.9 / 0s | PASS +1.5 / 0s |
| field_estimate realcam_harsh | ccw | severe_departure -12.1 / 9s | severe_departure -12.0 / 6s | severe_departure -12.0 / 5s |
| field_estimate realcam_harsh | cw | severe_departure -12.0 / 3s | PASS +0.4 / 0s | severe_departure -12.0 / 4s |
| weak_left_turn realcam | ccw | PASS -0.6 / 2s | PASS +1.5 / 0s | PASS +1.0 / 0s |
| weak_left_turn realcam | cw | PASS +0.8 / 0s | PASS +1.8 / 0s | PASS +1.6 / 0s |
| weak_left_turn realcam_harsh | ccw | severe_departure -12.0 / 19s | severe_departure -12.0 / 31s | severe_departure -12.0 / 11s |
| weak_left_turn realcam_harsh | cw | severe_departure -12.0 / 3s | PASS +0.4 / 0s | held +1.4 / 0s |
| sluggish realcam | ccw | PASS +0.7 / 0s | PASS +1.5 / 0s | PASS +1.4 / 0s |
| sluggish realcam | cw | PASS +0.7 / 0s | PASS +1.6 / 0s | PASS +2.0 / 0s |
| sluggish realcam_harsh | ccw | severe_departure -12.0 / 8s | severe_departure -12.0 / 12s | held -2.9 / 8s |
| sluggish realcam_harsh | cw | severe_departure -12.1 / 4s | severe_departure -12.1 / 4s | held -1.0 / 2s |
| random:500 realcam | ccw | PASS +1.3 / 0s | PASS +1.7 / 0s | PASS -3.5 / 12s |
| random:500 realcam | cw | PASS +0.2 / 0s | PASS +1.8 / 0s | PASS +1.5 / 0s |
| random:500 realcam_harsh | ccw | PASS -0.4 / 1s | severe_departure -12.0 / 21s | severe_departure -12.1 / 2s |
| random:500 realcam_harsh | cw | severe_departure -12.1 / 3s | severe_departure -12.0 / 20s | held +0.1 / 0s |