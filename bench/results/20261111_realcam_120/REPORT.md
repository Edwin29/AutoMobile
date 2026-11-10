
## H_realcam

- cnnA5b (own timing): laps 9/20, clean 7/20, min margin median -8.13 worst -12.06 cm, outside avg 6.3 s, held 0, departed 10
- cnnC5b (own timing): laps 1/20, clean 14/20, min margin median +2.74 worst -5.91 cm, outside avg 36.2 s, held 0, departed 0
- v4 (own timing): laps 8/20, clean 15/20, min margin median +1.64 worst -12.00 cm, outside avg 1.9 s, held 11, departed 1

| case | direction | cnnA5b (own) | cnnC5b (own) | v4 (own) |
|---|---|---|---|---|
| field_estimate realcam | ccw | PASS +0.4 / 0s | time_limit +0.9 / 0s | PASS +1.3 / 0s |
| field_estimate realcam | cw | PASS +0.5 / 0s | time_limit +3.5 / 0s | PASS +0.9 / 0s |
| field_estimate realcam_harsh | ccw | severe_departure -12.1 / 9s | time_limit -4.2 / 191s | held +3.1 / 0s |
| field_estimate realcam_harsh | cw | severe_departure -12.0 / 3s | time_limit +3.5 / 0s | held -1.1 / 3s |
| weak_left_turn realcam | ccw | PASS -0.6 / 2s | time_limit +2.0 / 0s | PASS +0.5 / 0s |
| weak_left_turn realcam | cw | PASS +0.8 / 0s | time_limit +3.5 / 0s | PASS +1.5 / 0s |
| weak_left_turn realcam_harsh | ccw | severe_departure -12.0 / 19s | time_limit -4.2 / 191s | held +3.0 / 0s |
| weak_left_turn realcam_harsh | cw | severe_departure -12.0 / 3s | time_limit +3.5 / 0s | held -7.2 / 14s |
| sluggish realcam | ccw | PASS +0.7 / 0s | PASS +0.3 / 0s | PASS +1.3 / 0s |
| sluggish realcam | cw | PASS +0.7 / 0s | time_limit +3.5 / 0s | PASS +1.8 / 0s |
| sluggish realcam_harsh | ccw | severe_departure -12.0 / 8s | time_limit -4.9 / 45s | held +3.3 / 0s |
| sluggish realcam_harsh | cw | severe_departure -12.1 / 4s | time_limit +3.5 / 0s | held -4.8 / 6s |
| random:500 realcam | ccw | PASS +1.3 / 0s | time_limit +0.8 / 0s | PASS -2.6 / 9s |
| random:500 realcam | cw | PASS +0.2 / 0s | time_limit +3.5 / 0s | PASS +1.9 / 0s |
| random:500 realcam_harsh | ccw | PASS -0.4 / 1s | time_limit -5.9 / 188s | held +3.7 / 0s |
| random:500 realcam_harsh | cw | severe_departure -12.1 / 3s | time_limit +3.5 / 0s | severe_departure -12.0 / 6s |
| field_estimate realcam_harsh (harsh) | ccw | severe_departure -12.0 / 7s | time_limit +4.2 / 0s | held +4.2 / 0s |
| field_estimate realcam_harsh (harsh) | cw | lap_complete -4.3 / 24s | time_limit -4.2 / 60s | held +4.0 / 0s |
| random:400 realcam_harsh (harsh) | ccw | severe_departure -12.0 / 35s | time_limit +4.2 / 0s | held +4.2 / 0s |
| random:400 realcam_harsh (harsh) | cw | severe_departure -12.0 / 8s | time_limit -1.8 / 49s | held +4.0 / 0s |

## H_realcam_factor

- cnnA5b (own timing): laps 20/20, clean 11/20, min margin median +0.13 worst -0.43 cm, outside avg 0.6 s, held 0, departed 0
- cnnC5b (own timing): laps 1/20, clean 14/20, min margin median +1.11 worst -1.95 cm, outside avg 1.4 s, held 0, departed 0
- v4 (own timing): laps 14/20, clean 14/20, min margin median +1.15 worst -6.08 cm, outside avg 2.1 s, held 6, departed 0

| case | direction | cnnA5b (own) | cnnC5b (own) | v4 (own) |
|---|---|---|---|---|
| field_estimate realcam_light | ccw | PASS -0.1 / 1s | time_limit +1.2 / 0s | PASS +1.4 / 0s |
| field_estimate realcam_light | cw | PASS -0.1 / 1s | time_limit +0.8 / 0s | PASS +1.7 / 0s |
| field_estimate realcam_shadow | ccw | PASS +0.8 / 0s | time_limit -2.0 / 6s | PASS +0.9 / 0s |
| field_estimate realcam_shadow | cw | PASS +0.1 / 0s | time_limit +3.5 / 0s | PASS +1.1 / 0s |
| field_estimate realcam_exposure | ccw | PASS -0.4 / 3s | time_limit +0.6 / 0s | held +3.7 / 0s |
| field_estimate realcam_exposure | cw | PASS -0.2 / 2s | time_limit +3.5 / 0s | held -6.1 / 7s |
| field_estimate realcam_blur | ccw | PASS +0.6 / 0s | PASS -0.2 / 2s | PASS +1.1 / 0s |
| field_estimate realcam_blur | cw | PASS +0.6 / 0s | time_limit +3.5 / 0s | PASS +1.2 / 0s |
| field_estimate realcam_optics | ccw | PASS -0.1 / 1s | time_limit +0.8 / 0s | held +3.7 / 0s |
| field_estimate realcam_optics | cw | PASS -0.2 / 2s | time_limit +3.5 / 0s | PASS -3.4 / 9s |
| random:500 realcam_light | ccw | PASS +1.0 / 0s | time_limit -1.7 / 7s | PASS -2.9 / 10s |
| random:500 realcam_light | cw | PASS +0.1 / 0s | time_limit +3.5 / 0s | PASS +1.9 / 0s |
| random:500 realcam_shadow | ccw | PASS +1.4 / 0s | time_limit +1.0 / 0s | PASS -1.1 / 5s |
| random:500 realcam_shadow | cw | PASS -0.0 / 0s | time_limit +3.5 / 0s | PASS +1.1 / 0s |
| random:500 realcam_exposure | ccw | PASS +0.9 / 0s | time_limit -1.8 / 8s | held +3.7 / 0s |
| random:500 realcam_exposure | cw | PASS -0.2 / 2s | time_limit +3.5 / 0s | held -3.3 / 6s |
| random:500 realcam_blur | ccw | PASS +1.2 / 0s | time_limit -1.8 / 5s | PASS -1.0 / 5s |
| random:500 realcam_blur | cw | PASS +0.4 / 0s | time_limit +3.5 / 0s | PASS +1.5 / 0s |
| random:500 realcam_optics | ccw | PASS +0.7 / 0s | time_limit -0.2 / 0s | held +3.7 / 0s |
| random:500 realcam_optics | cw | PASS -0.3 / 2s | time_limit +3.5 / 0s | PASS +0.6 / 0s |