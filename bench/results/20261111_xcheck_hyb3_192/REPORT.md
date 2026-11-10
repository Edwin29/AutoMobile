
## J_xcheck

- cnnA5b (own timing): laps 48/64, clean 29/64, min margin median -0.12 worst -12.06 cm, outside avg 3.5 s, held 0, departed 15
- hyb2 (own timing): laps 52/64, clean 37/64, min margin median +0.73 worst -12.06 cm, outside avg 4.5 s, held 0, departed 12
- hyb3 (own timing): laps 48/64, clean 31/64, min margin median -0.67 worst -12.12 cm, outside avg 4.5 s, held 0, departed 15

| case | direction | cnnA5b (own) | hyb2 (own) | hyb3 (own) |
|---|---|---|---|---|
| field_estimate realcam | ccw | PASS +0.4 / 0s | PASS +1.5 / 0s | PASS +1.7 / 0s |
| field_estimate realcam | cw | PASS +0.5 / 0s | PASS +1.9 / 0s | PASS +1.1 / 0s |
| field_estimate realcam_harsh | ccw | severe_departure -12.1 / 9s | severe_departure -12.0 / 6s | severe_departure -12.1 / 8s |
| field_estimate realcam_harsh | cw | severe_departure -12.0 / 3s | PASS +0.4 / 0s | severe_departure -12.0 / 14s |
| weak_left_turn realcam | ccw | PASS -0.6 / 2s | PASS +1.5 / 0s | PASS +1.2 / 0s |
| weak_left_turn realcam | cw | PASS +0.8 / 0s | PASS +1.8 / 0s | PASS +1.4 / 0s |
| weak_left_turn realcam_harsh | ccw | severe_departure -12.0 / 19s | severe_departure -12.0 / 31s | severe_departure -12.0 / 25s |
| weak_left_turn realcam_harsh | cw | severe_departure -12.0 / 3s | PASS +0.4 / 0s | PASS -4.6 / 12s |
| sluggish realcam | ccw | PASS +0.7 / 0s | PASS +1.5 / 0s | PASS +1.3 / 0s |
| sluggish realcam | cw | PASS +0.7 / 0s | PASS +1.6 / 0s | PASS +0.7 / 0s |
| sluggish realcam_harsh | ccw | severe_departure -12.0 / 8s | severe_departure -12.0 / 12s | severe_departure -12.1 / 8s |
| sluggish realcam_harsh | cw | severe_departure -12.1 / 4s | severe_departure -12.1 / 4s | severe_departure -12.1 / 4s |
| random:500 realcam | ccw | PASS +1.3 / 0s | PASS +1.7 / 0s | PASS +1.6 / 0s |
| random:500 realcam | cw | PASS +0.2 / 0s | PASS +1.8 / 0s | PASS +1.0 / 0s |
| random:500 realcam_harsh | ccw | PASS -0.4 / 1s | severe_departure -12.0 / 21s | PASS -1.0 / 7s |
| random:500 realcam_harsh | cw | severe_departure -12.1 / 3s | severe_departure -12.0 / 20s | severe_departure -12.1 / 3s |
