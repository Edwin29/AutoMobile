
## A_presets

- v3r (fast timing): laps 5/8, clean 2/8, min margin median -3.75 worst -6.29 cm, outside avg 13.7 s, held 3, departed 0
- v3r (own timing): laps 6/14, clean 3/14, min margin median -6.29 worst -7.49 cm, outside avg 13.4 s, held 6, departed 0
- v4 (own timing): laps 14/14, clean 0/14, min margin median -1.61 worst -7.55 cm, outside avg 12.2 s, held 0, departed 0
- v4 (slow timing): laps 14/14, clean 1/14, min margin median -2.48 worst -9.65 cm, outside avg 22.4 s, held 0, departed 0

| case | direction | v3r (fast) | v3r (own) | v4 (own) | v4 (slow) |
|---|---|---|---|---|---|
| ideal - | ccw | PASS +1.1 / 0s | PASS +1.1 / 0s | PASS -2.0 / 17s | PASS -1.4 / 9s |
| ideal - | cw | PASS +1.2 / 0s | PASS +1.2 / 0s | PASS -0.9 / 5s | PASS -2.6 / 25s |
| field_estimate - | ccw | PASS -5.4 / 32s | PASS -6.5 / 32s | PASS -2.4 / 15s | PASS -3.1 / 15s |
| field_estimate - | cw | PASS -5.6 / 33s | PASS -6.8 / 32s | PASS -1.2 / 7s | PASS -1.4 / 5s |
| weak_left_turn - | ccw | held -1.7 / 5s | held -7.5 / 7s | PASS -1.6 / 6s | PASS -3.2 / 24s |
| weak_left_turn - | cw | PASS -2.1 / 9s | PASS -3.0 / 19s | PASS -0.5 / 2s | PASS +0.3 / 0s |
| sluggish - | ccw | held -6.2 / 15s | held -7.1 / 16s | PASS -1.6 / 7s | PASS -1.0 / 5s |
| sluggish - | cw | held -6.3 / 16s | held -7.5 / 16s | PASS -0.6 / 4s | PASS -0.9 / 4s |
| high_start_duty - | ccw |  | time_limit +4.2 / 0s | PASS -6.3 / 29s | PASS -8.7 / 78s |
| high_start_duty - | cw |  | time_limit -2.0 / 14s | PASS -7.6 / 37s | PASS -9.7 / 81s |
| low_saturation - | ccw |  | held -6.0 / 7s | PASS -1.8 / 11s | PASS -2.4 / 6s |
| low_saturation - | cw |  | held -7.0 / 7s | PASS -0.5 / 3s | PASS -2.0 / 4s |
| unequal_sides - | ccw |  | PASS -4.3 / 31s | PASS -2.7 / 21s | PASS -2.9 / 22s |
| unequal_sides - | cw |  | held -6.6 / 7s | PASS -0.9 / 7s | PASS -4.4 / 35s |

## B_random_vehicle

- v3r (own timing): laps 11/24, clean 3/24, min margin median -4.38 worst -8.09 cm, outside avg 22.7 s, held 11, departed 0
- v4 (own timing): laps 23/24, clean 1/24, min margin median -3.42 worst -12.11 cm, outside avg 21.5 s, held 0, departed 1
- v4 (slow timing): laps 18/24, clean 1/24, min margin median -6.36 worst -12.07 cm, outside avg 31.7 s, held 1, departed 5

| case | direction | v3r (own) | v4 (own) | v4 (slow) |
|---|---|---|---|---|
| random:100 - | ccw | PASS -2.3 / 9s | PASS -2.3 / 16s | PASS -0.8 / 2s |
| random:100 - | cw | PASS +1.0 / 0s | PASS -0.8 / 4s | PASS -4.7 / 39s |
| random:101 - | ccw | lap_complete -5.6 / 62s | PASS -7.0 / 21s | held -11.9 / 7s |
| random:101 - | cw | held -4.5 / 15s | PASS -5.7 / 26s | severe_departure -12.0 / 3s |
| random:102 - | ccw | PASS -5.1 / 69s | PASS -6.5 / 24s | PASS -9.3 / 69s |
| random:102 - | cw | PASS +0.1 / 0s | PASS +0.1 / 0s | PASS -1.6 / 24s |
| random:103 - | ccw | held -6.4 / 7s | PASS -7.0 / 48s | PASS -11.0 / 66s |
| random:103 - | cw | lap_complete -4.7 / 110s | PASS -7.5 / 43s | PASS -11.6 / 82s |
| random:104 - | ccw | PASS +0.7 / 0s | PASS -2.5 / 12s | PASS -2.2 / 12s |
| random:104 - | cw | PASS -0.8 / 4s | PASS -1.1 / 3s | PASS +0.9 / 0s |
| random:105 - | ccw | PASS -5.7 / 33s | PASS -3.0 / 18s | PASS -2.3 / 21s |
| random:105 - | cw | PASS -4.5 / 54s | PASS -4.7 / 18s | severe_departure -12.0 / 10s |
| random:106 - | ccw | held -7.6 / 7s | PASS -2.2 / 19s | PASS -4.0 / 29s |
| random:106 - | cw | PASS -4.7 / 30s | PASS -2.0 / 17s | PASS -1.0 / 7s |
| random:107 - | ccw | held -0.9 / 2s | PASS -0.8 / 9s | PASS -2.0 / 25s |
| random:107 - | cw | held -2.5 / 12s | PASS -6.5 / 47s | severe_departure -12.0 / 9s |
| random:108 - | ccw | held -6.2 / 7s | PASS -4.9 / 36s | PASS -10.3 / 52s |
| random:108 - | cw | PASS -1.8 / 7s | PASS -1.7 / 7s | PASS -4.1 / 18s |
| random:109 - | ccw | PASS -4.6 / 73s | PASS -6.7 / 19s | severe_departure -12.1 / 65s |
| random:109 - | cw | held -8.1 / 7s | PASS -2.2 / 16s | PASS -5.8 / 43s |
| random:110 - | ccw | held -4.2 / 18s | PASS -5.6 / 15s | PASS -6.9 / 45s |
| random:110 - | cw | held -1.6 / 7s | PASS -3.6 / 13s | PASS -9.0 / 41s |
| random:111 - | ccw | held -0.7 / 2s | PASS -3.2 / 56s | PASS -4.1 / 62s |
| random:111 - | cw | held -1.9 / 10s | severe_departure -12.1 / 26s | severe_departure -12.0 / 27s |

## C_random_vehicle_and_effects

- v3r (own timing): laps 18/24, clean 2/24, min margin median -3.10 worst -12.01 cm, outside avg 26.5 s, held 4, departed 1
- v4 (own timing): laps 24/24, clean 0/24, min margin median -4.81 worst -9.07 cm, outside avg 18.7 s, held 0, departed 0
- v4 (slow timing): laps 13/24, clean 2/24, min margin median -8.50 worst -12.10 cm, outside avg 20.1 s, held 1, departed 9

| case | direction | v3r (own) | v4 (own) | v4 (slow) |
|---|---|---|---|---|
| random:200 glare | ccw | PASS -0.5 / 1s | PASS -2.4 / 20s | PASS -2.4 / 19s |
| random:200 glare | cw | PASS -4.1 / 27s | PASS -1.0 / 4s | PASS -0.5 / 1s |
| random:201 blur_warm+dropout+seam | ccw | PASS -3.1 / 8s | PASS -2.6 / 10s | PASS -1.9 / 8s |
| random:201 blur_warm+dropout+seam | cw | PASS -3.3 / 44s | PASS -5.1 / 20s | severe_departure -12.1 / 9s |
| random:202 low_power+occlusion+seam | ccw | held -2.1 / 8s | PASS -4.5 / 16s | PASS -10.3 / 61s |
| random:202 low_power+occlusion+seam | cw | held -2.1 / 13s | PASS -6.4 / 32s | severe_departure -12.1 / 28s |
| random:203 blur_warm+dropout+low_power | ccw | PASS +0.1 / 0s | PASS -1.8 / 15s | PASS -0.5 / 4s |
| random:203 blur_warm+dropout+low_power | cw | PASS -0.8 / 3s | PASS -1.2 / 8s | PASS -1.0 / 11s |
| random:204 blur_warm | ccw | PASS -4.0 / 70s | PASS -7.2 / 23s | PASS -9.8 / 76s |
| random:204 blur_warm | cw | severe_departure -12.0 / 6s | PASS -9.1 / 60s | severe_departure -12.0 / 27s |
| random:205 low_power | ccw | PASS -3.4 / 69s | PASS -5.4 / 19s | severe_departure -12.0 / 21s |
| random:205 low_power | cw | PASS -3.0 / 34s | PASS -1.5 / 8s | PASS +0.4 / 0s |
| random:206 dropout+glare+low_power | ccw | PASS -3.1 / 60s | PASS -5.2 / 24s | lap_complete -7.2 / 56s |
| random:206 dropout+glare+low_power | cw | PASS -1.1 / 7s | PASS -2.3 / 10s | PASS -5.9 / 32s |
| random:207 glare+seam | ccw | time_limit +4.6 / 0s | PASS -6.9 / 28s | severe_departure -12.0 / 10s |
| random:207 glare+seam | cw | held -4.0 / 40s | PASS -6.7 / 23s | severe_departure -12.0 / 43s |
| random:208 blur_warm+glare+occlusion | ccw | PASS -1.5 / 3s | PASS -2.3 / 16s | PASS -0.7 / 1s |
| random:208 blur_warm+glare+occlusion | cw | held -4.5 / 20s | PASS -5.4 / 24s | severe_departure -12.0 / 7s |
| random:209 glare+occlusion+seam | ccw | PASS -4.2 / 67s | PASS -5.5 / 16s | held -9.8 / 11s |
| random:209 glare+occlusion+seam | cw | PASS -0.5 / 2s | PASS -0.1 / 2s | PASS -2.1 / 24s |
| random:210 seam | ccw | PASS -3.1 / 17s | PASS -1.8 / 18s | PASS -1.0 / 5s |
| random:210 seam | cw | PASS -3.6 / 56s | PASS -6.3 / 18s | severe_departure -12.0 / 22s |
| random:211 blur_warm+glare+occlusion | ccw | PASS -0.9 / 4s | PASS -0.4 / 3s | PASS +0.5 / 0s |
| random:211 blur_warm+glare+occlusion | cw | PASS -5.6 / 78s | PASS -7.0 / 31s | severe_departure -12.1 / 8s |

## D_changing_vehicle

- v3r (own timing): laps 5/12, clean 0/12, min margin median -3.18 worst -5.65 cm, outside avg 20.6 s, held 7, departed 0
- v4 (own timing): laps 10/12, clean 1/12, min margin median -1.82 worst -12.01 cm, outside avg 19.0 s, held 1, departed 1
- v4 (slow timing): laps 5/12, clean 1/12, min margin median -11.30 worst -12.07 cm, outside avg 23.3 s, held 1, departed 6

| case | direction | v3r (own) | v4 (own) | v4 (slow) |
|---|---|---|---|---|
| field_estimate - (mild) | ccw | PASS -5.6 / 41s | PASS -1.6 / 7s | PASS -4.0 / 25s |
| field_estimate - (mild) | cw | PASS -2.8 / 24s | PASS -1.1 / 9s | PASS -5.9 / 41s |
| field_estimate - (harsh) | ccw | held -3.4 / 22s | PASS -4.0 / 34s | held -10.6 / 78s |
| field_estimate - (harsh) | cw | held -3.0 / 8s | severe_departure -12.0 / 9s | severe_departure -12.0 / 4s |
| random:300 - (mild) | ccw | PASS -1.2 / 3s | PASS -1.0 / 6s | PASS -0.2 / 1s |
| random:300 - (mild) | cw | PASS -4.0 / 21s | PASS +0.1 / 0s | PASS -7.0 / 16s |
| random:300 - (harsh) | ccw | held -4.6 / 14s | PASS -1.6 / 10s | severe_departure -12.1 / 15s |
| random:300 - (harsh) | cw | held -2.8 / 7s | held -7.3 / 20s | severe_departure -12.0 / 9s |
| random:301 - (mild) | ccw | held -0.5 / 6s | PASS -0.2 / 1s | PASS +0.1 / 0s |
| random:301 - (mild) | cw | PASS -4.8 / 69s | PASS -7.3 / 55s | severe_departure -12.0 / 10s |
| random:301 - (harsh) | ccw | held -0.7 / 3s | PASS -2.0 / 7s | severe_departure -12.1 / 74s |
| random:301 - (harsh) | cw | held -4.2 / 29s | PASS -9.6 / 71s | severe_departure -12.1 / 7s |