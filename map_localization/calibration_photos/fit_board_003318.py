"""Fit the visible partial lattice in the 00:33 photo, in local board coordinates."""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PHOTO = HERE / "board_ruler_20261106_003318_489430.png"
SQUARE_CM = 2.314285714
BUMPER_TO_FAR_GRID_CM = 16.5  # approximate user measurement along the board
FAR_ROW = 5  # first row of selected points is 0; last selected row is 4
# Approximate intersections selected in the original image. Column 4 is obscured.
# Rows advance away from the vehicle along the eleven-square dimension.
ROWS = [
    [(61,364),(113,369),(176,373),(249,374),None,(410,363),(475,354)],
    [(89,322),(134,324),(193,324),(256,322),None,(392,315),(451,310)],
    [(108,290),(152,290),(205,286),(260,286),None,(379,279),(432,275)],
    [(131,262),(169,260),(215,259),(265,256),None,(367,252),(416,249)],
    [(143,240),(183,238),(224,236),(269,235),None,(359,232),(402,229)],
]


def main():
    im = cv2.imread(str(PHOTO))
    assert im is not None and im.shape[:2] == (480,640)
    calib = np.load(HERE.parents[1]/"junction_logic/visibility_review/source/calib_checkerboard.npz")
    K,D = calib["K"],calib["D"]
    pairs = [((x,y),p) for y,row in enumerate(ROWS) for x,p in enumerate(row) if p]
    grid, seeds = map(np.asarray,zip(*pairs))
    obj = np.column_stack((grid*SQUARE_CM,np.zeros(len(grid))))
    pix = seeds.astype("float32").reshape(-1,1,2)
    cv2.cornerSubPix(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),pix,(7,7),(-1,-1),(3,60,.005))
    rect = cv2.fisheye.undistortPoints(pix.astype(float),K,D,P=K).reshape(-1,2)
    report = {"photo":PHOTO.name,"squares_forward_lateral":[11,8],"square_cm":SQUARE_CM,
              "ruler_zero":"board end","points":len(pix),"far_grid_row":FAR_ROW,
              "bumper_to_far_grid_cm":BUMPER_TO_FAR_GRID_CM}
    for name, use in (("all",np.ones(len(grid),bool)),("alternating",grid.sum(axis=1)%2==1)):
        ok,r,t = cv2.solvePnP(obj[use],rect[use],K,None)
        if not ok: raise RuntimeError("PnP failed")
        R = cv2.Rodrigues(r)[0]
        xyz = (-R.T@t).ravel()
        pred = cv2.fisheye.projectPoints(obj[:,None,:],r,t,K,D)[0]
        err = np.linalg.norm((pred-pix).reshape(-1,2),axis=1)
        far_distance=float(FAR_ROW*SQUARE_CM-xyz[1])
        report[name] = dict(camera_to_far_grid_forward_cm=far_distance,
                            camera_behind_front_cm=far_distance-BUMPER_TO_FAR_GRID_CM,
                            height_above_board_cm=float(xyz[2]),
                            pitch_down_deg=float(np.degrees(np.arcsin(-R[2,2]))),
                            fit_rms_px=float(np.sqrt(np.mean(err[use]**2))))
        if (~use).any():
            report[name]["withheld_rms_px"] = float(np.sqrt(np.mean(err[~use]**2)))
        if name=="all":
            # Reproject the selected far boundary for a visual check of row indexing.
            far_points=np.zeros((60,1,3))
            far_points[:,0,0]=np.linspace(0,6*SQUARE_CM,60)
            far_points[:,0,1]=FAR_ROW*SQUARE_CM
            far_uv=cv2.fisheye.projectPoints(far_points,r,t,K,D)[0]
            cv2.polylines(im,[np.rint(far_uv).astype(np.int32)],False,(255,100,0),2)
            for xy in np.rint(pix.reshape(-1,2)).astype(int):
                cv2.circle(im,tuple(xy),4,(0,255,255),1,cv2.LINE_AA)
            cv2.imwrite(str(HERE/"board_verified_points.png"),im)
    (HERE/"board_fit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__=="__main__":
    main()
