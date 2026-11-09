"""Apply the provisional lens model to an already rotated 640x480 image."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('output')
    p.add_argument('--model', default=str(Path(__file__).parent/'calibration/camera_640x480_rot180.npz'))
    args = p.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError('Cannot read input image')
    with np.load(args.model, allow_pickle=False) as model:
        if tuple(frame.shape[1::-1]) != tuple(model['image_size']):
            raise ValueError('Model requires 640x480; do not resize arbitrary sensor crops to fit it')
        result = cv2.remap(frame, model['map_x'], model['map_y'], cv2.INTER_LINEAR)
    if not cv2.imwrite(args.output, result):
        raise IOError('Could not write output image')


if __name__ == '__main__':
    main()
