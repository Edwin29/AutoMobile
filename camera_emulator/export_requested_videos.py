"""Export the two requested clips locally; never controls or modifies the Pi."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
from emulator import Emulator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'exports' / 'shared_videos_20261031_1651'
SOURCE = Path('C:/Users/USER/Documents/카카오톡 받은 파일')
FFMPEG = 'C:/Program Files/ffmpeg/bin/ffmpeg.exe'
FFPROBE = 'C:/Program Files/ffmpeg/bin/ffprobe.exe'
NAMES = ['KakaoTalk_20261031_165146762.mp4', 'KakaoTalk_20261031_165145372.mp4']


def probe(path):
    return json.loads(subprocess.check_output([
        FFPROBE, '-v', 'error', '-show_streams', '-of', 'json', str(path)
    ], encoding='utf-8'))


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    engine = Emulator()
    settings = dict(blur=1.0, noise=0.0, quality=100, gain=1.0, distort=True)
    manifest = {'filter': engine.metadata(), 'settings': settings,
                'timing': 'All decoded frames retained at source average frame rate (CFR); source audio copied.',
                'files': []}
    for name in NAMES:
        source = SOURCE / name
        original = OUT / name
        shutil.copy2(source, original)
        info = probe(source)
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        dest = OUT / (source.stem + '_distorted_blur.mp4')
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {source}')
        command = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '640x480',
                   '-framerate', video['avg_frame_rate'], '-i', 'pipe:0', '-i', str(source),
                   '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'medium',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
                   '-movflags', '+faststart', str(dest)]
        count = 0
        with (OUT / (source.stem + '_encode.log')).open('wb') as errors:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    _, result, crop = engine.apply(frame, **settings)
                    proc.stdin.write(result.tobytes())
                    count += 1
                    if count % 300 == 0:
                        print(f'{name}: {count} frames', flush=True)
            finally:
                cap.release()
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError(f'ffmpeg failed: {code}')
        if count != int(video['nb_frames']):
            raise RuntimeError(f'Incomplete decode: {count} frames')
        output_info = probe(dest)
        output_video = next(s for s in output_info['streams'] if s['codec_type'] == 'video')
        assert int(output_video['nb_frames']) == count
        assert (output_video['width'], output_video['height']) == (640, 480)
        assert abs(float(output_video['duration']) - float(video['duration'])) < .05
        assert any(s['codec_type'] == 'audio' for s in output_info['streams'])
        for path in (original, dest):
            manifest['files'].append({'name': path.name, 'bytes': path.stat().st_size,
                                      'sha256': sha256(path), 'frames': count,
                                      'crop_xywh': crop if path == dest else None})
        assert sha256(original) == sha256(source)
        print(f'Verified: {dest.name} ({count} frames)', flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['name']}\n" for f in manifest['files']), encoding='ascii')
    (OUT / 'README.txt').write_text(
        'Original MP4 files and synthetic OV5647 distortion + Gaussian blur versions.\n'
        'Processed: 640x480, center crop from 1280x720 (x=160,y=0,w=960,h=720).\n'
        'Gaussian sigma=1.0 output pixel; noise=0; gain=1; no extra JPEG degradation.\n'
        'H.264 CRF18, source AAC audio copied. All source frames retained at average FPS.\n'
        'Per-frame variable timing is normalized to constant frame rate.\n'
        'Calibration and hashes: manifest.json. Verify: sha256sum -c SHA256SUMS\n'
        'This is a synthetic camera approximation, not a validated exact camera match.\n', encoding='utf-8')
    print(str(OUT), flush=True)


if __name__ == '__main__':
    main()
