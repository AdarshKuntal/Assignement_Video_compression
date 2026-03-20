import cv2
import os
import json
import imagehash
import subprocess
from PIL import Image
import numpy as np
import argparse

DEFAULT_INPUT = "Class_8_cctv_video_1.mov"
FRAME_DIR = "f"
FFMPEG_EXE = r"C:\Users\prash\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"

os.makedirs(FRAME_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def phash_similarity(frame1, frame2):
    h1 = imagehash.phash(Image.fromarray(frame1))
    h2 = imagehash.phash(Image.fromarray(frame2))
    similarity = 1 - (h1 - h2) / 64
    return similarity

def get_motion_score(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    return len(faces) > 0

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    context_interval = int(10 * fps) 

    frame_index = 0
    prev_kept_frame = None
    prev_kept_gray = None
    kept_frames = []

    print(f"Processing: {input_path}...")
    count = 0
    count1 = 0
    while True:
        count1 += 1
        ret, frame = cap.read()
        if not ret:
            break

        keep = False

        if frame_index % context_interval == 0:
            keep = True

        if prev_kept_frame is None:
            keep = True
        else:
            similarity = phash_similarity(prev_kept_frame, frame)
            
            if similarity < 0.90:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = get_motion_score(prev_kept_gray, curr_gray)
                
                if motion > 0.20:
                    keep = True
            
        if keep:
            count += 1
            path = os.path.join(FRAME_DIR, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(path, frame)
            kept_frames.append(frame_index)
            print(count, count1)
            prev_kept_frame = frame.copy()
            prev_kept_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Frame {frame_index} processed... Kept: {len(kept_frames)}")

    cap.release()
    return kept_frames

def save_segments(frames):
    if not frames: return
    segments = []
    start = frames[0]
    for i in range(1, len(frames)):
        if frames[i] != frames[i-1] + 1:
            segments.append({"start_frame": start, "end_frame": frames[i-1]})
            start = frames[i]
    segments.append({"start_frame": start, "end_frame": frames[-1]})

    with open("segments_kept.json", "w") as f_out:
        json.dump({"segments": segments}, f_out, indent=4)

def create_video():
    list_file = "file_list.txt"
    images = sorted([img for img in os.listdir(FRAME_DIR) if img.endswith(".jpg")])
    
    if not images:
        print("No frames were kept. Adjust thresholds.")
        return

    with open(list_file, "w") as f_out:
        for img in images:
            f_out.write(f"file '{FRAME_DIR}/{img}'\n")

    cmd = [
        FFMPEG_EXE, "-y", "-r", "12",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "compressed_output.mp4"
    ]

    subprocess.run(cmd)
    if os.path.exists(list_file): os.remove(list_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", default=DEFAULT_INPUT, help="Video to compress")
    args = parser.parse_args()

    for f in os.listdir(FRAME_DIR):
        os.remove(os.path.join(FRAME_DIR, f))

    frames = process_video(args.input_video)
    if frames:
        save_segments(frames)
        create_video()
        print(f"\nCompression complete! Kept {len(frames)} frames.")
    else:
        print("\nProcess finished with zero frames kept.")

if __name__ == "__main__":
    main()