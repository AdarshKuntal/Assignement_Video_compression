
# Smart Behavioral Video Compression

## Objective
Reduce CCTV footage size while keeping frames containing humans.

## Algorithm Steps

1. Perceptual Hash (pHash) removes duplicate frames (>90% similarity).
2. Optical Flow calculates motion score and removes static frames (<0.20).
3. One context frame preserved every 3 seconds.
4. Surviving frames re-encoded into H.264 MP4 at 12 FPS.

## Files

solution.py – main compression script  
compressed_output.mp4 – compressed video output  
segments_kept.json – kept frame segments for pipeline integration  
compression_report.html – compression analysis report  
Adarsh_230054.mp4 – screen recording demonstration  

## Requirements

Python 3.9+

Libraries:

opencv-python  
imagehash  
numpy  
Pillow  

FFmpeg must be installed.
