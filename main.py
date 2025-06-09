import os
import argparse
from pathlib import Path
import urllib.request
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from scipy.signal import find_peaks
import pickle
from src.head_tracker import HeadTracker


def main(video_path, model_path, output_dir):
    # Get the model
    landmarker = HeadTracker(model_path)
    
    # prepare video input
    video_stem = Path(video_path).stem
    input = iio.imiter(video_path, plugin="pyav")
    meta = iio.immeta(video_path, plugin="pyav")
    fps = meta["fps"]
    duration = meta["duration"]
    
    # prepare video output
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"{video_stem}_raw_tracking.mp4")
    out_tracking_path = os.path.join(output_dir, f"{video_stem}_tracking.pkl")
    
    detection_data = []
    pitch_data = []
    print(f"Processing video {video_path}")
    with iio.imopen(out_video_path, "w", plugin="pyav") as output:
        output.init_video_stream("h264", fps=fps)  
        
        # decode each original frame
        for i, frame in enumerate(input):
            ts = int((i / fps) * 1000)  # timestamp in milliseconds
            
            # track facemesh
            detection, head_pitch, annotated = landmarker.track_frame(frame, ts)
            detection_data.append(detection)
            pitch_data.append(head_pitch)

            # save annotated frame to output
            output.write_frame(annotated)
            
            # Show progress bar
            progress = i / float(fps) / duration * 100
            print(f"Progress: {progress:.2f}%", end="\r")
    
    landmarker.close()       
        
    # save tracking
    with open(out_tracking_path, "wb") as f:
        pickle.dump(detection_data, f)
    
    print("Video procesing complete! Total samples:", len(detection_data))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count head nods using MediaPipe face mesh estimation.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="models/face_landmarker.task", 
                        help="Path to MediaPipe face mesh model. If not available, the model will be downloaded.")
    parser.add_argument("--output", default="out/", help="Directory to save output videos")
    args = parser.parse_args()

    main(args.video, args.model, args.output)