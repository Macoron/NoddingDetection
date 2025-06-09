# Nodding Detection

Detects nodding head movement.

## Examples

https://github.com/user-attachments/assets/26cad9d5-cd8e-48ee-bc72-190b26e25ef4

https://github.com/user-attachments/assets/6d198aad-6366-4476-9607-86314b6b81a5

## Getting Started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Macoron/NoddingDetection
cd NoddingDetection
pip install -r requirements.txt
```

Now you can run the script:

```bash
python main.py --video path/to/video.mp4
```

By default the script will save the results in the `./out` directory.

## Output Structure

After inference the program will save the results using the following format:
- Mediapipe tracking video: `<video_name>_raw_tracking.mp4`
- Raw tracking data frame by frame in Pickle: `<video_name>_tracking.pkl`
- Nodding detection video: `<video_name>_nodding.mp4`
- Noddings intervals (start, stop) frame in Pickle: `<video_name>_nodding.pkl`

## Assumptions

- User is expected to be facing the camera and stay not further away than 2 meters.
- The face is expected to be always visible and not occluded.
- The nodding is assumpted to be a pattern of moving head one time up and one time down.

To learn more about check the [research notebook](notebooks/research.ipynb).
