# Nodding Detection

Detects nodding head movement.

## Examples

https://github.com/user-attachments/assets/ebc60ecd-56fb-4249-81b2-641c670e02f6

https://github.com/user-attachments/assets/7cdc485a-5a58-4275-bd82-007ba813fe02

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
