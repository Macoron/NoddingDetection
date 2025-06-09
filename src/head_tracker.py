import os
import urllib.request
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" 

"""
Mediapipe face landmarker and head tracker.
"""
class HeadTracker:
    def __init__(self, model_path):
        self.landmarker = self.load_model(model_path)
    
    def load_model(self, model_path):
        # download model if needed
        if not os.path.exists(model_path):
            print(f"Downloading model from {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"Model downloaded and saved to {model_path}")
        
        print(f"Loading model from {model_path}...")
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_facial_transformation_matrixes=True
        )
        landmarker = FaceLandmarker.create_from_options(options)
        print("Model loaded!")
        
        return landmarker

    
    def track_frame(self, frame, ts):   
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
        detection = self.landmarker.detect_for_video(mp_image, timestamp_ms=ts)
        head_pitch = self.__calculate_head_pitch(detection)
        annotated = self.__draw_landmarks_on_image(frame, detection)
        annotated = self.__draw_head_pitch(annotated, head_pitch)
        
        return detection, head_pitch, annotated
    
    
    def close(self):
        self.landmarker.close()   
    
    
    def __calculate_head_pitch(self, detection):
        R = detection.facial_transformation_matrixes[0][:3, :3]
        rotation = Rotation.from_matrix(R)
        euler = rotation.as_euler('zyx', degrees=True)
        return euler[2]

    
    def __draw_head_pitch(self, rgb_image, head_pitch):
        font = ImageFont.truetype("DejaVuSans.ttf", size=64)
        img = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), f"Pitch: {head_pitch:.2f}", fill=(255, 0, 0), font=font)       
        return np.array(img)
    
    def __draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style()
            )
            
        # Draw axis
        matrix = detection_result.facial_transformation_matrixes[0]
        solutions.drawing_utils.draw_axis(
            annotated_image,
            -matrix[:3, :3],
            -matrix[:3, 3],
            axis_length=10,
        )

        return annotated_image