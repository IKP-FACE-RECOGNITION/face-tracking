import threading
import time
import os
import json

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from datetime import datetime

import httpx

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

import asyncio

from dotenv import load_dotenv

from db.mongo_connect import initialize_mongo_connection

# Config OS Env
os.environ["COMMANDLINE_ARGS"] = '--precision full --no-half'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load environment variables from the .env file (if present)
load_dotenv()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (choose one)
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(
    model_name="r50", path="face_recognition/arcface/weights/SFace-KT.pth", device=device
)

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Mapping of face IDs to names
id_face_mapping = {}

# Data mapping for tracking information
data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}

current_objId = list(data_mapping["tracking_ids"])

people = []


def load_config(file_name):
    """
    Load a YAML configuration file.

    Args:
        file_name (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """
    Process tracking for a frame.

    Args:
        frame: The input frame.
        detector: The face detector.
        tracker: The object tracker.
        args (dict): Tracking configuration parameters.
        frame_id (int): The frame ID.
        fps (float): Frames per second.

    Returns:
        numpy.ndarray: The processed tracking image.
    """
    # Face detection and tracking
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names=id_face_mapping,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image


@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Inference to get feature
    emb_img_face = recognizer(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    """
    Recognize a face image.

    Args:
        face_image: The input face image.

    Returns:
        tuple: A tuple containing the recognition score and name.
    """
    # Get feature from face
    query_emb = get_feature(face_image)

    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name


def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


async def call_deepface_service(url, files):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, files=files)
        if response :            
            return response.json()  # Ensure JSON response is returned correctly
        else:
            return {}
    
async def call_compreface_service(url,file,headers):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, files=file)
        if response :            
            return response.json()  # Ensure JSON response is returned correctly
        else:
            return {}
    
async def call_time_attendance_service(url,payload,headers):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        # response.raise_for_status()  # Raise an error for HTTP errors
        if response :            
            return response.json()  # Ensure JSON response is returned correctly
        else:
            return {}

async def recognize(detector, args):
    """Face recognition in a separate thread."""
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    count = 0

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Verify the settings
    print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))

    
    uri = os.getenv("ATTENDANCE_DB_URI")
    db_name = "Attendance"
    db = initialize_mongo_connection(uri,db_name)
    attendance_collection = db["attendance"]
    
    while True:
        count = count+1
        
        _, img = cap.read()
        
        current_objId = list(data_mapping["tracking_ids"])
        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)
        # print("data_mapping: ", data_mapping)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", tracking_image)
        

        for i in range(len(data_mapping["tracking_bboxes"])):
            for j in range(len(data_mapping["detection_bboxes"])):
                mapping_score = mapping_bbox(box1=data_mapping["tracking_bboxes"][i], box2=data_mapping["detection_bboxes"][j])
                
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=data_mapping["raw_image"],  landmark=data_mapping["detection_landmarks"][j])
                    # cv2.imshow("Transformed Image", face_alignment)
                    # score, name = recognition(face_image=face_alignment)
                    # print("score: ", score)
                    # if name is not None:
                    #     if score < 0.25:
                    #         caption = "UN_KNOWN"
                    #     else:
                    #         caption = f"{name}:{score:.2f}"
                    
                    # transform to img bytes
                    success, encoded_img = cv2.imencode('.jpg', face_alignment)
                    # print(encoded_img)
                    if not success:
                        continue
                    img_bytes = encoded_img.tobytes()
                    # url = "http://10.1.0.150:8088/recognize-face/?model_name=SFace&detector_backend=yunet&distance_metric=cosine&align=true"
                    # files = {
                    #     'img': ('result.jpg', img_bytes, 'image/jpeg')
                    #     }
                    # res = await call_deepface_service(url,files)
                    url = os.getenv('COMPARE_API_ENDPOINT') + "/api/v1/recognition/recognize?face_plugins=landmarks"
                    api_key = os.getenv('API_KEY')
                    files = {
                        'file': ('result.jpg', img_bytes, 'image/jpeg')
                        }
                    headers = {
                        'x-api-key': api_key
                    }
                    # res = ""
                    # if count%100 ==0 :
                    #     res = await call_compreface_service(url=url,headers=headers,file=files)
                    
                    #     if res.get("result") :
                    #         print("res.get(subjects)",res.get("result")[0].get("subjects"))
                    #         if len(res.get("result")[0].get("subjects")) >0 and res.get("result")[0].get("subjects")[0].get("similarity") >=0.96: 
                    #             caption = res.get("result")[0].get("subjects")[0].get("subject")
                    #             id_face_mapping[data_mapping["tracking_ids"][i]]  = caption
                    #             print ("caption: ", caption)
                    #             data_mapping["detection_bboxes"] = np.delete(data_mapping["detection_bboxes"], j, axis=0)
                    #             data_mapping["detection_landmarks"] = np.delete(data_mapping["detection_landmarks"], j, axis=0)
                                
                                
                                
                                
                    #             if caption not in people:
                    #                 attendance_check = attendance_collection.find_one({"pid": caption})
                    #                 print("attendance_check",attendance_check)
                                    
                    #                 if not attendance_check:
                    #                     # Get the current date and time
                    #                     now = datetime.now()

                    #                     # Format the date as 'YYYY-MM-DD'
                    #                     formatted_date = now.strftime('%Y-%m-%d')

                    #                     # Format the time as 'HH:MM'
                    #                     formatted_time = now.strftime('%H:%M')
                    #                     print("==== Add Schedule ====")
                    #                     attendance = {
                    #                         "pid": caption,
                    #                         "attendanceDate": formatted_date,
                    #                         "startWorkTime": formatted_time
                    #                     }
                                        
                    #                     attendace_result = attendance_collection.insert_one(attendance)
                    #                     print(f"Inserted document with ID: {attendace_result.inserted_id}")
                                
                    #             # if caption not in people:
                    #             #     # Get the current date and time
                    #             #     now = datetime.now()

                    #             #     # Format the date as 'YYYY-MM-DD'
                    #             #     formatted_date = now.strftime('%Y-%m-%d')

                    #             #     # Format the time as 'HH:MM'
                    #             #     formatted_time = now.strftime('%H:%M')
                    #             #     attendance_url = os.getenv('ATTENDANCE_URL') + "/TxGeTimeAttendance/import"
                    #             #     attendance_headers = {
                    #             #         'accept': '*/*',
                    #             #         'Content-Type': 'application/json-patch+json'
                    #             #         }
                    #             #     payload = json.dumps({
                    #             #         "pid": caption,
                    #             #         "deviceId": "FACE_RECOG_01",
                    #             #         "attendanceDate": formatted_date,
                    #             #         "startWorkTime": formatted_time,
                    #             #         "getOffWorkTime": None
                    #             #         })
                    #             #     print(payload)
                    #             #     await call_time_attendance_service(attendance_url,payload,attendance_headers)
                    #             #     people.append(caption)
                                
                    #         else:
                    #             caption = "UN_KNOWN"
                    #             print ("caption: ", caption)
                    #     else:
                    #         print("Error Response: ", res)
                    #     break

        if not data_mapping["tracking_bboxes"]:
            print("Waiting for a person...")
            # print("")
        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # """Main function to start face tracking and recognition threads."""
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)
    print("config_tracking: ",config_tracking)

    asyncio.run(recognize(detector, config_tracking))

if __name__ == "__main__":
    main()
