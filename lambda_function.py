import json
import base64
import torch
import cv2
import numpy as np
from pathlib import Path

from emotion import detect_emotion, init
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ===============================
# Inicialização GLOBAL (cold start)
# ===============================

DEVICE = select_device("cpu")
init(DEVICE)

MODEL = attempt_load("weights/yolov7-tiny.pt", map_location=DEVICE)
MODEL.eval()

IMG_SIZE = 512
CONF_THRES = 0.5
IOU_THRES = 0.45

NAMES = MODEL.names
COLORS = (
    (0,52,255),(121,3,195),(176,34,118),(87,217,255),
    (69,199,79),(233,219,155),(203,139,77),(214,246,255)
)

# ===============================
# Função principal da Lambda
# ===============================

def handler(event, context):
    try:
        body = json.loads(event["body"])
        image_base64 = body["image"]

        # Decode image
        image_bytes = base64.b64decode(image_base64)
        np_img = np.frombuffer(image_bytes, np.uint8)
        im0 = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if im0 is None:
            raise ValueError("Imagem inválida")

        # Preprocess
        img = cv2.resize(im0, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(DEVICE)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        # Inference YOLO
        with torch.no_grad():
            pred = MODEL(img)[0]
            pred = non_max_suppression(
                pred,
                CONF_THRES,
                IOU_THRES
            )

        results = []

        for det in pred:
            if det is None or len(det) == 0:
                continue

            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape
            ).round()

            faces = []

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                face_img = im0[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                faces.append(face_img)

            if faces:
                emotions = detect_emotion(faces, show_conf=True)

                for i, (*xyxy, conf, cls) in enumerate(det):
                    results.append({
                        "bbox": list(map(int, xyxy)),
                        "face_confidence": float(conf),
                        "emotion": emotions[i][0],
                        "emotion_id": int(emotions[i][1])
                    })

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "detections": results
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
