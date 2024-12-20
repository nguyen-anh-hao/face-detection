import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp

# Khởi tạo Flask và SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')  # Trang web chính

@socketio.on('video_feed')
def handle_video_stream(data):
    try:
        # Giải mã dữ liệu base64
        img_data = base64.b64decode(data)
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Xử lý khuôn mặt bằng Mediapipe
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            # Vẽ hình chữ nhật quanh khuôn mặt nếu phát hiện
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mã hóa hình ảnh đã xử lý và gửi về frontend
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_data = base64.b64encode(img_encoded).decode('utf-8')
        emit('processed_video', img_data)

    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
