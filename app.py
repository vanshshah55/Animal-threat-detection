
from flask import Flask, render_template, request, Response, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLOv3 model and COCO names (for detecting zoo animals)
YOLO_CONFIG_PATH = 'yolov3.cfg'  # Path to YOLOv3 config file
YOLO_WEIGHTS_PATH = 'yolov3.weights'  # Path to YOLOv3 weights file
COCO_NAMES_PATH = 'coco.names'  # Path to COCO class names file

net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(COCO_NAMES_PATH, 'r') as f:
    classes = f.read().strip().split("\n")

# Only zoo animals classes (you can add more as per the model's training)
ZOO_ANIMALS_CLASSES = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe','animal']



# Load Haar cascade for gun detection
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Check if the file has one of the allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process the video and stream it frame by frame
# Extend with human-related classes
THREAT_CLASSES = ['knife', 'gun']
NON_THREAT_CLASSES = ['camera']
ALL_CLASSES = ZOO_ANIMALS_CLASSES + ['person'] + THREAT_CLASSES + NON_THREAT_CLASSES

# Process the video and stream it frame by frame
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    skip_frames = 3  # Adjust this number based on desired speed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to speed up the process
        if frame_count % skip_frames != 0:
            continue

        # Get original frame dimensions
        orig_height, orig_width, _ = frame.shape

        # Reduce resolution for faster processing
        small_frame = cv2.resize(frame, (320, 320))
        small_height, small_width, _ = small_frame.shape

        blob = cv2.dnn.blobFromImage(small_frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Information to draw bounding boxes
        class_ids = []
        confidences = []
        boxes = []

        # Loop over the YOLO output layers
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in ALL_CLASSES:
                    center_x = int(detection[0] * small_width)
                    center_y = int(detection[1] * small_height)
                    w = int(detection[2] * small_width)
                    h = int(detection[3] * small_height)

                    # Scale coordinates back to the original frame size
                    x = int((center_x - w / 2) * (orig_width / small_width))
                    y = int((center_y - h / 2) * (orig_height / small_height))
                    w = int(w * (orig_width / small_width))
                    h = int(h * (orig_height / small_height))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Gun detection using Haar Cascade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        guns = gun_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=20, minSize=(100, 100))

        # Draw bounding boxes for gun detection
        for (gx, gy, gw, gh) in guns:
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)  # Red rectangle for guns
            cv2.putText(frame, 'threat', (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw the bounding boxes around animals and humans
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                color = (0, 255, 0)  # Default to green for animals
                if label == 'person':
                    color = (255, 0, 0)  # Blue for humans
                elif label in THREAT_CLASSES:
                    color = (0, 0, 255)  # Red for threats

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Redirect to the video stream page after uploading the file
        return redirect(url_for('video_feed', filename=filename))

    return redirect(request.url)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)




