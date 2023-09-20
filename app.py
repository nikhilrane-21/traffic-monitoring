import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import get_class_color, estimatedSpeed
import tempfile
import base64
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
st.set_page_config(page_title="Traffic Monitoring", page_icon="🤖", layout="wide")
hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

task_list = ["Camera", "Video", "RTSP"]

with st.sidebar:
    st.title('Source Selection')
    task_name = st.selectbox("Select your source type:", task_list)
    # Add a slider to configure the threshold speed limit
    speed_limit = st.slider("Set the speed limit (km/h):", 0, 200, 60)

st.title("Dashboard")

model = YOLO("./YoloWeights/yolov8l.pt")

mask = cv2.imread("static/mask.png")

tracker = DeepSort(
    max_iou_distance=0.7,
    max_age=2,
    n_init=3,
    nms_max_overlap=3.0,
    max_cosine_distance=0.2)

limitsUp = [210, 450, 600, 450]
limitsDown = [650, 450, 1000, 450]

totalCountUp = []
totalCountDown = []

coordinatesDict = dict()

clsCounterUp = {'car': 0, 'truck': 0, 'motorbike': 0}
clsCounterDown = {'car': 0, 'truck': 0, 'motorbike': 0}

# Create a dictionary to store the maximum speed of each vehicle
max_speeds = {}

# Create a Streamlit container to display the warning messages
warning_container = st.empty()

# Create Streamlit containers to display the values
counter1, counter2 = st.columns(2)
with counter1:
    st.markdown("**Total Count Up**")
    counter1_text = st.markdown("0")

    st.markdown("**Car Count Up**")
    counter3_text = st.markdown("0")

    st.markdown("**Truck Count Up**")
    counter4_text = st.markdown("0")

    st.markdown("**Motorbike Count Up**")
    counter5_text = st.markdown("0")

# Display the initial values in the second main column
with counter2:
    st.markdown("**Total Count Down**")
    counter2_text = st.markdown("0")

    st.markdown("**Car Count Down**")
    counter6_text = st.markdown("0")

    st.markdown("**Truck Count Down**")
    counter7_text = st.markdown("0")

    st.markdown("**Motorbike Count Down**")
    counter8_text = st.markdown("0")

def process_frame(frame):
    img = cv2.resize(frame, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = list()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            currentClass = model.names[cls]
            if currentClass == 'car' and conf > 0.5:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

            elif currentClass == "truck":
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

            elif currentClass == "motorbike":
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

            cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), thickness=5)
            cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), thickness=5)

            tracks = tracker.update_tracks(detections, frame=img)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id

                bbox = track.to_ltrb()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                w, h = x2 - x1, y2 - y1

                co_ord = [x1, y1]

                if track_id not in coordinatesDict:
                    coordinatesDict[track_id] = co_ord
                else:
                    if len(coordinatesDict[track_id]) > 2:
                        del coordinatesDict[track_id][-3:-1]
                    coordinatesDict[track_id].append(co_ord[0])
                    coordinatesDict[track_id].append(co_ord[1])
                estimatedSpeedValue = 0
                if len(coordinatesDict[track_id]) > 2:
                    location1 = [coordinatesDict[track_id][0], coordinatesDict[track_id][2]]
                    location2 = [coordinatesDict[track_id][1], coordinatesDict[track_id][3]]
                    estimatedSpeedValue = estimatedSpeed(location1, location2)

                cls = track.get_det_class()
                currentClass = model.names[cls]

                # Change the color of the bounding box to red if the vehicle is exceeding the speed limit
                if estimatedSpeedValue > speed_limit:
                    clsColor = (0, 0, 255)
                    # Update the maximum speed of the vehicle
                    if track_id not in max_speeds or estimatedSpeedValue > max_speeds[track_id]:
                        max_speeds[track_id] = estimatedSpeedValue
                        # Update the warning message for the vehicle
                        warning_container.warning(
                            f"Alarm: {currentClass} with ID {track_id} is exceeding the speed limit of {speed_limit} km/h with a maximum speed of {max_speeds[track_id]} km/h")
                else:
                    clsColor = get_class_color(currentClass)

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=clsColor)

                cvzone.putTextRect(
                    img,
                    text=f"{model.names[cls]} {estimatedSpeedValue} km/h",
                    pos=(max(0, x1), max(35, y1)),
                    colorR=clsColor,
                    scale=1,
                    thickness=1,
                    offset=2)

                cx, cy = x1 + w // 2, y1 + h // 2

                cv2.circle(img, (cx, cy), radius=5, color=clsColor, thickness=cv2.FILLED)

                if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[3] + 15:
                    if totalCountUp.count(track_id) == 0:
                        totalCountUp.append(track_id)
                        clsCounterUp[currentClass] += 1
                        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (255, 255, 255),
                                 thickness=3)

                if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[3] + 15:
                    if totalCountDown.count(track_id) == 0:
                        totalCountDown.append(track_id)
                        clsCounterDown[currentClass] += 1
                        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (255, 255, 255),
                                 thickness=3)
    # Update the values in real-time
    counter1_text.markdown(str(len(totalCountUp)))
    counter2_text.markdown(str(len(totalCountDown)))
    counter3_text.markdown(str(clsCounterUp["car"]))
    counter4_text.markdown(str(clsCounterUp["truck"]))
    counter5_text.markdown(str(clsCounterUp["motorbike"]))
    counter6_text.markdown(str(clsCounterDown["car"]))
    counter7_text.markdown(str(clsCounterDown["truck"]))
    counter8_text.markdown(str(clsCounterDown["motorbike"]))

    return img


if task_name == task_list[0]:
    cap = cv2.VideoCapture(0)
    if st.button("Start"):
        # Streamlit container to display video frames
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode()
            stframe.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
        cap.release()

elif task_name == task_list[1]:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file.seek(0)

            if st.button("Submit"):
                cap = cv2.VideoCapture(temp_file.name)
                stframe = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = process_frame(frame)
                    _, buffer = cv2.imencode('.jpg', img)
                    frame_base64 = base64.b64encode(buffer).decode()
                    stframe.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
                cap.release()

elif task_name == task_list[2]:
    rtsp_link = st.text_input("Enter the RTSP link")
    if rtsp_link:
        cap = cv2.VideoCapture(rtsp_link)
        if st.button("Submit"):
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = process_frame(frame)
                _, buffer = cv2.imencode('.jpg', img)
                frame_base64 = base64.b64encode(buffer).decode()
                stframe.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)

            cap.release()
