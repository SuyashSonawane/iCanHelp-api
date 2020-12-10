from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from absl import app, flags, logging
from absl.flags import FLAGS
import time, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import os
import pyttsx3

engine = None

flags.DEFINE_string("classes", "./data/labels/coco.names", "path to classes file")
flags.DEFINE_string("weights", "./weights/yolov3.tf", "path to weights file")
flags.DEFINE_boolean("tiny", True, "yolov3 or yolov3-tiny")
flags.DEFINE_integer("size", 416, "resize images to")
flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to video file or number for webcam)"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_string(
    "output_format", "XVID", "codec used in VideoWriter when saving video to file"
)
flags.DEFINE_integer("num_classes", 80, "number of classes in the model")

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

import os

dirname = os.path.dirname(__file__)


class ContentLoaderThread(QThread):
    signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        model_filename = os.path.join(dirname, "model_data/mars-small128.pb")
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        tracker = Tracker(metric)

        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        yolo = YoloV3Tiny(classes=80)
        yolo.load_weights(os.path.join(dirname, "weights/yolov3-tiny.tf"))

        class_names = [
            c.strip()
            for c in open(os.path.join(dirname, "data/labels/coco.names",)).readlines()
        ]

        if os.path.isfile(os.path.join(dirname, "detection.txt")):
            os.remove(os.path.join(dirname, "detection.txt"))

        self.signal.emit([yolo, class_names, encoder, tracker])


class CamLoaderThread(QThread):
    signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        cap = cv2.VideoCapture("http://192.168.43.1:8080/shot.jpg")
        rect, frame = cap.read()
        if rect:
            self.signal.emit([cap])
        else:
            self.signal.emit([])


class SpeakerThread(QThread):
    signal = pyqtSignal(list)
    text = None

    def __init__(self, parent=None):
        super().__init__(parent)

    def setText(self, text):
        self.text = text

    def run(self):
        engine.say(self.text)
        engine.runAndWait()


class Ui_MainWindow(QtWidgets.QWidget):

    speakerSignal = pyqtSignal(str)
    id_history = []
    running = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1038, 620)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(890, 210, 131, 141))
        self.startButton.setCheckable(False)
        self.startButton.setObjectName("startButton")
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(890, 210, 131, 141))
        self.stopButton.setCheckable(False)
        self.stopButton.setObjectName("stopButton")
        self.stopButton.hide()
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 20, 861, 551))
        self.label.setObjectName("label")
        self.fpsLabel = QtWidgets.QLabel(self.centralwidget)
        self.fpsLabel.setGeometry(QtCore.QRect(10, 10, 50, 50))
        self.fpsLabel.setObjectName("Fps label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1038, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.startButton.clicked.connect(self.camLoader)
        self.stopButton.clicked.connect(self.stop)
        self.startButton.setDisabled(True)
        # Threads

        self.speaker = SpeakerThread()
        self.speakerSignal.connect(self.speaker.setText)
        # self.speakerSignal.emit("Loading Contents")
        # self.speaker.start()
        self.loader = ContentLoaderThread()
        self.loader.signal.connect(self.contentLoaded)
        self.loader.start()
        self.statusbar.showMessage("Loading ML model and associated contents")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "I can Help"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.label.setText(_translate("MainWindow", "Images show here"))
        self.fpsLabel.setText(_translate("MainWindow", "FPS Counter"))

    def contentLoaded(self, data):
        self.speak("Contents Loaded!")
        self.statusbar.showMessage("Contents Loaded!")
        self.yolo = data[0]
        self.class_names = data[1]
        self.encoder = data[2]
        self.tracker = data[3]
        self.loader.terminate()
        self.startButton.setDisabled(False)

    def camLoader(self):
        self.speak("Attempting to connect to IP camera")
        self.statusbar.showMessage("Attempting to connect to IP camera")
        self.t = CamLoaderThread()
        self.t.signal.connect(self.show)
        self.t.start()

    def speak(self, string):
        self.speakerSignal.emit(string)
        self.speaker.start()

    def show(self, cap):
        self.t.terminate()
        if len(cap) > 0:
            self.speak("Connected")
            self.statusbar.showMessage("Connected")
            self.running = True
            self.startButton.hide()
            self.stopButton.show()
            while self.running:
                cap = cv2.VideoCapture("http://192.168.43.1:8080/shot.jpg")
                rect, frame = cap.read()
                if rect:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.performInference(rgbImage)
                    cv2.resize(rgbImage, (400, 300))
                    h, w, ch = rgbImage.shape
                    step = ch * w
                    convertToQtFormat = QtGui.QImage(
                        rgbImage.data, w, h, step, QtGui.QImage.Format_RGB888
                    )
                    self.label.setPixmap(QtGui.QPixmap.fromImage(convertToQtFormat))
                    if cv2.waitKey(0) & 0xFF == ord("q"):
                        pass
                else:
                    self.statusbar.showMessage("Internal Error")
                    return
            self.label.setText("Video Feed Stoped")
            cap.release()
        else:
            self.speak("Cannot connect to IP camera")
            self.statusbar.showMessage("Cannot connect to IP camera")

    def stop(self):
        self.running = False
        self.startButton.show()
        self.stopButton.hide()
        self.speak("Stopped")
        self.statusbar.showMessage("Stopped")

    def performInference(self, rgbImage):
        img_in = tf.expand_dims(rgbImage, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        boxes, scores, classes, nums = self.yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(rgbImage, boxes[0])
        features = self.encoder(rgbImage, converted_boxes)
        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(
                converted_boxes, scores[0], names, features
            )
        ]

        # initialize color map
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores
        )
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            midY = (bbox[3] + bbox[1]) / 2
            midX = (bbox[2] + bbox[0]) / 2
            if track.track_id not in self.id_history:
                if midX < 250:
                    self.speak(f"{class_name} on left")
                    # print(f"{class_name} on left")
                elif midX > 480:
                    self.speak(f"{class_name} on right")
                    # print(f"{class_name} on right")
                else:
                    self.speak(f"{class_name} in the middle")
                    # print(f"{class_name} in the middle")

                self.id_history.append(track.track_id)

            distance = round((1 - (bbox[3] - bbox[1])), 1)
            # print(distance)
            cv2.rectangle(
                rgbImage,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.rectangle(
                rgbImage,
                (int(bbox[0]), int(bbox[1] - 30)),
                (
                    int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                    int(bbox[1]),
                ),
                color,
                -1,
            )
            cv2.putText(
                rgbImage,
                class_name + "-" + str(track.track_id),
                (int(bbox[0]), int(bbox[1] - 10)),
                0,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                rgbImage, "0", (int(midX), int(midY)), 0, 0.75, (255, 255, 255), 2,
            )
            # cv2.imshow("output", rgbImage)


if __name__ == "__main__":
    import sys

    engine = pyttsx3.init()
    appctxt = ApplicationContext()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    exit_code = appctxt.app.exec_()  # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
