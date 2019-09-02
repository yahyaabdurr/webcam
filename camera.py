import cv2
import numpy as np
import pickle

class VideoCamera(object):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('E:\\PyCharm Project\\Face Recognition\\haarcascade_frontalface_alt.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("E:\\PyCharm Project\\Face Recognition\\train\\trainned.yml")
        with open("E:\\PyCharm Project\\Face Recognition\\pickle\\labellbph.pickle", "rb") as f:
            og_labels = pickle.load(f)
            self.labels = {v: k for k, v in og_labels.items()}
        self.font = cv2.FONT_HERSHEY_COMPLEX
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        cod = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('out.avi', cod, 20.0, (640, 480))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    def __del__(self):
        self.video.release()
        self.out.release()
    
    def get_frame(self):
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2, minSize=(30, 30))
        for (x, y, w, h) in faces:
            # Num = []
            roi_gray = gray[y:y + h, x:x + w]
            # roi_image = image[y:y+h, x:x+w]
            color = (255, 0, 0)  # BGR 0-255
            stroke = 3
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), color, stroke)
            id, conf = self.recognizer.predict(roi_gray)
            profile = self.labels
            nama = self.labels[id]
            if (profile != None):
                cv2.putText(image, nama, (x, y - 40), self.font, 2, (255, 255, 255), 3)

            else:
                Id = "Unknown"
                cv2.rectangle(image, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
                cv2.putText(image, str(Id), (x, y - 40), self.font, 2, (255, 255, 255), 3)
        self.out.write(image)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()