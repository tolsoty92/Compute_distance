import cv2
import os
import numpy as np
from  Libs.Detector import  Num_detector
from Libs.Init_platform import *
from  Libs.Platform import Platform
from Libs.Visualization_utils import *


# Uncommet to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PROJECT_DIR = os.getcwd()

# There are pathes to detector and classifier files.
DATA_PATH = os.path.join(PROJECT_DIR , "data")
DETECTOR_NAME = os.path.join(DATA_PATH, "detector/rcnn_22925.pb")
DETECTOR_LABELS_NAME = os.path.join(DATA_PATH, "detector/label_map.pbtxt")
CLASSIFIER_NAME = os.path.join(DATA_PATH, "classifier/output_graph.pb")
CLASSIFIER_LABELS_NAME = os.path.join(DATA_PATH, "classifier/output_labels.txt")
COLORS_PATH = os.path.join(DATA_PATH, "colors.txt")

# IP camera parametrs
camera_params = np.load(os.path.join(DATA_PATH, "ip_cam_params.npz"))

newcameramtx = camera_params["newcameramtx"]
roi = camera_params["roi"]
mtx = camera_params["mtx"]
dist = camera_params["dist"]

undist_params = [mtx, dist, newcameramtx, roi]

# Initialize camera's stream and visualization window.
stream = cv2.VideoCapture("/home/user/PycharmProjects/computing_distanse/246.mp4")
WIDTH, HEIGHT = 640, 480
IMAGE_CENTER = (WIDTH//2, HEIGHT//2)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("vis_image", cv2.WINDOW_NORMAL)


# Initialize detector.
detector =Num_detector(DETECTOR_NAME, DETECTOR_LABELS_NAME,
                                         CLASSIFIER_NAME, CLASSIFIER_LABELS_NAME)

# While RUN is True the programm will be running.
RUN = True

def main():
    RUN = True
    platforms_lst= init_platforms(stream, detector, undist_params,
                                  frames_count=30, img_resize=(640, 360),
                                  colors_path=COLORS_PATH)

    platforms = [Platform(p[0], p[1], p[2]) for p in platforms_lst]
    ret, img = stream.read()

    #Undistortion doesn't work!
    #img = undistort_img(img, undist_params)
    img = cv2.resize(img, (640, 360))
    h, w, _ = img.shape

    for p in platforms:
        pt1, pt2 = p.get_bbox(w, h)
        bbox = p.init_tracker_bbox(pt1, pt2)
        p.create_tracker(img, bbox)
        p.update_position(bbox)

    while RUN and stream.isOpened():
        ret, img = stream.read()
        #img = undistort_img(img,undist_params)
        img = cv2.resize(img, (640, 360))
        vis_img = img.copy()
        for p in platforms:
            bbox = p.update_tracker(img)
            if bbox:
                p.update_position(bbox)
                draw_trajectory(vis_img, p.trajectory, p._color)
        cv2.imshow("vis_image", vis_img)
        if cv2.waitKey(10) & 0xFF == 27 or not ret:
            RUN = not RUN

if __name__ == "__main__":
    main()