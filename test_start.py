import cv2
import os
import numpy as np
from  Libs.Detector import  Num_detector
from Libs.Init_platform import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PROJECT_DIR = os.getcwd()

# There are pathes to detector and classifier files.
DATA_PATH = os.path.join(PROJECT_DIR , "data")
DETECTOR_NAME = os.path.join(DATA_PATH, "detector/rcnn_22925.pb")
DETECTOR_LABELS_NAME = os.path.join(DATA_PATH, "detector/label_map.pbtxt")
CLASSIFIER_NAME = os.path.join(DATA_PATH, "classifier/output_graph.pb")
CLASSIFIER_LABELS_NAME = os.path.join(DATA_PATH, "classifier/output_labels.txt")

# IP camera parametrs

camera_params = np.load(os.path.join(DATA_PATH, "ip_cam_params.npz"))

newcameramtx = camera_params["newcameramtx"]
roi = camera_params["roi"]
mtx = camera_params["mtx"]
dist = camera_params["dist"]

# Initialize camera's stream and visualization window.
stream = cv2.VideoCapture("/home/user/PycharmProjects/computing_distanse/569.mp4")
WIDTH, HEIGHT = 640, 480
IMAGE_CENTER = (WIDTH//2, HEIGHT//2)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Initialize detector.
detector =Num_detector(DETECTOR_NAME, DETECTOR_LABELS_NAME,
                                         CLASSIFIER_NAME, CLASSIFIER_LABELS_NAME)

# While RUN is True the programm will be running.
RUN = True

# Draw X mark in the center of image.

def main():
    RUN = True
    platform_last = {}
    while RUN:
        ret, img = stream.read()        # Read data from camera
        #img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        #x,y,w,h = roi
        #img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (640, 360))
        height, width, _ = img.shape    # Get image shape
        if ret:                                                                                                                                    # If camea returns data
            boxes_lst  = detector.detect_objects(img)
            for box in boxes_lst:
                print(box)
                #draw_simple_box(img, box[])
            boxes_dict = detect_platforms(img, detector, boxes_lst)                            # Get boxes of objects and drawing them
            platform_dict = find_nearest(platform_last, boxes_dict)
            if len(boxes_dict):
                platform_last = platform_dict
            for box in boxes_lst:
                number = detector.classificate_number(img, box, height, width)        # Classificate data
                cntr = detector.get_box_center(box, width, height)                              # Compute boxes center
                #cv2.circle(img2, cntr, 5, (0, 0, 255), 5)                                                                       # Draw center
                #cv2.putText(img2, number, (10, 30), 2, 1, (255, 0, 0), thickness=2)                        # Put class data on image
                #distance_in_pix = detector.compute_distance(cntr, IMAGE_CENTER)
                #cv2.putText(img2, str(distance_in_pix), (10, 40), 1, 1, (255, 0, 0), thickness=3)  # Put distance data on image
            #cv2.imshow("Image", img2)                                                                                          # Show image
            if cv2.waitKey() & 0xFF == 27:                                                                                 # If press ESC RUN = False
                RUN = not RUN
        else:
            print("Cannot fint camera!")
            RUN = not RUN


if __name__ == "__main__":
    main()
    stream.release()
    cv2.destroyAllWindows()
