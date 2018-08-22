import cv2
import os
from  Libs.Detector import  Num_detector

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PROJECT_DIR = os.getcwd()


# There are pathes to detector and classifier files.
DATA_PATH = PROJECT_DIR + "/data/"
DETECTOR_NAME = "detector/ssd13.pb"
DETECTOR_LABELS_NAME = "detector/label_map.pbtxt"
CLASSIFIER_NAME = "classifier/output_graph.pb"
CLASSIFIER_LABELS_NAME = "classifier/output_labels.txt"

# Initialize camera's stream and visualization window.
stream = cv2.VideoCapture(0)
WIDTH, HEIGHT = 640, 480
IMAGE_CENTER = (WIDTH//2, HEIGHT//2)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Initialize detector.
detector =Num_detector(DATA_PATH+DETECTOR_NAME, DATA_PATH+DETECTOR_LABELS_NAME,
                       DATA_PATH+CLASSIFIER_NAME, DATA_PATH+CLASSIFIER_LABELS_NAME)

# While RUN is True the programm will be running.
RUN = True

# Draw X mark in the center of image.
def draw_img_center(img, w, h):
    cv2.line(img, (0, h//2), (w, h//2), (255, 0, 0), thickness=2)
    cv2.line(img, (w//2, 0), (w//2, h), (255, 0, 0), thickness=2)


def main(RUN):
    ret, img = stream.read()        # Read data from camera
    img = cv2.resize(img, (WIDTH, HEIGHT))
    heigth, width, _ = img.shape    # Get image shape
    print(heigth, width)
    if ret:                         # If camea returns data
        img2, boxes_dict  = detector.detect_objects(img)    # Get boxes of objects and drawing them
        #draw_img_center(img2, width, heigth)
        for box in boxes_dict:                         
            number = detector.classificate_number(img, boxes_dict[box], heigth, width)      # Classificate data
            cntr = detector.get_box_center(boxes_dict[box], width, heigth)                  # Compute boxes center
            #print(cntr)
            cv2.circle(img2, cntr, 5, (0, 0, 255), 5)                                       # Draw center
            cv2.putText(img2, number, (10, 30), 2, 1, (255, 0, 0), thickness=2)             # Put class data on image
            distance_in_pix = detector.compute_distance(cntr, IMAGE_CENTER)
            #cv2.putText(img2, str(distance_in_pix), (10, 40), 1, 1, (255, 0, 0), thickness=3)# Put distance data on image
        cv2.imshow("Image", img2)                                                           # Show image
        if cv2.waitKey(10) & 0xFF == 27:                                                    # If press ESC RUN = False
            RUN = not RUN
    else:
        print("Cannot fint camera!")
        RUN = not RUN
    return RUN

if __name__ == "__main__":
    while RUN:
        RUN = main(RUN)

    stream.release()
    cv2.destroyAllWindows()
