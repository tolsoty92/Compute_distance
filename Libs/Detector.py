import cv2
import os
import tensorflow as tf
import numpy as np
from math import sqrt
from Label_map_util import *
from Visualization_utils import visualize_boxes_and_labels_on_image_array

class Num_detector:

    def __init__(self, path_to_graph, path_to_labels, classifier, cl_labels):
        # Init detector and classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        self.label_map = load_labelmap(path_to_labels)
        categories = convert_label_map_to_categories(self.label_map, max_num_classes=10,
                                                                    use_display_name=True)
        self.category_index = create_category_index(categories)

        self.label_lines = [line.rstrip() for line
                       in tf.gfile.GFile(cl_labels)]

        with tf.gfile.FastGFile(classifier, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as self.classifier_sess:
            self.softmax_tensor = self.classifier_sess.graph.get_tensor_by_name('final_result:0')

    def detect_objects(self, img):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np = img.copy()
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        img, boxes_dict = visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4, max_boxes_to_draw=10)
        return img, boxes_dict      # Return visualized image and boxes              

    def classificate_number(self, img, box, heigth, width):
        # Classificate data in box
        ymin, xmin, ymax, xmax = box
        ymin, ymax = int(ymin*heigth), int(ymax*heigth)
        xmax, xmin =  int(xmax*width), int(xmin*width)
        number = img[ymin:ymax, xmin:xmax]
        number_string = cv2.imencode('.jpg', number)[1].tostring()
        predictor = self.classifier_sess.run(self.softmax_tensor,
                                        {'DecodeJpeg/contents:0': number_string})
        top_k = predictor[0].argsort()[-len(predictor[0]):][::-1]
        scores = []
        numbers = []
        for node_id in top_k:
             numbers.append(self.label_lines[node_id])
             scores.append(predictor[0][node_id])
        return(numbers[scores.index(max(scores))])  # Return class of object in box

    def get_box_center(self, box, img_w, img_h):
        # Get box center
        y_min, x_min, y_max, x_max = box

        c_x = int(np.mean([x_min, x_max]) * img_w)
        c_y = int(np.mean([y_min, y_max]) * img_h)

        return (c_x, c_y)

    def compute_distance(self, point1, point2):
        distance = sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
        return distance