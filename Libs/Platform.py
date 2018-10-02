import cv2

class Platform():
    def __init__(self, name, position, color):
        self._name = name
        self.position = position
        self._color = color
        self.trajectory = []

    def __str__(self):
        return self._name

    def points_to_position(self, x, y, w, h):
        cnt = (int(x) + int(w // 2), int(y) + int(h // 2))
        return cnt

    def update_position(self, bbox):
        x, y, w, h = bbox
        self.position = self.points_to_position(x, y, w, h)
        self.trajectory.append( self.position)

    def get_bbox(self, w, h):
        pt1 = (int(self.position[1]*w), int(self.position[0]*h))
        pt2 = (int(self.position[3]*w), int(self.position[2]*h))
        return pt1, pt2

    def init_tracker_bbox(self, pt1, pt2):
        x_min, x_max = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y_min, y_max = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
        w, h = x_max -x_min, y_max -y_min
        return x_min, y_min, w, h

    def append_trajectory(self, point):
        self.trajectory.append(point)


###############################
###############################
                    #TRACKERS
###############################
###############################

    def create_tracker(self, img, bbox):
        self.tracker = cv2.TrackerMedianFlow_create()
        self.tracker.init(img, bbox)

    def update_tracker(self, img):
        ret, bbox = self.tracker.update(img)
        if ret:
            return bbox
        else:
            return None




