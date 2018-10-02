from math import sqrt
import cv2
from Libs.Visualization_utils import draw_simple_box, load_colors, undistort_img


def detect_platforms(img, detector, boxes_lst):
    heigth, width, _ = img.shape
    platforms_dict = {}
    for box in boxes_lst:
        number = detector.classificate_number(img, box, heigth, width)
        platforms_dict[tuple(box)] = list(number)
    return platforms_dict


def get_distance(xy1, xy2):
    X = xy1[0] - xy2[0]
    Y = xy1[1] - xy2[1]
    distance = sqrt(X**2 + Y**2)
    return distance


def find_nearest(platform_dict_last, platform_dict_now):
    if len(platform_dict_last) == len(platform_dict_now):
        for xy_last in platform_dict_last:
            min_distance = 5000
            actual_xy = (None, None)
            for xy_now in platform_dict_now:
                distance = get_distance(xy_last, xy_now)
                if distance <= min_distance:
                    min_distance = distance
                    actual_xy = xy_now
            for num in platform_dict_last[xy_last]:
                platform_dict_now[actual_xy].append(num)
        return platform_dict_now

    elif len(platform_dict_last) < len(platform_dict_now):
        d_now = platform_dict_now.copy()
        new_dict = {}
        for xy_last in platform_dict_last:
            min_distance = 5000
            actual_xy = (None, None)
            for xy_now in d_now:
                distance = get_distance(xy_last, xy_now)
                if distance <= min_distance:
                    min_distance = distance
                    actual_xy = xy_now
            for num in platform_dict_last[xy_last]:
                platform_dict_now[actual_xy].append(num)
            new_dict[actual_xy] = platform_dict_now[actual_xy]
            del platform_dict_now[actual_xy]
        for key in platform_dict_now:
            new_dict[key] = platform_dict_now[key]
        return new_dict

    elif len(platform_dict_last) > len(platform_dict_now):
        dict_last = platform_dict_last.copy()
        for xy_now in platform_dict_now:
            min_distance = 5000
            actual_xy = (None, None)
            for xy_last in dict_last:
                distance = get_distance(xy_last, xy_now)
                if distance <= min_distance:
                    min_distance = distance
                    actual_xy = xy_last
            for num in dict_last[actual_xy]:
                platform_dict_now[xy_now].append(num)
            del dict_last[actual_xy]
        for key in dict_last:
            platform_dict_now[key] = dict_last[key]
        return platform_dict_now

def get_class(platform_dict):
    # platform_dict is a dict with structure:
    #   key is a platforms' position on image
    #   value is a platform's numbers list
    predicted_classes = list(platform_dict.values())
    platform_number = max(predicted_classes)
    return platform_number

def init_platforms(video_stream, detector, undistort_params, frames_count=100, img_resize=(640, 360),
                   visualize=True, colors_path="colors.txt", ):
    colors = load_colors(colors_path)
    counter = 0
    RUN = True
    platform_last = {}
    if visualize:
        cv2.namedWindow("init", cv2.WINDOW_NORMAL)
    if frames_count < 0 or frames_count > 2000:
        print("frames_count  invalidate! \n 0 < frame_count < 2000")
    while counter < frames_count and RUN:
        ret, img = video_stream.read()
        if not ret:
            print("Cannot fint camera!")
            RUN = not RUN
        else:
            #img = undistort_img(img, undistort_params)
            img = cv2.resize(img, img_resize)
            boxes_lst = detector.detect_objects(img)
            if visualize:
                vis_img = img.copy()
                for box in boxes_lst:
                    pt1, pt2 = detector.tf_box_to_img_box(box, img_resize[0], img_resize[1])
                    draw_simple_box(vis_img, pt1, pt2)
                cv2.imshow("init", vis_img)
                cv2.waitKey(5)
            boxes_dict = detect_platforms(img, detector, boxes_lst)
            platform_dict = find_nearest(platform_last, boxes_dict)
            if len(boxes_dict):
                platform_last = platform_dict
        counter += 1

    platforms = []
    for platform in platform_dict:
        if len(platform_dict[platform]) > frames_count*0.4:
            platforms.append([max(platform_dict[platform]), platform, colors[list(platform_dict.keys()).index(platform)]])
    return  platforms

