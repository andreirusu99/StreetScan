import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pynput import keyboard
import time

# Hyper-parameters
FILL_TH = 0.8
IMAGE_PATH = '../data/images'
ANNOTATION_PATH = '../data/annotations/min_max_depth.txt'

pressed_key = None
depth, color = None, None
minD, maxD = 0, 0


def capture_frame():
    timestamp = time.time()
    cv2.imwrite(IMAGE_PATH + "/color/" + "{}_bgr8.jpg".format(timestamp), color)
    cv2.imwrite(IMAGE_PATH + "/depth/" + "{}_z16.jpg".format(timestamp), depth)
    annotation_file = open(ANNOTATION_PATH, 'a')
    annotation_file.write("{} {} {}\n".format(timestamp, minD, maxD))
    annotation_file.close()
    print(time.time(), "SAVED!")


def on_press(key):
    global pressed_key
    try:
        if key.char == 'g':
            print(time.time(), key.char, "SAVING...")
            pressed_key = key.char
            capture_frame()

    except AttributeError:
        # for special keys
        pass


# map a distance between min and max to between 0 and 1
def map_dist(dist, min_dist, max_dist):
    return (dist - min_dist) / (max_dist - min_dist)


def crop_roi(img, roi):
    y, x = img.shape
    startx = int(x * (1 - roi))
    endx = int(x * roi)
    starty = int(y * (1 - roi))
    endy = int(y * roi)
    return img[starty:endy, startx:endx]


def depth_to_grayscale(depth_map):
    # multiply depth image with the depth scale
    depth_map = depth_map.astype(float)
    depth_map *= depth_scale * 100  # cm

    # crop the images to a central ROI
    # depth_map = crop_roi(depth_map, 0.9)

    # compute the max and min distances in the image and the fill percentage
    max_depth = np.max(depth_map)
    valid_pixels = depth_map[np.nonzero(depth_map)]
    fill_percentage = float(np.size(valid_pixels)) / np.size(depth_map)

    min_depth = np.min(valid_pixels)

    # map distance values between 0 and 1
    depth_map = map_dist(depth_map, min_depth, max_depth)
    depth_map = 1.0 - depth_map

    # filtering (smoothing the image)
    depth_map = cv2.convertScaleAbs(depth_map, None, 255, 0)
    depth_image = cv2.medianBlur(depth_map, 7)

    return depth_image, min_depth, max_depth, fill_percentage


if __name__ == '__main__':
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Set mode to high density
    depth_sensor.set_option(rs.option.visual_preset, 4)
    print(depth_sensor.get_option_value_description(rs.option.visual_preset, 4))

    # Collect keyboard events
    with keyboard.Listener(
            on_press=on_press,
            on_release=None) as listener:

        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_map = np.asanyarray(depth_frame.as_frame().get_data())
                color_image = np.asanyarray(color_frame.as_frame().get_data())

                depth_image, min_depth, max_depth, fill_percentage = depth_to_grayscale(depth_map)

                print("\t{:.2f}cm - {:.2f}cm, Fill: {:.4}%".format(min_depth, max_depth, fill_percentage * 100))

                # TODO: De-noise the color image

                # visualize both streams
                cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Color', color_image)
                cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Depth', depth_image)

                depth = depth_image
                color = color_image
                minD = min_depth
                maxD = max_depth

                cv2.waitKey(50)

        finally:
            pipeline.stop()
            listener.join()

    # End main
