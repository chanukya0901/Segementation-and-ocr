import numpy as np
import cv2


def draw_polygon(image, points):
    cv2.fillPoly(image, [points], color=255)

def get_segmask(points,shape):
    mask= np.zeros(shape=shape, dtype=np.uint8)
    points_x=points[0]
    points_y=points[1]
    points = [(x, y) for x, y in zip(points_x, points_y)]
    draw_polygon(mask, np.array(points, dtype=np.int32))
    return mask