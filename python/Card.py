import cv2
import numpy as np

import image_processing as ip


class Card:

    def __init__(self, cid, loc):
        self.cid = cid
        self.loc = loc    # 4 vertices of quadrilateral in image
        self.attributes = {}

    def update_attribute(self, attribute):
        self.attributes[attribute.name] = attribute

    def visualize_card(self, card_size):
        im = np.ones(card_size)*255
        cy = int(im.shape[0] / 2)
        cx = int(im.shape[1] / 2)
        # get all attribute data and convert it to viable versions

        # TODO! Make this so it uses centroid data! That way the different shapes and colors match up

        count = self.attributes['count'].data
        count = count[0]
        radii = self.attributes['shape'].data
        thetas = map(lambda t: t * np.pi / 180, range(-180, 180, 360 / 180))
        shape = zip(thetas, radii)
        shape_contour = ip.contour_polar2xy(shape, scale=100, center=(cy, cx))

        color = self.attributes['color'].data
        hue = np.mod((np.arctan2(color[0], color[1]) + 2 * np.pi) * 180 / np.pi, 360) * 100 / 360
        color = cv2.cvtColor(np.array((int(hue), 100, 255), dtype=np.uint8).reshape((1, 1, 3)),
                             cv2.COLOR_HSV2BGR_FULL)
        color = tuple(map(int, color[0, 0, :]))


        infill = self.attributes['infill'].data
        saturation = infill[0] ** 2
        infill = cv2.cvtColor(np.array((int(hue), int(saturation), 255), dtype=np.uint8).reshape((1, 1, 3)),
                              cv2.COLOR_HSV2BGR_FULL)
        infill = tuple(map(int, infill[0, 0, :]))

        cv2.fillPoly(im, [shape_contour], infill)
        cv2.drawContours(im, [shape_contour], -1, color, 2)
        im = im.astype(np.uint8)

        return im
