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
        im = np.ones(card_size) * 255
        # get all attribute data and convert it to viable versions

        count = int(self.attributes['count'].class_data[0])

        radii = self.attributes['shape'].data
        thetas = map(lambda t: t * np.pi / 180, range(-180, 180, 360 / 180))
        shape = zip(thetas, radii)

        color = self.attributes['color'].class_data
        hue = np.mod(np.arctan2(color[1], color[0]) * 180 / np.pi + 360, 360) / 2
        color = cv2.cvtColor(np.array((int(hue), 255, 255), dtype=np.uint8).reshape((1, 1, 3)),
                             cv2.COLOR_HSV2BGR)
        color = tuple(map(int, color[0, 0, :]))

        infill = self.attributes['infill'].class_data
        saturation = infill[0] ** 2
        infill = cv2.cvtColor(np.array((int(hue), int(saturation), 255), dtype=np.uint8).reshape((1, 1, 3)),
                              cv2.COLOR_HSV2BGR)
        infill = tuple(map(int, infill[0, 0, :]))

        # figure out how to fit all count shapes into im
        for i in xrange(count):
            h = shape[45][1] + shape[135][1]
            w = shape[0][1] + shape[90][1]
            scale = min(im.shape[0] / h * 0.8, im.shape[1] / w * .8 / count)

            cy = int(im.shape[0] / 2)
            cx = int((2 * i + 1) * im.shape[1] / (2 * count))

            shape_contour = ip.contour_polar2xy(shape, scale, (cy, cx))

            cv2.fillPoly(im, [shape_contour], infill)
            cv2.drawContours(im, [shape_contour], -1, color, 5)


        im = im.astype(np.uint8)

        #cv2.imshow("TEST", im)
        #cv2.waitKey(0)

        return im
