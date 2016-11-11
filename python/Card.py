import cv2
import numpy as np

from image_processing import im_mask, contour_polar2xy, contour_xy2polar


class Card:
    def __init__(self, cid, loc, card_image):
        self.cid = cid
        self.loc = loc  # 4 vertices of quadrilateral in image
        self.contours = self.find_contours(card_image)
        self.attributes = {}
        self.update_attribute(Count(self.contours))
        self.update_attribute(Shape(self.contours))
        self.update_attribute(Color(card_image, self.contours))
        self.update_attribute(Infill(card_image, self.contours))

    @staticmethod
    def find_contours(card_image):
        card_image_mask = (255 - im_mask(card_image))
        contours, _ = cv2.findContours(card_image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return filter(lambda contour: (card_image_mask.size / 20) < cv2.contourArea(contour), contours)

    def update_attribute(self, attribute):
        self.attributes[attribute.name] = attribute

    def visualize_card(self, card_size):
        im = np.ones(card_size) * 255
        # get all attribute data and convert it to viable versions

        count = int(self.attributes['count'].class_data[0])

        radii = self.attributes['shape'].class_data
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

            shape_contour = contour_polar2xy(shape, scale, (cy, cx))

            cv2.fillPoly(im, [shape_contour], infill)
            cv2.drawContours(im, [shape_contour], -1, color, 5)

        return im.astype(np.uint8)


class AbstractAttribute:
    def __init__(self, name=None, default_data=None):
        self.name = name
        self.data = default_data
        self.class_data = None
        self.contours = None
        self.classification = None

    def classify(self, classification, centroid):
        self.classification = classification
        self.class_data = centroid

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.classification == other.classification
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.classification is None:
            raise ValueError('Card not classified')
        return self.classification


class Count(AbstractAttribute):
    def __init__(self, contours):
        AbstractAttribute.__init__(self, name='count', default_data=None)
        self.parse_image(contours)

    def parse_image(self, contours):
        self.data = [len(contours), len(contours) ** 2]


class Shape(AbstractAttribute):
    def __init__(self, contours):
        AbstractAttribute.__init__(self, name='shape', default_data=None)
        self.parse_image(contours)

    def parse_image(self, contours):

        shapes_radii = np.mean(
            map(lambda contour: map(lambda (t, r): r, contour_xy2polar(contour, n_points=180)), contours),
            axis=0)

        # smooth
        kernel_size = 7
        kernel = np.ones(kernel_size) / float(kernel_size)
        shapes_radii = np.hstack((shapes_radii[-kernel_size / 2 + 1:], shapes_radii, shapes_radii[:kernel_size / 2]))
        shapes_radii = np.convolve(shapes_radii, kernel, mode='valid')
        self.data = shapes_radii.tolist()


class Color(AbstractAttribute):
    def __init__(self, card_image, contours):
        AbstractAttribute.__init__(self, name='color', default_data={})
        self.parse_image(card_image, contours)

    def parse_image(self, card_image, contours):

        shape_mask = np.zeros(card_image.shape)
        cv2.drawContours(shape_mask, contours, -1, (1, 1, 1), -1)
        card_image_hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

        shape_hsv = card_image_hsv * shape_mask
        shape_mask = shape_hsv[:, :, 1] > 64
        shape_hue = shape_hsv[shape_mask, 0].astype(np.int32) * 2  # rescale to be [0, 360)
        shape_hue_radians = shape_hue * (np.pi / 180)
        xs = np.cos(shape_hue_radians)
        x = np.mean(xs)
        ys = np.sin(shape_hue_radians)
        y = np.mean(ys)
        self.data = [x, y]


class Infill(AbstractAttribute):
    def __init__(self, card_image, contours):
        AbstractAttribute.__init__(self, name='infill', default_data=None)
        self.parse_image(card_image, contours)

    def parse_image(self, card_image, contours):

        shape_mask = np.zeros((card_image.shape[0], card_image.shape[1]))
        cv2.drawContours(shape_mask, contours, -1, 1, -1)
        shape_mask = cv2.erode(shape_mask, np.ones((16, 16), np.uint8), iterations=1)
        card_image_hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

        shape_hsv = card_image_hsv[shape_mask == 1]
        shape_saturation_mean = np.mean(shape_hsv[:, 1])
        shape_saturation_median = np.median(shape_hsv[:, 1])

        self.data = [shape_saturation_mean ** 0.5, np.log2(shape_saturation_median)]
