import numpy as np
import cv2


# auto canny function
# from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(im, sigma=0.33):
    v = np.median(im)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(im, lower, upper)


def bgr2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def rgb2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def im_close(im, n=5):
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def im_open(im, n=5):
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def threshold_image(im, color_mu, color_std=0, sigma=1):
    nim = im.copy()
    _, ima = cv2.threshold(im, color_mu - (color_std * sigma), 255, cv2.THRESH_BINARY)
    _, imb = cv2.threshold(im, color_mu + (color_std * sigma), 255, cv2.THRESH_BINARY_INV)
    nim[np.equal(ima, imb)] = 255
    nim[np.not_equal(ima, imb)] = 0
    nim = im_open(nim)
    nim = im_open(nim)
    return nim


def im_mask(im, sigma=1):
    im_gray = rgb2gray(im)
    card_color_mu, card_color_std = cv2.meanStdDev(im_gray)
    return threshold_image(im_gray, card_color_mu, card_color_std, sigma).astype("uint8")


def contour_xy2polar(contour, n_points=180):
    # get current polar coordinates
    m = cv2.moments(contour)
    cy = int(m['m01'] / m['m00'])
    cx = int(m['m10'] / m['m00'])
    polar = map(lambda (x, y):
                (np.arctan2(cy - y, cx - x), np.sqrt((cx - x) ** 2 + (cy - y) ** 2)),
                map(lambda pt: pt[0], contour))
    polar = sorted(polar, key=lambda (t, r): t)
    thetas, radii = zip(*polar)
    # interpolate to new polar coordinates based on number of critical points to use (n_points)
    new_thetas = np.pi / 180 * np.arange(-180, 180, 360 / n_points)
    new_radii = np.interp(new_thetas, thetas, radii)
    # normalize
    new_radii /= max(new_radii)
    new_polar = zip(new_thetas, new_radii)
    return new_polar


def contour_polar2xy(contour_polar, scale=1, center=(0, 0)):
    cy = center[0]
    cx = center[1]
    contour_xy = map(lambda (theta, radius):
                     [[int(np.cos(theta) * radius * scale + cx), int(np.sin(theta) * radius * scale + cy)]],
                     contour_polar)
    return np.array(contour_xy)
