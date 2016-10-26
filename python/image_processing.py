import numpy as np
import cv2


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
    nim[np.equal(ima,imb)] = 255
    nim[np.not_equal(ima,imb)] = 0
    nim = im_open(nim)
    nim = im_open(nim)
    return nim


def im_mask(im, sigma=1):
    im_gray = bgr2gray(im)
    card_color_mu, card_color_std = cv2.meanStdDev(im_gray)
    return threshold_image(im_gray, card_color_mu, card_color_std).astype("uint8")