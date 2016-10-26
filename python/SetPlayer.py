import numpy as np
from itertools import combinations
import cv2

import image_processing as ip
import Transformation
from Card import Card
from CardAttributes import Count, Color, Shape


def find_card_color(im):
    bg = np.hstack((im[[0, -1], :].flatten(), im[:, [0, -1]].flatten()))
    background_color_mu, background_color_std = np.mean(bg), np.std(bg)
    hist = np.histogram(im,bins=16)
    card_color = None
    for c, v in zip(hist[0], hist[1]):
        if c > (im.size/8) and not (abs(v - background_color_mu) < 32):
            card_color = v
            break
    card_color_stddev = max(32, abs(card_color - background_color_mu) - background_color_std * 1.5)
    card_color_stddev = min(32, card_color_stddev)
    return card_color, card_color_stddev


def find_cards_in_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    color_mu, color_std = find_card_color(im)
    im = ip.threshold_image(im, color_mu, color_std)
    _, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours
                if cv2.contourArea(contour) > (im.shape[0] * im.shape[1]) / 15 / 1000 * 1000 / 3]
    trapezoids = map(Transformation.simple_trapezoid, contours)
    return trapezoids


def parse_image(card_image, cid, source_vertices):

    card = Card(cid, source_vertices)
    count = Count(card_image)
    card.update_attribute(count)
    color = Color(card_image)
    card.update_attribute(color)
    shape = Shape(card_image)
    card.update_attribute(shape)
    return card


def get_card_features(target_dimensions, im, card_borders):
    transformer = Transformation.Transformer(target_image_dimensions=target_dimensions)
    cards = {}
    for cid, source_vertices in enumerate(card_borders):
        card_image = transformer.transform(im, source_vertices)
        border = np.fliplr(source_vertices).reshape((-1, 1, 2))
        card = parse_image(card_image, cid, border)
        cards[cid] = card
    return cards


# board is the group of cards that sets should be found in
def generate_valid_sets(board, n=3):
    def check_set(potential_set):
        all_match = False
        none_match = False
        for attributeA, attributeB in combinations(potential_set, 2):
            if attributeA == attributeB:
                if none_match:
                    return False
                all_match = True
            else:
                if all_match:
                    return False
                none_match = True
        return all_match or none_match

    # iterate through all combinations of n SET cards
    # note that this only creates combinations of cids
    for cids in combinations(board.keys(), n):
        cards = [board[c] for c in cids]
        # assume all cards will have the same attributes, so just pull from first card
        attributes = cards[0].attributes.keys() 
        valid_set = True
        for a in attributes:
            # get the values for the current attribute for all cards
            # there should be 1 unique value (all same) or n unique values (all different)
            # if not, break early and set valid_set to False
            valid_set = check_set(map(lambda x: x.attributes[a], cards))
            if not valid_set:
                break
        if valid_set:
            yield sorted(cids)


def visualize_set(card_set, im):
    nim = im.copy()
    color = (255, 0, 0)
    cv2.drawContours(nim, map(lambda card: card.loc, card_set), -1, color, 3)
    cv2.imshow("SET", nim)
    cv2.waitKey(0)

fName = "../data/SetCards.jpg"
oim = cv2.imread(fName)
card_borders = find_cards_in_image(oim)
all_cards = get_card_features(target_dimensions=(int(270), int(420), 3), im=oim, card_borders=card_borders)

for cid_set in generate_valid_sets(all_cards, n=3):
    card_set = map(lambda cid: all_cards[cid], cid_set)
    visualize_set(card_set, oim)

cv2.destroyAllWindows()
