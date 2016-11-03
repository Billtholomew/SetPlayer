import numpy as np
from itertools import combinations
import cv2

import image_processing as ip
import transformation
import learning
from Card import Card
from CardAttributes import Count, Color, Shape, Infill


def find_cards_in_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = ip.auto_canny(im)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = map(lambda contour: cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True), contours)
    trapezoids = filter(lambda contour: len(contour) == 4, contours)
    # To assume a contour is a card, it must be less than 10% the size of the image (no fewer than 12 cards allowed)
    # and must be more than 2% of the image (no more than 24 cards allowed, with the whole set taking 1/2 the image)
    return filter(lambda t: edges.size / 50 < cv2.contourArea(t) < edges.size / 10, trapezoids)


def parse_card_image(card_image, cid, source_vertices):
    card = Card(cid, source_vertices)
    count = Count(card_image)
    card.update_attribute(count)
    color = Color(card_image)
    card.update_attribute(color)
    shape = Shape(card_image)
    card.update_attribute(shape)
    infill = Infill(card_image)
    card.update_attribute(infill)
    return card


def get_card_features(target_dimensions, im):
    card_borders = find_cards_in_image(im)
    transformer = transformation.Transformer(target_image_dimensions=target_dimensions)
    cards = {}
    for cid, border in enumerate(card_borders):
        card_image = transformer.transform(im, border.copy())
        card = parse_card_image(card_image, cid, border)
        cards[cid] = card
    return cards


# board is the group of cards that sets should be found in
def generate_valid_sets(board, n=3):
    # iterate through all combinations of n SET cards
    # note that this only creates combinations of card_ids
    for card_ids in combinations(board.keys(), n):
        cards = [board[card_id] for card_id in card_ids]
        # assume all cards will have the same attributes, so just pull from first card
        attributes = cards[0].attributes.keys()
        valid_set = True
        for a in attributes:
            # get the values for the current attribute for all cards
            # there should be 1 unique value (all same) or n unique values (all different)
            # if not, break early and set valid_set to False
            a_values = set(map(lambda x: x.attributes[a], cards))
            if not (len(a_values) == 1 or len(a_values) == n):
                valid_set = False
                break
        if valid_set:
            yield sorted(card_ids)


def visualize_set(card_set, im):
    nim = im.copy()
    color = (255, 0, 0)
    print '======================='
    print '======================='
    print '======================='
    for card in card_set:
        count_name = (card.attributes['count'].data[0], card.attributes['count'].classification)
        color_name = (np.arctan2(card.attributes['color'].data[0], card.attributes['color'].data[1]) * 180 / np.pi,
                      card.attributes['color'].classification)
        print 'CARD', count_name, color_name
        for k, v in card.attributes.iteritems():
            if k=='color' or k=='count':
                continue
            print k, v.data, '->', v.classification
    cv2.drawContours(nim, np.fliplr(map(lambda card: card.loc, card_set)), -1, color, 3)
    cv2.imshow("SET", nim)
    cv2.waitKey(0)


fName = "../data/setTest.jpg"
oim = cv2.imread(fName, cv2.CV_LOAD_IMAGE_COLOR)
all_cards = get_card_features(target_dimensions=(int(270), int(420), 3), im=oim)

learning.classify_attributes(all_cards, ['shape', 'color', 'count', 'infill'])

for cid_set in generate_valid_sets(all_cards, n=3):
    visualize_set(map(lambda cid: all_cards[cid], cid_set), oim)

cv2.destroyAllWindows()
