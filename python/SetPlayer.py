import argparse
from itertools import combinations
import numpy as np
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
    transformer = transformation.Transformer(is_target=True, image_dimensions=target_dimensions)
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
            a_values = set(map(lambda x: x.attributes[a].classification, cards))
            if not (len(a_values) == 1 or len(a_values) == n):
                valid_set = False
                break
        if valid_set:
            yield sorted([board[cid] for cid in card_ids], key=lambda c: c.cid)


def visualize_set(card_set, im):
    nim = im.copy()
    color = (255, 0, 0)
    cv2.drawContours(nim, np.fliplr(map(lambda card: card.loc, card_set)), -1, color, 3)
    cv2.imshow("SET", nim)
    cv2.waitKey(0)


def overlay_ar_board(cards, im):
    rows, columns = (int(270), int(420))
    source_vertices = np.array([[0, 0],
                                [0, columns - 1],
                                [rows - 1, 0],
                                [rows - 1, columns - 1]])
    transformer = transformation.Transformer(False, (int(270), int(420), 3), source_vertices)
    for card in cards:
        new_card_image = card.visualize_card((int(270), int(420), 3))
        target_vertices = card.loc.reshape((-1, 2))
        im = transformer.transform(new_card_image, target_vertices, im.copy())
    return im


def get_sets_from_image(oim, set_size=3):
    all_cards = get_card_features(target_dimensions=(int(270), int(420), 3), im=oim)
    learning.classify_attributes(all_cards, ['shape', 'color', 'count', 'infill'])
    oim = overlay_ar_board(all_cards.values(), oim)
    valid_sets = generate_valid_sets(all_cards, set_size)
    return oim, valid_sets


def get_image_from_camera():
    im = None
    raise Exception('get_image_from_camera() not yet implemented')
    return im


def main(filename=None, set_size=3):
    try:
        if filename is None:
            oim = get_image_from_camera()
        else:
            oim = cv2.imread(filename, cv2.CV_LOAD_IMAGE_COLOR)
        oim, valid_set_generator = get_sets_from_image(oim, set_size)
        for card_set in valid_set_generator:
            visualize_set(card_set, oim)
    except Exception, e:
        print e
    finally:
        cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Finds Sets in image of Set cards')
parser.add_argument('--source', '-s', nargs='?', choices=['camera', 'file'], dest='source', required=True,
                    help='where to get image from, if "file" --input <full path> is required')
parser.add_argument('--input', '-i', nargs='?', dest='fName', const=str, default=None,
                    help='full path of image to process')
parser.add_argument('--set_size', nargs='?', dest='set_size', const=int, default=3,
                    help='number of cards per set, Default: 3')
args = parser.parse_args()

if __name__ == '__main__':
    if args.source == 'camera':
        if args.fName is not None:
            print 'Reading from camera. Option "--input/-i', args.fName+'"','will be ignored'
        main(filename=None, set_size=args.set_size)
    elif args.source == 'file':
        if args.fName is not None:
            main(filename=args.fName, set_size=args.set_size)
        else:
            print 'ERROR: With source set to "file", --input/-i must be set to the full path to file'
