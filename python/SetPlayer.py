import argparse
from multiprocessing import Process, Queue
from itertools import combinations
from math import factorial

import cv2
import numpy as np

import transformation
import learning
from Card import Card
from image_processing import auto_canny


def find_cards_in_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(im)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = map(lambda contour: cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True), contours)
    trapezoids = filter(lambda contour: len(contour) == 4, contours)
    # To assume a contour is a card, it must be less than 10% the size of the image (no fewer than 12 cards allowed)
    # and must be more than 2% of the image (no more than 24 cards allowed, with the whole set taking 1/2 the image)
    return filter(lambda t: edges.size / 50 < cv2.contourArea(t) < edges.size / 10, trapezoids)


def get_card_features(target_dimensions, im, card_borders, shape_points):
    transformer = transformation.Transformer(is_target=True, image_dimensions=target_dimensions)
    cards = {cid: Card(cid, border, transformer.transform(im, border.copy()), shape_points)
             for cid, border in enumerate(card_borders)}
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
    map(lambda card:
        transformer.transform(card.visualize_card((int(270), int(420), 3)), card.loc.reshape((-1, 2)), im),
        cards)

    return im


def distraction(oim, borders, set_size, time_to_waste=5):
    n = len(borders)
    k = set_size
    nCr = factorial(n) / (factorial(k) * factorial(n-k))
    show_time = 6000 / nCr
    cv2.namedWindow('Searching for Sets')
    for border_set in combinations(range(len(borders)), set_size):
        im = oim.copy()
        cv2.drawContours(im, [borders[i] for i in border_set], -1, (0, 0, 255), 3)
        cv2.imshow('Searching for Sets', im)
        cv2.waitKey(show_time)


def get_card_data(oim, borders, results):
    all_cards = get_card_features(target_dimensions=(int(270), int(420), 3), im=oim, card_borders=borders,
                                  shape_points=180)  # 1.34 s
    learning.classify_attributes(all_cards, ['shape', 'color', 'count', 'infill'], max_k=5)  # 1.05 s, but f(max_k)
    results.put(all_cards)


def get_sets_from_image(oim, set_size=3, view_ar=False, distract=True):
    import time
    t = time.time()
    borders = find_cards_in_image(oim)
    results = Queue()
    processes = []
    process_c = Process(target=get_card_data, args=(oim.copy(), borders, results))
    processes.append(process_c)
    if distract:
        process_d = Process(target=distraction, args=(oim.copy(), borders, set_size))
        processes.append(process_d)
    map(lambda process: process.start(), processes)
    map(lambda process: process.join(5), processes)
    all_cards = results.get()
    if view_ar:
        oim = overlay_ar_board(all_cards.values(), oim)  # 1.03 s
    print time.time() - t
    valid_sets = generate_valid_sets(all_cards, set_size)  # 0.00 s
    return oim, valid_sets


def main(filename=None, set_size=3, view_ar=False):
    try:
        if filename is None:
            raise Exception('get_image_from_camera() not yet implemented')
        else:
            oim = cv2.imread(filename, flags=1)
        oim, valid_set_generator = get_sets_from_image(oim, set_size, view_ar, distract=True)
        map(lambda card_set: visualize_set(card_set, oim), valid_set_generator)
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
parser.add_argument('--view_ar', dest='view_ar', default=False, action='store_true',
                    help='include tag to view augmented reality version (see program''s version of cards)')
args = parser.parse_args()

if __name__ == '__main__':
    if args.source == 'camera':
        if args.fName is not None:
            print 'Reading from camera. Option "--input/-i', args.fName+'"', 'will be ignored'
        main(filename=None, set_size=args.set_size, view_ar=args.view_ar)
    elif args.source == 'file':
        if args.fName is not None:
            main(filename=args.fName, set_size=args.set_size, view_ar=args.view_ar)
        else:
            print 'ERROR: With source set to "file", --input/-i must be set to the full path to file'
