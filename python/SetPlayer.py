import numpy as np
from itertools import combinations
import cv2

import ImageProcessing as ip
import Transformation
from Card import Card
from CardAttributes import Count, Color


def find_card_color(im):
    bg = np.hstack((im[[0,-1],:].flatten(),im[:,[0,-1]].flatten()))
    bgColor, bgstd = np.mean(bg),np.std(bg)
    hist = np.histogram(im,bins=16)
    cardColor = None
    for c,v in zip(hist[0],hist[1]):
        if c>(im.size/8) and not (abs(v-bgColor)<32):
            cardColor = v
            break
    cardStd = max(32,abs(cardColor-bgColor) - bgstd*1.5)
    cardStd = min(32,cardStd)
    return cardColor,cardStd


def parse_image(cardImage, cid, sourceVertices):

    card = Card(cid,sourceVertices)
    count = Count(cardImage)
    card.update_attribute(count)
    color = Color(cardImage)
    card.update_attribute(color)

    return card

    shape = None
    infill = None
    color = None
    count = None
    
    # get mask for just shape
    cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_BGR2GRAY)
    _,cardImageMask = cv2.threshold(cardImageBW,np.mean(cardImageBW)-np.std(cardImageBW),1,cv2.THRESH_BINARY_INV)
    cardColor,cardStd = cv2.meanStdDev(cardImageBW)
    cardImageMask = (255 - ip.threshold_image(cardImageBW,cardColor,cardStd).astype("uint8"))
    # count & shape
    contours, _ = cv2.findContours(cardImageMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    sAreas = np.array([cv2.contourArea(shape) for shape in contours])
    contours = contours[sAreas > (targetDims[0] * targetDims[1]) / 20]
    sAreas = sAreas[sAreas > (targetDims[0] * targetDims[1]) / 20]
    count = len(contours)
    shape = contours[np.argmax(sAreas)]
    # color and infill
    cardImageMask = np.zeros(cardImageMask.shape)
    cv2.drawContours(cardImageMask,contours,-1,(255,0,0),-1)
    shapeShade = np.multiply(cardImageMask/255.0,cardImageBW)    

    color = cv2.mean(cardImage,(cardImageMask/255.0).astype("uint8"))
    color = cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)
    color = color[0][0]
    print color
    color2 = color.copy()
    color[0] = color[0].astype('float')/180.0*360.0
    color[1] = 255.0
    color[2] = 255.0
    color = cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_HSV2RGB)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)[0][0]
    color = color.astype('float')
    swap = color[0]
    color[0] = color[1]
    color[1] = swap

    cardImage = cv2.GaussianBlur(cardImage,(15,15),0)
    
    cardMask = (1-cv2.dilate(cardImageMask,np.ones((10,10),np.uint8),iterations=1)/255.0).astype("uint8")
    cColor = cv2.mean(cardImage,cardMask)
    infillMask = (cv2.erode(cardImageMask,np.ones((20,20),np.uint8),iterations=1)/255.0).astype("uint8")

    sColor = cv2.mean(cardImage,infillMask)
    sColor2 = cv2.cvtColor(np.uint8([[color2]]),cv2.COLOR_HSV2BGR)[0][0]

    dA = np.absolute(np.subtract(np.int16([sColor[:3]]),np.int16([cColor[:3]])))
    dB = np.absolute(np.subtract(np.int16([sColor2[:3]]),np.int16([cColor[:3]])))
    infill = np.max(np.divide(dA.astype("float"),dB.astype("float")))
    
    return {'shape':shape,'infill':infill,'color':color,'count':count}


def get_card_features(transformer, oim, allCardVertices):
    cards = {}

    for cid, sourceVertices in enumerate(allCardVertices):
        cardImage = transformer.transform(oim, sourceVertices)

        border = np.fliplr(sourceVertices).reshape((-1, 1, 2))
        card = parse_image(cardImage, cid, border)

        cards[cid] = card

    return cards


# board is the group of cards that sets should be found in
def find_all_sets(board, n=3):
    def check_set(potentialSet):
        pairs = combinations(potentialSet, 2)
        allMatch = False
        noneMatch = False
        for attributeA, attributeB in pairs:
            if attributeA == attributeB:
                if noneMatch:
                    return False
                allMatch = True
            else:
                if allMatch:
                    return False
                noneMatch = True
        return allMatch or noneMatch

    # iterate through all combinations of n SET cards
    # note that this only creates combinations of cids
    for cids in combinations(board.keys(), n):
        cards = [board[c] for c in cids]
        # assume all cards will have the same attributes, so just pull from first card
        attributes = cards[0].attributes.keys() 
        validSet = True
        for a in attributes:
            # get the values for the current attribute for all cards
            # there should be 1 unique value (all same) or n unique values (all different)
            # if not, break early and set validSet to False
            validSet = check_set(map(lambda x: x.attributes[a], cards))
            if not validSet:
                break
        if validSet:
            cards = [board[c] for c in cids]
            attributes = cards[0].attributes.keys()
            for a in attributes:
                for card in cards:
                    print card.attributes[a].data
            yield sorted(cids)

fName = "../data/SetCards.jpg"
oim = cv2.imread(fName)
im = cv2.cvtColor(oim, cv2.COLOR_BGR2GRAY)
colorMu, colorStd = find_card_color(im)
im = ip.threshold_image(im, colorMu, colorStd)
_, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours
            if cv2.contourArea(contour) > (im.shape[0] * im.shape[1]) / 15 / 1000 * 1000 / 3]

targetDims = (int(270), int(420), 3)
transformer = Transformation.Transformer(targetImageDimensions=targetDims)
trapezoids = map(transformer.simple_trapezoid, contours)
allCards = get_card_features(transformer, oim, trapezoids)

for cidSet in find_all_sets(allCards, n=3):
    nim = oim.copy()
    color = (255, 0, 0)
    cv2.drawContours(nim, map(lambda cid: allCards[cid].loc, cidSet), -1, color, 3)
    cv2.imshow("SET",nim)
    cv2.waitKey(0)

cv2.destroyAllWindows()
