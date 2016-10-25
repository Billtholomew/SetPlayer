import numpy as np
from scipy.spatial import Delaunay
from scipy import stats
from itertools import combinations
import cv2

def imclose(im,n=5):
  im = cv2.dilate(im,np.ones((n,n),np.uint8),iterations=1)
  im = cv2.erode(im,np.ones((n,n),np.uint8),iterations=1)
  return im

def imopen(im,n=5):
  im = cv2.erode(im,np.ones((n,n),np.uint8),iterations=1)
  im = cv2.dilate(im,np.ones((n,n),np.uint8),iterations=1)
  return im
  
def find_card_color(im):
  bg = np.hstack((im[[0,-1],:].flatten(),im[:,[0,-1]].flatten()))
  bgColor,bgstd = np.mean(bg),np.std(bg)
  hist = np.histogram(im,bins=16)
  cardColor = None
  for c,v in zip(hist[0],hist[1]):
    if c>(im.size/8) and not (abs(v-bgColor)<32):
      cardColor = v
      break
  cardStd = max(32,abs(cardColor-bgColor)-bgstd*1.5)
  cardStd = min(32,cardStd)
  return cardColor,cardStd

def threshold_image(im,colorMu,colorStd):
  nim = im.copy()
  _,ima = cv2.threshold(im,colorMu-colorStd,255,cv2.THRESH_BINARY)
  _,imb = cv2.threshold(im,colorMu+colorStd,255,cv2.THRESH_BINARY_INV)
  nim[np.equal(ima,imb)] = 255
  nim[np.not_equal(ima,imb)] = 0
  nim = imopen(nim)
  nim = imopen(nim)
  return nim

# standardize points to make it a more reliable rectangle to work with
# actually probably more of a trapezoid than a rectangle
def rectify(contour):
  def worker(contour):
    contour = np.reshape(contour,(len(contour),2))
    rect = np.zeros((4, 2), dtype = "int32")
    s = contour.sum(axis=1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]
    diff = np.diff(contour,axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]
    return np.fliplr(rect)
  return [worker(rect) for rect in contour]

# get barycentric coordinates and corresponding points in target (new) image
# assumes a rectangle
# tDims should be dimensions of rectangle to write to
def normalize_preprocess(tDims):
  nys = np.arange(tDims[0])
  nxs = np.arange(tDims[1])
  pts = np.transpose([np.repeat(nys, len(nxs)), np.tile(nxs, len(nys))])

  targetVerts = np.array([[0,0],[0,tDims[1]],[tDims[0],tDims[1]],[tDims[0],0]])
  tri = Delaunay(targetVerts)
  ss = tri.find_simplex(pts)
  Ts = tri.transform[ss,:2]
  prs = pts-tri.transform[ss,2]
  idxs = tri.simplices[ss]

  barys = np.array([Ts[i].dot(pr) for i,pr in enumerate(prs)])
  barys = np.hstack((barys, 1-np.sum(barys,axis=1,keepdims=True)))

  return pts,barys,idxs

def parse_image(cardImage):

  shape = None
  infill = None
  color = None
  count = None
  
  # get mask for just shape
  cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_BGR2GRAY)
  _,cardImageMask = cv2.threshold(cardImageBW,np.mean(cardImageBW)-np.std(cardImageBW),1,cv2.THRESH_BINARY_INV)
  cardColor,cardStd = cv2.meanStdDev(cardImageBW)
  cardImageMask = (255-threshold_image(cardImageBW,cardColor,cardStd).astype("uint8"))
  # count & shape
  contours,_ = cv2.findContours(cardImageMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contours = np.array(contours)
  sAreas = np.array([cv2.contourArea(shape) for shape in contours])
  contours = contours[sAreas>(tDims[0]*tDims[1])/20]
  sAreas = sAreas[sAreas>(tDims[0]*tDims[1])/20]
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

def get_card_features(normData,oim):
  pts,barys,idxs = normData
  cards = {}
  zeroImage = np.zeros(tDims,np.uint8)

  nrs = pts[:,0]
  ncs = pts[:,1]
  
  for cid,rect in enumerate(rects):

    cardImage = zeroImage.copy()
    # target (rectangle) x,y
    rpts = rect[idxs,:]
    # source (original) x,y
    ors = np.multiply(rpts[:,:,0],barys).sum(axis=1,keepdims=True).astype("int32")
    ocs = np.multiply(rpts[:,:,1],barys).sum(axis=1,keepdims=True).astype("int32")

    cardImage[nrs,ncs,:] = oim[ors,ocs,:].reshape((-1,3))

    cardInfo = parse_image(cardImage)
    border = np.fliplr(rect).reshape((-1,1,2))
    cardInfo['loc'] = [border]
    cards[cid] = cardInfo
    print cid
  
  return cards

# board is the group of cards that sets should be found in
def find_all_sets(board,n=3):
  # iterate through all combinations of n SET cards
  # note that this only creates combinations of cids
  for cids in combinations(board.keys(),n):
    cards = [board[c] for c in cids]
    # assume all cards will have the same attributes, so just pull from first card
    attributes = cards[0].attributes.keys() 
    validSet = True
    for a in attributes:
      # get the values for the current attribute for all cards
      # there should be 1 unique value (all same) or n unique values (all different)
      # if not, break early and set validSet to False
      aVals = set(map(lambda x:x.attributes[a],cards))
      if not (len(aVals)==1 or len(aVals)==n):
        validSet = False
        break
    if validSet:
      yield sorted(cids)

fName = "../data/angled.jpg"
oim = cv2.imread(fName)
im = cv2.cvtColor(oim,cv2.COLOR_BGR2GRAY)
colorMu,colorStd = find_card_color(im)
im = threshold_image(im,colorMu,colorStd)
contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
locs = [shape for shape in contours if cv2.contourArea(shape)>(im.shape[0]*im.shape[1])/15/1000*1000/3]
rects = rectify(locs)

tDims = (int(270),int(420),3)
normData = normalize_preprocess(tDims)
cards = get_card_features(normData,oim)

for cidSet in find_all_sets(cards,n=3):
  nim = oim.copy()
  c = (255,0,0)
  for cid in cidSet:
    card = cards[cid]
    cv2.drawContours(nim,card.loc,-1,c,3)
  cv2.imshow("SET",nim)
  cv2.waitKey(0)

cv2.destroyAllWindows()
