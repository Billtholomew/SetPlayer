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

def set_check(cardA,cardB,cardC,debug=False):
  def color_check(cardA,cardB):
    if np.size(np.intersect1d(np.where(cardA['color']==255)[0],np.where(cardB['color']==255)[0]))==0:
      return False
    return True
  def infill_check(cardA,cardB):
    return abs(cardA['infill']-cardB['infill'])<1
  def shape_check(cardA,cardB):
    return cv2.matchShapes(cardA['shape'],cardB['shape'],1,0.0)<0.05
  def count_check(cardA,cardB):
    return cardA['count']==cardB['count']
  def tri_check(cardA,cardB,cardC,func):
    AB = func(cardA,cardB)
    AC = func(cardA,cardC)
    BC = func(cardB,cardC)
    if (AB and BC):
      return True
    if not AB and not BC and not AC:
      return True
    return False
  # actual body
  if not tri_check(cardA,cardB,cardC,count_check):
    if debug:
      print 'Count not set'
    return False
  if not tri_check(cardA,cardB,cardC,infill_check):
    if debug:
       print 'infill not set'
    return False
  if not tri_check(cardA,cardB,cardC,shape_check):
    if debug:
       print 'shape not set'
    return False
  if not tri_check(cardA,cardB,cardC,color_check):
    if debug:
      print 'Color not set'
    return False
  return True

def find_all_sets(cards,oim):
  for c in combinations(cards.keys(),3):
    cardA = cards[c[0]]
    cardB = cards[c[1]]
    cardC = cards[c[2]]
    if set_check(cardA,cardB,cardC,debug=False):
      print "SET FOUND!"
      print cardA['color'],cardA['infill'],cardA['count']
      print cardB['color'],cardB['infill'],cardB['count']
      print cardC['color'],cardC['infill'],cardC['count']
      nim = oim.copy()
      c = (255,0,0)
      cv2.drawContours(nim,cardA['loc'],-1,cardA['color'].tolist(),3)
      cv2.drawContours(nim,cardB['loc'],-1,cardB['color'].tolist(),3)
      cv2.drawContours(nim,cardC['loc'],-1,cardC['color'].tolist(),3)
      cv2.imshow("SET",nim)
      cv2.waitKey(0)

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
find_all_sets(cards,oim)

cv2.destroyAllWindows()
