import numpy as np
from scipy.spatial import Delaunay
from scipy import stats
from itertools import combinations
import matplotlib as plt
import cv2

def resize(im,maxSize=512):
  dims = im.shape
  scale = float(512)/max(dims)
  return cv2.resize(im,(0,0),fx=scale,fy=scale)

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

def threshold_image(im):
  cardColor,cardStd = find_card_color(im)
  _,ima = cv2.threshold(im,cardColor-cardStd,255,cv2.THRESH_BINARY)
  _,imb = cv2.threshold(im,cardColor+cardStd,255,cv2.THRESH_BINARY_INV)
  im[np.equal(ima,imb)] = 255
  im[np.not_equal(ima,imb)] = 0
  im = imopen(im)
  im = imopen(im)
  return im

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
  cardImage = cv2.GaussianBlur(cardImage,(5,5),0)
  # get mask for just shape
  cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_RGB2GRAY)
  _,cardImageMask = cv2.threshold(cardImageBW,np.mean(cardImage)-np.std(cardImage),255,cv2.THRESH_BINARY_INV)
  #cardColor = cv2.mean(cardImage,cardImageMask-255)
  # count & shape
  contours,_ = cv2.findContours(cardImageMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contours = np.array(contours)
  sAreas = np.array([cv2.contourArea(shape) for shape in contours])
  contours = contours[sAreas>(tDims[0]*tDims[1])/20]
  sAreas = sAreas[sAreas>(tDims[0]*tDims[1])/20]
  count = len(contours)
  shape = contours[np.argmax(sAreas)]
  # infill
  shapeArea = sum(cv2.contourArea(shape) for shape in contours)
  infill = (np.sum(cardImageMask)/255.0)/shapeArea
  # average color
  bounds = np.zeros(cardImageMask.shape,dtype='uint8')
  cv2.drawContours(bounds,contours,-1,[255])
  #bounds[np.not_equal(ima,imb)] = 0
  color = np.uint8([[cv2.mean(cardImage,bounds)]])
  #print color
  # convert to cieLAB
  #color = np.squeeze(cv2.cvtColor(color,cv2.COLOR_RGB2LAB))
  # the scale is supposed to be ([0,100],[-128,128],[-128,128])
  #color = np.subtract(np.multiply(np.divide(color,[255.0,1.,1.0]),[100,1,1]),[0,128,128])
  return {'shape':shape,'infill':infill,'color':color,'count':count}

def get_card_data(normData,oim):
  pts,barys,idxs = normData
  cards = {}
  zeroImage = np.zeros(tDims,np.uint8)
  for cid,rect in enumerate(rects):
    cardImage = zeroImage.copy()
    # target (rectangle) x,y
    rpts = rect[idxs,:]
    nrs = pts[:,0]
    ncs = pts[:,1]
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
  def infill_check(cardA,cardB):
    return abs(cardA['infill']-cardB['infill'])<0.1
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
  # color needs a special checker
  #print cardA['color'],cardB['color'],cardC['color']
  dAB = np.sum(np.subtract(cardA['color'],cardB['color'])**2)**0.5
  dAC = np.sum(np.subtract(cardA['color'],cardC['color'])**2)**0.5
  dBC = np.sum(np.subtract(cardB['color'],cardC['color'])**2)**0.5
  print dAB,dAC,dBC
  if not ( (dAB>50 and dAC>50 and dBC>50) or (dAB<50 and dAC<50 and dBC<50) ):
    return False
  return True

def find_all_sets(cards,oim):
  for c in combinations(cards.keys(),3):
    cardA = cards[c[0]]
    cardB = cards[c[1]]
    cardC = cards[c[2]]
    if set_check(cardA,cardB,cardC,debug=False):
      print "SET FOUND!"
      nim = oim.copy()
      cv2.drawContours(nim,cardA['loc'],-1,(0,255,0),3)
      cv2.drawContours(nim,cardB['loc'],-1,(0,255,0),3)
      cv2.drawContours(nim,cardC['loc'],-1,(0,255,0),3)
      cv2.imshow("SET",nim)
      cv2.waitKey(0)

fName = "../data/angled.jpg"
oim = cv2.imread(fName)
oim = resize(oim)
im = cv2.cvtColor(oim,cv2.COLOR_RGB2GRAY)
im = threshold_image(im)
contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
locs = [shape for shape in contours if cv2.contourArea(shape)>(im.shape[0]*im.shape[1])/15/1000*1000/3]
rects = rectify(locs)

tDims = (int(270/1.5),int(420/1.5),3)
normData = normalize_preprocess(tDims)
cards = get_card_data(normData,oim)
find_all_sets(cards,oim)

cv2.destroyAllWindows()
