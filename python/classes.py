import cv2

#### ATTRIBUTE CLASSES FOR USE IN MAIN CARD CLASS ####

class Color:

  def __init__(self,cardImage=None):
    if cardImage is not None:
      self.parse_image(cardImage)
    else:
      self.data = None    

  def parse_image(self,cardImage):
    # card image will be in BGR
    cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_BGR2GRAY)
    cardColor,cardStd = cv2.meanStdDev(cardImageBW)
    cardImageMask = (255-threshold_image(cardImageBW,cardColor,cardStd).astype("uint8"))
    contours,_ = cv2.findContours(cardImageMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    sAreas = np.array([cv2.contourArea(shape) for shape in contours])
    contours = contours[sAreas>(tDims[0]*tDims[1])/20]
    cardImageMask = np.zeros(cardImageMask.shape)
    cv2.drawContours(cardImageMask,contours,-1,(255,0,0),-1)
    color = cv2.mean(cardImage,(cardImageMask/255.0).astype("uint8"))
    color = cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)
    color = color[0][0]
    color = [color[0].astype('float')/180.0*360.0, 255, 255]
    self.data['hsv'] = color
    color = cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_HSV2RGB)
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)[0][0]
    color = color.astype('float')
    swap = color[0]
    color[0] = color[1]
    color[1] = swap
    self.data['bgr'] = color

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return abs(self.data['hsv'][0]-other.data['hsv'][0])<30
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

###
  
class Shape:

  def __init__(self,cardImage=None):
    if cardImage is not None:
      self.parse_image(cardImage)
    else:
      self.data = None

  def parse_image(self,cardImage):
    cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_BGR2GRAY)
    cardColor,cardStd = cv2.meanStdDev(cardImageBW)
    cardImageMask = (255-threshold_image(cardImageBW,cardColor,cardStd).astype("uint8"))
    contours,_ = cv2.findContours(cardImageMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    sAreas = np.array([cv2.contourArea(shape) for shape in contours])
    contours = contours[sAreas>(tDims[0]*tDims[1])/20]
    sAreas = sAreas[sAreas>(tDims[0]*tDims[1])/20]
    self.data = contours[np.argmax(sAreas)]

###
    
class Count:

  def __init__(self,cardImage=None):
    if cardImage is not None:
      self.parse_image(cardImage)
    else:
      self.data = None

  def parse_image(self,cardImage):
    cardImageBW = cv2.cvtColor(cardImage,cv2.COLOR_BGR2GRAY)
    cardColor,cardStd = cv2.meanStdDev(cardImageBW)
    cardImageMask = (255-threshold_image(cardImageBW,cardColor,cardStd).astype("uint8"))
    contours,_ = cv2.findContours(cardImageMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    sAreas = np.array([cv2.contourArea(shape) for shape in contours])
    contours = contours[sAreas>(tDims[0]*tDims[1])/20]
    sAreas = sAreas[sAreas>(tDims[0]*tDims[1])/20]
    self.data = len(contours)

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.data == other.data
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

#### MAIN CARD CLASS ####

class Card:

  def __init__(self,cid,loc):
    self.cid = cid
    self.loc = loc # 4 vertices of quadrilateral in image
    self.attributes = {}

  def update_attribute(self,attrName,attrData):
    self.attributes[attrName] = attrData

  def parse_image(self,cardImage):
    
