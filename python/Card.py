#### MAIN CARD CLASS ####
class Card:

  def __init__(self, cid, loc):
    self.cid = cid
    self.loc = loc # 4 vertices of quadrilateral in image
    self.attributes = {}

  def update_attribute(self, attribute):
    self.attributes[attribute.name] = attribute
