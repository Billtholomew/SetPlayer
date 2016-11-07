# SetPlayer
This finds set of three cards in the game Set

# Requirements

- OpenCV v2.4.13
- Numpy v1.8.0
- Scipy v0.18.2

# Running

Takes no input through command line, instead it is hardcoded.
Usage is as
fName = "../data/setTest.jpg"

Image used for testing can be acquired here:
http://www.theboardgamefamily.com/wp-content/uploads/2013/03/SetCards.jpg


Parses most images tested correctly. 
Works best with images with less specular reflection, and greater difference between card color and table sruface.

Speedily isolates, normalizes, parses cards for card attributes.
Normalization uses Delaunay trianguation and has been vectorized to greatly increase speed from early versions.

Uses kmeans to classify attribtues. This allows us to not have to set thresholds, and fine-tune the attributes themselves for comparison. It should be more robust to deviation in individual attributes. However, kmeans itself still needs some tweaking. I don't set k, and try to learn it ad hoc, which may be unreliable if only two colors show up. It is slightly biased towards more clusters.
