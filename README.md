# SetPlayer
This finds set of three cards in the game Set

Takes no input through command line, instead it is hardcoded.
Usage is as
fName = "../data/setTest.jpg"

Parses most images tested correctly. 
Works best with images with less specular reflection, and greater difference between card color and table sruface.

Speedily isolates, normalizes, parses cards for card attributes.
Normalization uses Delaunay trianguation and has been vectorized to greatly increase speed from early versions.
  - More vectorization may be possible to squeeze a few more ounces out

Intent is to have any attributes of cards handled, and be extensible from standard 3 shapes, 3 colors, 3 infills, and 3 counts.
  - Shape and count should work here, thought as shapes appear more similar, the code may get confused as it drops below similarity threshold
  
Color and infill are still the hardest to resolve. Both are close to working, and should work if cards are restricted to being Red, Blue, or Green.
Please see "Issues" page for more notes.
