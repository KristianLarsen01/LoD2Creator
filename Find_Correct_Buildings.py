#run conda activate lod2creator

import laspy
import glob

for f in glob.glob("*.laz"):
    las = laspy.read(f)
    print(f, las.header.min, las.header.max)
