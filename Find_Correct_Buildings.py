import laspy
import glob
import numpy as np

paths = glob.glob("sample_roofdata_50/*.laz")

mins = []
maxs = []

for f in paths:
    las = laspy.read(f)
    mins.append(las.header.min)  # [minX, minY, minZ]
    maxs.append(las.header.max)  # [maxX, maxY, maxZ]

mins = np.array(mins)
maxs = np.array(maxs)

overall_min = mins.min(axis=0)
overall_max = maxs.max(axis=0)

print("GLOBAL BBOX:")
print("min:", overall_min)
print("max:", overall_max)

las = laspy.read("sample_roofdata_50/10444144.laz")

print(las.header)
print(las.point_format)
print(las.point_format.dimension_names)
