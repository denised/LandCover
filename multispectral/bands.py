# Lists of what are at each index in a Landsat file, a Corine file, etc.
# This is mostly documentation, but might be used for code too.

# The data that is in the landsat files has a 'qa' band at the end.
# We replace that with a 'mask' band instead.
LANDSAT_BANDS = ('band1','band2','band3','band4','band5','band7','mask')
LANDSAT_LOGICAL_BANDS = ('blue','green','red','NIR','SWIR1','SWIR2','mask')

# The corine bands 
CORINE_BANDS = (
    'mask',    # 0     1 if data, 0 if none
    'water',   # 1
    'barren',  # 2
    'grass',   # 3
    'shrub',   # 4
    'wetlands',# 5
    'forest',  # 6
    'farm',    # 7
    'urban',   # 8
    'cloud',   # 9,  added by merge
    'shadow'   # 10, added by merge
    )

# Bit masks for items in the pixel qa band
PIXEL_QA = {
    'fill': 0x1,    # this is mask
    'clear': 0x2,   # not water, cloud or shadow
    'water': 0x4,
    'shadow': 0x8,
    'snow': 0x10,
    'cloud': 0x20,
    'cloud_confidence': 0xc0  # cloud is the same as cloud confidence == high
}


def band_index(bandtuple,bandname):
    return bandtuple.index(bandname)