# Lists of what are at each index in a Landsat file, a Corine file, etc.
# This is mostly documentation, but might be used for code too.

LANDSAT_BANDS = ('band1','band2','band3','band4','band5','band7','qa')
LANDSAT_LOGICAL_BANDS = ('blue','green','red','NIR','SWIR1','SWIR2','qa')

# The corine bands 
CORINE_BANDS = (
    'mask',   # 1 if data, 0 if none
    'water',
    'barren',
    'grass',
    'shrub',
    'wetlands',
    'forest',
    'farm',
    'urban',
    'cloud',  # added by merge
    'shadow'  # added by merge
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