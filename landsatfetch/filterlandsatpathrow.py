import csv

# This script was used to generate the original list of landsat tiles to look at
# Kept for historical purposes

thefile = "./WRSCornerPoints.csv"

spans = {
    'utm29n' : { 'xmin': -12, 'xmax': -6, 'ymin': 35, 'ymax': 60 },
    'utm30n' : { 'xmin': -6, 'xmax': 0, 'ymin': 35, 'ymax': 60 },
    'utm31n' : { 'xmin': 0, 'xmax': 6, 'ymin': 40, 'ymax': 53 },
    'utm32n' : { 'xmin': 6, 'xmax': 12, 'ymin': 38, 'ymax': 63 },
    'utm33n' : { 'xmin': 12, 'xmax': 18, 'ymin': 35, 'ymax': 63 },
    'utm34n' : { 'xmin': 18, 'xmax': 24, 'ymin': 32, 'ymax': 63 },
    'utm35n' : { 'xmin': 24, 'xmax': 30, 'ymin': 30, 'ymax': 63 },
    'utm36n' : { 'xmin': 30, 'xmax': 36, 'ymin': 33, 'ymax': 43 },
    'utm37n' : { 'xmin': 36, 'xmax': 42, 'ymin': 33, 'ymax': 43 },
    'utm38n' : { 'xmin': 42, 'xmax': 48, 'ymin': 34, 'ymax': 43 }
}

paths = {}

with open(thefile) as fp:
    reader = csv.DictReader(fp)
    for tile in reader:
        # skip any tiles that don't have data
        if not all( tile.values() ):
            continue
        for zone in spans.values():
            # check longitude (x) first since it is unique
            if zone['xmin'] <= float(tile['CTR LON']) < zone['xmax']:
                # for latitude, check for actual boundary overlap
                if float(tile['UL LAT']) > zone['ymin'] and float(tile['LR LAT']) < zone['ymax']:
                    path = int(tile['PATH'])
                    row = int(tile['ROW'])
                    if path not in paths:
                        paths[path] = { 'min': row, 'max': row }
                    else:
                        if row < paths[path]['min']:
                            paths[path]['min'] = row
                        if row > paths[path]['max']:
                            paths[path]['max'] = row

for path in paths:
    print("path {}, rows {} to {}".format(path, paths[path]['min'], paths[path]['max']))
                    