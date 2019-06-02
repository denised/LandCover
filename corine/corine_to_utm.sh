#!/bin/bash

gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32629' -te -13 35 -5 60 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32629.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32630' -te -7 35 1 60 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32630.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32631' -te -1 40 7 53 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32631.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32632' -te 5 38 13 63 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32632.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32633' -te 11 35 19 63 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32633.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32634' -te 17 32 25 63 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32634.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32635' -te 23 30 31 63 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32635.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32636' -te 29 33 37 43 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32636.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32637' -te 35 33 43 43 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32637.tif
gdalwarp -co "SPARSE_OK=TRUE" -co "COMPRESS=LZW" -co "TILED=YES" -co "INTERLEAVE=BAND"  -t_srs 'EPSG:32638' -te 41 34 49 43 -te_srs EPSG:4326 -tr 30 30 -r bilinear -multi corine/corine.tif corine_32638.tif