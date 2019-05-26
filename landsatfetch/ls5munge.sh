#!/bin/bash

# Usgage: ls5munge tarfile+
#
# Convert tarfiles in the format we retrieve from usgs into a single
# multi-band tiff file.

tmpdir=`mktemp -d`
echo "Working in temporary directory $tmpdir"

for file in "$@"
do
    destfile=`echo $file | sed s/.tar.*/.tif/`
    if [ -e $destfile ]; then
        echo "Skipping $file: $destfile already exists"
    else
        echo "Processing $file"
        
        rm -f $tmpdir/*
        tar xf "$file" -C $tmpdir
    
        # gdal_translate only works on a single input file, so
        # create a virtual tif that has all the bands in it
        gdalbuildvrt -separate $tmpdir/bands.vrt $tmpdir/*_sr_band* 

        # change the data type and scale of the band data.
        # the range [0, 10000] is mapped to [1 255].  
        # 0 is reserved for a "nodata" value
        gdal_translate -ot Byte -scale 0 10000 1 255 $tmpdir/bands.vrt $tmpdir/bands.tif

        # add in the pixelqa band, and write out with all file format options
        # See https://www.gdal.org/frmt_gtiff.html for the co options
        # Note we use the default 256x256 tile size
        gdal_merge.py -o "$destfile" -separate -a_nodata 0 -co INTERLEAVE=BAND -co TILED=YES -co SPARSE_OK=TRUE -co COMPRESS=LZW $tmpdir/bands.tif $tmpdir/*pixel_qa.tif
    fi
done

#echo "Removing temporary directory $tmpdir"
#rm -rf $tmpdir