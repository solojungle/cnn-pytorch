#!/bin/bash
FILES=
for f in $FILES
do
  convert $f -resize "256^>" $f
  convert $f -gravity center -crop 256x256+0+0 +repage $f
done

