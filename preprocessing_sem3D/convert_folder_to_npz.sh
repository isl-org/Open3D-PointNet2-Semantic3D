#!/bin/bash
for file in ${1%/}/*.txt
do
  filepath=${file:0:${#file}-4}
  filename=${file:${#1}:${#file}}
  filename=${filename:0:${#filename}-4}
  echo "converting file " $filename
  python convert_to_npz.py --file=$file --out=$2
done

