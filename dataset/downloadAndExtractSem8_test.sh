declare -a arr=("birdfountain_station1_xyz_intensity_rgb.7z" "castleblatten_station1_intensity_rgb.7z" "castleblatten_station5_intensity_rgb.7z" "marketplacefeldkirch_station1_intensity_rgb.7z" "marketplacefeldkirch_station4_intensity_rgb.7z" "marketplacefeldkirch_station7_intensity_rgb.7z" "sg27_station10_intensity_rgb.7z" "sg27_station3_intensity_rgb.7z" "sg27_station6_intensity_rgb.7z" "sg27_station8_intensity_rgb.7z" "sg28_station2_intensity_rgb.7z" "sg28_station5_intensity_rgb.7z" "stgallencathedral_station1_intensity_rgb.7z" "stgallencathedral_station3_intensity_rgb.7z" "stgallencathedral_station6_intensity_rgb.7z")

if [ ! -d "semantic_raw_data" ]; then
  echo mkdir "raw_semantic_data"
fi
cd "raw_semantic_data"

# now loop through the above array to download the data
for i in "${arr[@]}"
do
   wget "semantic3d.net/data/point-clouds/testing1/$i"
   # or do whatever with individual element of the array
done

wget "semantic3d.net/data/sem8_labels_training.7z"

## now loop through the above array to extract the data
for i in "${arr[@]}"
do
   7z e "$i"
   # or do whatever with individual element of the array
done

7z e sem8_labels_training.7z

# renaming; one file gets its name wrong after unzipping
mv station1_xyz_intensity_rgb.txt neugasse_station1_xyz_intensity_rgb.txt
