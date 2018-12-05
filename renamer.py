"""
Rename the prediction label files for submission
"""

import os
import glob


conversion_dict = {
    "birdfountain_station1_xyz_intensity_rgb.labels": "birdfountain1.labels",
    "castleblatten_station1_intensity_rgb.labels": "castleblatten1.labels",
    "castleblatten_station5_xyz_intensity_rgb.labels": "castleblatten5.labels",
    "marketplacefeldkirch_station1_intensity_rgb.labels": "marketsquarefeldkirch1.labels",
    "marketplacefeldkirch_station4_intensity_rgb.labels": "marketsquarefeldkirch4.labels",
    "marketplacefeldkirch_station7_intensity_rgb.labels": "marketsquarefeldkirch7.labels",
    "sg27_station10_intensity_rgb.labels": "sg27_10.labels",
    "sg27_station3_intensity_rgb.labels": "sg27_3.labels",
    "sg27_station6_intensity_rgb.labels": "sg27_6.labels",
    "sg27_station8_intensity_rgb.labels": "sg27_8.labels",
    "sg28_station2_intensity_rgb.labels": "sg28_2.labels",
    "sg28_station5_xyz_intensity_rgb.labels": "sg28_5.labels",
    "stgallencathedral_station1_intensity_rgb.labels": "stgallencathedral1.labels",
    "stgallencathedral_station3_intensity_rgb.labels": "stgallencathedral3.labels",
    "stgallencathedral_station6_intensity_rgb.labels": "stgallencathedral6.labels",
}


if __name__ == "__main__":
    # for src_path in glob.glob("result_archive/result_dec_03_full_trainset/dense/*"):
    for src_path in glob.glob("result/dense/*"):
        dir_name = os.path.dirname(src_path)
        src_name = os.path.basename(src_path)
        dst_name = conversion_dict.get(src_name, None)
        if dst_name is not None:
            dst_path = os.path.join(dir_name, dst_name)
            os.rename(src_path, dst_path)
            print("Moved {} to {}".format(src_path, dst_path))
        else:
            print("src_name not found in conversion_dict:", src_name)
