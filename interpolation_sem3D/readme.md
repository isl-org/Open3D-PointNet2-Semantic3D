```bash
mkdir build
cd build
cmake ..
make -j

./interpolate $HOME/data/semantic3d \
              $HOME/repo/Open3D-PointNet-Semantic/visu/semantic_test/full_scenes_predictions \
              $HOME/repo/Open3D-PointNet-Semantic/visu/semantic_test/full_scenes_predictions_all_points \
              0.1 1
```