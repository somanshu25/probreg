import copy
import numpy as np
import open3d as o3
from probreg import gmmtree
from probreg import l2dist_regs
from probreg import callbacks

# load source and target point cloud
source = o3.read_point_cloud('bunny.pcd')
target = copy.deepcopy(source)
sCheck = np.asarray(source.points)
tCheck = np.asarray(target.points)
print(sCheck[0:5])
print(tCheck[0:5])
# transform target point cloud
th = np.deg2rad(30.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))
source = o3.voxel_down_sample(source, voxel_size=0.005)
target = o3.voxel_down_sample(target, voxel_size=0.005)
#source.paint_uniform_color([1, 0, 0])
#target.paint_uniform_color([0, 1, 0])
#o3.draw_geometries([source, target])

# compute cpd registration
cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param, _ = gmmtree.registration_gmmtree(source, target,callbacks = cbs)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)
print(tf_param)
# draw result
#source.paint_uniform_color([1, 0, 0])
#target.paint_uniform_color([0, 1, 0])
#result.paint_uniform_color([0, 0, 1])
#o3.draw_geometries([source, target, result])
