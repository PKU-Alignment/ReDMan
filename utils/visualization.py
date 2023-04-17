import open3d as o3d
import torch
import numpy as np
data = torch.load("/home/shengjie/gyr/Safety-DexterousHands/temp/color62.pth")
print(data)
data = np.array(np.array(data[0][0][0].cpu()))
pt1 = o3d.geometry.PointCloud()
pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
# pt1.colors=o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pt1],'part of cloud',width=500,height=500)


