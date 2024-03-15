import cv2
import numpy as np

# Example 3D points
object_points = np.array([[1, 1, 1],
                           [2, 2, 2],
                           [3, 3, 3]], dtype=np.float32)

focal_length = 1000
image_width,image_height = 600,400
#focal_length = 150
intrinsic_matrix = np.array([ 
    [focal_length, 0, image_width/2], 
    [0, focal_length, image_height/2], 
    [0, 0, 1] 
]) 

#define extrinsic camera parameters
rvec = np.array([0, 0, 0], dtype=np.float32) 
tvec = np.array([0, -2, 2], dtype=np.float32)

points_3d = np.dstack([splats[:,0],splats[:,1],splats[:,2]])[0]

#calculate distances from points to the camera
distances = np.array(np.abs(points_3d[:, 2] - tvec[2]))

#sort points and colors by distance in descending order
sorted_indices = np.argsort(-distances)
points_3d = points_3d[sorted_indices]
splats = splats[sorted_indices]

#project 3D points onto 2D plane 
points_2d, _ = cv2.projectPoints(points_3d, 
                                rvec,
                                tvec.reshape(-1, 1), 
                                intrinsic_matrix, 
                                None) 