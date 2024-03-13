import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Read binary file
def read_binary_file(file_path):
    format_string = "3f 3f 4B 4b"

    with open(file_path, "rb") as file:
        data = file.read()

    num_splats = len(data) // 32
    splat_list = []

    for i in range(num_splats):
        start_index = i * 32

        splat_data = struct.unpack_from(format_string, data, start_index)
        position = splat_data[:3]
        scale = splat_data[3:6]
        color = splat_data[6:10]
        rotation = splat_data[10:]

        rotation = [(c - 128) / 128 for c in rotation]

        # Store the splat data in a dictionary
        splat_dict = {
            'position': position,
            'scale': scale,
            'color': color,
            'rotation': rotation
        }

        # Append the dictionary to the splat list
        splat_list.append(splat_dict)
    return splat_list



splat_list = read_binary_file("nike.splat")

def create_view_matrix(eye, target, up):
    forward = (target - eye)
    forward /= np.linalg.norm(forward)
    
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    
    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    view_matrix = np.array([
        [right[0], new_up[0], -forward[0], eye[0]],
        [right[1], new_up[1], -forward[1], eye[1]],
        [right[2], new_up[2], -forward[2], eye[2]],
        [0, 0, 0, 1]
    ])

    return view_matrix



def create_perspective_matrix():
    width, height = 800, 600 
    fov = 60
    near_plane, far_plane = 10, 100.0 
    aspect_ratio = width / height
    fov_rad = np.deg2rad(fov)
    f = 15 / np.tan(fov_rad / 2)
    z_range = near_plane - far_plane
    perspective_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (near_plane + far_plane) / z_range, (2 * near_plane * far_plane) / z_range],
        [0, 0, -1, 0]
    ])
    return perspective_matrix

# view_matrix = np.array([
#         [1, 0, 0, -np.dot(np.array([1, 0, 0]), np.array([0, 0, 0]))],
#         [0, 1, 0, -np.dot( np.array([0, 1, 0]), np.array([0, 0, 0]))],
#         [0, 0, 1, -np.dot(np.array([0, 0, -1]) , np.array([0, 0, 0]))],
#         [0, 0, 0, 1]
#     ])


print("********", view_matrix)
'''
| Xx  Xy  Xz  -dot(X, eye) |
| Yx  Yy  Yz  -dot(Y, eye) |
| Zx  Zy  Zz  -dot(Z, eye) |
|  0   0   0         1     |
'''


# perspective_matrix = np.array([
#     [f / aspect_ratio, 0, 0, 0],
#     [0, f, 0, 0],
#     [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
#     [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
# ])



eye = np.array([0, 0, 10], dtype=np.float64)     # Camera position: (0, 0, 5)
target = np.array([0, 0, 0], dtype=np.float64)  # Camera target: (0, 0, 0) - looking at the origin
up = np.array([0, 100, 0], dtype=np.float64)      # Up direction: (0, 1, 0) - assuming Y is up
joint_matrix =  perspective_matrix @ create_view_matrix(eye,target,up) 

def to_homogeneous(point):
    homogeneous_point = np.array([point['position'][0], point['position'][1], point['position'][2], 1])
    return homogeneous_point

# position mnozena z view in perspective matrix
for point in splat_list:
    homogeneous_point = to_homogeneous(point)
    transformed_point = np.dot(joint_matrix, homogeneous_point)
    point['position'] = transformed_point[:3] / transformed_point[3] 



positions = [splat['position'] for splat in splat_list]
colors = [tuple(x / 255 for x in splat['color']) for splat in splat_list]

x = [point['position'][0] for point in splat_list]
y = [point['position'][1] for point in splat_list]
z = [point['position'][2] for point in splat_list]

plt.figure(figsize=(width, height),  facecolor='black')
plt.scatter(x,y,c=colors,s=2)
plt.show()


# Plotting 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=colors, s=2) 

# for idx in range(1000):
#     ax.scatter(positions[idx][0],positions[idx][1],positions[idx][2],c=colors[idx],s=2 )

# plt.show()



# Y = np.array([0, 1, 0])  # Up direction
# Z = np.array([0, 0, -1]) 

# # Example parameters, you can adjust them as needed
# width, height = 800, 600  # Image dimensions
# fov = 60  # Field of view in degrees
# near_plane, far_plane = 0.1, 100.0  # Near and far clipping planes

# # Construct projection matrix
# aspect_ratio = width / height
# fov_rad = np.deg2rad(fov)
# f = 1.0 / np.tan(fov_rad / 2)
# projection_matrix = np.array([
#     [f / aspect_ratio, 0, 0, 0],
#     [0, f, 0, 0],
#     [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
#     [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
# ])

# Read binary data

# # Define transformation functions
# def decode_float32(data):
#     return struct.unpack('f', data)[0]

# def decode_uint8(data):
#     return struct.unpack('B', data)[0]

# def decode_quaternion_component(data):
#     return (decode_uint8(data) - 128) / 128

# # Process splat data
# splat_size = 32
# splat_count = len(binary_data) // splat_size

# for i in range(splat_count):
#     offset = i * splat_size
#     splat_data = binary_data[offset:offset + splat_size]

#     # Decode splat data
#     position = np.array([
#         decode_float32(splat_data[0:4]),
#         decode_float32(splat_data[4:8]),
#         decode_float32(splat_data[8:12]),
#         1.0
#     ])

#     # Apply transformation
#     transformed_position = projection_matrix @ position

#     # Normalize coordinates
#     transformed_position /= transformed_position[3]
#     print(transformed_position)

#     # Convert to screen space
#     x = int((transformed_position[0] + 1) / 2 * width)
#     y = int((1 - transformed_position[1]) / 2 * height)

#     # Get color
#     color = splat_data[12:16]

#     # Draw pixel (ignoring depth for now)
#     canvas = np.zeros((300, 300, 3), dtype="uint8")
#     cv2.circle(canvas, (x, y), 7, (225,0,0))  # OpenCV uses BGR format, so reverse the color order

# # Display image
# cv2.imshow('Splat Image', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
