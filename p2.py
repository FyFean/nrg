import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

from util import *



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



def create_perspective_matrix(width, height):
    
    fov = 90
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

def to_homogeneous(point):
    homogeneous_point = np.array([point['position'][0], point['position'][1], point['position'][2], 1])
    return homogeneous_point

# view_matrix = np.array([
#         [1, 0, 0, -np.dot(np.array([1, 0, 0]), np.array([0, 0, 0]))],
#         [0, 1, 0, -np.dot( np.array([0, 1, 0]), np.array([0, 0, 0]))],
#         [0, 0, 1, -np.dot(np.array([0, 0, -1]) , np.array([0, 0, 0]))],
#         [0, 0, 0, 1]
#     ])


# perspective_matrix = np.array([
#     [f / aspect_ratio, 0, 0, 0],
#     [0, f, 0, 0],
#     [0, 0, (far_plane + near_plane) / (near_plane - far_plane), -1],
#     [0, 0, (2 * far_plane * near_plane) / (near_plane - far_plane), 0]
# ])
def create_viewport_matrix(width, height):
    # Define the viewport dimensions
    viewport_width = width
    viewport_height = height
    
    # Define the viewport origin (usually (0, 0))
    viewport_x = 0
    viewport_y = 0
    
    # Define the depth range of the viewport (usually 0 to 1)
    viewport_min_depth = 0
    viewport_max_depth = 1
    
    # Note: We multiply the scaling factor by -1 to perform the flip
    viewport_matrix = np.array([
        [viewport_width / 2, 0, 0, viewport_x + viewport_width / 2],
        [0, viewport_height / 2, 0, viewport_y + viewport_height / 2],  # Changed sign here
        [0, 0, (viewport_max_depth - viewport_min_depth) / 2, (viewport_max_depth + viewport_min_depth) / 2],
        [0, 0, 0, 1]
    ])
    
    return viewport_matrix
width = 600
height = 400
viewport_matrix = create_viewport_matrix(width, height)

splat_list = read_binary_file("nike.splat")

eye = np.array([0, 0, 15], dtype=np.float64)     # Camera position: (0, 0, 5)
target = np.array([0, 0, 0], dtype=np.float64)  # Camera target: (0, 0, 0) - looking at the origin
up = np.array([0, 100, 0], dtype=np.float64)      # Up direction: (0, 1, 0) - assuming Y is up
width, height = 600, 400 

persp = create_perspective_matrix(width, height)
view = create_view_matrix(eye,target,up) 

joint_matrix =  persp @ view


x = []
y = []
og_z = []

# position mnozena z view in perspective matrix
for point in splat_list:
    homogeneous_point = to_homogeneous(point)
    transformed_point = np.dot(joint_matrix, homogeneous_point)
    transformed_point2 = np.dot(viewport_matrix,transformed_point)
    point['z'] = transformed_point[2]
    # Perspective division
    viewport_point = transformed_point2[:3] / transformed_point2[3]

    
    # viewport_point = np.dot(viewport_matrix[:3, :3], ndc_point)
    point['position'] = viewport_point

    x.append(point['position'][0])
    y.append(point['position'][1])
    og_z.append(point['z'])

x = np.array(x)
y = np.array(y)
print(np.max(x))
print(np.max(y))
print(x[600:630])
og_z = np.array(og_z)
colors = [tuple(x / 255 for x in splat['color']) for splat in splat_list]

canvas = np.zeros((height, width, 4), dtype=np.float32)

def render_points(x, y, z, colors, scaling_parameter): 

    for x_val, y_val, color in zip(x, y, colors):
        x_idx = int(x_val)
        y_idx = int(y_val)
        # print(y_idx, x_idx, color)
        canvas[y_idx, x_idx, :] = color
    print(canvas[:10,:10,:])
    plt.imshow(canvas)
    plt.axis('off')  # Turn off axis
    plt.show()



    # print("s",scaling_parameter)
    # scaling_factor = scaling_parameter / z
    # print("z",z[0:5])
    # side_length = 2 * scaling_factor
    # print(scaling_factor[0:5])
    # print()
  

    # # Calculate coordinates of square vertices for all points 5 x 2 x 27000
    # # [x1,y1]
    # # [x2,y2]
    # # [x3,y3]
    # # [x4,y4]
    
    # square_vertices = np.array([
    #     [x - side_length / 2, y - side_length / 2],  # Bottom-left
    #     [x + side_length / 2, y - side_length / 2],  # Bottom-right
    #     [x + side_length / 2, y + side_length / 2],  # Top-right
    #     [x - side_length / 2, y + side_length / 2],  # Top-left
    # ])

    # # print("squares1",square_vertices[:,:,30], "scaling f ", side_length[30])
    # # print("squares2",square_vertices[:,:,800], "scaling f ", side_length[800])

    # plt.figure(figsize=(8, 6))
    # # for i in range(1000):
    # for i in range(square_vertices.shape[2]):
    #     plt.fill([square_vertices[0][0][i], square_vertices[1][0][i], square_vertices[2][0][i], square_vertices[3][0][i], square_vertices[0][0][i]], [square_vertices[0][1][i], square_vertices[1][1][i], square_vertices[2][1][i], square_vertices[3][1][i], square_vertices[0][1][i]], color=colors[i]) 








# square1 = [[-0.11789554, -4.7795533 ],
#  [-0.14580792, -4.7795533 ],
#  [-0.14580792, -4.80746568],
#  [-0.11789554, -4.80746568],
#  [-0.11789554, -4.7795533 ]]  # Close the square

# # Coordinates of the second square vertices
# square2 = [[ 1.11286813, -4.89220175],
#  [ 1.08481644,-4.89220175],
#  [ 1.08481644, -4.92025345],
#  [ 1.11286813, -4.92025345],
#  [ 1.11286813, -4.89220175]]  # Close the square

# # Extract x and y coordinates for both squares
# x1 = [point[0] for point in square1]
# y1 = [point[1] for point in square1]

# x2 = [point[0] for point in square2]
# y2 = [point[1] for point in square2]

# # Plot both squares
# plt.figure(figsize=(6, 4))
# plt.plot(x1, y1, 'b-')  # 'b-' for blue lines
# plt.plot(x2, y2, 'r-')  # 'r-' for red lines
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Two Squares')
# plt.grid(True)
# plt.axis('equal')  # Equal aspect ratio
# plt.show()



fig, ax = plt.subplots(figsize=(6, 4))

# render_points(x,y,og_z,colors,0.5)

plt.gca().set_aspect('equal', adjustable='box')

# Slider
scatter = ax.scatter(x, y, c=colors, s=2)
# ax_slider = plt.axes([0.2, 0.03, 0.65, 0.03], facecolor='gray')
# slider = Slider(ax_slider, 'Point Size',0, 60, valinit=1, valstep=1)

# def update(val):
#     size = slider.val
#     ax.clear()

#     # scatter.set_sizes([size] * len(x))
#     render_points(x,y,og_z,size)
#     fig.canvas.draw_idle()
    
# slider.on_changed(update)
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
