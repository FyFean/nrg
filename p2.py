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
    
    new_up = -np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    view_matrix = np.array([
        [right[0], new_up[0], -forward[0], eye[0]],
        [right[1], new_up[1], -forward[1], eye[1]],
        [right[2], new_up[2], -forward[2], eye[2]],
        [0, 0, 0, 1]
    ])

    return view_matrix


def create_perspective_matrix():
    aspect_ratio = 1
    fov = 60
    near_plane, far_plane = 10, 100.0 
    
    fov_rad = np.deg2rad(fov)
    f = 1.0 / np.tan(fov_rad / 2)
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

def normalize_pts(points, width):

    # Step 1: Find the minimum and maximum values of x and y coordinates
    min_x, min_y,_ = np.min(points, axis=0)
    max_x, max_y,_ = np.max(points, axis=0)

    aspect_ratio = (max_x - min_x) / (max_y - min_y)

    new_width = width
    new_height = int(new_width / aspect_ratio)

    normalized_points = points.copy()
    normalized_points[:, 0] = ((points[:, 0] - min_x) / (max_x - min_x)) * new_width
    normalized_points[:, 1] = ((points[:, 1] - min_y) / (max_y - min_y)) * new_height

    return new_width,new_height, normalized_points



# def calculate_og_aspect_ratio(splat_list):
#     orix = []
#     oriy = []

#     for point in splat_list:
#         orix.append(point['position'][0])
#         oriy.append(point['position'][1])

#     colors = [tuple(x / 255 for x in splat['color']) for splat in splat_list]

#     oripts = np.array([orix, oriy]).T

#     min_x, min_y = np.min(oripts, axis=0)
#     max_x, max_y = np.max(oripts, axis=0)
#     print(min_x, min_y )
#     print(max_x, max_y )

#     # Step 2: Calculate the aspect ratio
#     aspect_ratioooo =  (max_y - min_y) /(max_x - min_x)
#     # fig, ax = plt.subplots(figsize=(8, 4))
#     # plt.gca().set_aspect('equal', adjustable='box')
#     # scatter = ax.scatter(orix, oriy, c=colors, s=2)
#     # plt.show()

#     return aspect_ratioooo


width = 1200

splat_list = read_binary_file("nike.splat")
eye = np.array([0, 0, 15], dtype=np.float64)     # Camera position: (0, 0, 5)
target = np.array([0, 0, 0], dtype=np.float64)  # Camera target: (0, 0, 0) - looking at the origin
up = np.array([0, 100, 0], dtype=np.float64)      # Up direction: (0, 1, 0) - assuming Y is up



persp = create_perspective_matrix()
view = create_view_matrix(eye,target,up) 
joint_matrix =  persp @ view

x = []
y = []
og_z = []

# multiplying positions with view, projection and viewport matrix
for point in splat_list:
    homogeneous_point = to_homogeneous(point)
    transformed_point = np.dot(joint_matrix, homogeneous_point)
    point['z'] = transformed_point[2]
    # Perspective division
    divided_pt = transformed_point[:3] / transformed_point[3]  

    point['position'] = divided_pt
    x.append(point['position'][0])
    y.append(point['position'][1])
    og_z.append(point['z'])




points = np.array([x, y, og_z]).T


width, height, normalized_points = normalize_pts(points, width)
colors = [tuple(x / 255 for x in splat['color']) for splat in splat_list]
colors_array = np.array(colors)

#plot with scatter
# x = normalized_points[:,0]
# y = normalized_points[:,1]
# print(np.min(x), np.max(x))
# print(np.min(y), np.max(y))
# fig, ax = plt.subplots(figsize=(6, 4))
# plt.gca().set_aspect('equal', adjustable='box')
# scatter = ax.scatter(x, y, c=colors, s=2)
# plt.show()


# def render_points(xyz_color, width, height, scaling_parameter):
#     # Initialize canvas
#     canvas = np.zeros((height + 1, width + 1, 3), dtype=np.float32)

#     xyz_color_sorted = sorted(xyz_color, key=lambda point: point[2], reverse=True)


#     # Iterate through points
#     for point_color in xyz_color_sorted:
#         x, y, z, r, g, b, _ = point_color
        
#         # Calculate side length based on z-axis and scaling parameter
#         side_length = 2 * scaling_parameter / abs(z)
#         # print("2 * ",scaling_parameter,"/", abs(z), "=", side_length)
#         # Calculate pixel coordinates
#         x_min = max(0, round(x - side_length / 2))
#         x_max = min(width, round(x + side_length / 2))
#         y_min = max(0, round(y - side_length / 2))
#         y_max = min(height, round(y + side_length / 2))
#         # Assign color to pixels
#         # print(round(x - side_length / 2))
#         # print(round(x + side_length / 2))

#         canvas[y_min:y_max, x_min:x_max] = [r, g, b]
#     # Plot canvas
#     fig = plt.figure(facecolor='black')  # Set figure face color to black
#     ax = fig.add_subplot(111, facecolor='black')  # Set axis background color to black
#     ax.set_aspect('equal', adjustable='box')
#     ax.imshow(canvas)
#     ax.axis('off')  # Turn off axis
#     plt.show()

def render_points(xyz_color, width, height, scaling_parameter):
    # Initialize canvas with white color and full opacity
    canvas = np.ones((height + 1, width + 1, 4), dtype=np.float32)
    canvas[..., 3] = 1.0  # Set alpha channel to 1 (full opacity)

    # Sort points by z-values
    xyz_color_sorted = sorted(xyz_color, key=lambda point: point[2], reverse=True)

    # Iterate through points
    for point_color in xyz_color_sorted:
        x, y, z, r, g, b, a = point_color
        # print("trenutni kvadratek",r, g, b, a)
        
        # Calculate side length based on z-axis and scaling parameter
        side_length = 2 * scaling_parameter / abs(z)

        # Calculate pixel coordinates
        x_min = max(0, round(x - side_length / 2))
        x_max = min(width, round(x + side_length / 2))
        y_min = max(0, round(y - side_length / 2))
        y_max = min(height, round(y + side_length / 2))
        
        # Alpha blending
        # source_color = np.array([1, 1, 1, 1.0])  # Source color with full opacity, r,g,b,alpha (1,1,1,1.0)
        
        #cez vse piksle kvadratka
        for y_coord in range(y_min, y_max):
            for x_coord in range(x_min, x_max):
                RGBd = canvas[y_coord, x_coord][:3] #trenutna barva canvasa samo r,g,b

                RGBs =  np.array([r, g, b]) #tole farbamo cez torej trenutni kvadratek
                As = a #alfa trenutnega kvadratka

                new_color =  (1 - As) * RGBd + As * RGBs
                new_alpha = 1
                canvas[y_coord, x_coord] =np.append(new_color, new_alpha)

                # print("RGBd",RGBd[:3])
                # print("RGBs",RGBs)
                # print("canvas[y_coord, x_coord]",(1 - As) * RGBd + As * RGBs)
                # print("As",As)
                # print()
    # Plot canvas without alpha channel
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(canvas[..., :3])  # Discard alpha channel for display
    plt.axis('off')  # Turn off axis
    plt.show()


# Join the arrays (270491, 7) -> x,y,z,r,g,b,a [477.50265052 121.0596521  -42.04483991   0.83529412   0.59215686   0.           1.        ]
xyz_color = np.concatenate((normalized_points, colors_array), axis=1)

scaling_parameter_init = 100 

render_points(xyz_color, width, height, scaling_parameter_init)

# Create slider probs dj plt.show stran iz render funkcije
# plt.subplots_adjust(bottom=0.2)  # Adjust plot area for the slider
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Define slider position
# slider_scaling = Slider(ax_slider, 'Scaling Parameter', 25.0, 50.0, valinit=scaling_parameter_init, valstep=5)

# # Function to update plot when slider value changes
# def update(val):
#     scaling_parameter = slider_scaling.val
#     render_points(xyz_color, width, height, scaling_parameter)

# # Connect slider to update function
# slider_scaling.on_changed(update)

# plt.show()



# def render_points(x, y, z, colors, scaling_parameter): 

#     for x_val, y_val, color in zip(x, y, colors):
#         x_idx = int(x_val)
#         y_idx = int(y_val)
#         # print(y_idx, x_idx, color)
#         canvas[y_idx, x_idx, :] = color
#     print(canvas[:10,:10,:])
#     plt.gca().set_aspect('equal', adjustable='box')

#     plt.imshow(canvas)
#     plt.axis('off')  # Turn off axis
#     plt.show()



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






# fig, ax = plt.subplots(figsize=(6, 4))

# # render_points(x,y,og_z,colors,0.5)

# plt.gca().set_aspect('equal', adjustable='box')

# # Slider
# scatter = ax.scatter(x, y, c=colors, s=2)
# # ax_slider = plt.axes([0.2, 0.03, 0.65, 0.03], facecolor='gray')
# # slider = Slider(ax_slider, 'Point Size',0, 60, valinit=1, valstep=1)

# # def update(val):
# #     size = slider.val
# #     ax.clear()

# #     # scatter.set_sizes([size] * len(x))
# #     render_points(x,y,og_z,size)
# #     fig.canvas.draw_idle()
    
# # slider.on_changed(update)
# plt.show()


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
