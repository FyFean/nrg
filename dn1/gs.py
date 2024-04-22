import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

format_string = "3f 3f 4B 4b"

# Define camera parameters
camera_position = np.array([0, 0, 10])  # Example camera position
camera_direction = np.array([0, 0, -1])  # Example camera direction
fov = 60  # Example field of view in degrees
aspect_ratio = 16 / 9  # Example aspect ratio
near_plane = 1  # Example near plane
far_plane = 100  # Example far plane

# Define view transformation matrix
view_matrix = np.eye(4)
view_matrix[:3, 3] = -camera_position

# Define perspective transformation matrix
perspective_matrix = np.eye(4)
perspective_matrix[0, 0] = 1 / np.tan(np.radians(fov) / 2)
perspective_matrix[1, 1] = 1 / np.tan(np.radians(fov) / 2) * aspect_ratio
perspective_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
perspective_matrix[2, 3] = -2 * far_plane * near_plane / (far_plane - near_plane)
perspective_matrix[3, 2] = -1
perspective_matrix[3, 3] = 0

# Construct joint transformation matrix
joint_matrix = perspective_matrix @ view_matrix




print(joint_matrix)
# Open the binary file for reading
with open("nike.splat", "rb") as file:
    # Read the entire content of the file
    data = file.read()

# Calculate the number of splats based on the size of the file
num_splats = len(data) // 32

splat_list = []

# Iterate over each splat and unpack its data
for i in range(num_splats):
    # Calculate the starting index of the current splat
    start_index = i * 32

    # Unpack the data for the current splat
    splat_data = struct.unpack_from(format_string, data, start_index)
    # position homogene koordinate, 4d vektor z 1ko na koncu, ostale tri xyz so iste
    # splat_center = np.array(splat_data[:3] + (1,))
    # mnozimo nas position (4d vektor) z matriko (4x4 pomoje) da applyjamo view in perspective transformation, dobimo (4d vektor)
    # transformed_center = joint_matrix @ splat_center

    # delimo z homogeno koordinato da normaliziramo nazaj v 3d vektor
    # screen_space_coords = transformed_center[:3] / transformed_center[3]
    # Extract individual components from the unpacked data
    position = splat_data[:3]
    scale = splat_data[3:6]
    color = splat_data[6:10]
    rotation = splat_data[10:]

    # Perform any necessary decoding on rotation components
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

def init():
    camera_position = np.array([0, 0, 10]) #position kamere
    camera_target = np.array([0, 0, 0]) #position kam gleda kamera gledamo v 0,0,0
    camera_up = np.array([0, 1, 0]) #kok gor gledamo, y vector

    # Define perspective parameters
    fov = 60  # Field of view
    aspect_ratio = 16 / 9  # Aspect ratio
    near_plane = 1  # Near plane distance
    far_plane = 100  # Far plane distance
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect_ratio, near_plane, far_plane)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(*camera_position, *camera_target, *camera_up)

# Render function
def render():
    # You allocate memory on the GPU and store your vertex data (positions, colors, normals, etc.) in the VBO.

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw your objects here
    # Example: Draw a colored cube
    glColor3f(1.0, 0.0, 0.0)  # Set color to red
    glutWireCube(2.0)  # Draw a wireframe cube
    # Draw splats
    glPointSize(5)  # Set the size of points
    #  Begin drawing points
    glBegin(GL_POINTS)

    for splat in splat_list:
        # Extract data from splat
        position = splat['position']
        scale = splat['scale']
        color = splat['color']
        rotation = splat['rotation']
        
        # Apply transformations
        glPushMatrix()
        glTranslatef(*position)  # Translate to the position
        glRotatef(rotation[0], 1, 0, 0)  # Rotate around X-axis
        glRotatef(rotation[1], 0, 1, 0)  # Rotate around Y-axis
        glRotatef(rotation[2], 0, 0, 1)  # Rotate around Z-axis
        glScalef(*scale)  # Scale
        
        # Set color
        glColor3f(1.0, 0.0, 0.0)
        
        # Draw point
        glVertex3f(0, 0, 0)  # Draw at the origin
        glPopMatrix()
    glEnd()

    
    glutSwapBuffers()

# Main function
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Transformation Visualization")
    glEnable(GL_DEPTH_TEST)
    init()
    glutDisplayFunc(render)
    glutMainLoop()

if __name__ == "__main__":
    main()

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Simple PyQt Canvas")
#         self.setGeometry(100, 100, 800, 600)

#         self.canvas = PlotCanvas(self)
#         self.setCentralWidget(self.canvas)

# class PlotCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
#         super().__init__(fig)
#         self.setParent(parent)
#         self.plot_random_points()

#     def plot_random_points(self):
#         # Generate random data
#         x = np.random.rand(10)
#         y = np.random.rand(10)

#         # Plot the points
#         self.ax.scatter(x, y)
#         self.ax.set_title('Random Scatter Plot')
#         self.ax.set_xlabel('X')
#         self.ax.set_ylabel('Y')

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mainWindow = MainWindow()
#     mainWindow.show()
#     sys.exit(app.exec_())
