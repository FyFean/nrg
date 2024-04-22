import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import struct

def readData():
    format_string = "3f 3f 4B 4b"
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
    return splat_list

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube(l):
    glPointSize(0.11) 
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    for splat in l:
        p = splat['position']
        s = splat['scale']
        c = splat['color']
        r = splat['rotation']
        # print(p)
        p[0] -= screen_center_x
        p[1] -= screen_center_y
        p[2] -= screen_center_z
        glVertex3fv(p)
    glEnd()


def main():
    l = readData()
    pygame.init()
    display = (800,600) # -> |
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    #view transformation - moves the camera to desired location
    gluLookAt(0, 0, 10,  # Camera position
         0, 0, 0,  # Target position, position kam gleda kamera gledamo v 0,0,0
         0, 1, 0)  # Up vector, kam gor premaknemo v y direction

    #perspective transformation
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0) #FOV, aspect, z near, z far

    glTranslatef(0.0,0.0, -5) # lokacija kamere, -5 v z direction

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube(l)
        pygame.display.flip()
        pygame.time.wait(10)


main()
