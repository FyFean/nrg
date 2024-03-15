import numpy as np
import struct
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
