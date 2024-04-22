width = 600
height = 400

# Calculate the aspect ratio
aspect_ratio = width / height

# Define the normalized view volume extents
view_volume_width = 2
view_volume_height = 2

# Determine the boundaries of the viewport
# Assuming symmetric view volume around the origin
vr = view_volume_width / 2
vl = -vr

# Adjust top and bottom boundaries based on aspect ratio
vt = vr / aspect_ratio
vb = -vt

print("vr:", vr)
print("vl:", vl)
print("vt:", vt)
print("vb:", vb)