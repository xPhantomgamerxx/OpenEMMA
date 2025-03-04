import cv2
import numpy as np

# 1. Load the image (update the path as needed)
image_path = '/home/ubuntu/project_ws/OpenEMMA/lane_test/frame_00000011.jpg'
img = cv2.imread(image_path)

# Below are four polylines representing the main orange lane-dividing lines
# visible in the image. Each is defined from top (y ~ 0) down to bottom (y ~ image height).
# The coordinates (x,y) were eyeballed from the provided image.
# You might tweak them slightly to align perfectly.

line1 = np.array([[ 443,    0],
                  [ 402,  302],
                  [ 362,  603],
                  [ 302,  905],
                  [ 241, 1206],
                  [ 161, 1544]], dtype=np.int32)

line2 = np.array([[ 805,    0],
                  [ 744,  302],
                  [ 704,  603],
                  [ 664,  905],
                  [ 604, 1206],
                  [ 563, 1544]], dtype=np.int32)

line3 = np.array([[1308,    0],
                  [1268,  302],
                  [1237,  603],
                  [1207,  905],
                  [1187, 1206],
                  [1167, 1544]], dtype=np.int32)

line4 = np.array([[1710,    0],
                  [1670,  302],
                  [1650,  603],
                  [1620,  905],
                  [1589, 1206],
                  [1559, 1544]], dtype=np.int32)

# Reshape for cv2.polylines()
line1 = line1.reshape((-1,1,2))
line2 = line2.reshape((-1,1,2))
line3 = line3.reshape((-1,1,2))
line4 = line4.reshape((-1,1,2))

# Draw each line in a different color
cv2.polylines(img, [line1], False, (0, 0, 255),   3)  # red
cv2.polylines(img, [line2], False, (0, 255, 0),   3)  # green
cv2.polylines(img, [line3], False, (255, 0, 0),   3)  # blue
cv2.polylines(img, [line4], False, (0, 255, 255), 3)  # yellowish

# 3. Visualize the result
cv2.imwrite("outtest.jpg", img)
# cv2.imshow("Lane Dividers", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
