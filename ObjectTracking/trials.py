from collections import deque
import numpy as np


pts = deque(maxlen=32)
counter = 0
(dX, dY) = (0, 0)


for i in range(32):
	pts.appendleft((i, i))

print(pts)


print(pts[-10][0])
print(pts[-33][0])

# print(pts[len(pts)-10][0])

# for i in np.arange(1, len(pts)):
	# dX = pts[-10][0] - pts[i][0]
