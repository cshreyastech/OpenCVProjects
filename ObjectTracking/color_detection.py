import cv2
image = cv2.imread("/home/shreyas/Downloads/blue_ball.png", cv2.IMREAD_UNCHANGED)
print(image.shape)
b, g, r = (image[103, 102])[-3:]
# if image type is b g r, then b g r value will be displayed.
# if image is gray then color intensity will be displayed.
print("BGR",b, g, r)
cv2.imshow("Frame", image)
key = cv2.waitKey(5000) & 0xFF

cv2.destroyAllWindows()
