import cv2

# Create a Cascade Classifier object
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

img = cv2.imread("images/arpan.jpg", 1)

# Search the co-ordinates of the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()