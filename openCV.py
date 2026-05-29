import cv2

img = cv2.imread("face_1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = cv2.CascadeClassifier(

    "haarcascade_frontalface_default.xml"

)

faces = face.detectMultiScale(gray)

for (x, y, w, h) in faces:

    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(img, "face", (x, y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8, (0,255,0), 2)

cv2.imshow("face", img)

cv2.waitKey(0)

cv2.destroyAllWindows()