import cv2

from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(

    "haarcascade_frontalface_default.xml"

)

img = cv2.imread("face_1.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:

    face_img = img[y:y+h, x:x+w]

    result = DeepFace.analyze(

        img_path=face_img,

        actions=["emotion"],

        enforce_detection=False,

        detector_backend="skip"

    )

    emotion = result[0]["dominant_emotion"]

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(img, emotion, (x, y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                1, (0, 255, 0), 2)

cv2.imshow("face", img)

cv2.waitKey(0)

cv2.destroyAllWindows()