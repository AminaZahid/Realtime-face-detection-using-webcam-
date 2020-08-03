import matplotlib.pyplot as plt
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/zahid/Desktop/DL_Projects/Face_Detection/Facial_Trained_Classifier/haarcascade_frontalface_default.xml')
def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
    return frame



video_capture=cv2.VideoCapture(0)
while True:
    _, frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,frame)
    cv2.imshow('Face Detection in Video',canvas)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()