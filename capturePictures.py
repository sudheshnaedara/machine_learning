import cv2
import os
name = "suedara"
path = "Images/training/"+name
cam = cv2.VideoCapture(0)
cv2.namedWindow("capture image")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        if os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
        img_name = "{}/{}_{}.png".format(path,name,img_counter)
        cv2.rotate(frame,cv2.ROTATE_180)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
