import cv2
import imutils
import numpy as np


# asssigment 1 
# write the code for download the image from the url 
# read it without saving it in cv2
# reading image

def image_pro_pipline(img):
    # imgc = cv2.resize(img, (700, 500))  
    resized_img = imutils.resize(img, height=400)
    # by default images in RGB format
    # but cv2 read images in BGR format
    # cv2.imshow("cv2", imgc)
    gray_scale = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_scale, (5, 5), 0)

    edg = cv2.Canny(blur_image, 100, 200)
    return resized_img, edg

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("error while reading video")
        # img = cv2.imread("dd.jpg")
        resized_img, edg = image_pro_pipline(frame)
        cv2.imshow("cv2", resized_img)
        cv2.imshow("edge detection", edg)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()