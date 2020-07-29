from os import listdir
import shutil
import os
import cv2

#####################################################################
# Below code for match txt file name into image directory and save into another directory.
#####################################################################

# dst = "/home/dell/work/projects/images/new/yolo_19_05_2020_10_02_40/"
# txt_path = "/home/dell/work/projects/images/new/yolo_19_05_2020_10_02_40/"
# img_path = "/home/dell/work/projects/images/new/19_05_2020_10_02_40/"
# for file in listdir(txt_path):
#     img_name = file.split(".")[0] + ".jpg"
#     for root, dirs, files in os.walk(img_path):
#         if img_name in files:
#             image = img_path + img_name
#             shutil.copy(image, dst)
#             print(image)


#################################################
#  for save frame in interval
#################################################

if __name__ == "__main__":
    count = 0
    path = "/home/dell/work/projects/images/21052020/22-05-2020 16_59_17 (UTC+05_30).mkv"
    to_save_frame = "/home/dell/work/projects/images/21052020/frame_16_59_17/"
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("error while reading video")
        
        if count % 29 == 0:
            cv2.imwrite(to_save_frame + "cent2_16_59_17%d.jpg" % count, frame)    
        count += 1
        # cv2.imshow("image", frame)
        print("count: ", count)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

##################################################################
# for rename the file in same directory
##################################################################
# if __name__ == "__main__":
#     path = "/home/dell/work/projects/images/new/frame_14_50_44/"
#     for file in listdir(path):
#         name = file[6:]
#         os.rename(path + file, path + "rec" + name)
