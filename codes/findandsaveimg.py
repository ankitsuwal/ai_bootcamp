from os import listdir
import shutil
import os
import cv2

#####################################################################
    # Below code for match file name and save into another directory.
#####################################################################

dst = "/home/dell/work/projects/images/video/to_check/dst/"
txt_path = "/home/dell/work/projects/images/video/to_check/2000/"
img_path = "/home/dell/work/projects/images/video/to_check/2000_img/"
for file in listdir(img_path):
    txt_name = file.split(".")[0] + ".txt"
    for root, dirs, files in os.walk(txt_path):
        if txt_name in files:
            txt_file = txt_path + txt_name
            shutil.copy(txt_file, dst)
            print(txt_name)


#################################################
#  for save frame in interval
#################################################
# f10_15_05_20_<count>

# if __name__ == "__main__":
#     count = 0
#     path = "/home/dell/work/projects/images/video/5_14_2020 8_00_13 AM (UTC+05_30).mkv"
#     to_save_frame = "/home/dell/work/projects/images/video/5_14_2020/"
#     cap = cv2.VideoCapture(path)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("error while reading video")
        
#         if count % 29 == 0:
#             cv2.imwrite(to_save_frame + "ailab_15_5_%d.jpg" % count, frame)    
#         count += 1
#         # cv2.imshow("image", frame)
#         print("count: ", count)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

##################################################################
# for rename the file in same directory
##################################################################
# # ailab_15_5_0.jpg
# if __name__ == "__main__":
#     path = "/home/dell/work/projects/images/video/5_13_2020/"
#     # path = "/home/dell/work/projects/images/video/ats/"
#     for file in listdir(path):
#         name = file[5:]
#         os.rename(path + file, path + "recep" + name)
