from os import listdir
import shutil
import os
dst = "/home/dell/work/projects/images/150_img/"
txt_path = "/home/dell/work/projects/images/14_05_1_03/"
img_path = "/home/dell/work/projects/images/14_05_3_/"
for file in listdir(txt_path):
	txt = file.split(".")[0] + ".jpg"
	for root, dirs, files in os.walk(img_path):
		if txt in files:
			image = img_path + txt
			shutil.copy(image, dst)
			print(txt)




