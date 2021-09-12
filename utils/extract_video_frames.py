import cv2
import os

video_path = "KTH FSD Perception- Intro/campus.mp4"
name = "VID_20180619_175221224_HDR"

try:
    os.mkdir(f"./frames/{name}")
except:
    pass


vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(f"./frames/{name}/frame%d.jpg" % count, image)     # save frame as JPG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1