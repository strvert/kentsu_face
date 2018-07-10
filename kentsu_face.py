import sys
import os
import cv2
from PIL import Image


expands = ['.jpg', '.png', '.gif', '.svg', '.tiff', '.bmp']

cascade_path = "haarcascade_frontalface_alt.xml"
image_path = sys.argv[1]
file_name = os.path.basename(image_path)
for x in expands:
    file_name = file_name.replace(x, '')
print(file_name)

image = cv2.imread(image_path)

kentsu_path = "kentsu_face.png"
kentsu = cv2.imread(kentsu_path, -1)

color = (255, 255, 255)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cascade_path)

face_detect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

kora_image = image
kora_image_array = cv2.cvtColor(kora_image, cv2.COLOR_BGR2RGBA)
kora_image_p = Image.fromarray(kora_image_array)
if len(face_detect) > 0:
    for rect in face_detect:
        width, height = int(rect[2]*1.25), int(rect[3]*1.5)
        kentsu_scaled = cv2.resize(kentsu, (width, height))
        kentsu_scaled_array = cv2.cvtColor(kentsu_scaled, cv2.COLOR_BGRA2RGBA)
        kentsu_scaled_p = Image.fromarray(kentsu_scaled_array)
        kora_image_p.paste(kentsu_scaled_p, (rect[0]-int(kentsu_scaled_p.size[0]*0.2), rect[1]-int(kentsu_scaled_p.size[1]*0.25)), mask=kentsu_scaled_p)

kora_image_p.save(file_name + '_kentsued.png')
