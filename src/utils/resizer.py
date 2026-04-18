import os
from PIL import Image

folder_name = input()
f = 'output/' + folder_name

new_d = 160
for file in os.listdir(f):
    f_img = f+'/'+file
    try:
        img = Image.open(f_img)
        img = img.resize((new_d, new_d))
        img.save(f_img)
    except IOError:
        pass
