"""
dataset.py — captures face images from webcam or video for a named person.
Called during enrollment to build the training dataset.
"""

import os
import sys
import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN


def dataset_creation(parameters):
    path1, webcam, face_dim, gpu, username, vid_path = parameters
    path = ""
    personNo = 1
    folder_name = ""

    # Resolve output directory
    if os.path.isdir(path1):
        path = path1 + '/output'
        if not os.path.isdir(path):
            os.makedirs(path)
            print("Directory successfully made in: " + path)
        else:
            print("Directory already exists. Using it")
    else:
        path = 'output'
        if not os.path.isdir(path):
            if os.makedirs(path):
                print("Error in making directory.")
                sys.exit()
            else:
                print("Directory successfully made: " + path)
        else:
            print("Directory already exists. Using it")

    detector = MTCNN()

    res = tuple(map(int, webcam.split('x'))) if webcam else (640, 480)
    gpu_fraction = round(float(gpu), 1) if gpu else 0.8
    face_size = tuple(map(int, face_dim.split('x'))) if face_dim else (160, 160)

    while True:
        ask = username.replace(" ", "_")
        folder_name = ask if ask else 'person' + str(personNo)
        personNo += 1

        users_folder = path + "/" + folder_name
        image_no = 1

        if not os.path.isdir(users_folder):
            if os.makedirs(users_folder):
                print("Error in making directory.")
                sys.exit()
            else:
                print("Directory successfully made: " + users_folder)
        else:
            print("Directory already exists. Using it")

        data_type = int(vid_path) if vid_path == "" else vid_path
        loop_type = vid_path == ""
        total_frames = 0

        device = cv2.VideoCapture(data_type if vid_path != "" else 0)

        if vid_path == "":
            device.set(3, res[0])
            device.set(4, res[1])
        else:
            total_frames = int(device.get(cv2.CAP_PROP_FRAME_COUNT))
            loop_type = False

        while loop_type or (total_frames > 0):
            if not loop_type:
                total_frames -= 1

            ret, image = device.read()

            if (cv2.waitKey(1) & 0xFF) == ord("s"):
                detect = detector.detect_faces(image)
                if detect:
                    bb = detect[0]['box']
                    x, y, w, h = bb
                    aligned_image = image[y:y+h, x:x+w]
                    image_name = users_folder + "/" + folder_name + "_" + str(image_no).zfill(4) + ".png"
                    cv2.imwrite(image_name, aligned_image)
                    image_no += 1

            cv2.imshow("Output", image)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                device.release()
                cv2.destroyAllWindows()
                _resize_saved_images(path, folder_name)
                return 1


def _resize_saved_images(path: str, folder_name: str, size: int = 160):
    folder = path + '/' + folder_name
    for file in os.listdir(folder):
        f_img = folder + '/' + file
        try:
            img = Image.open(f_img)
            img = img.resize((size, size))
            img.save(f_img)
        except IOError:
            pass
