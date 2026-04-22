"""
recognizer.py — real-time face recognition via webcam, video, or image batch.
Draws bounding boxes, predicts identity, and marks attendance via sheet.py.
"""

import os
import sys
import pickle
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import facenet
import detect_face
from face_aligner import FaceAligner
from sheet import mark_present


def recognize(mode: str, parameters: tuple) -> str:
    """
    mode: 'w' = webcam, 'v' = video, 'i' = image folder
    Returns comma-separated string of recognised names.
    """
    path1, path2, face_dim, gpu, thresh1, thresh2, resolution, img_path, out_img_path, vid_path, vid_save, vid_see = parameters

    classifier_filename = os.path.expanduser(path1 if path1 else 'classifier.pkl')
    model = (path2 + "/20180402-114759/20180402-114759.pb") if path2 else "20180402-114759/20180402-114759.pb"
    image_size = tuple(map(int, face_dim.split('x'))) if face_dim else (160, 160)
    gpu_fraction = round(float(gpu), 1) if gpu else 0.8

    affine = FaceAligner(desiredLeftEye=(0.33, 0.33), desiredFaceWidth=image_size[0], desiredFaceHeight=image_size[1])

    # MTCNN params
    minsize = 20
    threshold = [0.6, 0.7, 0.8]
    factor = 0.709
    if thresh1 and float(thresh1) < 1:
        threshold[2] = round(float(thresh1), 2)

    classifier_threshold = float(thresh2) if thresh2 else 0.50

    # Build separate TF graphs for FaceNet and MTCNN
    g1, g2 = tf.Graph(), tf.Graph()
    with g1.as_default():
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, log_device_placement=False)):
            facenet.load_model(model)

    with g2.as_default():
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts, log_device_placement=False))
        with sess2.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess2, None)

    with open(classifier_filename, 'rb') as f:
        (modelSVM, class_names) = pickle.load(f)

    # Input setup
    loop_type = False
    res = (640, 480)
    total_frames = 0
    save_video, save_images, display_output = False, False, True
    frame_no = 1
    image_list, out_img_folder, output_video = [], "", []
    st_name = ''

    if mode == 'w':
        data_type = 0
        loop_type = True
        if resolution:
            res = tuple(map(int, resolution.split('x')))

    elif mode == 'i':
        image_folder = img_path if img_path else ''
        image_list = os.listdir(image_folder)
        total_frames = len(image_list)
        save_images = True
        out_img_folder = _setup_output_images_folder(out_img_path)
        display_output = vid_see != 'y'

    else:  # video
        data_type = vid_path
        save_video = vid_save == 'y'
        display_output = vid_see != 'y'

    device = cv2.VideoCapture(data_type if mode != 'i' else 0)

    if mode == 'w':
        device.set(3, res[0])
        device.set(4, res[1])
    elif mode == 'v':
        total_frames = int(device.get(cv2.CAP_PROP_FRAME_COUNT))
        if save_video:
            fps = device.get(cv2.CAP_PROP_FPS)
            fmt = int(device.get(cv2.CAP_PROP_FOURCC))
            sz = (int(device.get(cv2.CAP_PROP_FRAME_WIDTH)), int(device.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            output_video = cv2.VideoWriter("output_videos/Output_" + data_type, fmt, fps, sz)

    while loop_type or (frame_no <= total_frames):
        image = cv2.imread(image_folder + "/" + image_list[frame_no - 1]) if mode == 'i' else device.read()[1]

        image = _preprocess(image)
        bb, points = _detect_faces(image, g2, pnet, rnet, onet, minsize, threshold, factor)

        if bb.shape[0] > 0:
            embedding = _embed_faces(image, bb, points, affine, g1, image_size)
            st_name = _predict_and_draw(
                image, bb, points, embedding, modelSVM, class_names,
                classifier_threshold, save_video, save_images, display_output, st_name
            )

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if display_output:
            cv2.imshow("Output", image)
        if save_video:
            output_video.write(image)
        if save_images and image_list:
            cv2.imwrite(out_img_folder + image_list[frame_no - 1], image)
        if not loop_type:
            print("\nProgress: %.2f%%" % (100 * frame_no / total_frames))
            frame_no += 1
        if cv2.waitKey(1) == ord('q'):
            if save_video:
                output_video.release()
            device.release()
            cv2.destroyAllWindows()
            break

    return st_name


# ── Private helpers ───────────────────────────────────────────────────────────

def _preprocess(image):
    image = cv2.resize(image, (800, 600))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v -= 0
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return np.asarray(image, dtype='uint8')


def _detect_faces(image, g2, pnet, rnet, onet, minsize, threshold, factor):
    g2.as_default()
    with tf.Session(graph=g2):
        return detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)


def _embed_faces(image, bb, points, affine, g1, image_size):
    img_list = []
    for col in range(points.shape[1]):
        aligned = affine.align(image, points[:, col])
        mean, std = np.mean(aligned), np.std(aligned)
        std_adj = np.maximum(std, 1.0 / np.sqrt(aligned.size))
        img_list.append(np.multiply(np.subtract(aligned, mean), 1 / std_adj))

    images = np.stack(img_list)
    g1.as_default()
    with tf.Session(graph=g1) as sess:
        ip = tf.get_default_graph().get_tensor_by_name("input:0")
        emb = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        pt = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        return sess.run(emb, feed_dict={ip: images, pt: False})


def _predict_and_draw(image, bb, points, embedding, modelSVM, class_names,
                      threshold, save_video, save_images, display_output, st_name):
    predictions = modelSVM.predict_proba(embedding)
    best_idx = np.argmax(predictions, axis=1)
    best_prob = predictions[np.arange(len(best_idx)), best_idx]

    if save_video or display_output or save_images:
        for i in range(bb.shape[0]):
            cv2.rectangle(image, (int(bb[i][0]), int(bb[i][1])), (int(bb[i][2]), int(bb[i][3])), (0, 255, 0), 1)
            if best_prob[i] > threshold:
                name = class_names[best_idx[i]]
                cv2.putText(image, name, (int(bb[i][0] + 1), int(bb[i][1]) + 10),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                st_name += ',' + name
                mark_present(st_name)
        for col in range(points.shape[1]):
            for i in range(5):
                cv2.circle(image, (int(points[i][col]), int(points[i + 5][col])), 1, (0, 255, 0), 1)

    return st_name


def _setup_output_images_folder(out_img_path: str) -> str:
    path = out_img_path if out_img_path and os.path.isdir(out_img_path) else ""
    if path:
        path += '/output_images'
    else:
        path = 'output_images'
    if not os.path.isdir(path):
        os.makedirs(path)
    return path + "/"
