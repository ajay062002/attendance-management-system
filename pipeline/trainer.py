"""
trainer.py — trains an SVM classifier on FaceNet embeddings.
Optionally splits dataset into train/test and reports accuracy.
"""

import os
import math
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import facenet
from sklearn.svm import SVC


def get_embeddings(model: str, paths: list, batch_size: int, image_size: int, gpu_fraction: float) -> np.ndarray:
    """Run FaceNet forward pass to produce 128-d embeddings for each image path."""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            print('\nLoading feature extraction model')
            facenet.load_model(model)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            print('Calculating features for images')
            for i in range(nrof_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, nrof_images)
                batch_paths = paths[start:end]
                images = facenet.load_data(batch_paths, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start:end, :] = sess.run(embeddings, feed_dict=feed_dict)

    return emb_array


def train(parameters):
    path1, path2, batch, img_dim, gpu, svm_name, split_percent, split_data = parameters

    path = path1 if path1 else 'output'
    gpu_fraction = round(float(gpu), 1) if gpu else 0.8
    model = (path2 + "/20180402-114759/20180402-114759.pb") if path2 else "20180402-114759/20180402-114759.pb"
    batch_size = int(batch) if batch else 90
    image_size = int(img_dim) if img_dim else 160
    classifier_filename = os.path.expanduser((svm_name + '.pkl') if svm_name else 'classifier.pkl')
    split_dataset = split_data
    percentage = float(split_percent) if split_percent else 70.0

    dataset = facenet.get_dataset(path)
    train_set, test_set = [], []

    if split_dataset == 'y':
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            n_train = int(percentage * len(paths) * 0.01)
            train_set.append(facenet.ImageClass(cls.name, paths[:n_train]))
            test_set.append(facenet.ImageClass(cls.name, paths[n_train:]))

    if split_dataset == 'y':
        paths_train, labels_train = facenet.get_image_paths_and_labels(train_set)
        paths_test, labels_test = facenet.get_image_paths_and_labels(test_set)
        print('\nClasses: %d | Train: %d | Test: %d' % (len(train_set), len(paths_train), len(paths_test)))
    else:
        paths_train, labels_train = facenet.get_image_paths_and_labels(dataset)
        paths_test, labels_test = [], []
        print('\nClasses: %d | Images: %d' % (len(dataset), len(paths_train)))

    emb_array = get_embeddings(model, paths_train, batch_size, image_size, gpu_fraction)

    print('\nTraining SVM classifier')
    model_svc = SVC(kernel='linear', probability=True)
    model_svc.fit(emb_array, labels_train)

    class_names = [cls.name.replace('_', ' ') for cls in (train_set if split_dataset == 'y' else dataset)]

    with open(classifier_filename, 'wb') as f:
        pickle.dump((model_svc, class_names), f)
    print('\nSaved classifier to: "%s"' % classifier_filename)

    if split_dataset == 'y' and paths_test:
        test_emb = get_embeddings(model, paths_test, batch_size, image_size, gpu_fraction)
        _evaluate(classifier_filename, test_emb, labels_test)

    return 1


def _evaluate(classifier_filename: str, emb_array: np.ndarray, labels_test: list):
    with open(classifier_filename, 'rb') as f:
        (modelSVM, class_names) = pickle.load(f)

    predictions = modelSVM.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

    for i in range(len(best_class_indices)):
        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

    accuracy = np.mean(np.equal(best_class_indices, labels_test))
    print('\nAccuracy: %.3f' % accuracy)
