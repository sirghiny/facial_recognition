"""Scripts to load the data."""

from itertools import combinations
from os import getcwd
from pickle import load
from random import choice, sample, shuffle

from cv2 import COLOR_BGR2GRAY, cvtColor, imread
from numpy import array
from tqdm import tqdm

cwd = getcwd() + '/facial_recognition'


def create_pairs(sample_size):
    """
    Generate pairs to be used for training and testing.

    Some of the pictures should only appear in training or testing only.
    If a face has more than one picture, it can appears in both.
    No combination should be repeated.
    [1,0] - Pair of same person's pictures.
    [0,1] - Pair of different people's pictures.
    Number of [1,0] and [0,1] should be equal.
    Consider data generation by lateral inversion of pictures.
    Return [(path1, path2, [1,0]/[0,1]), ...]
    """
    paths = load(open(cwd + '/data/more_than_one.pkl', 'rb'))
    faces = [key for key in paths]

    def create_positive_pairs():
        """
        Create pairs of images of the same face.

        This will be used as the [1, 0] outcomes.
        A sample of path combinations is selected for use.
        """
        paths_ = [paths[face] for face in faces]
        path_combinations = []
        for i in tqdm(paths_):
            i_comb = list(combinations(i, 2))
            i_comb_rev = [list(reversed(j)) for j in i_comb]
            path_combinations.append(i_comb + i_comb_rev)
        path_combinations = sample(
            [j for i in path_combinations for j in i], int(sample_size / 2))
        return path_combinations

    def create_negative_pairs():
        """
        Create pairs of images of different faces.

        This will be used as the [0, 1] outcomes.
        A sample of 10000 path combinations is selected for use.
        """
        face_combinations = sample(
            list(combinations(faces, 2)), int(sample_size / 2))
        path_combinations = [(choice(paths[i[0]]), choice(paths[i[1]]))
                             for i in tqdm(face_combinations)]
        return path_combinations

    print('\nCreating pairs.')
    positives = create_positive_pairs()
    negatives = create_negative_pairs()
    return positives, negatives


def normalize(image_ndarray):
    """Normalize pixel values to 0-1."""
    return array([array([float(j) / 255 for j in i]) for i in image_ndarray])


def load_data(sample_size):
    """
    Load and prepare data for the Convolutional Neural Network.

    Return x_train1, x_train2, y_train, x_test1, x_test2, y_test.
    x contain numpy arrays of shape (128, 128)
    y contain numpy arrays of binary outcomes ([0, 1], [1, 0])
    """
    positives, negatives = create_pairs(sample_size)
    print('\nOpening images.')
    positive_pairs = [[normalize(cvtColor(imread(j), COLOR_BGR2GRAY))
                       for j in i] for i in tqdm(positives)]
    positive_pairs = [(i, [1, 0]) for i in positive_pairs]
    negative_pairs = [[normalize(cvtColor(imread(j), COLOR_BGR2GRAY))
                       for j in i] for i in tqdm(negatives)]
    negative_pairs = [(i, [0, 1]) for i in negative_pairs]
    pairs = positive_pairs + negative_pairs
    shuffle(pairs)
    train_size = int(0.7 * len(pairs))
    test_size = len(pairs) - train_size
    x_train1 = array([i[0][0] for i in pairs[:train_size]]
                     ).reshape([train_size, 128, 128, 1])
    x_train2 = array([i[0][1] for i in pairs[:train_size]]
                     ).reshape([train_size, 128, 128, 1])
    y_train = array([i[1] for i in pairs[:train_size]])
    x_test1 = array([i[0][0] for i in pairs[train_size:]]
                    ).reshape([test_size, 128, 128, 1])
    x_test2 = array([i[0][1] for i in pairs[train_size:]]
                    ).reshape([test_size, 128, 128, 1])
    y_test = array([i[1] for i in pairs[train_size:]])
    return x_train1, x_train2, y_train, x_test1, x_test2, y_test

x_train1, x_train2, y_train, x_test1, x_test2, y_test = load_data(20000)
