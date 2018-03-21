"""
This is a collection of experimantal statements to view data.
"""

from os import listdir
from pickle import load
from operator import itemgetter
from sys import exit

from cv2 import COLOR_BGR2GRAY, cvtColor, imread
from tqdm import tqdm

from prep_data import get_face_landmarks


def general_view():
    """
    General view of structure of data.
    """
    face = 'Face00006'
    distances = load(open('data/distances.pkl', 'rb'))
    landmarks = load(open('data/landmarks.pkl', 'rb'))
    paths = load(open('data/paths_with_landmarks.pkl', 'rb'))
    locations = load(open('data/face_locations.pkl', 'rb'))
    print(landmarks[face], '\n')
    print(distances[face], '\n')
    print(paths[face], '\n')

# general_view()


def sizes():
    """
    Find sizes of cropped images.
    """
    image_sizes = []
    faces = [i for i in listdir('data/cropped') if i.startswith('Face')]
    for face in tqdm(faces):
        images = [i for i in listdir(
            'data/cropped/' + face) if i.endswith('.jpg')]
        for image in images:
            image_sizes.append(
                cvtColor(imread('data/cropped/' + face +
                                '/' + image), COLOR_BGR2GRAY).shape)
    print(set(image_sizes))

# sizes()


def face_nums():
    """
    Find number of faces with more than one image.
    """
    faces = [i for i in listdir('data/cropped') if i.startswith('Face')]
    more_than_one = []
    for face in tqdm(faces):
        images = [i for i in listdir(
            'data/cropped/' + face) if i.endswith('.jpg')]
        if len(images) > 1:
            more_than_one.append(face)
    print(len(more_than_one))

# face_nums()


def view_landmarks_in_cropped():
    """
    Find cropped images with no landmarks and print their number.
    """
    faces = [i for i in listdir('data/cropped') if i.startswith('Face')]
    no_landmarks = []
    for face in tqdm(faces):
        images = [i for i in listdir(
            'data/cropped/' + face) if i.endswith('.jpg')]
        for image in images:
            path = 'data/cropped/' + face + '/' + image
            landmarks = get_face_landmarks(path)
            if not landmarks:
                no_landmarks.append(path)
    dump()
    print('\n', len(no_landmarks), '\n')

view_landmarks_in_cropped()
