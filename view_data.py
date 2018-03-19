"""
This is a collection of experimantal statements to view data.
"""

from os import listdir
from pickle import load
from PIL import Image
from sys import exit

from face_recognition import load_image_file
from prep_data import get_face_locations, euclidean_distance
from tqdm import tqdm


def general_view():
    face = 'Face00006'
    distances = load(open('data/distances.pkl', 'rb'))
    landmarks = load(open('data/landmarks.pkl', 'rb'))
    paths = load(open('data/paths_with_landmarks.pkl', 'rb'))
    locations = load(open('data/face_locations.pkl', 'rb'))
    print(landmarks[face], '\n')
    print(distances[face], '\n')
    print(paths[face], '\n')

general_view()
