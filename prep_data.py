from itertools import combinations
from math import sqrt
from os import getcwd, listdir, system
from pickle import dump, load
from sys import exit

from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imwrite
from face_recognition import face_landmarks, face_locations
from tqdm import tqdm


def euclidean_distance(point1, point2):
    """
    Calculate the euclidean distance between two points.
    """
    return sqrt((((point1[0] - point2[0])**2)+((point1[1] - point2[1])**2)))


def get_face_landmarks(path):
    """
    Get 72 key landmarks on faces.
    Return a dictionary with landmark and coordinates of points.
    """
    image = imread(path)
    try:
        landmarks = face_landmarks(image)[0]
    except IndexError:
        landmarks = {}
    return landmarks


def get_face_locations(path):
    """
    Given an image path, return locations of faces in the image.
    """
    image = imread(path)
    return face_locations(image)


def store_image_paths():
    """
    Get all paths of images to be used.
    Return a dictionary of each unique face and its images.
    {'face': [path1, path2...]}
    """
    paths = {
        j: [('data/faces/' + j + '/' + k)
            for k in listdir('data/faces/' + j) if k.endswith('.jpg')]
        for j in tqdm(
            [i for i in listdir('data/faces') if i.startswith('Face')])
    }
    dump(paths, open('data/paths.pkl', 'wb'))


def store_image_data():
    """
    Create and store pickle of image data.
    Each face has an image_url, location, missing attributes and landmarks.
    {'Face': {'missing': [], 'images': [{}]}, ...}
    """
    data = {}
    face_paths = load(open('data/paths.pkl', 'rb'))
    system('rm -rf temp && mkdir temp')
    for face in tqdm(face_paths):
        paths = face_paths[face]
        images = []
        for path in paths:
            image_object = {
                'image_url': '',
                'location': [],
                'missing': [],
                'landmarks': {}
            }
            image_object['image_url'] = path
            image = imread(path)
            gray_image = cvtColor(image, COLOR_BGR2GRAY)
            face_loci = get_face_locations(path)
            if len(face_loci) == 0:
                image_object['missing'].extend(['location', 'landmarks'])
            else:
                top, right, bottom, left = face_loci[0]
                image_object['location'] = face_loci
                face_image = image[top:bottom, left:right]
                imwrite('temp/current.png', face_image)
                landmarks = get_face_landmarks('temp/current.png')
                if not landmarks:
                    image_object['missing'].append('landmarks')
                else:
                    image_object['landmarks'].update(landmarks)
                    system('rm temp/current.png')
                images.append(image_object)
        data.update({face: images})
    dump(data, open('data/image_data.pkl', 'wb'))
    system('rm -rf temp')


def store_distances():
    """
    Create and store a pickle of distances between landmarks.
    {Face: [[()...], ...], ...}
    """
    landmarks_ = sorted(
        ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
         'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'])
    combinations_ = sorted(list(combinations(landmarks_, 2)))
    data = load(open('data/image_data.pkl', 'rb'))
    distances = {}
    for face in tqdm(data):
        faces = data[face]
        face_distances = []
        for i in faces:
            if i['missing']:
                pass
            else:
                i_landmarks = i['landmarks']
                vectors = [
                    tuple([i_landmarks[pair[0]], i_landmarks[pair[1]]])
                    for pair in combinations_]
                i_distances = []
                for vector_pairs in vectors:
                    for j in vector_pairs[0]:
                        for k in vector_pairs[1]:
                            i_distances.append(euclidean_distance(j, k))
                i_ratios = [j/i_distances[0] for j in i_distances]
                face_distances.append(i_ratios)
            distances.update({face: face_distances})
    dump(distances, open('data/distances.pkl', 'wb'))


def store_face_locations():
    """
    Store location of faces on an image.
    {'Face': [{'image_url': '', 'location': ()}, ...], ...}
    """
    data = load(open('data/image_data.pkl', 'rb'))
    faces_locations = {}
    for face in tqdm(data):
        faces = data[face]
        face_locations = []
        for i in faces:
            if i['location']:
                face_locations.append(
                    {'image_url': i['image_url'],
                     'location': i['location']})
            else:
                pass
        faces_locations.update({face: face_locations})
    dump(faces_locations, open('data/face_locations.pkl', 'wb'))


def store_paths_with_landmarks():
    """
    Get the paths with landmark data.
    """
    data = load(open('data/image_data.pkl', 'rb'))
    paths = {}
    for face in tqdm(data):
        faces = data[face]
        face_paths = []
        for i in faces:
            if not i['landmarks']:
                pass
            face_paths.append(i['image_url'])
        if not face_paths:
            pass
        paths.update({face: face_paths})
    dump(paths, open('data/paths_with_landmarks.pkl', 'wb'))


def store_cropped_faces():
    """
    Crop out and store faces.
    """
    system('mkdir data/cropped')
    paths = load(open('data/face_locations.pkl', 'rb'))
    for face in tqdm(paths):
        face_data = paths[face]
        i_count = 0
        system('mkdir data/cropped/' + face)
        for i in face_data:
            image = imread(i['image_url'])
            top, right, bottom, left = i['location'][0]
            face_image = image[top:bottom, left:right]
            new_path = 'data/cropped/' + face + '/' + str(i_count) + '.jpg'
            i_count = i_count + 1
            imwrite(new_path, face_image)
    return True


print('\nStoring image paths.')
store_image_paths()
print('\nStoring image data.')
store_image_data()
print('\nStoring distances.')
store_distances()
print('\nStoring face locations.')
store_face_locations()
print('\nStoring paths with landmarks.')
store_paths_with_landmarks()
print('\nStoring cropped faces.')
store_cropped_faces()
