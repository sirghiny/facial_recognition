"""Scripts to prepare the data."""

from itertools import combinations
from math import sqrt
from os import getcwd, listdir, system
from pickle import dump, load

from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imwrite, resize
from face_recognition import face_landmarks, face_locations
from tqdm import tqdm

cwd = getcwd() + '/facial_recognition'


def euclidean_distance(point1, point2):
    """Calculate the euclidean distance between two points."""
    return sqrt((((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2)))


def get_face_landmarks(path):
    """Return a dictionary with landmarks(72) and coordinates of points."""
    image = imread(path)
    try:
        landmarks = face_landmarks(image)[0]
    except IndexError:
        landmarks = {}
    return landmarks


def get_face_locations(path):
    """Given an image path, return locations of faces in the image."""
    image = imread(path)
    return face_locations(image)


def store_image_paths():
    """
    Get all paths of images to be used.

    Return a dictionary of each unique face and its images.
    {'face': [path1, path2...]}
    """
    paths = {
        j: [(cwd + '/data/faces/' + j + '/' + k)
            for k in listdir(cwd + '/data/faces/' + j) if k.endswith('.jpg')]
        for j in tqdm(
            [i for i in listdir(cwd + '/data/faces') if i.startswith('Face')])
    }
    dump(paths, open(cwd + '/data/paths.pkl', 'wb'))


def store_image_data():
    """
    Create and store pickle of image data.

    Each face has an image_url, location, missing attributes and landmarks.
    {'Face': {'missing': [], 'images': [{}]}, ...}
    """
    data = {}
    face_paths = load(open(cwd + '/data/paths.pkl', 'rb'))
    system('rm -rf ' + cwd + '/' + 'temp && mkdir ' + cwd + '/' + 'temp')
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
            face_loci = get_face_locations(path)
            if len(face_loci) == 0:
                image_object['missing'].extend(['location', 'landmarks'])
            else:
                top, right, bottom, left = face_loci[0]
                image_object['location'] = face_loci
                face_image = image[top:bottom, left:right]
                imwrite(cwd + '/temp/current.png', face_image)
                landmarks = get_face_landmarks(cwd + '/temp/current.png')
                if not landmarks:
                    image_object['missing'].append('landmarks')
                else:
                    image_object['landmarks'].update(landmarks)
                    system('rm ' + cwd + '/' 'temp/current.png')
                images.append(image_object)
        data.update({face: images})
    dump(data, open(cwd + '/data/image_data.pkl', 'wb'))
    system('rm -rf ' + cwd + '/' + 'temp')


def store_distances():
    """
    Create and store a pickle of distances between landmarks.

    Only necessary if you wish to experiment with feed-forward networks.
    {Face: [[()...], ...], ...}
    """
    landmarks_ = sorted(
        ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
         'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'])
    combinations_ = sorted(list(combinations(landmarks_, 2)))
    data = load(open(cwd + '/data/image_data.pkl', 'rb'))
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
                i_ratios = [j / i_distances[0] for j in i_distances]
                face_distances.append(i_ratios)
            distances.update({face: face_distances})
    dump(distances, open(cwd + '/data/distances.pkl', 'wb'))


def store_face_locations():
    """
    Store location of faces on an image.

    {'Face': [{'image_url': '', 'location': ()}, ...], ...}
    """
    data = load(open(cwd + '/data/image_data.pkl', 'rb'))
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
    dump(faces_locations, open(cwd + '/data/face_locations.pkl', 'wb'))


def store_paths_with_landmarks():
    """Get the paths with landmark data."""
    data = load(open(cwd + '/data/image_data.pkl', 'rb'))
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
    dump(paths, open(cwd + '/data/paths_with_landmarks.pkl', 'wb'))


def adjust_crop(coordinates, image):
    """
    Make a cropped image square.

    To add height increase y2 or reduce y1.
    To add width increase x2 or reduce x1.
    Return cropped image adjusted to a square shape.
    """
    def get_change(diff):
        """Get upper and lower extensions to adjust crop."""
        if diff > 1:
            upper = int(diff / 2)
            lower = diff - upper
            return [upper, lower]
        return [diff, 0]

    y1, x2, y2, x1 = coordinates
    face_image = cvtColor(image[y1:y2, x1:x2], COLOR_BGR2GRAY)
    shape = face_image.shape
    if shape[0] != shape[1]:
        if shape[0] < shape[1]:
            h_diff = abs(shape[0] - shape[1])
            h_change = get_change(h_diff)
            y1, y2 = (y1 - h_change[1]), (y2 + h_change[0])
        else:
            w_diff = abs(shape[0] - shape[1])
            w_change = get_change(w_diff)
            x1, x2 = (x1 - w_change[1]), (x2 + w_change[0])
        adjusted_face_image = cvtColor(image[y1:y2, x1:x2], COLOR_BGR2GRAY)
        return adjusted_face_image
    return face_image


def store_cropped_faces():
    """
    Crop out and store faces.

    The commands are for Unix-based systems.
    The cropped images are of shape (128, 128).
    """
    system('rm -rf ' + cwd + '/' + 'data/cropped && mkdir ' +
           cwd + '/' + 'data/cropped')
    paths = load(open(cwd + '/data/face_locations.pkl', 'rb'))
    for face in tqdm(paths):
        face_data = paths[face]
        i_count = 0
        system('mkdir ' + cwd + '/' + 'data/cropped/' + face)
        for i in face_data:
            image = imread(i['image_url'])
            y1, x2, y2, x1 = i['location'][0]
            face_image = adjust_crop(i['location'][0], image)
            face_image = resize(face_image, (128, 128))
            new_path = cwd + '/data/cropped/' + \
                face + '/' + str(i_count) + '.jpg'
            i_count = i_count + 1
            imwrite(new_path, face_image)
    return True


def store_faces_with_many_images():
    """From the cropped images, store face names with more than one image."""
    faces = [i for i in listdir(cwd + '/data/cropped') if i.startswith('Face')]
    more_than_one = {}
    for face in tqdm(faces):
        face_paths = []
        images = [i for i in listdir(
            cwd + '/data/cropped/' + face) if i.endswith('.jpg')]
        if len(images) > 1:
            for i in images:
                face_paths.append(cwd + '/data/cropped/' + face + '/' + i)
            more_than_one.update({face: face_paths})
    dump(more_than_one, open(cwd + '/data/more_than_one.pkl', 'wb'))

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
print('\nStoring faces with more than one image.')
store_faces_with_many_images()
