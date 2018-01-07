from constants import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  gray_border = np.zeros((150, 150), np.uint8)
  gray_border[:,:] = 200
  gray_border[((150 / 2) - (SIZE_FACE/2)):((150/2)+(SIZE_FACE/2)), ((150/2)-(SIZE_FACE/2)):((150/2)+(SIZE_FACE/2))] = image
  image = gray_border

  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is we don't found an image
  if not len(faces) > 0:
    #print "No hay caras"
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size

  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  print image.shape
  return image


def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def flip_image(image):
    return cv2.flip(image, 1)

def data_to_image(data):
    #print data
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy() 
    data_image = format_image(data_image)
    return data_image

FILE_PATH = 'fer2013.csv'
data = pd.read_csv(FILE_PATH)

train_labels = []
train_images = []
test_labels = []
test_images = [] 
valid_labels = []
valid_images = [] 
index = 1
total = data.shape[0]
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    if image is not None:
        if row['Usage'] == 'Training':
          train_labels.append(emotion)
          train_images.append(image)
          #data augmentation could be done here theoretically
          #labels.append(emotion)
          #images.append(flip_image(image))
        elif row['Usage'] == 'PrivateTest':
          test_labels.append(emotion)
          test_images.append(image)
        elif row['Usage'] == 'PublicTest':
          valid_labels.append(emotion)
          valid_images.append(image)
    else:
        print "Error"
    index += 1
    print "Progreso: {}/{} {:.2f}%".format(index, total, index * 100.0 / total)
'''
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])
    if image is not None:

        labels.append(emotion)
        images.append(image)
        #labels.append(emotion)
        #images.append(flip_image(image))
    else:
        print "Error"
    index += 1
    print "Progreso: {}/{} {:.2f}%".format(index, total, index * 100.0 / total)
'''

train_images = (train_images - np.min(train_images)) / (np.max(train_images) - np.min(train_images))
test_images = (test_images - np.min(test_images)) / (np.max(test_images) - np.min(test_images))
valid_images = (valid_images- np.min(valid_images)) / (np.max(valid_images) - np.min(valid_images))
#test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
#sanity check
print "Total amount of train images: " + str(len(train_images))
print "Total amount of test images: " + str(len(test_images))
print "Total amount of train labels: " + str(len(test_images))
print "Total amount of test labels: " + str(len(test_labels))
#np.save('data_kike.npy', images)
#np.save('labels_kike.npy', labels)
np.save('fer_train_data_output.npy', train_images)
np.save('fer_test_data_output.npy', test_images)
np.save('fer_valid_data_output.npy', valid_images)
np.save('fer_train_labels_output.npy', train_labels)
np.save('fer_test_labels_output.npy', train_labels)
np.save('fer_valid_labels_output.npy', valid_labels)
'''
np.save('fer_train_data_output.npy', train_images)
np.save('fer_test_data_output.npy', test_images)
np.save('fer_train_labels_output.npy', train_labels)
np.save('fer_test_labels_output.npy', train_labels)
'''
