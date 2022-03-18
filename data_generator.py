import os
import cv2
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_load(path):
  x_train = sorted(glob(os.path.join(path,"train","image","*.jpg")))
  y_train = sorted(glob(os.path.join(path,"train","mask","*.jpg")))

  x_test = sorted(glob(os.path.join(path,"test","image","*.jpg")))
  y_test = sorted(glob(os.path.join(path,"test","mask","*.jpg")))

  return (x_train,y_train),(x_test, y_test)



  if __name__ == "__main__":
    "seeding"
    np.random.seed(42)

    " data load "
    data_path = r"data_path"
    
    (x_train,y_train),(x_test, y_test) =  data_load(data_path)
    print(f"train:{len(x_train)}-{len(y_train)}")
    print(f"test:{len(x_test)}-{len(x_test)}")


    "data _generator"

    from keras_preprocessing import image
datagen = ImageDataGenerator(
        rotation_range=40,
        featurewise_center=False,
        horizontal_flip=False,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest')


train_data_aug = datagen.flow_from_directory('/content/mask_folder', target_size=(256, 256),
                                             color_mode='rgb', 
                                             classes=None,
                                             class_mode='input',
                                             batch_size=32, 
                                             shuffle=False, 
                                             seed=None, 
                                             save_to_dir='/content/mask_aug',
                                             save_prefix='eye', 
                                             save_format='jpg', 
                                             follow_links=False,
                                             subset=None,
                                             interpolation='nearest')


"data enumerate"
for i , batch in enumerate(train_data_aug):
  print(batch)