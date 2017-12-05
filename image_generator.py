#-*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def loadImage():
    image = load_img('image.jpg')
    image_array = img_to_array(image) 
    image_array = image_array.reshape((1,) + image_array.shape)
    return image_array

def imageGenerator(image):
    datagen = ImageDataGenerator(rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                fill_mode='nearest')
    counter = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir='images', save_prefix='1', save_format='jpg'):
        counter += 1
        if counter > 17: 
            break      

if __name__ == "__main__":
    image = loadImage()
    imageGenerator(image)