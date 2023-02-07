# load the image and resize it to input size of the model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import keras
from PIL import Image, ImageOps

model = keras.models.load_model('cats_and_dogs')

cat_image = load_img('PetImages\Cat\3.jpg', target_size=(
    150, 150), color_mode='grayscale')
cat_image1 = load_img('PetImages\Cat\49.jpg', target_size=(
    150, 150), color_mode='grayscale')
cat_image2 = load_img('PetImages\Cat\30.jpg', target_size=(
    150, 150), color_mode='grayscale')

dog_image = load_img('PetImages\Dog\3.jpg', target_size=(
    150, 150), color_mode='grayscale')
dog_image1 = load_img('PetImages\Cat\6.jpg', target_size=(
    150, 150), color_mode='grayscale')
dog_image2 = load_img('PetImages\Dog\1.jpg', target_size=(
    150, 150), color_mode='grayscale')

# convert the image to array
cat_image_array = img_to_array(cat_image)
cat_image_array1 = img_to_array(cat_image1)
cat_image_array2 = img_to_array(cat_image2)

dog_image_array = img_to_array(dog_image)
dog_image_array1 = img_to_array(dog_image1)
dog_image_array2 = img_to_array(dog_image2)


image_array = [cat_image_array, cat_image_array1, cat_image_array2,
               dog_image_array, dog_image_array1, dog_image_array2]
# pre-process the image for the model
for image in image_array:
    # image = image/255
    image = np.expand_dims(image, axis=0)

# for image in image_array:
    # make a prediction
    prediction = model.predict(image)

    if prediction > 0.5:
        print('Cat')
    else:
        print('Dog')
# image = image/255
# image = np.expand_dims(image, axis=0)


# # make a prediction
# prediction = cats_.predict(image)

# if prediction > 0.5:
#     print('Cat')
# else:
#     print('Dog')
