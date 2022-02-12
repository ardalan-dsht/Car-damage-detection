from flask import Flask
from tensorflow import keras
import tensorflow as tf
from flask import request
from keras.preprocessing.image import load_img , img_to_array
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np

'''
The webapp uses keras pre-trained model "VGG16" and my trained model called
"detective". The entire training process is available in
car_damage_detection.ipynb which is stored in this repository.
HTTP request is also stored in this repository as a json file.
For further info read the README.md file or contact me at ardalan.dsht@google.com
'''

#Loading our pre-trained models.
detective = keras.models.load_model("detective.h5") #Loading our model.
base_model = VGG16(weights='imagenet') #Downloading base model from the net.

#Creating flask server object.
app = Flask(__name__)

#Pre-processing input image.
def prepare_image(image_path):
    image = load_img(image_path, target_size=(224, 224)) #Creates a PIL object.
    image = img_to_array(image) #Converting PIL image to a numpy array.
    image = np.expand_dims(image, axis=0) #Expanding the x-axis by one dimention.
    image = preprocess_input(image) #Encoding array.(Only works with imagenet)
    return image

#Classifying cars as damaged or not damaged.
@app.route("/damage_detection",methods=["POST"])
def damage_detection(detective,base_model):

    pic = request.get_data(["pic"]) #Storing input photo in variable.
    pic = prepare_image(pic) #Pre-processing image for models.

    #These classes represent cars in imagenet dataset.
    cars = [656, 627, 817, 511, 468, 751, 705, 757, 717, 734, 654, 675, 864, 609, 436]
    if np.argmax(base_model.predict(pic)) in cars: #Is the uploaded photo a car?
        if detective.predict(pic) <= 0.5: #Is the car damaged?
            return "Your car is damaged!"
        else :
            return "Your car is NOT damaged!"
    else:
        return "This is not a car!Please re-upload your photo"


#Start
if __name__ == "__main__":
    app.run()
