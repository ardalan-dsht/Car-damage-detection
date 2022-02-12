
# Car Damage Detector (CDD)

The CDD is a webapp which uses deep neural networks to classify cars as damaged or not damaged.
The task is done by two different DNNs.

### 1. Base Model
The base model task is to classify the input image as car or not a car.
It is trained on the "imagenet" dataset with VGG16 architecture by François Chollet (imported from keras).
### 2. Detective
Detective is the second classifier which performs the main task. Its a DNN based on the base model (VGG16) and trained on a 
dataset called "Damaged and Whole cars image dataset" made by Anuj shah.The dataset includes images of damaged cars as 
well as normal cars.It is well balanced and the only necessary preprocessing task shall be resizing the images.

ATTENTION!

The models dense layer is pretty small
do to lack of training data.Since theres not enough training data I skipped the evaluation part on the test set.
I understand that theres no good way to judge this models performance without test set data but  since 
this is for educational purposes it can be acceptable.(couldn't find a better dataset)

You can download the dataset from here :

https://www.kaggle.com/anujms/car-damage-detection

Please check notebook file for further info.

## APP
The app is based on flask framework and uses the pretrained models to perform the task.
It can be tested with postman with the sample "request.json" file.