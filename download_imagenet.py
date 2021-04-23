from keras.applications import VGG16
'''
This will download VGG16 Model locally.
This script must be run before running rest of the scripts.
'''
image_model = VGG16(include_top=True, weights='imagenet')  
image_model.summary()