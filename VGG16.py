import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def create( input_shape ):
    return VGG16(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=input_shape
    )

def predict( vgg_model, frame ):
    return np.squeeze(
        vgg_model.predict(np.expand_dims( frame, axis=0 ), verbose=False ) # shape (1 2 2 512)
    )

def zVectorize( vggoutput ):
    return np.reshape(
        np.transpose(
            np.squeeze( vggoutput )
        ), -1
    )

