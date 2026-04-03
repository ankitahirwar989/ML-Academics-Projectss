import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import traceback
import tensorflow as tf
print('TF version:', tf.__version__)

try:
    m = tf.keras.models.load_model('fruit_veg_classifier_final.keras')
    print('Model loaded OK!')
    print('Output shape:', m.output_shape)
except Exception as e:
    print('ERROR:')
    traceback.print_exc()
