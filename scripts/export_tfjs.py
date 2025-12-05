import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model('models/final_model.h5')
tf.saved_model.save(model, 'saved_model')
tfjs.converters.convert_tf_saved_model('saved_model', 'web_model')
