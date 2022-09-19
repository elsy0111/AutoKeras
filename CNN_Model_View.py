from keras.utils import plot_model
import keras
 
model = keras.models.load_model("image_regressor/best_model")
plot_model(model, to_file='model.png')