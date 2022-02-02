from tensorflow.keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):
    emo_list =  ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    
    def __init__ (self, model_json_file, model_weights_file):
        with open (model_weights_file,"r",encoding="cp437", errors='ignore') as json_file:
            loaded_model_json=json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()
    
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.emo_list[np.argmax(self.preds)]
