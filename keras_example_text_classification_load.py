from keras.models import model_from_json
from keras.models import model_from_json
import pickle
from keras.preprocessing.sequence import pad_sequences

maxlen = 100
# loading
with open('./data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load json and create model
json_file = open('./data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./data/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#test using new data
sentence_test = ["this is good, yes, so good", "I love it so much", "I believe its ok, great, beautiful","hate it with all my soul, boring", "not a good experience, real bad", "so bad wrong no no please worst thing ever"]
xnew = tokenizer.texts_to_sequences(sentence_test)
xnew = pad_sequences(xnew, padding='post', maxlen=maxlen)

ynew = loaded_model.predict_classes(xnew)
print(ynew)

ynew = loaded_model.predict(xnew)
print(ynew)

