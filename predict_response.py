import nltk
import random
from data_preprocessing import get_stem_words
import tensorflow

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
#list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tensorflow.keras.models.load_model('chatbot_model.h5')
def process_user_input(user_input):
    input1 = nltk.word_tokenize(user_input)
    input2 = get_stem_words(input1, ignore_words)
    input2 = sorted(list(set(input2)))
    bag = []
    bag_of_word = []

    for i in words:
        if(i in input2):
            bag_of_word.append(1)

        else:
            bag_of_word.append(0)

    bag.append(bag_of_word)
    return np.array(bag)

def chatbot_prediction(user_input):
    input1 = process_user_input(user_input)
    prediction = model.predict(input1)
    predict_class = np.argmax(prediction[0])
    return predict_class

def bot_response(user_input):
    prediction1 = chatbot_prediction(user_input)
    predicted_class = classes[prediction1]

    for i in intents['intents']:
        if(i['tag'] == predicted_class):
            bot_response1 = random.choice(i['responses'])
            return bot_response1
        
print('HI! HOW CAN I HELP YOU')

while True:
    user_input = input('type the mssg here : ')
    response = bot_response(user_input)
    print('bot_response', response)
