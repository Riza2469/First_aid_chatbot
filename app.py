import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# model_path = os.path.join('project_folder', 'chatbot_model.h5')
# model = load_model(model_path)

model = load_model('D:\projects\Custom_Chatbot\project_folder\chatbot_model.h5')
words = pickle.load(open('D:\projects\Custom_Chatbot\project_folder\words.pkl', 'rb'))
classes = pickle.load(open('D:\projects\Custom_Chatbot\project_folder\classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\projects\Custom_Chatbot\project_folder\data\intents.json').read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        # Fallback when no intent is recognized
        return {"response": "Sorry! I do not know the answer."}
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if 'tag' in i and i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def index():
    # Serve the HTML file
    return app.send_static_file('index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    try:
        data = request.get_json()
        user_message = data.get('userMessage', '')

        # Use your chatbot functions to process the user's message
        ints = predict_class(user_message)
        res = get_response(ints, intents)
        print(res)

        # Return the chatbot's response as JSON
        return jsonify({'chatbotResponse': res})

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("GO! Bot is running!")
    # Run the Flask app
    app.run(debug=True)

# import numpy as np
# from tensorflow.keras.models import load_model
# import pickle
# from nltk.stem import WordNetLemmatizer
# import nltk
# import json

# nltk.download('punkt')
# nltk.download('wordnet')

# def clean_up_sentence(sentence):
#     lemmatizer = WordNetLemmatizer()
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0]*len(words)
#     for s in sentence_words:
#         if s in words:
#             bag[words.index(s)] = 1
#             if show_details:
#                 print(f"found in bag: {s}")
#     return(np.array(bag))

# def predict_class(sentence, model, words, classes):
#     p = bag_of_words(sentence, words, show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

# def get_response(intents_list, intents_json):
#     if intents_list:
#         tag = intents_list[0]["intent"]
#         list_of_intents = intents_json["intents"]
#         for i in list_of_intents:
#             if i.get("tag") == tag:
#                 result = {"response": np.random.choice(i["responses"])}
#                 break
#         else:
#             result = {"response": "Sorry, I didn't understand that."}
#     else:
#         result = {"response": "Sorry, I didn't understand that."}
#     return result


# def main():
#     model_path = 'D:\projects\Custom_Chatbot\project_folder\chatbot_model.h5'
#     model = load_model(model_path)

#     words = pickle.load(open('D:\projects\Custom_Chatbot\project_folder\words.pkl', 'rb'))
#     classes = pickle.load(open('D:\projects\Custom_Chatbot\project_folder\classes.pkl', 'rb'))
#     intents = json.loads(open('D:\projects\Custom_Chatbot\project_folder\chatbot\intents.json').read())

#     print("Chatbot Tester")
#     print("Type 'exit' to end the conversation.\n")

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break

#         intents_list = predict_class(user_input, model, words, classes)
#         chatbot_response = get_response(intents_list, intents)["response"]

#         print(f"Chatbot: {chatbot_response}\n")

# if __name__ == "__main__":
#     main()
