import re
import tensorflow as tf
import numpy as np 
import json
import requests

import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from flask_cors import CORS


data = open("data/data.md", encoding="utf8").read()


new_data = re.sub(r'[#]+', '', data)


corpus = new_data.lower().split("\n")
unused = [" ", "", "----------"]
corpus = [x for x in corpus if x not in unused]


# Initialize the Tokenizer class
tokenizer = Tokenizer()

# Generate the word index dictionary
tokenizer.fit_on_texts(corpus)

# Define the total words. You add 1 for the index `0` which is just the padding token.
total_words = len(tokenizer.word_index) + 1

# Initialize the sequences list
input_sequences = []

#load model
model = keras.models.load_model("my_model")

# Loop over every line
for line in corpus:

	# Tokenize the current line
	token_list = tokenizer.texts_to_sequences([line])[0]

	# Loop over the line several times to generate the subphrases
	for i in range(1, len(token_list)):
		
		# Generate the subphrase
		n_gram_sequence = token_list[:i+1]

		# Append the subphrase to the sequences list
		input_sequences.append(n_gram_sequence)

# Get the length of the longest line
max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)

# Pad all sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create inputs and label by splitting the last token in the subphrases
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# Convert the label into one-hot arrays
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


app = Flask(__name__)
cors = CORS(app)

@app.route("/predict",  methods=["POST"])
def predict():
    seed_text = request.json["predict"]
    next_words = 5
    for _ in range(next_words):
    # Convert the seed text to a token sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        data = json.dumps({"signature_name": "serving_default", "instances": token_list.tolist()})

        # Feed to the model and get the probabilities for each index
        # headers = {"content-type": "application/json"}
        # probabilities = requests.post('http://test-type-suggest.herokuapp.com/v1/models/saved_model:predict', data=data, headers=headers)

        predictions = model.predict(token_list)
        
        
        # Get the index with the highest probability
        predicted = np.argmax(predictions, axis=-1)[0]

        # Ignore if index is 0 because that is just the padding.
        if predicted != 0:
            
            # Look up the word associated with the index. 
            output_word = tokenizer.index_word[predicted]

            # Combine with the seed text
            seed_text += " " + output_word
    return jsonify({"prediction": seed_text})




@app.route("/")
def index():
    return "Welcome to the TensorFlow 2.0 Flask Server!"


if __name__ == "__main__":
    app.run(ssl_context='adhoc')






