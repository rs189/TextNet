import json
import time
import numpy as np
from tensorflow import config, keras, distribute
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app import textnet_pipe

# Check if a GPU is available
gpus = config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            config.experimental.set_memory_growth(gpu, True)
        logical_gpus = config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU device found. Using CPU.")

class TextNetDataset:
    def __init__(self, path):
        self.path = path

        self.inputs = []
        self.outputs = []

    def load(self):
        with open(self.path) as f:
            self.data = json.load(f)

        self.inputs = [item['input'] for item in self.data['dataset']]
        self.outputs = [item['output'] for item in self.data['dataset']]

textnet_dataset = None
#textnet_dataset.load()

class TextNetTokenizer:
    def __init__(self, dataset):
        self.dataset = dataset

        self.texts = []
        self.input_sequences = []
        self.output_sequences = []

        self.vocab_size = None
        self.max_length = None

        self.tokenizer = None

    def tokenize(self):
        for i in range(len(self.dataset.outputs)):
            self.texts.append("Question: " + self.dataset.inputs[i] + " Answer: " + self.dataset.outputs[i] + " endofstring")

        # Tokenize the texts
        self.tokenizer = Tokenizer(oov_token='<OOV>', 
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(self.texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(self.texts)

        # Generate input-output pairs
        for sequence in sequences:
            for i in range(1, len(sequence)):
                self.input_sequences.append(sequence[:i])
                self.output_sequences.append(sequence[i])

        # Pad sequences to the same length
        self.max_length = max(len(seq) for seq in self.input_sequences)
        self.input_sequences = pad_sequences(self.input_sequences, maxlen=self.max_length, padding='pre')
        self.output_sequences = np.array(self.output_sequences)

textnet_tokenizer = None
#textnet_tokenizer.tokenize()

class TextNetTrainer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.model = None

    def train(self, epochs=100, dataset=None, load=None):
        model = keras.Sequential([
            keras.layers.Embedding(self.tokenizer.vocab_size, 100, input_length=self.tokenizer.max_length),
            keras.layers.LSTM(100),
            keras.layers.Dense(self.tokenizer.vocab_size, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        # Create the distributed model
        self.model = self.distribute_model(model)

        # Load the model
        if load is not None and load != 'None':
            self.model = keras.models.load_model('models/' + load)
            return

        # Train the model
        self.model.fit(self.tokenizer.input_sequences, self.tokenizer.output_sequences, epochs=epochs, verbose=2)

        if dataset is not None:
            # Remove json extension
            dataset = dataset[:-5]
            self.model.save('models/' + dataset + '.h5')

    # Define a function to distribute the model across multiple GPUs
    def distribute_model(self, model):
        strategy = distribute.MirroredStrategy()
        with strategy.scope():
            self.model = model
        return self.model
    
textnet_trainer = None
#textnet_trainer.train()

class TextNetPredictor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict_next_word(self, input_text, temperature=1.0):
        input_sequence = self.tokenizer.tokenizer.texts_to_sequences([input_text])[0]
        input_sequence = pad_sequences([input_sequence], maxlen=self.tokenizer.max_length, padding='pre')

        predicted_probs = self.model.predict(input_sequence, verbose=0)[0]
        predicted_probs = np.log(predicted_probs) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        predicted_index = np.random.choice(range(self.tokenizer.vocab_size), p=predicted_probs)
        predicted_word = self.tokenizer.tokenizer.index_word.get(predicted_index, '')

        # if predicted_word == '<OOV>':
        #     # Use attention weights to copy the word from input sequence
        #     attention_weights = attention_weights[0, -1, :]
        #     max_attention_index = np.argmax(attention_weights)
        #     predicted_word = self.tokenizer.tokenizer.index_word.get(input_sequence[0, max_attention_index], '')

        return predicted_word

textnet_predictor = None

# Test the model
while True:
    if textnet_pipe.train_event.is_set():
        textnet_pipe.train_event.clear()

        if not textnet_pipe.dataset_queue.empty():
            dataset = textnet_pipe.dataset_queue.get()
            textnet_pipe.dataset_queue.queue.clear()

        print('Training TextNet on dataset ' + dataset)

        textnet_dataset = TextNetDataset('datasets/' + dataset)
        textnet_dataset.load()

        textnet_tokenizer = TextNetTokenizer(textnet_dataset)
        textnet_tokenizer.tokenize()

        epochs = textnet_pipe.epochs_queue.get()
        textnet_pipe.epochs_queue.queue.clear()

        model = "None"
        if not textnet_pipe.model_queue.empty():
            model = textnet_pipe.model_queue.get()
            textnet_pipe.model_queue.queue.clear()

        textnet_trainer = TextNetTrainer(textnet_tokenizer)
        textnet_trainer.train(epochs=epochs, dataset=dataset, load=model)

        textnet_predictor = TextNetPredictor(textnet_tokenizer, textnet_trainer.model)

    if textnet_pipe.predict_event.is_set():
        textnet_pipe.predict_event.clear()

        if not textnet_pipe.input_queue.empty():
            input_text = 'Question: ' + textnet_pipe.input_queue.get() + ' Answer:'
            textnet_pipe.input_queue.queue.clear()
            textnet_pipe.output_queue.queue.clear()

            temperature = textnet_pipe.temperature_queue.get()
            textnet_pipe.temperature_queue.queue.clear()

            predicted_word = ''
            predicted_output = ''

            while predicted_word != 'endofstring':
                predicted_word = textnet_predictor.predict_next_word(input_text, temperature)
                input_text += ' ' + predicted_word
                if predicted_word == 'endofstring':
                    predicted_output = predicted_output.strip()
                else:
                    textnet_pipe.output_queue.put(predicted_word + ' ')
                    predicted_output += ' ' + predicted_word

            print('Output: ' + predicted_output)

    time.sleep(0.1)
    #input_text = 'Question: ' + input('Enter input question: ') + ' Answer:'
    #print(textnet_predictor.predict(input_text))