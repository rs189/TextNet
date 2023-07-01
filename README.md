# 📙 TextNet

TextNet is a utility that uses a neural network to generate text based on the provided input. It allows for the loading of different datasets and offers the functionality to save and load pre-trained models. The model has been designed to be very lightweight and easy to run.

## Requirements

- Python 3.10

Additionally you have to install dependencies for Python from the requirements.txt file, you can do so by running.

```
pip install -r requirements.txt
```

## Model

The TextNet model consists of 3 layers: embedding, lstm, and dense. These layers are wrapped inside a sequential class. TextNet can be trained on any dataset to predict an output based on the given input. It is particularly designed for natural text and conversation, but it could also be used for other purposes.

![Screenshot](https://github.com/rs189/TextNet/blob/main/Graph.png?raw=true)

For the visualisation of the neural network structure we assume that the vocabulary size and the maximum length is 100, in the actual implementation however that is decided by the Tokenizer based on the dataset provided. The first, Embedding layer has a vocab_size x 100 size with input length being determined by the maximum length. The second, LSTM layer has a fixed size of 100. Lastly the third, Dense layer has size of vocab_size and SoftMax activation. 

## Usage

To use TextNet simply launch either the run.bat file or manually run the app.py python file.

If you want to load a specific dataset to train on, select the dataset from the dataset dropdown menu.

If you wish to load an already trained model, you first must select the dataset from the dataset dropdown menu that the model has been trained on, after which you must select the model you want to load and press the "Train/Load" button.

In order to generate a prediction simply put your input text in the input text field, select your temperature value or leave it at the default value and press the "Predict" button.

## Screenshots

![Screenshot](https://github.com/rs189/TextNet/blob/main/Thumbnail.png?raw=true)
