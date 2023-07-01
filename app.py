import tkinter
import customtkinter

from threading import Thread, Event
from queue import Queue
import time
import os

import textnet as tn

class TextNetPipe:
    def __init__(self):
        self.dataset_queue = Queue()
        self.model_queue = Queue()
        self.epochs_queue = Queue()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.temperature_queue = Queue()

        self.train_event = Event()
        self.predict_event = Event()

        # A thread to synchronise the output from the textnet thread to the ui
        output_text = None
        self.output_thread = Thread(target=self.output_callback)
        self.output_thread.start()

    def output_callback(self):
        while True:
            if not self.output_queue.empty():
                output = self.output_queue.get()
                self.output_text.insert(tkinter.INSERT, output)
            time.sleep(0.1)

textnet_pipe = TextNetPipe()

class TextNetUI:
    def __init__(self):
        pass

    def run(self):
        customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        app = customtkinter.CTk()  # create CTk window like you do with the Tk window
        app.geometry("800x600")
        app.title("TextNet")
        app.resizable(False, False)

        self.draw_training_frame(app)
        self.draw_dataset_frame(app)

        self.draw_prediction_frame(app)

        app.mainloop()

    def list_datasets(self):
        cwd = os.getcwd()
        datasets = os.listdir(cwd + "/datasets")
        all_datasets = []
        for dataset in datasets:
            if dataset.endswith(".json"):
                all_datasets.append(dataset)
        return all_datasets
    
    def list_models(self):
        cwd = os.getcwd()
        models = os.listdir(cwd + "/models")
        all_models = []
        all_models.append("None")
        for model in models:
            if model.endswith(".h5"):
                all_models.append(model)
        return all_models

    def draw_dataset_frame(self, app):
        # Training frame and its widgets on the left side, margin 10 pixels on each side of the frame
        dataset_frame = customtkinter.CTkFrame(master=app, width=380, height=280)
        dataset_frame.place(relx=0.25, rely=0.25, anchor=tkinter.CENTER)

        dataset_frame_title = customtkinter.CTkLabel(master=dataset_frame, text="Dataset", font=("Arial", 20))
        dataset_frame_title.place(relx=0.5, rely=0.015, anchor=tkinter.N)

        dataset_combobox_label = customtkinter.CTkLabel(master=dataset_frame, text="Dataset", anchor=tkinter.W, justify=tkinter.LEFT)
        dataset_combobox_label.place(relx=0.035, rely=0.15, anchor=tkinter.W)

        def dataset_combobox_callback(data):
            textnet_pipe.dataset_queue.queue.clear()
            textnet_pipe.dataset_queue.put(data)

        datasets = self.list_datasets()   

        dataset_combobox = customtkinter.CTkOptionMenu(master=dataset_frame, command=dataset_combobox_callback,
                                                        width=355, state="readonly")
        dataset_combobox.configure(values=datasets)
        dataset_combobox.set(datasets[0])
        textnet_pipe.dataset_queue.put(datasets[0]) 
        dataset_combobox.place(relx=0.035, rely=0.25, anchor=tkinter.W)

    def draw_training_frame(self, app):
        # Training frame and its widgets on the left side, margin 10 pixels on each side of the frame
        training_frame = customtkinter.CTkFrame(master=app, width=380, height=280)
        training_frame.place(relx=0.25, rely=0.75, anchor=tkinter.CENTER)

        training_frame_title = customtkinter.CTkLabel(master=training_frame, text="Training", font=("Arial", 20))
        training_frame_title.place(relx=0.5, rely=0.015, anchor=tkinter.N)

        model_label = customtkinter.CTkLabel(master=training_frame, text="Model", anchor=tkinter.W, justify=tkinter.LEFT)
        model_label.place(relx=0.035, rely=0.15, anchor=tkinter.W)

        # model_entry = customtkinter.CTkEntry(master=training_frame, width=355, textvariable=tkinter.StringVar(value="100"))
        # model_entry.place(relx=0.035, rely=0.25, anchor=tkinter.W)
        def model_combobox_callback(data):
            textnet_pipe.model_queue.queue.clear()
            textnet_pipe.model_queue.put(data)

        models = self.list_models()   

        model_combobox = customtkinter.CTkOptionMenu(master=training_frame, command=model_combobox_callback,
                                                        width=355, state="readonly")
        model_combobox.configure(values=models)
        model_combobox.set(models[0])
        textnet_pipe.model_queue.put(models[0]) 
        model_combobox.place(relx=0.035, rely=0.25, anchor=tkinter.W)


        epochs_label = customtkinter.CTkLabel(master=training_frame, text="Epochs", anchor=tkinter.W, justify=tkinter.LEFT)
        epochs_label.place(relx=0.035, rely=0.65, anchor=tkinter.W)

        epochs_entry = customtkinter.CTkEntry(master=training_frame, width=355, textvariable=tkinter.StringVar(value="100"))
        epochs_entry.place(relx=0.035, rely=0.75, anchor=tkinter.W)

        def train_button_callback():
            textnet_pipe.epochs_queue.queue.clear()
            textnet_pipe.epochs_queue.put(int(epochs_entry.get()))

            textnet_pipe.train_event.clear()
            textnet_pipe.train_event.set()

        train_button = customtkinter.CTkButton(master=training_frame, text="Train/Load", command=train_button_callback, width=355)
        train_button.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)

    def draw_prediction_frame(self, app):
        prediction_frame = customtkinter.CTkFrame(master=app, width=380, height=580)
        prediction_frame.place(relx=0.75, rely=0.5, anchor=tkinter.CENTER)

        prediction_frame_title = customtkinter.CTkLabel(master=prediction_frame, text="Prediction", font=("Arial", 20))
        prediction_frame_title.place(relx=0.5, rely=0.015, anchor=tkinter.N)

        input_label = customtkinter.CTkLabel(master=prediction_frame, text="Input", anchor=tkinter.W, justify=tkinter.LEFT)
        input_label.place(relx=0.035, rely=0.07, anchor=tkinter.W)

        input_text = customtkinter.CTkEntry(master=prediction_frame, width=355)
        input_text.configure(textvariable=tkinter.StringVar(value="Hello!"))
        input_text.place(relx=0.5, rely=0.096, anchor=tkinter.N)

        output_label = customtkinter.CTkLabel(master=prediction_frame, text="Output", anchor=tkinter.W, justify=tkinter.LEFT)
        output_label.place(relx=0.035, rely=0.175, anchor=tkinter.W)

        output_text = customtkinter.CTkTextbox(master=prediction_frame, width=355)
        output_text.place(relx=0.5, rely=0.2, anchor=tkinter.N)
        textnet_pipe.output_text = output_text

        def predict_button_callback():
            output_text.delete("0.0", "end")

            input = input_text.get()
            input.strip()

            textnet_pipe.input_queue.queue.clear()
            textnet_pipe.input_queue.put(input)

            temperature = temperature_slider.get()
            textnet_pipe.temperature_queue.queue.clear()
            textnet_pipe.temperature_queue.put(temperature)

            textnet_pipe.predict_event.clear()
            textnet_pipe.predict_event.set()

        temperature_label = customtkinter.CTkLabel(master=prediction_frame, text="Temperature", anchor=tkinter.W, justify=tkinter.LEFT)
        temperature_label.place(relx=0.035, rely=0.85, anchor=tkinter.W)

        temperature_slider = customtkinter.CTkSlider(master=prediction_frame, from_=0.01, to=1.0, number_of_steps=1000, width=355, variable=tkinter.IntVar(value=1))
        temperature_slider.place(relx=0.5, rely=0.8775, anchor=tkinter.N)

        predict_button = customtkinter.CTkButton(master=prediction_frame, text="Predict", command=predict_button_callback, width=355)
        predict_button.place(relx=0.5, rely=0.95, anchor=tkinter.CENTER)

textnet_ui = TextNetUI()
textnet_ui_thread = Thread(target=textnet_ui.run)
textnet_ui_thread.start()