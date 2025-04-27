from wsgiref.util import request_uri

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import whisper
from glob import glob

#test
# Load the audio files
all_audio = glob('data/*.wav')

def audio_to_image(list_of_pathnames):
    for i in range(len(list_of_pathnames)):
        y, sr = librosa.load(list_of_pathnames[i])
        y_trimmed, silence = librosa.effects.trim(y, top_db=20)
        figure, ax = plt.subplots()

        ax.plot(y_trimmed)
        ax.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        plt.gca().set_frame_on(False)
        plt.savefig(f"output/waveform_image{i}.png", pad_inches = 0, bbox_inches='tight', transparent=True)
        plt.close()



def audio_to_text(list_of_pathnames):
    returned_words = []
    model = whisper.load_model("base")
    for i in range(len(list_of_pathnames)):
        result = model.transcribe(list_of_pathnames[i], word_timestamps= True)
        for segment in result["segments"]:
            words = segment.get("words", [])
            for word_data in words:
                word = word_data['word']
                start = word_data['start']
                end = word_data['end']
                returned_words.append(word)
        return returned_words


def audio_to_face(list_of_pathnames):
    words_to_draw = audio_to_text(list_of_pathnames)
    fig, ax = plt.subplots()
    x = np.linspace(-32, 32, 400)
    for word in words_to_draw:
        if "A" in word:
            y = (5 / 1024) * x ** 2
            y_straight = (np.full_like(x, 5))
            plt.plot(x, y)
            plt.plot(x, y_straight)
            plt.fill_between(x, y, y_straight, color='blue')
        elif "O" in word:
            plt.plot()
        elif "U" in word:
            plt.plot()
        elif ("E" or "I") in word:
            plt.plot()
        else:
            output_plot = plt.plot([-32,0], [32,0])
        ax.axis('off')
        ax.set_aspect('equal')
        plt.savefig(f"output/frame{i}.png", pad_inches=0, bbox_inches='tight', transparent=True)
        plt.close()




