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
    model = whisper.load_model("base")
    for i in range(len(list_of_pathnames)):
        result = model.transcribe(list_of_pathnames[i])
        print(result["text"])



def audio_to_face(list_of_pathnames):
    default_face = plt.plot([-32,0], [32,0])


model = whisper.load_model("base")
result = model.transcribe("data/test.wav", word_timestamps=True)
for segment in result["segments"]:
    words = segment.get("words", [])
    for word_data in words:
        word = word_data['word']
        start = word_data['start']
        end = word_data['end']
        print(word, start, end)