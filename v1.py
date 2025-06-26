import ollama
import os
from dotenv import load_dotenv
import sounddevice as sd
# from scipy.io.wavfile import write
import wavio as wv
from groq import Groq
import json
import numpy as np
import keyboard
import pyperclip

load_dotenv()

# Paramètres audio
freq = 44100
channels = 1
chunk_duration = 0.1  # Durée des chunks (seconds)
chunk_size = int(freq * chunk_duration)

print("Press and hold the spacebar to start recording...")

all_chunks = []

keyboard.wait('space')

# Start stream
stream = sd.InputStream(samplerate=freq, channels=channels)
stream.start()

# Continue recording while space is held
while keyboard.is_pressed('space'):
  audio_chunk, _ = stream.read(chunk_size)
  all_chunks.append(audio_chunk)

stream.stop()
print("* Done recording")

# Concatenation des extraits audio avec numpy
recording = np.concatenate(all_chunks, axis=0)

wv.write("recording1.wav", recording, freq, sampwidth=2)

client = Groq()

filename = os.path.dirname(__file__) + "/recording1.wav"

with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()),
      model="whisper-large-v3-turbo",
            language="en",
      response_format="verbose_json",
    )
    print(transcription.text)
      

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
      {
        "role": "system",
        "content": "Create a professional looking mail for the user. Just anwsers the mail in JSON with { mail: \"This is a mail\" }"
      },
      {
        "role": "user",
        "content": f"{transcription.text}\n"
      }
    ],
    temperature=0.3,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
)




result = completion.choices[0].message.content
json_result = json.loads(result)
print(json_result["mail"])
pyperclip.copy(json_result["mail"])
