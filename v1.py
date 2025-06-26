import ollama
import os
from dotenv import load_dotenv
# import required libraries
import sounddevice as sd
# from scipy.io.wavfile import write
import wavio as wv
from groq import Groq
import json
load_dotenv()

# Sampling frequency
freq = 44100

# Recording duration
duration = 5

# Start recorder with the given values of 
# duration and sample frequency
recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2)



# Record audio for the given number of seconds
sd.wait()

# Convert the NumPy array to audio file
wv.write("recording1.wav", recording, freq, sampwidth=2)

client = Groq()

filename = os.path.dirname(__file__) + "/recording1.wav"

with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()),
      model="whisper-large-v3",
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
