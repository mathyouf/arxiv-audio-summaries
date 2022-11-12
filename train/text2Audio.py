## Tortoise TTS Replicate - https://replicate.com/afiaka87/tortoise-tts
import replicate
import os
from dotenv import load_dotenv

def convertToMp3(audio_file):
    # Convert to mp3 using ffmpeg
    output_filename = audio_file[:-4] + ".mp3"
    # CHeck if file already exists
    if os.path.exists(output_filename):
        return output_filename
    os.system("ffmpeg -i " + audio_file + " " + output_filename)
    return output_filename

def trainTortoiseTTS(audio_file):
    # Check if audio file is mp3, if not, convert to mp3
    if audio_file[-4:] != ".mp3":
        audio_file = convertToMp3(audio_file)
    # Get absolute path of audio file
    audio_file = os.path.abspath(audio_file)
    # Convert to data:: URL scheme
    audio_file = "data::" + audio_file
    # Load Replicate Token from .env
    load_dotenv()
    token = os.getenv("REPLICATE_TOKEN")
    model = replicate.models.get("afiaka87/tortoise-tts")
    version = model.versions.get("e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71")
    output = version.predict(custom_voice=audio_file, text="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    # Save output audio
    audio_file_title = audio_file[:-4] + "_tortoise.mp3"
    output.save(audio_file_title)

trainTortoiseTTS("./data/audio/episode1/out000.wav")