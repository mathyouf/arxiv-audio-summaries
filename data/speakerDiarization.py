# https://github.com/pyannote/pyannote-audio
# https://huggingface.co/pyannote/speaker-diarization
# conda activate pyannote
import os
from pyannote.audio import Pipeline

# MP3 to WAV: ffmpeg -i input.mp3 output.wav 

def diarize_mp3(audio_file="./data/audio/part1.mp3", output_dir="./data/audio"):

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1") ## Broken here, model not found

    # Check if audio file isn't WAV, convert to WAV
    if audio_file[-3:] != "wav":
        audio_file = audio_file[:-3] + "wav"
        os.system("ffmpeg -i " + audio_file + " " + audio_file)

    # Check if audio files are greater than 200MB, split into 200MB chunks
    if os.path.getsize(audio_file) > 200000000:
        os.system("ffmpeg -i " + audio_file + " -f segment -segment_time 1000 -c copy " + output_dir + "/%03d.wav")

    # apply the pipeline to an audio file
    diarization = pipeline(audio_file, num_speakers=2)

    # save diarization as RTTM file
    output_file = os.path.join(output_dir, "diarization.rttm")

    # dump the diarization output to disk using RTTM format
    with open(output_file, "w") as rttm:
        diarization.write_rttm(rttm)