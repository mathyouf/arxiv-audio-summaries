# https://huggingface.co/pyannote/speaker-diarization
# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# https://github.com/pyannote/pyannote-audio
# conda activate pyannote
import os
from pyannote.audio import Pipeline

def splitAudioByRTTM(rttm_file, audio_file, output_dir="./data/audio"):
    # Read RTTM file
    with open(rttm_file, "r") as f:
        lines = f.readlines()
    # Split audio file by RTTM file
    for line in lines:
        line = line.split(" ")
        start = line[3]
        end = line[4]
        speaker = line[7]
        os.system("ffmpeg -i " + audio_file + " -ss " + start + " -to " + end + " -c copy " + output_dir + "/speaker_" + speaker + ".wav")

def diarize_mp3(audio_file="./data/audio/part1.mp3", num_speakers=2, output_dir="./data/audio"):

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization") ## Broken here, model not found

    # Check if audio file isn't WAV, convert to WAV
    if audio_file[-3:] != "wav":
        audio_file_wav = audio_file[:-3] + "wav"
        os.system("ffmpeg -i " + audio_file + " " + audio_file_wav)

    # Check if audio files are greater than 200MB, split into 200MB chunks
    if os.path.getsize(audio_file) > 200000000:
        os.system("ffmpeg -i " + audio_file + " -f segment -segment_time 1000 -c copy " + output_dir + "/%03d.wav")

    # apply the pipeline to an audio file
    diarization = pipeline(audio_file, num_speakers=num_speakers)

    # Get data of audio time splits
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    # save diarization as RTTM file
    output_file = os.path.join(output_dir, "diarization.rttm")

    # dump the diarization output to disk using RTTM format
    with open(output_file, "w") as rttm:
        diarization.write_rttm(rttm)

    splitAudioByRTTM(output_file, audio_file, output_dir)
