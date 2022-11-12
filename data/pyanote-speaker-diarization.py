# https://github.com/pyannote/pyannote-audio
# https://huggingface.co/pyannote/speaker-diarization
# conda activate pyannote
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

def diarize_mp3(audio_file="./data/audio/part1.mp3", output_dir="./data/audio"):

    load_dotenv()

    HF_TOKEN = os.getenv('HF_TOKEN')

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=HF_TOKEN)

    # apply the pipeline to an audio file
    diarization = pipeline(audio_file, num_speakers=2)

    # save diarization as RTTM file
    output_file = os.path.join(output_dir, "diarization.rttm")

    # dump the diarization output to disk using RTTM format
    with open(output_file, "w") as rttm:
        diarization.write_rttm(rttm)