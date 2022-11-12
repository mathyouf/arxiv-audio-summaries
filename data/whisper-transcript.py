import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

def whisperTranscribe(mp3_file, output_dir):
    model = whisper.load_model("base")
    result = model.transcribe(mp3_file)
    print(result["text"])
    output_file = os.path.join(output