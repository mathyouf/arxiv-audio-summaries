# arxiv-audio-summaries
Want to get a summary of a new ARXIV paper? Don't even want to read the summary? Have the best read it out loud for you!

Have Adrej Karpathy and Justin Johnson explain ML Papers from ARXIV

## How It Works

1. Download recorded audio conversations where Justin Johnson and Adrej Karpathy are discussing ML related topics (should be most of the ones they have lol)
    - [Deep Learning Deep Dive](https://www.listennotes.com/podcasts/deep-learning-deep-dive-deep-learning-deep-MY3s4NlQ7D-/)

2. Process the audio files to get separate audio files for each speaker (using Diarization)
    - [HF: speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)

3. Process the seperate audio files to get the transcripts (using Whisper)

4. Fine-tune a language model on the transcripts
    - GPT-3

5. Fine-tune an audio model on the audio files
    -    Using [Tortoise-TTS](https://replicate.com/afiaka87/tortoise-tts)

6. Prompt the language model with the paper title and abstract and generate a summary
    - GPT-3

7. Given this prompt to the Fine-tuned GPT-3 to get the transcript

8. Given the transcript to the Fine-tuned audio model to get the audio summary