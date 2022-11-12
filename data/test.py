from speakerDiarization import diarize_mp3

audio_file = "./data/audio/episode1/out000.wav"
speaker1, speaker2 = diarize_mp3(audio_file, num_speakers=2)
