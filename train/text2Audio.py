import replicate
model = replicate.models.get("afiaka87/tortoise-tts")
version = model.versions.get("e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71")
output = version.predict(text="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
