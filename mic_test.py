import sounddevice as sd

print(sd.query_devices())

print("Testing microphone...")

audio = sd.rec(
    16000 * 5,
    samplerate=16000,
    channels=2,
    dtype='int16',
    device=1
)

sd.wait()

print("Microphone works!")