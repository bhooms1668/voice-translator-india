import argparse
import os
import sys
import tempfile
import time
import warnings

# Suppress Whisper FP16 warning before importing whisper
warnings.filterwarnings("ignore", message=".*FP16.*")

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame

print("🚀 Program started")

# =========================
# WHISPER INIT
# =========================
try:
    import whisper
    print("⏳ Loading Whisper model (first run may take a moment)...")
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded")
except Exception as e:
    whisper_model = None
    print(f"⚠️  Whisper not available: {e}")

# =========================
# PYGAME INIT
# =========================
try:
    pygame.mixer.init()
    print("✅ Pygame mixer initialized")
except Exception as e:
    print(f"⚠️  Pygame mixer failed: {e}")

# =========================
# LANGUAGE MAPS
# All 22 scheduled languages of the Indian Constitution + English
# =========================
LANG_MAP = {
    # Indo-Aryan
    "english":    "en",
    "hindi":      "hi",
    "bengali":    "bn",
    "marathi":    "mr",
    "gujarati":   "gu",
    "punjabi":    "pa",
    "odia":       "or",
    "assamese":   "as",
    "maithili":   "mai",   # Scheduled since 2003
    "sindhi":     "sd",
    "kashmiri":   "ks",
    "konkani":    "gom",   # Goan Konkani (Google code)
    "nepali":     "ne",
    "sanskrit":   "sa",
    "dogri":      "doi",
    "bodo":       "brx",   # may fall back to transliteration
    # Dravidian
    "tamil":      "ta",
    "telugu":     "te",
    "kannada":    "kn",
    "malayalam":  "ml",
    # Tibeto-Burman
    "manipuri":   "mni-Mtei",  # Meitei/Manipuri
    "santhali":   "sat",
    # Urdu (also official in several Indian states)
    "urdu":       "ur",
}

# Google Speech Recognition locale codes
# Languages without an Indian locale fall back to a close match
SPEECH_RECOGNITION_LANG_MAP = {
    "en":       "en-IN",
    "hi":       "hi-IN",
    "bn":       "bn-IN",
    "mr":       "mr-IN",
    "gu":       "gu-IN",
    "pa":       "pa-IN",
    "or":       "or-IN",
    "as":       "as-IN",
    "mai":      "hi-IN",   # no dedicated locale; Hindi is closest
    "sd":       "sd-IN",
    "ks":       "ur-IN",   # Kashmiri STT not widely supported; Urdu fallback
    "gom":      "hi-IN",   # Konkani STT fallback
    "ne":       "ne-NP",
    "sa":       "sa-IN",
    "doi":      "hi-IN",   # Dogri STT fallback
    "brx":      "hi-IN",   # Bodo STT fallback
    "ta":       "ta-IN",
    "te":       "te-IN",
    "kn":       "kn-IN",
    "ml":       "ml-IN",
    "mni-Mtei": "hi-IN",   # Manipuri STT fallback
    "sat":      "hi-IN",   # Santhali STT fallback
    "ur":       "ur-IN",
}

# =========================
# DEVICE UTILITIES
# =========================

def list_input_devices():
    print("\n📋 Available INPUT devices:")
    found = False
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " ◀ default" if idx == sd.default.device[0] else ""
            print(f"  [{idx:2d}] {dev['name']}  "
                  f"({dev['max_input_channels']} ch, "
                  f"{int(dev['default_samplerate'])} Hz){marker}")
            found = True
    if not found:
        print("  ❌ No input devices found!")
    print()


def pick_input_device(requested=None):
    """Return a verified input-capable device index."""
    if requested is not None:
        dev = sd.query_devices(requested)
        if dev["max_input_channels"] < 1:
            raise RuntimeError(f"Device {requested} '{dev['name']}' has no input channels.")
        return requested

    # Try sounddevice default first
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            dev = sd.query_devices(default_in)
            if dev["max_input_channels"] > 0:
                return default_in
    except Exception:
        pass

    # Walk all devices and take first with input channels
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            return idx

    raise RuntimeError("No input audio device found.")

# =========================
# AUDIO RECORDING
# =========================

def record_audio(output_path, duration=5, samplerate=16000, device=None):
    """
    Record mic audio and save as a 16-bit mono WAV.
    Records at device's native samplerate then resamples to 16kHz.
    Uses soundfile for writing so Whisper can always read it correctly.
    """
    device = pick_input_device(device)
    dev_info = sd.query_devices(device)
    print(f"🎛  Device [{device}]: {dev_info['name']}")

    # Always record at the device's native sample rate to avoid driver issues
    native_sr = int(dev_info["default_samplerate"])

    print("📢 Prepare to speak. Recording starts in 2 seconds…")
    time.sleep(2)
    print(f"🎙  Recording for {duration} seconds at {native_sr} Hz…")

    try:
        audio = sd.rec(
            int(duration * native_sr),
            samplerate=native_sr,
            channels=1,
            dtype="float32",   # float32 works on all devices; int16 can fail on some
            device=device,
        )
        sd.wait()
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        list_input_devices()
        raise

    audio = audio.flatten()

    # ── Silence / level check ──────────────────────────────────
    peak = float(np.abs(audio).max())
    rms  = float(np.sqrt(np.mean(audio ** 2)))
    print(f"📊 Peak amplitude: {peak:.4f}   RMS: {rms:.6f}")
    if peak < 0.01:
        print("⚠️  WARNING: Recorded audio is near-silent!")
        print("    Possible causes:")
        print("    1. Microphone muted in Windows Sound Settings → Recording tab")
        print("    2. Wrong device selected — run: python voice_translator.py --list-devices")
        print("       Then re-run with: --device N  (use the correct index)")
        print("    3. Microphone volume set to 0 in Windows mixer")
        list_input_devices()
    # ──────────────────────────────────────────────────────────

    # Resample to 16 kHz if needed
    if native_sr != samplerate:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=samplerate)
            print(f"🔄 Resampled {native_sr} → {samplerate} Hz (librosa)")
        except ImportError:
            # Simple decimation fallback (good enough for speech)
            ratio = native_sr / samplerate
            indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
            print(f"🔄 Resampled {native_sr} → {samplerate} Hz (simple)")

    # Convert float32 → int16
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    # Write WAV — prefer soundfile (Whisper reads it perfectly)
    try:
        import soundfile as sf
        sf.write(output_path, audio_int16, samplerate, subtype="PCM_16")
        print(f"✅ Audio saved (soundfile): {output_path}")
    except ImportError:
        from scipy.io.wavfile import write as wav_write
        wav_write(output_path, samplerate, audio_int16)
        print(f"✅ Audio saved (scipy): {output_path}")

    return output_path

# =========================
# SPEECH RECOGNITION
# =========================

def whisper_transcribe(audio_path, src_lang_code="en"):
    if whisper_model is None:
        return None
    print("🧠 Whisper transcribing…")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = whisper_model.transcribe(
                audio_path,
                language=src_lang_code,
                task="transcribe",
                fp16=False,        # ← critical fix: forces CPU-safe float32
                verbose=False,
            )
        text = result.get("text", "").strip()
        if text:
            print(f"✅ Whisper: {text}")
            return text
        print("⚠️  Whisper returned no text (audio may be silent or too noisy).")
    except Exception as e:
        print(f"❌ Whisper error: {e}")
    return None


def speech_to_text(audio_path, src_lang_code="en"):
    # ── Try Google STT ─────────────────────────────────────────
    print("🌐 Trying Google STT…")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
        lang = SPEECH_RECOGNITION_LANG_MAP.get(src_lang_code, "en-US")
        text = recognizer.recognize_google(audio_data, language=lang)
        if text:
            print(f"✅ Google STT: {text}")
            return text
    except sr.UnknownValueError:
        print("⚠️  Google STT: speech unclear.")
    except sr.RequestError as e:
        print(f"⚠️  Google STT: network error ({e}).")
    except Exception as e:
        print(f"⚠️  Google STT: {e}")

    # ── Fallback: Whisper ───────────────────────────────────────
    print("🔄 Falling back to Whisper…")
    return whisper_transcribe(audio_path, src_lang_code)


# =========================
# TRANSLATION & TTS
# =========================

def translate_text(text, target_lang):
    if not text:
        return ""
    print("🌐 Translating…")
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        print(f"💬 Translation: {translated}")
        return translated
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return ""


def speak_text(text, lang_code, output_path):
    if not text:
        print("❌ Nothing to speak.")
        return
    print("🔊 Generating TTS…")
    try:
        gTTS(text=text, lang=lang_code, slow=False).save(output_path)
        print(f"✅ TTS saved: {output_path}")
        play_audio(output_path)
    except Exception as e:
        print(f"❌ TTS error: {e}")


def play_audio(path):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        # Windows holds a file lock on the MP3 after playback.
        # unload() releases it so TemporaryDirectory can delete the file cleanly.
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"❌ Playback error: {e}")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="🎤 Voice Translator — speak in one language, hear another",
    )
    parser.add_argument("--src",          default=None, help="Source language (e.g. english)")
    parser.add_argument("--to",           default=None, help="Target language (e.g. hindi)")
    parser.add_argument("--duration",     type=int, default=5,    help="Recording seconds (default: 5)")
    parser.add_argument("--device",       type=int, default=None, help="Input device index")
    parser.add_argument("--list-devices", action="store_true",    help="Show all input devices and exit")

    # Use parse_known_args so --list-devices always works even without --src/--to
    args, _ = parser.parse_known_args()

    if args.list_devices:
        list_input_devices()
        return

    if not args.src or not args.to:
        print("❌ --src and --to are required.\n")
        print("Usage examples:")
        print("  python voice_translator.py --src english --to hindi")
        print("  python voice_translator.py --src english --to hindi --device 1 --duration 8")
        print("  python voice_translator.py --list-devices")
        sys.exit(1)

    src = args.src.lower()
    tgt = args.to.lower()

    if src not in LANG_MAP:
        print(f"❌ Unknown source language '{src}'.\nSupported: {', '.join(LANG_MAP)}")
        sys.exit(1)
    if tgt not in LANG_MAP:
        print(f"❌ Unknown target language '{tgt}'.\nSupported: {', '.join(LANG_MAP)}")
        sys.exit(1)

    src_code = LANG_MAP[src]
    tgt_code = LANG_MAP[tgt]

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "input.wav")
        tts_path   = os.path.join(tmpdir, "translation.mp3")

        try:
            record_audio(audio_path, duration=args.duration, device=args.device)

            text = speech_to_text(audio_path, src_lang_code=src_code)
            if not text:
                print("\n❌ No speech recognised. Try:")
                print("   Step 1 — list devices:  python voice_translator.py --list-devices")
                print("   Step 2 — pick your mic: python voice_translator.py --src english --to hindi --device N")
                print("   Step 3 — speak longer:  add --duration 8")
                return

            translated = translate_text(text, tgt_code)
            if not translated:
                print("❌ Translation failed. Check internet connection.")
                return

            speak_text(translated, tgt_code, tts_path)
            print("\n✅ Done!")

        except KeyboardInterrupt:
            print("\n⛔ Interrupted.")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            raise


if __name__ == "__main__":
    main()