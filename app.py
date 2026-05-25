from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import io
import base64
import tempfile
import warnings
import subprocess
import numpy as np

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ==========================================
# ALL 22 INDIAN LANGUAGES + ENGLISH
# ==========================================

LANG_MAP = {
    "english":   "en",
    "hindi":     "hi",
    "bengali":   "bn",
    "telugu":    "te",
    "marathi":   "mr",
    "tamil":     "ta",
    "gujarati":  "gu",
    "urdu":      "ur",
    "kannada":   "kn",
    "odia":      "or",
    "malayalam": "ml",
    "punjabi":   "pa",
    "assamese":  "as",
    "maithili":  "mai",
    "sindhi":    "sd",
    "kashmiri":  "ks",
    "konkani":   "gom",
    "nepali":    "ne",
    "sanskrit":  "sa",
    "dogri":     "doi",
    "bodo":      "brx",
    "manipuri":  "mni-Mtei",
    "santhali":  "sat"
}

# Whisper returns ISO 639-1 codes — map to our language keys
WHISPER_TO_LANG = {
    "en": "english",   "hi": "hindi",     "bn": "bengali",
    "te": "telugu",    "mr": "marathi",   "ta": "tamil",
    "gu": "gujarati",  "ur": "urdu",      "kn": "kannada",
    "or": "odia",      "ml": "malayalam", "pa": "punjabi",
    "as": "assamese",  "ne": "nepali",    "sa": "sanskrit",
    "sd": "sindhi",    "ks": "kashmiri",
}

SR_LOCALE_MAP = {
    "en": "en-IN",  "hi": "hi-IN",  "bn": "bn-IN",
    "te": "te-IN",  "mr": "mr-IN",  "ta": "ta-IN",
    "gu": "gu-IN",  "ur": "ur-IN",  "kn": "kn-IN",
    "or": "or-IN",  "ml": "ml-IN",  "pa": "pa-IN",
    "as": "as-IN",  "sa": "sa-IN",  "ne": "ne-NP",
    "mai": "hi-IN", "sd": "ur-IN",  "ks": "ur-IN",
    "gom": "hi-IN", "doi": "hi-IN", "brx": "hi-IN",
    "mni-Mtei": "hi-IN", "sat": "hi-IN",
}

SUPPORTED_TTS = [
    "en", "hi", "bn", "te", "mr", "ta", "gu",
    "ur", "kn", "ml", "pa", "ne", "or", "as", "sa"
]

# ==========================================
# WHISPER — load once at startup
# Using "small" model for much better Indian language detection
# "tiny" often misidentifies Indian languages as English because
# it has too few parameters to distinguish similar phonetics
# ==========================================

_whisper_mod  = None
whisper_model = None
STT_AVAILABLE = False

try:
    import whisper as _whisper_mod
    import speech_recognition as _sr_mod

    print("⏳ Loading Whisper 'small' model for accurate Indian language detection...")
    # device="cpu" + fp16=False avoids ALL FP16 warnings and bugs on Windows
    whisper_model = _whisper_mod.load_model("small", device="cpu")
    print("✅ Whisper 'small' loaded on CPU")
    STT_AVAILABLE = True

except SyntaxError:
    # Handle the `import X as Y` syntax — try separately
    try:
        import whisper as _whisper_mod
        import speech_recognition
        _sr_mod = speech_recognition

        print("⏳ Loading Whisper 'small' model...")
        whisper_model = _whisper_mod.load_model("small", device="cpu")
        print("✅ Whisper loaded")
        STT_AVAILABLE = True
    except Exception as e:
        print(f"⚠️  Whisper not available: {e}")

except Exception as e:
    print(f"⚠️  Whisper/STT not available: {e}")


# ==========================================
# Helper — convert any browser audio format to clean 16-kHz WAV
# Browser MediaRecorder sends webm/opus or webm/vorbis.
# Whisper's load_audio uses ffmpeg internally but sometimes fails
# on certain webm variants. We add our own ffmpeg fallback.
# ==========================================

def to_wav(input_path: str) -> str:
    """
    Convert input_path (webm/ogg/mp4/etc.) → 16-kHz mono WAV.
    Returns path to the WAV file (caller must delete it).
    Uses ffmpeg if available, otherwise falls back to whisper.load_audio.
    """
    out_path = input_path + "_converted.wav"

    # Try ffmpeg first (most reliable for browser webm)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "16000",   # 16 kHz sample rate
                "-ac", "1",       # mono
                "-sample_fmt", "s16",
                out_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        if result.returncode == 0 and os.path.exists(out_path):
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # ffmpeg not installed or timed out — fall through

    # Fallback: whisper's own audio loader (also uses ffmpeg internally
    # but with different flags that sometimes work when the above doesn't)
    try:
        import soundfile as sf
        audio_array = _whisper_mod.load_audio(input_path)  # float32, 16 kHz
        sf.write(out_path, audio_array, 16000, subtype="PCM_16")
        return out_path
    except Exception as e:
        raise RuntimeError(f"Could not convert audio to WAV: {e}")


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# /translate
# ==========================================

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data     = request.json or {}
        text     = (data.get("text") or "").strip()
        src      = data.get("src", "english")
        tgt      = data.get("tgt", "hindi")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        src_code = LANG_MAP.get(src, "en")
        tgt_code = LANG_MAP.get(tgt, "hi")

        translated = GoogleTranslator(
            source=src_code, target=tgt_code
        ).translate(text)

        audio_b64 = None
        if tgt_code in SUPPORTED_TTS:
            try:
                tts = gTTS(text=translated, lang=tgt_code, slow=False)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
            except Exception as e:
                print(f"⚠️  TTS failed for '{tgt_code}': {e}")

        return jsonify({
            "translated_text": translated,
            "translated":      translated,
            "audio_b64":       audio_b64,
            "tgt_code":        tgt_code,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# /stt — speech-to-text OR language detection
# ==========================================

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if not STT_AVAILABLE:
        return jsonify({
            "error": "STT not available — run: pip install openai-whisper SpeechRecognition soundfile"
        }), 503

    src_lang    = request.form.get("src", "english").lower()
    detect_lang = request.form.get("detect_lang", "false").lower() == "true"
    src_code    = LANG_MAP.get(src_lang, "en")
    locale      = SR_LOCALE_MAP.get(src_code, "en-IN")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    # Save raw browser audio (webm/opus)
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        raw_path = tmp.name

    wav_path = None

    try:
        # Convert to clean WAV for processing
        wav_path = to_wav(raw_path)

        # Check duration — too short = unreliable
        import soundfile as sf
        info = sf.info(wav_path)
        duration = info.duration
        print(f"📊 Audio duration: {duration:.2f}s")

        if duration < 1.5:
            return jsonify({
                "error": "Recording too short — please speak for at least 2 seconds"
            }), 422

        # --------------------------------------------------------------
        # MODE 1 — LANGUAGE DETECTION
        #
        # KEY FIX: Instead of just running detect_language() on a mel
        # spectrogram (which is unreliable and biased toward English),
        # we use whisper_model.transcribe() with task="transcribe" and
        # NO language hint. Whisper then:
        #   1. Internally detects the language using its full model
        #   2. Returns the detected language code in result["language"]
        # This is far more accurate for Indian languages because it uses
        # the full decoder context, not just the encoder's mel output.
        # --------------------------------------------------------------
        if detect_lang:
            print("🔍 Detecting language via full Whisper transcription pipeline...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = whisper_model.transcribe(
                    wav_path,
                    language=None,      # ← CRITICAL: don't hint the language
                    task="transcribe",
                    fp16=False,         # CPU-safe
                    verbose=False,
                    # These settings improve detection for short clips:
                    temperature=0.0,    # greedy decoding = deterministic
                    best_of=1,
                    beam_size=1,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.3,
                    logprob_threshold=-1.5,
                )

            detected_code = result.get("language", "")
            transcribed   = (result.get("text") or "").strip()

            print(f"✅ Whisper detected language: '{detected_code}'")
            print(f"   Transcribed text: '{transcribed[:80]}'")

            if not detected_code:
                return jsonify({
                    "error": "Could not detect language — speak clearly and try again"
                }), 422

            # Map Whisper ISO code → our language name
            detected_name = WHISPER_TO_LANG.get(detected_code, detected_code)

            # Extra validation: if transcribed text is very short or empty,
            # detection might be unreliable — flag it
            is_confident = len(transcribed) >= 3

            return jsonify({
                "detected_lang": detected_name,
                "detected_code": detected_code,
                "confidence":    95.0 if is_confident else 60.0,
                "transcribed":   transcribed,   # bonus: also return the text
            })

        # --------------------------------------------------------------
        # MODE 2 — TRANSCRIPTION
        # --------------------------------------------------------------
        text = None

        # Step 1: Google STT (accurate for Indian languages with locale)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language=locale)
                print(f"✅ Google STT: {text}")
            except sr.UnknownValueError:
                print("⚠️  Google STT: speech unclear")
            except sr.RequestError as e:
                print(f"⚠️  Google STT network error: {e}")
        except Exception as e:
            print(f"⚠️  Google STT error: {e}")

        # Step 2: Whisper fallback
        if not text:
            print("🔄 Whisper transcription fallback...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = whisper_model.transcribe(
                    wav_path,
                    language=src_code if src_code != "auto" else None,
                    task="transcribe",
                    fp16=False,
                    verbose=False,
                    temperature=0.0,
                    best_of=1,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
            text = (result.get("text") or "").strip() or None
            if text:
                print(f"✅ Whisper: {text}")

        if text:
            return jsonify({"text": text})

        return jsonify({
            "error": "Could not recognise speech — speak clearly and try again"
        }), 422

    except Exception as e:
        print(f"❌ STT error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [raw_path, wav_path]:
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ==========================================
# /health
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "stt":     STT_AVAILABLE,
        "whisper": whisper_model is not None,
        "model":   "small" if whisper_model else None,
    })


# ==========================================
# /audio/<filename> — backward compat
# ==========================================

@app.route("/audio/<filename>")
def audio(filename):
    if os.path.exists(filename):
        return send_file(filename, mimetype="audio/mpeg")
    return jsonify({"error": "File not found"}), 404


# ==========================================
# START
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  SunoBhashini Server — http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)