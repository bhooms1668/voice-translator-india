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

SR_LOCALE_MAP = {
    "en":        "en-IN",
    "hi":        "hi-IN",
    "bn":        "bn-IN",
    "te":        "te-IN",
    "mr":        "mr-IN",
    "ta":        "ta-IN",
    "gu":        "gu-IN",
    "ur":        "ur-IN",
    "kn":        "kn-IN",
    "or":        "or-IN",
    "ml":        "ml-IN",
    "pa":        "pa-IN",
    "as":        "as-IN",
    "sa":        "sa-IN",
    "ne":        "ne-NP",
    "mai":       "hi-IN",
    "sd":        "ur-IN",
    "ks":        "ur-IN",
    "gom":       "hi-IN",
    "doi":       "hi-IN",
    "brx":       "hi-IN",
    "mni-Mtei":  "hi-IN",
    "sat":       "hi-IN",
}

# ==========================================
# LANGUAGES SUPPORTED BY gTTS
# ==========================================

SUPPORTED_TTS = [
    "en", "hi", "bn", "te", "mr", "ta", "gu",
    "ur", "kn", "ml", "pa", "ne", "or", "as", "sa"
]

# ==========================================
# WHISPER — for STT (transcription) ONLY
# Language detection is handled in the browser
# using Unicode script ranges — 100% accurate
# for Indian languages, no model bias issues.
# ==========================================

_whisper_mod  = None
whisper_model = None
STT_AVAILABLE = False

try:
    import whisper as _whisper_mod
    import speech_recognition as _sr_mod
    import soundfile as _sf_mod

    print("⏳ Loading Whisper model (base)...")
    whisper_model = _whisper_mod.load_model("base", device="cpu")
    print("✅ Whisper loaded")
    STT_AVAILABLE = True

except Exception as e:
    print(f"⚠️  Whisper/STT not available: {e}")


# ==========================================
# HELPER — convert browser audio → 16kHz WAV
# ==========================================

def to_wav(input_path):
    out_path = input_path + "_conv.wav"

    # Try ffmpeg first
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )
        if r.returncode == 0 and os.path.exists(out_path):
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Whisper loader fallback
    try:
        audio_array = _whisper_mod.load_audio(input_path)
        _sf_mod.write(out_path, audio_array, 16000, subtype="PCM_16")
        return out_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {e}")


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# /translate — text + TTS audio
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
# /stt — speech-to-text (transcription only)
# Language detection is done in the browser.
# ==========================================

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if not STT_AVAILABLE:
        return jsonify({
            "error": "STT not available — install openai-whisper SpeechRecognition soundfile"
        }), 503

    src_lang = request.form.get("src", "english").lower()
    src_code = LANG_MAP.get(src_lang, "en")
    locale   = SR_LOCALE_MAP.get(src_code, "en-IN")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        raw_path = tmp.name

    wav_path = None

    try:
        wav_path = to_wav(raw_path)

        # Duration check
        info     = _sf_mod.info(wav_path)
        duration = info.duration
        print(f"📊 Audio: {duration:.2f}s  locale: {locale}")

        if duration < 1.0:
            return jsonify({"error": "Recording too short — speak for at least 2 seconds"}), 422

        text = None

        # ── Step 1: Google STT (best for Indian languages with locale) ──────
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
                print("⚠️  Google STT: unclear audio")
            except sr.RequestError as e:
                print(f"⚠️  Google STT network error: {e}")
        except Exception as e:
            print(f"⚠️  Google STT error: {e}")

        # ── Step 2: Whisper fallback ─────────────────────────────────────────
        if not text:
            print("🔄 Whisper fallback...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = whisper_model.transcribe(
                    wav_path,
                    language=src_code,
                    task="transcribe",
                    fp16=False,
                    verbose=False,
                    temperature=0.0,
                    condition_on_previous_text=False,
                )
            text = (result.get("text") or "").strip() or None
            if text:
                print(f"✅ Whisper: {text}")

        if text:
            return jsonify({"text": text})

        return jsonify({"error": "Could not recognise speech — speak clearly and try again"}), 422

    except Exception as e:
        print(f"❌ STT error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [raw_path, wav_path]:
            if p:
                try: os.unlink(p)
                except: pass


# ==========================================
# /health
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "stt":     STT_AVAILABLE,
        "whisper": whisper_model is not None,
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
    print("  SunoBhashini — http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)