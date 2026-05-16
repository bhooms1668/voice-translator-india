from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import os
import io
import base64
import tempfile
import warnings

warnings.filterwarnings("ignore", message=".*FP16.*")

app = Flask(__name__)
CORS(app)

# ==========================================
# ALL 22 INDIAN LANGUAGES + ENGLISH
# ==========================================

LANG_MAP = {
    "english":  "en",
    "hindi":    "hi",
    "bengali":  "bn",
    "telugu":   "te",
    "marathi":  "mr",
    "tamil":    "ta",
    "gujarati": "gu",
    "urdu":     "ur",
    "kannada":  "kn",
    "odia":     "or",
    "malayalam":"ml",
    "punjabi":  "pa",
    "assamese": "as",
    "maithili": "mai",
    "sindhi":   "sd",
    "kashmiri": "ks",
    "konkani":  "gom",
    "nepali":   "ne",
    "sanskrit": "sa",
    "dogri":    "doi",
    "bodo":     "brx",
    "manipuri": "mni-Mtei",
    "santhali": "sat"
}

# Whisper language code → our language key
WHISPER_TO_LANG = {
    "en":"english","hi":"hindi","bn":"bengali","te":"telugu","mr":"marathi",
    "ta":"tamil","gu":"gujarati","ur":"urdu","kn":"kannada","or":"odia",
    "ml":"malayalam","pa":"punjabi","as":"assamese","ne":"nepali",
    "sa":"sanskrit","sd":"sindhi",
}

SR_LOCALE_MAP = {
    "en":"en-IN","hi":"hi-IN","bn":"bn-IN","te":"te-IN","mr":"mr-IN",
    "ta":"ta-IN","gu":"gu-IN","ur":"ur-IN","kn":"kn-IN","or":"or-IN",
    "ml":"ml-IN","pa":"pa-IN","as":"as-IN","sa":"sa-IN","ne":"ne-NP",
    "mai":"hi-IN","sd":"ur-IN","ks":"ur-IN","gom":"hi-IN","doi":"hi-IN",
    "brx":"hi-IN","mni-Mtei":"hi-IN","sat":"hi-IN",
}

# ==========================================
# LANGUAGES SUPPORTED BY gTTS AUDIO
# ==========================================

SUPPORTED_TTS = [
    "en","hi","bn","te","mr","ta","gu","ur","kn","ml","pa","ne","or","as","sa"
]

# ==========================================
# OPTIONAL: Whisper for STT + language detection
# ==========================================

try:
    import whisper
    import speech_recognition as sr
    print("⏳ Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("✅ Whisper loaded")
    STT_AVAILABLE = True
except Exception as e:
    whisper_model = None
    STT_AVAILABLE = False
    print(f"⚠️  Whisper/STT not available: {e}")

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# TRANSLATE — returns translated text + base64 audio
# ==========================================

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data     = request.json
        text     = data.get("text", "").strip()
        src      = data.get("src", "english")
        tgt      = data.get("tgt", "hindi")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        src_code = LANG_MAP.get(src, "en")
        tgt_code = LANG_MAP.get(tgt, "hi")

        # ── Translation ────────────────────────────────────────────────────────
        translated = GoogleTranslator(
            source=src_code, target=tgt_code
        ).translate(text)

        # ── Audio — return as base64 so no file cleanup needed ─────────────────
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
            "translated_text": translated,   # kept for backward compatibility
            "translated":      translated,
            "audio_b64":       audio_b64,    # base64 MP3, played directly in browser
            "tgt_code":        tgt_code,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# STT — speech-to-text OR language detection from audio
# ==========================================

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if not STT_AVAILABLE:
        return jsonify({"error": "STT not available — install openai-whisper and SpeechRecognition"}), 503

    src_lang    = request.form.get("src", "english").lower()
    detect_lang = request.form.get("detect_lang", "false").lower() == "true"
    src_code    = LANG_MAP.get(src_lang, "en")
    locale      = SR_LOCALE_MAP.get(src_code, "en-IN")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── MODE 1: Detect language from audio using Whisper ──────────────────
        if detect_lang:
            audio_array = whisper.load_audio(tmp_path)
            audio_array = whisper.pad_or_trim(audio_array)
            mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, probs = whisper_model.detect_language(mel)

            detected_code = max(probs, key=probs.get)
            confidence    = round(probs[detected_code] * 100, 1)
            detected_name = WHISPER_TO_LANG.get(detected_code, detected_code)
            print(f"✅ Detected: {detected_name} ({detected_code}) — {confidence}%")

            return jsonify({
                "detected_lang": detected_name,
                "detected_code": detected_code,
                "confidence":    confidence,
            })

        # ── MODE 2: Transcribe audio to text ──────────────────────────────────
        text = None

        # Try Google STT (convert webm → wav first via whisper's loader)
        try:
            import soundfile as sf
            audio_array = whisper.load_audio(tmp_path)
            wav_path = tmp_path + ".wav"
            sf.write(wav_path, audio_array, 16000, subtype="PCM_16")

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
                print(f"⚠️  Google STT error: {e}")
            try:
                os.unlink(wav_path)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️  Google STT pipeline error: {e}")

        # Whisper fallback
        if not text and whisper_model:
            print("🔄 Whisper transcription fallback...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = whisper_model.transcribe(
                    tmp_path, language=src_code,
                    task="transcribe", fp16=False, verbose=False
                )
            text = result.get("text", "").strip() or None
            if text:
                print(f"✅ Whisper: {text}")

        if text:
            return jsonify({"text": text})
        return jsonify({"error": "Could not recognise speech — speak clearly and try again"}), 422

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "stt":     STT_AVAILABLE,
        "whisper": whisper_model is not None,
    })


# ==========================================
# AUDIO ROUTE (kept for backward compatibility)
# ==========================================

@app.route("/audio/<filename>")
def audio(filename):
    if os.path.exists(filename):
        return send_file(filename, mimetype="audio/mpeg")
    return jsonify({"error": "File not found"}), 404


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SunoBhashini Server — http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True)