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

# Whisper detected code → our language key
WHISPER_TO_LANG = {
    "en": "english",  "hi": "hindi",     "bn": "bengali",
    "te": "telugu",   "mr": "marathi",   "ta": "tamil",
    "gu": "gujarati", "ur": "urdu",      "kn": "kannada",
    "or": "odia",     "ml": "malayalam", "pa": "punjabi",
    "as": "assamese", "ne": "nepali",    "sa": "sanskrit",
    "sd": "sindhi",
}

SR_LOCALE_MAP = {
    "en": "en-IN",  "hi": "hi-IN",  "bn": "bn-IN",
    "te": "te-IN",  "mr": "mr-IN",  "ta": "ta-IN",
    "gu": "gu-IN",  "ur": "ur-IN",  "kn": "kn-IN",
    "or": "or-IN",  "ml": "ml-IN",  "pa": "pa-IN",
    "as": "as-IN",  "sa": "sa-IN",  "ne": "ne-NP",
    "mai": "hi-IN", "sd": "ur-IN",  "ks": "ur-IN",
    "gom":"hi-IN",  "doi":"hi-IN",  "brx":"hi-IN",
    "mni-Mtei": "hi-IN", "sat": "hi-IN",
}

# ==========================================
# LANGUAGES SUPPORTED BY gTTS
# ==========================================

SUPPORTED_TTS = [
    "en", "hi", "bn", "te", "mr", "ta", "gu",
    "ur", "kn", "ml", "pa", "ne", "or", "as", "sa"
]

# ==========================================
# WHISPER — load once at startup, CPU-safe
# FIX: load model at module level so it is a true global
#      and never re-assigned inside a function (which would
#      create a local shadow and break UnboundLocalError).
# ==========================================

whisper_model  = None
STT_AVAILABLE  = False
_whisper_mod   = None   # reference to the whisper module itself

try:
    import whisper as _whisper_mod
    import speech_recognition as sr

    print("⏳ Loading Whisper model (tiny — fast, good for language detection)...")
    # FIX: always load on CPU explicitly so fp16 is never attempted
    whisper_model = _whisper_mod.load_model("tiny", device="cpu")
    print("✅ Whisper loaded on CPU")
    STT_AVAILABLE = True

except Exception as e:
    print(f"⚠️  Whisper/STT not available: {e}")

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# /translate — text → translated text + base64 MP3
# ==========================================

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data     = request.json
        text     = (data.get("text") or "").strip()
        src      = data.get("src", "english")
        tgt      = data.get("tgt", "hindi")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        src_code = LANG_MAP.get(src, "en")
        tgt_code = LANG_MAP.get(tgt, "hi")

        # Translation
        translated = GoogleTranslator(
            source=src_code, target=tgt_code
        ).translate(text)

        # TTS — stream to BytesIO, encode as base64 (no disk files)
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
            "translated_text": translated,  # backward compat
            "translated":      translated,
            "audio_b64":       audio_b64,
            "tgt_code":        tgt_code,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# /stt — speech-to-text  OR  language detection
#
# FIX SUMMARY
# -----------
# 1. Removed  whisper_model = whisper_model.to("cpu")  inside the
#    function — that line created a LOCAL variable that shadowed the
#    global, causing UnboundLocalError before detect_language() ran.
#    The model is already on CPU from load_model("tiny", device="cpu").
#
# 2. Used the module-level reference `_whisper_mod` for helper calls
#    (load_audio, pad_or_trim, log_mel_spectrogram) instead of the
#    bare name `whisper`, which is not in scope inside the function.
#
# 3. Added a minimum-duration guard: clips shorter than 1 s often
#    produce meaningless probabilities — we return a clear error
#    instead of a wrong answer.
#
# 4. Improved confidence threshold: if the top language probability
#    is below 15 % we report "uncertain" rather than a wrong guess.
# ==========================================

@app.route("/stt", methods=["POST"])
def speech_to_text():
    if not STT_AVAILABLE:
        return jsonify({
            "error": "STT not available — install openai-whisper and SpeechRecognition"
        }), 503

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

        # ------------------------------------------------------------------
        # MODE 1 — Language detection
        # ------------------------------------------------------------------
        if detect_lang:
            # FIX: use _whisper_mod (module ref), NOT whisper (not in scope)
            audio_array = _whisper_mod.load_audio(tmp_path)   # float32, 16 kHz

            # Guard: need at least ~1 second of audio for a reliable result
            if len(audio_array) < 16000:
                return jsonify({
                    "error": "Recording too short — speak for at least 2 seconds"
                }), 422

            # pad_or_trim clips/pads to 30 s (Whisper's expected window)
            audio_trimmed = _whisper_mod.pad_or_trim(audio_array)

            # Build mel spectrogram on CPU
            mel = _whisper_mod.log_mel_spectrogram(audio_trimmed).to(
                whisper_model.device   # always "cpu"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # FIX: use global whisper_model directly — no reassignment
                _, probs = whisper_model.detect_language(mel)

            # Top detected language
            detected_code = max(probs, key=probs.get)
            confidence    = round(probs[detected_code] * 100, 1)

            # FIX: low-confidence guard — avoid confidently wrong answers
            if confidence < 15.0:
                return jsonify({
                    "error": "Could not detect language confidently — speak more clearly"
                }), 422

            detected_name = WHISPER_TO_LANG.get(detected_code, detected_code)

            # Log top-5 for debugging
            top5 = sorted(probs.items(), key=lambda x: -x[1])[:5]
            print(f"✅ Detected: {detected_name} ({detected_code}) — {confidence}%")
            print(f"   Top 5: {[(k, round(v*100,1)) for k,v in top5]}")

            return jsonify({
                "detected_lang": detected_name,   # e.g. "hindi"
                "detected_code": detected_code,   # e.g. "hi"
                "confidence":    confidence,       # e.g. 94.2
            })

        # ------------------------------------------------------------------
        # MODE 2 — Transcription (speech → text)
        # ------------------------------------------------------------------
        text = None

        # Step 1: convert webm → 16-kHz WAV using Whisper's loader,
        #         then run Google STT (more accurate for Indian languages)
        try:
            import soundfile as sf

            audio_array = _whisper_mod.load_audio(tmp_path)
            wav_path    = tmp_path + ".wav"
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
                print(f"⚠️  Google STT network error: {e}")

            try:
                os.unlink(wav_path)
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️  Google STT pipeline error: {e}")

        # Step 2: Whisper fallback if Google STT failed
        if not text and whisper_model:
            print("🔄 Whisper transcription fallback...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = whisper_model.transcribe(
                    tmp_path,
                    language=src_code,
                    task="transcribe",
                    fp16=False,       # CPU-safe
                    verbose=False,
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
        try:
            os.unlink(tmp_path)
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
    })


# ==========================================
# /audio/<filename>  — backward compatibility
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