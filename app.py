from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import io
import base64

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
    "en":       "en-IN",
    "hi":       "hi-IN",
    "bn":       "bn-IN",
    "te":       "te-IN",
    "mr":       "mr-IN",
    "ta":       "ta-IN",
    "gu":       "gu-IN",
    "ur":       "ur-IN",
    "kn":       "kn-IN",
    "or":       "or-IN",
    "ml":       "ml-IN",
    "pa":       "pa-IN",
    "as":       "as-IN",
    "sa":       "sa-IN",
    "ne":       "ne-NP",
    "mai":      "hi-IN",
    "sd":       "ur-IN",
    "ks":       "ur-IN",
    "gom":      "hi-IN",
    "doi":      "hi-IN",
    "brx":      "hi-IN",
    "mni-Mtei": "hi-IN",
    "sat":      "hi-IN",
}

# ==========================================
# LANGUAGES SUPPORTED BY gTTS
# ==========================================

SUPPORTED_TTS = [
    "en", "hi", "bn", "te", "mr", "ta", "gu",
    "ur", "kn", "ml", "pa", "ne", "or", "as", "sa"
]



# ==========================================
# HELPER — convert browser audio to 16kHz WAV
# ==========================================
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
# ==========================================

# ==========================================
# /health
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok"
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