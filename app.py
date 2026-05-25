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


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
    "status": "ok"
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