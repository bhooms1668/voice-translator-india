from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import os

app = Flask(__name__)
CORS(app)

# ==========================================
# ALL 22 INDIAN LANGUAGES + ENGLISH
# ==========================================

LANG_MAP = {

    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "telugu": "te",
    "marathi": "mr",
    "tamil": "ta",
    "gujarati": "gu",
    "urdu": "ur",
    "kannada": "kn",
    "odia": "or",
    "malayalam": "ml",
    "punjabi": "pa",
    "assamese": "as",
    "maithili": "mai",
    "sindhi": "sd",
    "kashmiri": "ks",
    "konkani": "gom",
    "nepali": "ne",
    "sanskrit": "sa",
    "dogri": "doi",
    "bodo": "brx",
    "manipuri": "mni-Mtei",
    "santhali": "sat"
}

# ==========================================
# LANGUAGES SUPPORTED BY gTTS AUDIO
# ==========================================

SUPPORTED_TTS = [
    "en", "hi", "bn", "te", "mr",
    "ta", "gu", "ur", "kn",
    "ml", "pa", "ne", "or"
]

# ==========================================
# TRANSLATE API
# ==========================================

@app.route("/translate", methods=["POST"])
def translate():

    try:

        data = request.json

        text = data.get("text")
        src = data.get("src")
        tgt = data.get("tgt")

        src_code = LANG_MAP.get(src, "en")
        tgt_code = LANG_MAP.get(tgt, "hi")

        # ==========================================
        # TRANSLATION
        # ==========================================

        translated = GoogleTranslator(
            source=src_code,
            target=tgt_code
        ).translate(text)

        audio_url = None

        # ==========================================
        # GENERATE AUDIO IF SUPPORTED
        # ==========================================

        if tgt_code in SUPPORTED_TTS:

            filename = f"{uuid.uuid4()}.mp3"

            tts = gTTS(
                text=translated,
                lang=tgt_code
            )

            tts.save(filename)

            audio_url = f"http://127.0.0.1:5000/audio/{filename}"

        return jsonify({
            "translated_text": translated,
            "audio_url": audio_url
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ==========================================
# AUDIO ROUTE
# ==========================================

@app.route("/audio/<filename>")
def audio(filename):

    return send_file(
        filename,
        mimetype="audio/mpeg"
    )

# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)