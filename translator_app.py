"""
translator_app.py — Flask web server for the Apatani translator.

Routes:
  GET  /           → serves the UI (index.html)
  POST /translate  → accepts JSON { text, direction }, returns JSON { result, method }
  GET  /health     → quick status check (useful for debugging)
"""

from flask import Flask, render_template, request, jsonify
from translator import translate_text, model

app = Flask(__name__)


@app.route('/')
def index():
    # render_template looks inside the /templates folder automatically
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()

    text      = data.get('text', '').strip()
    direction = data.get('direction', 'en_to_ap')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result, method = translate_text(text, direction)

    return jsonify({
        'result': result,
        'method': method,          # 'ml' or 'dictionary' — shown in the UI
        'model_available': model is not None
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'mode': 'ml + dictionary' if model is not None else 'dictionary only'
    })


if __name__ == '__main__':
    print("Starting Apatani Translator...")
    print("Visit http://localhost:5000")
    app.run(debug=True, port=5001)
