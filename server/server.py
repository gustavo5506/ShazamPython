# server.py
import logging
from flask import Flask, request, jsonify

from convertAudioWav import convert_youtube_to_wav
from fingerprinting import fingerprinting
from Query import init_db, store_fingerprints

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
db = init_db()

@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json(force=True)
    url = data.get('url')
    if not url:
        return jsonify({'error': "Campo 'url' em falta"}), 400

    try:
        # 1) descarrega+converte → devolve caminho completo do .wav
        wav_path = convert_youtube_to_wav(url)
        app.logger.debug(f"WAV gerado em: {wav_path}")

        # 2) fingerprinting recebe caminho completo do ficheiro
        fp_map = fingerprinting(wav_path)
        app.logger.debug(f"Gerados {len(fp_map)} hashes")

        # 3) guarda na DB
        store_fingerprints(fp_map, db)

        return jsonify({'status': 'ok', 'song': wav_path}), 200

    except Exception as e:
        app.logger.error("Erro ao processar", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
