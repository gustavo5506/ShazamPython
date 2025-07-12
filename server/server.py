import logging
import os
import uuid
import sqlite3
from collections import defaultdict

from flask import Flask, request, jsonify

from server.convertAudioWav import convert_youtube_to_wav
from fingerprintingPaste.fingerprinting import fingerprinting
from Query import init_db, store_fingerprints

BASE_DIR  = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "audios")
TMP_DIR   = os.path.join(BASE_DIR, "tmp")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# ATENÇÃO: init_db deve usar check_same_thread=False internamente
db = init_db()

@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json(force=True)
    url  = data.get("url")
    if not url:
        return jsonify({"error": "Campo 'url' em falta"}), 400

    try:
        wav_path = convert_youtube_to_wav(url, output_dir=AUDIO_DIR)
        app.logger.debug(f"[CONVERT] WAV gerado: {wav_path}")

        fp_map = fingerprinting(wav_path)
        app.logger.debug(f"[CONVERT] {len(fp_map)} hashes únicos gerados")

        store_fingerprints(fp_map, db)
        app.logger.debug("[CONVERT] Fingerprints armazenados")

        return jsonify({"status":"ok","song":os.path.basename(wav_path)}),200
    except Exception as e:
        app.logger.error("Erro no /convert", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/identify', methods=['POST'])
def identify():
    # Limpa tmp de requests anteriores
    for fn in os.listdir(TMP_DIR):
        try: os.remove(os.path.join(TMP_DIR, fn))
        except: pass

    if "file" not in request.files:
        return jsonify({"error":"Envia o snippet como campo 'file'"}), 400

    f   = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    tmp_name = f"{uuid.uuid4()}{ext}"
    tmp_path = os.path.join(TMP_DIR, tmp_name)
    f.save(tmp_path)
    app.logger.debug(f"[IDENTIFY] chegou snippet `{f.filename}` → `{tmp_name}` ({os.path.getsize(tmp_path)} bytes)")

    try:
        # 1) fingerprinting do snippet
        snip_map = fingerprinting(tmp_path)
        total_hashes = len(snip_map)
        app.logger.debug(f"[IDENTIFY] snippet gerou {total_hashes} hashes únicos")

        # 2) consulta BD e constrói sets para cada offset
        cur = db.cursor()
        # candidatos[song_id][offset] = set(hashes)
        candidatos = defaultdict(lambda: defaultdict(set))

        for hsh, entries in snip_map.items():
            cur.execute("SELECT song_id, time FROM fingerprints WHERE hash = ?", (hsh,))
            for song_id, time_db in cur.fetchall():
                for time_snip, _ in entries:
                    offset = round(time_db - time_snip, 3)
                    candidatos[song_id][offset].add(hsh)
        cur.close()
        app.logger.debug(f"[IDENTIFY] músicas candidatas: {len(candidatos)}")

        # 3) constrói ranking com contagem de hashes únicos por offset
        ranking = []
        for song_id, offs in candidatos.items():
            for offset, hashes in offs.items():
                ranking.append((song_id, offset, len(hashes)))
        # ordena por votos desc
        ranking.sort(key=lambda x: x[2], reverse=True)
        app.logger.debug(f"[IDENTIFY] ranking (top 5): {ranking[:5]}")

        # 4) escolhe melhor
        if not ranking or ranking[0][2] == 0:
            app.logger.debug("[IDENTIFY] Sem match confiável")
            return jsonify({"match": None}), 200

        best_song, best_offset, best_votes = ranking[0]
        confidence = best_votes / total_hashes
        app.logger.debug(
            f"[IDENTIFY] melhor: {best_song} | offset={best_offset}s | "
            f"votes={best_votes} ({confidence:.2%} conf.)"
        )

        return jsonify({
            "match":   best_song,
            "offset":  best_offset,
            "votes":   best_votes,
            "confidence": round(confidence, 3)
        }), 200

    except Exception as e:
        app.logger.error("Erro no /identify", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
