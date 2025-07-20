import logging
import os
import uuid
import subprocess
from collections import defaultdict

from flask import Flask, request, jsonify

from server.convertAudioWav import convert_youtube_to_wav
from fingerprintingPaste.fingerprinting import fingerprinting
from Query import init_db, store_fingerprints

# --- Configuration and directories ---
BASE_DIR  = os.path.dirname(__file__)
# ffmpeg.exe is located in the project root (one level above this server folder)
FFMPEG    = os.path.abspath(os.path.join(BASE_DIR, '..', 'ffmpeg.exe'))
AUDIO_DIR = os.path.join(BASE_DIR, "audios")
TMP_DIR   = os.path.join(BASE_DIR, "tmp")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# --- Flask app and database setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
# Initialize SQLite DB (check_same_thread=False must be set internally)
db = init_db()

# --- Home route serving minimal HTML interface ---
@app.route('/', methods=['GET'])
def home():
    return """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ShazamPython</title></head>
<body>
  <h1>🎵 ShazamPython</h1>

  <!-- Form to add a YouTube URL -->
  <form id="f1">
    YouTube URL: <input id="url" type="url" required>
    <button>➕ Add to DB</button>
  </form>

  <!-- Form to upload an audio snippet -->
  <form id="f2" enctype="multipart/form-data">
    Snippet: <input id="file" type="file" accept="audio/*" required>
    <button>🔍 Identify</button>
  </form>

  <!-- Output area -->
  <pre id="out" style="border:1px solid #ccc; padding:1em;"></pre>

  <!-- Basic JavaScript to call endpoints -->
  <script>
    const out = document.getElementById('out');

    document.getElementById('f1').onsubmit = async e => {
      e.preventDefault();
      out.textContent = '⏳ Processing...';
      let r = await fetch('/convert', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url: url.value})
      });
      out.textContent = await r.text();
    };

    document.getElementById('f2').onsubmit = async e => {
      e.preventDefault();
      out.textContent = '⏳ Identifying...(this may take a while)';
      let fd = new FormData();
      fd.append('file', file.files[0]);
      let r = await fetch('/identify', {method: 'POST', body: fd});
      out.textContent = await r.text();
    };
  </script>
</body>
</html>
    """

# --- Route to convert YouTube URL into WAV, fingerprint and store ---
@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json(force=True)
    url  = data.get("url")
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    try:
        # Download and convert YouTube to WAV
        wav_path = convert_youtube_to_wav(url, output_dir=AUDIO_DIR)
        app.logger.debug(f"[CONVERT] WAV generated: {wav_path}")

        # Generate fingerprints
        fp_map = fingerprinting(wav_path)
        app.logger.debug(f"[CONVERT] {len(fp_map)} unique hashes generated")

        # Store fingerprints in database
        store_fingerprints(fp_map, db)
        app.logger.debug("[CONVERT] Fingerprints stored in DB")

        return jsonify({"status": "ok", "song": os.path.basename(wav_path)}), 200

    except Exception as e:
        app.logger.error("Error in /convert", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Route to identify an uploaded snippet ---
@app.route('/identify', methods=['POST'])
def identify():
    # Clear temporary folder
    for fn in os.listdir(TMP_DIR):
        try:
            os.remove(os.path.join(TMP_DIR, fn))
        except:
            pass

    # Check for uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "Send snippet as 'file' field"}), 400

    # Save raw upload
    f = request.files['file']
    name, ext = os.path.splitext(f.filename.lower())
    tmp_raw = os.path.join(TMP_DIR, f"{uuid.uuid4()}{ext}")
    f.save(tmp_raw)
    app.logger.debug(f"[IDENTIFY] Received `{f.filename}` as `{os.path.basename(tmp_raw)}`")

    # Convert non-WAV formats to WAV (mono, 16kHz)
    if ext != '.wav':
        tmp_wav = tmp_raw.replace(ext, '.wav')
        subprocess.run([
            FFMPEG, '-y', '-i', tmp_raw,
            '-ac', '1', '-ar', '16000', tmp_wav
        ], check=True)
        os.remove(tmp_raw)
    else:
        tmp_wav = tmp_raw

    try:
        # Generate fingerprint map of snippet
        snip_map = fingerprinting(tmp_wav)
        total_hashes = len(snip_map)
        app.logger.debug(f"[IDENTIFY] Snippet produced {total_hashes} unique hashes")

        # Query DB and build candidates by offset
        cur = db.cursor()
        candidates = defaultdict(lambda: defaultdict(set))  # song_id -> offset -> set(hashes)
        for hsh, entries in snip_map.items():
            cur.execute(
                "SELECT song_id, time FROM fingerprints WHERE hash = ?",
                (hsh,)
            )
            for song_id, time_db in cur.fetchall():
                for time_snip, _ in entries:
                    offset = round(time_db - time_snip, 3)
                    candidates[song_id][offset].add(hsh)
        cur.close()
        app.logger.debug(f"[IDENTIFY] {len(candidates)} candidate songs found")

        # Build ranking (song, offset, vote_count)
        ranking = []
        for song_id, offs in candidates.items():
            for ofs, hs in offs.items():
                ranking.append((song_id, ofs, len(hs)))
        ranking.sort(key=lambda x: x[2], reverse=True)
        app.logger.debug(f"[IDENTIFY] Top 5 ranking: {ranking[:5]}")

        # No reliable match?
        if not ranking or ranking[0][2] == 0:
            return jsonify({"match": None}), 200

        # Return best match (strip file extension)
        best_song, best_ofs, best_votes = ranking[0]
        song_name = os.path.splitext(best_song)[0]
        confidence = best_votes / total_hashes
        return jsonify({
            "match": song_name,
            "offset": best_ofs,
            "votes": best_votes,
            "confidence": round(confidence, 3)
        }), 200

    except Exception as e:
        app.logger.error("Error in /identify", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- Run server ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
