import os
import subprocess
import requests

# Aponta para o teu ffmpeg.exe na raiz do projeto:
FFMPEG      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg.exe'))
SERVER_URL  = 'http://localhost:5000'
BASE_DIR    = os.path.dirname(__file__)
SNIPPET_DIR = os.path.join(BASE_DIR, 'snippets')


def convert_to_wav(input_path):
    base, _  = os.path.splitext(input_path)
    wav_path = base + '.wav'
    print(f"→ Convertendo {os.path.basename(input_path)} para WAV…")
    subprocess.run([
        FFMPEG, '-y', '-i', input_path,
        '-ac', '1', '-ar', '16000', wav_path
    ], check=True)
    return wav_path


def identify_snippet(audio_path):
    ext = os.path.splitext(audio_path)[1].lower()
    wav = convert_to_wav(audio_path) if ext != '.wav' else audio_path

    print(f"→ Enviando `{os.path.basename(wav)}` para /identify...")
    resp = requests.post(f"{SERVER_URL}/identify",
                         files={'file': (os.path.basename(wav), open(wav,'rb'), 'audio/wav')})
    print("← Resposta recebida!")
    resp.raise_for_status()
    data = resp.json()

    if data.get('match'):
        print(
            f"🔍 Identificada: {data['match']} | "
            f"offset {data['offset']}s | votes {data['votes']} | "
            f"conf {data['confidence']:.0%}"
        )
    else:
        print("❌ Nenhum match encontrado.")
    if wav != audio_path:
        os.remove(wav)
    return data


if __name__ == '__main__':
    if not os.path.isdir(SNIPPET_DIR):
        print(f"Cria `{SNIPPET_DIR}` e coloca aí apenas o teu snippet .m4a")
        exit(1)

    # processa apenas o primeiro .m4a (o teu snippet)
    for fname in os.listdir(SNIPPET_DIR):
        if fname.lower().endswith('.m4a'):
            identify_snippet(os.path.join(SNIPPET_DIR, fname))
            break
