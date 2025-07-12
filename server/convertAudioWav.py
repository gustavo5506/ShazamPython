import os, re
from yt_dlp import YoutubeDL

def sanitize_filename(s: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', s)

def convert_youtube_to_wav(youtube_url: str, output_dir="audios") -> str:
    os.makedirs(output_dir, exist_ok=True)
    template = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "noplaylist": True,
        "ffmpeg_location": os.path.abspath("ffmpeg.exe"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        orig = ydl.prepare_filename(info)

    # Corrige aqui para devolveres o .wav e não o .webm
    wav_fn = os.path.splitext(orig)[0] + ".wav"
    if not os.path.isfile(wav_fn):
        raise FileNotFoundError(f"Voz não encontrada: {wav_fn}")

    # Sanitiza o nome se tiveres caracteres especiais
    safe = sanitize_filename(os.path.basename(wav_fn))
    final = os.path.join(output_dir, safe)
    if final != wav_fn:
        os.replace(wav_fn, final)

    return final
