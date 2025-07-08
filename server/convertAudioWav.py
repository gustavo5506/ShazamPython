import os
import re
import subprocess

def sanitize_filename(s: str) -> str:
    """
    Remove caracteres inválidos para nomes de ficheiro.
    """
    return re.sub(r'[\\/*?:"<>|]', '_', s)

def convert_youtube_to_wav(youtube_url: str, output_dir="audios") -> str:
    """
    Baixa apenas o áudio do YouTube (sem playlist), converte para WAV mono 16 kHz
    e guarda em output_dir com o título do vídeo como nome.

    Retorna o caminho do ficheiro gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Template para saida: usa o campo %(title)s que o yt-dlp preenche automaticamente,
    # mas sanitize_filename assegura que não haja caracteres proibidos
    template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # 1) baixa e extrai áudio em WAV, forçando mono + 16 kHz
    cmd = [
        "yt-dlp",
        "--no-playlist",                     # ignora &list=… e descarrega só esse vídeo
        "-x", "--audio-format", "wav",       # extrai para WAV
        "--audio-quality", "0",              # melhor qualidade
        "--postprocessor-args", "-ac 1 -ar 16000",
        "-o", template,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

    # 2) para saber exatamente o nome de saída (já com .wav), usamos --get-filename
    filename = subprocess.check_output([
        "yt-dlp",
        "--no-playlist",
        "--get-filename",
        "-o", template,
        youtube_url
    ], text=True).strip()

    # opcional: renomear para garantir filename sanitizado
    safe = sanitize_filename(os.path.basename(filename))
    final_path = os.path.join(output_dir, safe)
    if final_path != filename:
        os.replace(filename, final_path)

    return final_path
