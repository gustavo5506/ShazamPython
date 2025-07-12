from server.convertAudioWav import convert_youtube_to_wav
from fingerprintingPaste.fingerprinting import fingerprinting

if __name__ == "__main__":
    wav = convert_youtube_to_wav("https://www.youtube.com/watch?v=Qzl2wvW8R7A")
    fp_map = fingerprinting(wav)
    print("→ Total de hashes:", len(fp_map))
    # mostra os primeiros 10 pairs
    for h, entries in list(fp_map.items())[:10]:
        print(f"hash={h} → {entries}")
