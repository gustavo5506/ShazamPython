# Query.py

import sqlite3

def init_db(db_path="fingerprints.db"):
    # Permite usar esta conexão em várias threads
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS fingerprints (
        hash INTEGER,
        time REAL,
        song_id TEXT
      );
    """)
    conn.commit()
    return conn


def store_fingerprints(fingerprint_map, conn):
    """
    fingerprint_map: dict[int, list[(time, song_id)]]
    """
    cur = conn.cursor()
    for h, postings in fingerprint_map.items():
        for time1, song_id in postings:
            cur.execute(
                "INSERT INTO fingerprints(hash, time, song_id) VALUES (?, ?, ?)",
                (h, time1, song_id)
            )
    conn.commit()

def query_by_hash(h, conn):
    cur = conn.cursor()
    cur.execute("SELECT time, song_id FROM fingerprints WHERE hash = ?", (h,))
    return cur.fetchall()
