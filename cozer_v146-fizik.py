#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        YT GUARDIAN v2.0 — TAM ÇALIŞAN TEK-DOSYA MODERASYON SİSTEMİ        ║
║  @ShmirchikArt · NLP·BART·RL·Graf·Bayes·HMM·Oyun Kuramı·Stilometri·GMM   ║
║  Lokal AI (Ollama phi4:14b SADECE yorum için) · ROCm GPU · Selenium FF     ║
╚══════════════════════════════════════════════════════════════════════════════╝

KURULUM (Ubuntu):
  pip install flask flask-socketio flask-cors eventlet selenium yt-dlp requests
              numpy scipy scikit-learn torch transformers sentence-transformers
              spacy langdetect bertopic umap-learn hdbscan networkx
              python-louvain hmmlearn chromadb ollama pillow
  pip install fasttext-wheel   # veya fasttext
  python -m spacy download xx_ent_wiki_sm
  ollama pull phi4:14b

BAŞLATMA:
  python yt_guardian_full.py                    # Web paneli (port 5000)
  python yt_guardian_full.py --scrape           # Sadece kanal tarama
  python yt_guardian_full.py --analyze-all      # Tüm kullanıcıları analiz et
  python yt_guardian_full.py --port 8080        # Farklı port

KONFİGÜRASYON (yt_guardian_config.json — opsiyonel):
  {
    "yt_email":    "physicus93@hotmail.com",
    "yt_password": "%C7JdE4,)$MS;4'",
    "channel_url": "https://www.youtube.com/@ShmirchikArt/streams",
    "date_from":   "2023-01-01",
    "date_to":     "2026-12-31",
    "flask_port":  5000,
    "device":      "auto"
  }
"""

# ═══════════════════════════════════════════════════════════════════════════════
# § 1 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os, sys, re, json, time, math, hashlib, threading, logging, unicodedata

# 1) Imports bölümüne ekle
import shutil

import sqlite3, subprocess, argparse, random, traceback, base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter, deque
from typing import Optional, List, Dict, Tuple, Any
from unittest import result
import warnings

import socketio; warnings.filterwarnings("ignore")

from flask import app
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import wasserstein_distance


from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.naive_bayes import ComplementNB
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.model_selection import cross_val_score
import networkx as nx

# ── Opsiyonel kütüphaneler (graceful degradation) ────────────────────────────
def _try_import(name, pkg=None):
    try:
        import importlib
        return importlib.import_module(name), True
    except ImportError:
        return None, False

community_louvain, _LOUVAIN   = _try_import("community")
hmmlearn_hmm,     _HMM        = _try_import("hmmlearn.hmm")
chromadb_mod,     _CHROMA     = _try_import("chromadb")
sbert_mod,        _SBERT      = _try_import("sentence_transformers")
transformers_mod, _TRANS      = _try_import("transformers")
torch_mod,        _TORCH      = _try_import("torch")
spacy_mod,        _SPACY      = _try_import("spacy")
langdetect_mod,   _LANGDETECT = _try_import("langdetect")
fasttext_mod,     _FASTTEXT   = _try_import("fasttext")
bertopic_mod,     _BERTOPIC   = _try_import("bertopic")
ollama_mod,       _OLLAMA     = _try_import("ollama")
selenium_mod,     _SELENIUM   = _try_import("selenium")
flask_mod,        _FLASK      = _try_import("flask")
flask_sio,        _FLASK_SIO  = _try_import("flask_socketio")
flask_cors,       _FLASK_CORS = _try_import("flask_cors")

if _LANGDETECT:
    from langdetect import detect as langdetect_detect, DetectorFactory
    DetectorFactory.seed = 42
if _SBERT:
    from sentence_transformers import SentenceTransformer
if _TRANS:
    from transformers import pipeline as hf_pipeline
if _TORCH:
    import torch, torch.nn as nn, torch.optim as optim
if _SPACY:
    import spacy as _spacy_lib
if _FASTTEXT:
    import fasttext as _fasttext_lib
if _BERTOPIC:
    from bertopic import BERTopic
if _OLLAMA:
    import ollama as ollama_sdk
if _SELENIUM:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import (NoSuchElementException, TimeoutException,
                                             StaleElementReferenceException,
                                             InvalidSessionIdException,
                                             WebDriverException)
if _FLASK:
    from flask import Flask, render_template_string, request, jsonify
if _FLASK_SIO:
    from flask_socketio import SocketIO, emit
if _FLASK_CORS:
    from flask_cors import CORS
if _CHROMA:
    import chromadb

import requests as http_req

# ═══════════════════════════════════════════════════════════════════════════════
# § 2 — LOGGING & KONFİGÜRASYON
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("yt_guardian.log", encoding="utf-8")]
)
log = logging.getLogger("YTG")


# 2) _DEFAULT_CFG bloğunu bununla değiştir
_DEFAULT_CFG = {
    "yt_email":             "physicus93@hotmail.com",
    "yt_password":          "%C7JdE4,)$MS;4'",
    "channel_url":          "https://www.youtube.com/@ShmirchikArt/streams",
    "channel_handle":       "@ShmirchikArt",
    "db_path":              "yt_guardian.db",
    "chroma_path":          "./chromadb_data",
    "data_dir":             "./yt_data",
    "ollama_model":         "phi4:14b",
    "ollama_host":          "http://localhost:11434",
    "flask_port":           5000,
    "flask_secret":         "ytg_secret_2024_xk9m",
    "date_from":            "2023-01-01",
    "date_to":              "2026-12-31",
    "similarity_threshold": 0.65,
    "bot_threshold":        0.70,
    "hate_threshold":       0.65,
    "stalker_threshold":    0.55,
    "device":               "auto",
    "fasttext_model":       "lid.176.bin",
    "retrain_threshold":    500,
    "new_account_months":   6,
    "chromium_binary":      "",
    "chromium_user_data_dir": "",
    "chromium_profile_directory": "Default",
    "manual_login_timeout_sec": 180,
    "cookies_file":         "",
    "cookies_from_browser": "",
}

def load_config(cfg_file: str = "yt_guardian_config.json") -> dict:
    cfg = _DEFAULT_CFG.copy()
    if Path(cfg_file).exists():
        try:
            cfg.update(json.load(open(cfg_file, encoding="utf-8")))
        except Exception as e:
            log.warning("Config dosyası okunamadı: %s", e)
    # env override (güvenli)
    if os.environ.get("YT_EMAIL"):    cfg["yt_email"]    = os.environ["YT_EMAIL"]
    if os.environ.get("YT_PASSWORD"): cfg["yt_password"] = os.environ["YT_PASSWORD"]
    return cfg

CFG = load_config()

COLOR_MAP = {"GREEN":"#2ECC71","YELLOW":"#F1C40F","ORANGE":"#E67E22",
             "RED":"#E74C3C","BLUE":"#3498DB","PURPLE":"#9B59B6","CRIMSON":"#8B0000"}

THREAT_LABELS = [
    "antisemitic content","hate speech against Jewish people",
    "islamophobic content","white supremacist content",
    "groyper movement content","harassment and stalking behavior",
    "identity impersonation","coordinated bot attack",
    "neutral friendly message","spam content",
]
BOT_LABELS    = ["human-like conversation","spam or bot-like message"]
MOD_ACTIONS   = ["BAN","WARN","IGNORE","MONITOR"]
ACTOR_ACTIONS = ["BEHAVE","TROLL","IMPERSONATE","FLOOD"]
ACTION_NAMES  = {0:"HUMAN",1:"BOT",2:"HATER",3:"STALKER",4:"IMPERSONATOR",5:"COORDINATED"}

PAYOFF = np.array([
    [(-1,-5),(3,-3),(5,-4),(4,-3)],
    [( 1, 0),(-1,-1),(1,-2),(0,-1)],
    [( 2, 2),(-3, 3),(-4,4),(-3,3)],
    [( 1, 0),(2,-1),(3,-2),(2,-1)],
], dtype=object)

# ═══════════════════════════════════════════════════════════════════════════════
# § 3 — VERİTABANI (SQLite + ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════
_db_lock = threading.Lock()

def _get_conn() -> sqlite3.Connection:
    c = sqlite3.connect(CFG["db_path"], check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c

def init_db():
    with _get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS messages(
            id TEXT PRIMARY KEY, video_id TEXT NOT NULL, title TEXT,
            video_date TEXT, author TEXT NOT NULL, author_cid TEXT,
            message TEXT NOT NULL, timestamp INTEGER, lang TEXT,
            script_type TEXT, source_type TEXT,
            is_live INTEGER DEFAULT 0, deleted INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(author,message,content='messages',content_rowid='rowid');
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid,author,message) VALUES(new.rowid,new.author,new.message);
        END;
        CREATE TABLE IF NOT EXISTS user_profiles(
            author TEXT PRIMARY KEY, author_cid TEXT,
            msg_count INTEGER DEFAULT 0, human_score REAL DEFAULT 0.5,
            bot_prob REAL DEFAULT 0.0, hate_score REAL DEFAULT 0.0,
            stalker_score REAL DEFAULT 0.0, impersonator_prob REAL DEFAULT 0.0,
            antisemitism_score REAL DEFAULT 0.0, groyper_score REAL DEFAULT 0.0,
            identity_vector TEXT DEFAULT '{}', cluster_id INTEGER DEFAULT -1,
            threat_level TEXT DEFAULT 'GREEN', threat_score REAL DEFAULT 0.0,
            tfidf_json TEXT DEFAULT '{}', ngram_json TEXT DEFAULT '{}',
            typo_json TEXT DEFAULT '{}', pos_json TEXT DEFAULT '{}',
            temporal_json TEXT DEFAULT '{}',
            account_created TEXT, subscriber_count INTEGER DEFAULT 0,
            is_new_account INTEGER DEFAULT 0, video_count INTEGER DEFAULT 0,
            hmm_state TEXT DEFAULT 'NORMAL', q_state TEXT DEFAULT '00000',
            game_strategy TEXT DEFAULT 'BEHAVE', kalman_score REAL DEFAULT 0.0,
            gmm_component INTEGER DEFAULT -1, pagerank_score REAL DEFAULT 0.0,
            ollama_summary TEXT, ollama_action TEXT DEFAULT 'MONITOR',
            first_seen INTEGER, last_seen INTEGER,
            updated_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS identity_links(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_a TEXT NOT NULL, user_b TEXT NOT NULL,
            sim_score REAL, method TEXT, confidence REAL,
            emb_sim REAL DEFAULT 0, ngram_sim REAL DEFAULT 0,
            time_sim REAL DEFAULT 0, typo_sim REAL DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS graph_clusters(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER, members TEXT, algorithm TEXT,
            pagerank_leaders TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS game_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT,
            mod_action TEXT, actor_action TEXT, payoff_m REAL, payoff_a REAL,
            ts INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS training_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT,
            version INTEGER, accuracy REAL, f1_score REAL,
            dataset_size INTEGER,
            trained_at INTEGER DEFAULT (strftime('%s','now')), notes TEXT
        );
        CREATE TABLE IF NOT EXISTS dataset(
            id INTEGER PRIMARY KEY AUTOINCREMENT, msg_id TEXT, author TEXT,
            message TEXT, label TEXT, confirmed INTEGER DEFAULT 0, source TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS scraped_videos(
            video_id TEXT PRIMARY KEY, title TEXT, video_date TEXT,
            source_type TEXT, comment_count INTEGER DEFAULT 0,
            chat_count INTEGER DEFAULT 0,
            scraped_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS rag_cache(
            id INTEGER PRIMARY KEY AUTOINCREMENT, query_hash TEXT UNIQUE,
            query TEXT, response TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_msg_author ON messages(author);
        CREATE INDEX IF NOT EXISTS idx_msg_video  ON messages(video_id);
        CREATE INDEX IF NOT EXISTS idx_msg_ts     ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_msg_src    ON messages(source_type);
        CREATE INDEX IF NOT EXISTS idx_up_threat  ON user_profiles(threat_level);
        CREATE INDEX IF NOT EXISTS idx_up_threat_score ON user_profiles(threat_score DESC);
        CREATE INDEX IF NOT EXISTS idx_link_ab    ON identity_links(user_a,user_b);
        CREATE INDEX IF NOT EXISTS idx_ds_conf    ON dataset(confirmed,created_at);
        """)
        _migrate_legacy_schema(c)
    log.info("✅ SQLite hazır: %s", CFG["db_path"])

def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r["name"]) for r in rows}
    except Exception:
        return set()

def _migrate_legacy_schema(conn: sqlite3.Connection):
    """
    Eski sürümlerdeki kolon adlarını yeni şema ile uyumlu hale getir.
    Özellikle `user_profiles.username` -> `user_profiles.author` geçişini düzeltir.
    """
    cols = _table_columns(conn, "user_profiles")
    if not cols:
        return

    if "author" not in cols and "username" in cols:
        try:
            conn.execute("ALTER TABLE user_profiles RENAME COLUMN username TO author")
            log.info("✅ DB migration: user_profiles.username -> author")
        except Exception as e:
            log.warning("Kolon rename başarısız, fallback uygulanıyor: %s", e)
            try:
                conn.execute("ALTER TABLE user_profiles ADD COLUMN author TEXT")
                conn.execute("UPDATE user_profiles SET author=username WHERE author IS NULL OR author=''")
            except Exception as e2:
                log.error("DB migration başarısız: author kolonu oluşturulamadı: %s", e2)

    cols = _table_columns(conn, "user_profiles")
    required_user_profile_columns = {
        # Eski veritabanlarında bulunmayan ama yeni kod yollarında kullanılan kolonlar.
        # Bu listeyi tek merkezde tutarak query bazlı "hard-code" kırılmalarını önleriz.
        "author_cid": "TEXT",
        "subscriber_count": "INTEGER DEFAULT 0",
        "account_created": "TEXT",
        "is_new_account": "INTEGER DEFAULT 0",
        "video_count": "INTEGER DEFAULT 0",
        "pagerank_score": "REAL DEFAULT 0.0",
        "ollama_summary": "TEXT",
        "ollama_action": "TEXT DEFAULT 'MONITOR'",
    }

    for col_name, col_def in required_user_profile_columns.items():
        if col_name in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {col_name} {col_def}")
            log.info("✅ DB migration: user_profiles.%s eklendi", col_name)
        except Exception as e:
            log.warning("user_profiles.%s kolonu eklenemedi: %s", col_name, e)

def db_exec(sql: str, params: tuple = (), fetch: str = None):
    with _db_lock:
        with _get_conn() as c:
            cur = c.execute(sql, params)
            if fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None
            if fetch == "all":
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            return cur.lastrowid

def upsert_message(msg: dict):
    sql = ("INSERT OR IGNORE INTO messages"
           "(id,video_id,title,video_date,author,author_cid,message,timestamp,"
           "lang,script_type,source_type,is_live) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")
    db_exec(sql, (msg["msg_id"],msg.get("video_id",""),msg.get("title",""),
                  msg.get("video_date",""),msg["author"],msg.get("author_channel_id",""),
                  msg["message"],msg.get("timestamp_utc",0),msg.get("lang_detected",""),
                  msg.get("script",""),msg.get("source_type","comment"),
                  int(msg.get("is_live",False))))

def upsert_profile(author: str, upd: dict):
    if not db_exec("SELECT 1 FROM user_profiles WHERE author=?", (author,), fetch="one"):
        db_exec("INSERT OR IGNORE INTO user_profiles(author) VALUES(?)", (author,))
    if upd:
        sets = ", ".join(f"{k}=?" for k in upd)
        db_exec(f"UPDATE user_profiles SET {sets}, updated_at=strftime('%s','now') WHERE author=?",
                tuple(upd.values())+(author,))

def get_user_msgs(author: str) -> List[Dict]:
    rows = db_exec("SELECT * FROM messages WHERE author=? AND deleted=0 ORDER BY timestamp",
                   (author,), fetch="all")
    return [dict(r) for r in rows] if rows else []

# ── ChromaDB ─────────────────────────────────────────────────────────────────
_chroma_client = _ch_msgs = _ch_users = None

def init_chroma():
    global _chroma_client, _ch_msgs, _ch_users
    if not _CHROMA: return
    try:
        Path(CFG["chroma_path"]).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CFG["chroma_path"])
        _ch_msgs  = _chroma_client.get_or_create_collection("messages",  metadata={"hnsw:space":"cosine"})
        _ch_users = _chroma_client.get_or_create_collection("user_profiles", metadata={"hnsw:space":"cosine"})
        log.info("✅ ChromaDB hazır: %s", CFG["chroma_path"])
    except Exception as e:
        log.warning("ChromaDB başlatılamadı: %s", e)

def chroma_upsert(collection, uid: str, emb: list, meta: dict):
    if collection is None or not emb: return
    try:
        safe = {k: str(v)[:500] for k,v in meta.items()}
        collection.upsert(ids=[uid], embeddings=[emb], metadatas=[safe])
    except: pass

def chroma_query(collection, emb: list, n: int = 10) -> list:
    if collection is None or not emb: return []
    try:
        r = collection.query(query_embeddings=[emb], n_results=min(n,collection.count()))
        return [{"id":r["ids"][0][i],"dist":r["distances"][0][i],"meta":r["metadatas"][0][i]}
                for i in range(len(r["ids"][0]))]
    except: return []

# ═══════════════════════════════════════════════════════════════════════════════
# § 4 — MODEL YÖNETİMİ (Lazy Loading)
# ═══════════════════════════════════════════════════════════════════════════════
_models = {}
_mlock  = threading.Lock()

def _device():
    d = CFG.get("device","auto")
    if d != "auto": return d
    if _TORCH:
        if torch.cuda.is_available(): return "cuda"
        try:
            if torch.version.hip: return "cuda"  # ROCm
        except: pass
    return "cpu"

DEVICE = _device()

def get_sbert():
    with _mlock:
        if "sbert" not in _models:
            if not _SBERT: return None
            try:
                _models["sbert"] = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2", device=DEVICE)
                log.info("✅ SBERT yüklendi [%s]", DEVICE)
            except Exception as e:
                log.error("SBERT yüklenemedi: %s", e); _models["sbert"] = None
        return _models["sbert"]

def get_bart():
    with _mlock:
        if "bart" not in _models:
            if not _TRANS: return None
            try:
                dev = 0 if DEVICE in ("cuda","mps") else -1
                _models["bart"] = hf_pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli", device=dev)
                log.info("✅ BART zero-shot yüklendi [%s]", DEVICE)
            except Exception as e:
                log.error("BART yüklenemedi: %s", e); _models["bart"] = None
        return _models["bart"]

def get_spacy():
    with _mlock:
        if "spacy" not in _models:
            if not _SPACY: return None
            try:
                _models["spacy"] = _spacy_lib.load("xx_ent_wiki_sm")
                log.info("✅ spaCy xx_ent_wiki_sm yüklendi")
            except:
                try:
                    _models["spacy"] = _spacy_lib.blank("xx")
                except: _models["spacy"] = None
        return _models["spacy"]

def get_fasttext():
    with _mlock:
        if "fasttext" not in _models:
            if not _FASTTEXT: return None
            mp = CFG.get("fasttext_model","lid.176.bin")
            if not Path(mp).exists(): return None
            try:
                _models["fasttext"] = _fasttext_lib.load_model(mp)
                log.info("✅ fasttext yüklendi")
            except Exception as e:
                log.warning("fasttext yüklenemedi: %s", e); _models["fasttext"] = None
        return _models["fasttext"]

# ═══════════════════════════════════════════════════════════════════════════════
# § 5 — NORMALİZASYON & DİL TESPİTİ
# ═══════════════════════════════════════════════════════════════════════════════
_SCRIPT_RE = {
    "Hebrew":     re.compile(r"[\u0590-\u05FF]"),
    "Arabic":     re.compile(r"[\u0600-\u06FF]"),
    "Cyrillic":   re.compile(r"[\u0400-\u04FF]"),
    "Devanagari": re.compile(r"[\u0900-\u097F]"),
    "CJK":        re.compile(r"[\u4E00-\u9FFF]"),
}
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F"
    r"\U00002300-\U000023FF\U0001F600-\U0001F64F]", re.UNICODE)

def detect_script(text: str) -> str:
    for name, pat in _SCRIPT_RE.items():
        if pat.search(text): return name
    return "Latin"

def detect_lang(text: str) -> Tuple[str, float]:
    ft = get_fasttext()
    if ft:
        try:
            labels, probs = ft.predict(text.replace("\n"," ")[:300], k=1)
            return labels[0].replace("__label__",""), float(probs[0])
        except: pass
    if _LANGDETECT:
        try: return langdetect_detect(text[:300]), 0.75
        except: pass
    return "und", 0.0

def norm_text(raw: str) -> str:
    t = unicodedata.normalize("NFC", raw)
    t = re.sub(r"&amp;","&",t); t = re.sub(r"&lt;","<",t)
    t = re.sub(r"&gt;",">",t); t = re.sub(r"&quot;",'"',t)
    t = re.sub(r"&#39;","'",t)
    return t.strip()

def norm_username(name: str) -> str:
    return unicodedata.normalize("NFKC", name).lower().strip()

def msg_id(video_id: str, author: str, ts: int, message: str) -> str:
    return hashlib.sha256(f"{video_id}|{author}|{ts}|{message}".encode()).hexdigest()

def extract_emojis(text: str) -> List[str]:
    return _EMOJI_RE.findall(text)

def process_raw(raw: dict) -> Optional[dict]:
    try:
        message = norm_text(raw.get("message","") or raw.get("text","") or "")
        if not message: return None
        author = (raw.get("author","") or raw.get("username","")).strip()
        if not author: return None
        ts = int(raw.get("timestamp_utc", raw.get("timestamp",0)) or 0)
        lang, conf = detect_lang(message[:200])
        return {
            "msg_id":            msg_id(raw.get("video_id",""), author, ts, message),
            "video_id":          raw.get("video_id",""),
            "title":             raw.get("title",""),
            "video_date":        raw.get("video_date",""),
            "author":            author,
            "author_channel_id": raw.get("author_channel_id",""),
            "message":           message,
            "timestamp_utc":     ts,
            "lang_detected":     lang,
            "lang_confidence":   conf,
            "script":            detect_script(message),
            "source_type":       raw.get("source_type","comment"),
            "emojis":            extract_emojis(message),
            "is_live":           raw.get("is_live",False),
        }
    except: return None

# ═══════════════════════════════════════════════════════════════════════════════
# § 6 — YOUTUBE SCRAPER (yt-dlp + Selenium)
# ═══════════════════════════════════════════════════════════════════════════════
_driver    = None
_drv_lock  = threading.Lock()
_acct_cache: Dict[str,dict] = {}

def _sanitize_chromium_env():
    bad = []
    for key in ("CHROME_BINARY", "CHROMIUM_BINARY"):
        val = os.environ.get(key, "").strip()
        if not val:
            continue
        p = Path(val)
        if not p.exists() or p.is_dir() or not os.access(str(p), os.X_OK):
            bad.append((key, val))
            os.environ.pop(key, None)
    for key, val in bad:
        log.warning("Geçersiz %s temizlendi: %s", key, val)


# 3) make_driver'dan önce bu yardımcıları ekle
def _is_chromium_binary(path: str) -> bool:
    if not path:
        return False
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=8)
        out = f"{r.stdout} {r.stderr}".lower()
        return r.returncode == 0 and any(k in out for k in ("chromium", "chrome"))
    except Exception:
        return False


def _resolve_chromium_binary() -> str:
    candidates = []

    for key in ("CHROME_BINARY", "CHROMIUM_BINARY", "CHROMIUM_BIN"):
        v = os.environ.get(key, "").strip()
        if v:
            candidates.append(v)

    for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
            real = os.path.realpath(p)
            if real and real != p:
                candidates.append(real)

    candidates.extend([
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/usr/bin/google-chrome",
        "/snap/bin/chromium",
    ])

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        resolved = os.path.realpath(cand)
        for probe in [cand, resolved]:
            p = Path(probe)
            if p.exists() and p.is_file() and os.access(str(p), os.X_OK) and _is_chromium_binary(str(p)):
                return str(p)

    return ""


def _yt_dlp_base_cmd() -> List[str]:
    ytdlp_bin = shutil.which("yt-dlp")
    cmd = [
        ytdlp_bin if ytdlp_bin else sys.executable,
        "-m", "yt_dlp",
        "--no-warnings",
        "--ignore-errors",
        "--skip-download",
    ]
    if ytdlp_bin:
        cmd = cmd[:1] + cmd[3:]

    cookie_file = (CFG.get("cookies_file") or "").strip()
    if cookie_file and Path(cookie_file).exists():
        cmd += ["--cookies", cookie_file]
    else:
        browser = (CFG.get("cookies_from_browser") or "").strip()
        if browser:
            cmd += ["--cookies-from-browser", browser]

    return cmd


def _strip_cookies_from_browser_args(cmd: List[str]) -> List[str]:
    """Komuttan --cookies-from-browser <browser> çiftini güvenli biçimde kaldır."""
    out: List[str] = []
    skip_next = False
    for i, part in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if part == "--cookies-from-browser":
            if i + 1 < len(cmd):
                skip_next = True
            continue
        out.append(part)
    return out


def _run_ytdlp(cmd: List[str], timeout: int):
    """
    yt-dlp çalıştır.
    Eğer tarayıcı cookie DB hatası alırsa aynı komutu cookies-from-browser olmadan tekrar dener.
    """
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    stderr = (res.stderr or "").lower()
    cookie_db_err = (
        "could not find chromium cookies database" in stderr
        or "cookies-from-browser" in stderr and "error" in stderr
    )
    has_cfb = "--cookies-from-browser" in cmd

    if res.returncode != 0 and has_cfb and cookie_db_err:
        log.warning("yt-dlp browser cookie hatası; cookies-from-browser olmadan tekrar deneniyor.")
        retry_cmd = _strip_cookies_from_browser_args(cmd)
        res = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=timeout)
    return res


def export_cookies_from_driver(driver, cookie_file: str = None) -> bool:
    if not driver:
        return False
    path = Path(cookie_file or CFG.get("cookies_file", "") or (Path(CFG["data_dir"]) / "cookies.txt"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cookies = driver.get_cookies()
        if not cookies:
            return False

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Netscape HTTP Cookie File\n")
            for c in cookies:
                domain = c.get("domain", "")
                include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
                pth = c.get("path", "/")
                secure = "TRUE" if c.get("secure") else "FALSE"
                expiry = int(c.get("expiry", 0) or 0)
                name = c.get("name", "")
                value = c.get("value", "")
                f.write(
                    f"{domain}\t{include_subdomains}\t{pth}\t{secure}\t{expiry}\t{name}\t{value}\n"
                )

        CFG["cookies_file"] = str(path)
        log.info("✅ Cookies export edildi: %s", path)
        return True
    except Exception as e:
        log.warning("Cookie export başarısız: %s", e)
        return False
        

        
        
        
        
        

## 4) make_driver() — Anti-bot stealth + hızlı bağlantı
def make_driver(headless: bool = False):
    """
    Chromium WebDriver başlat.
    - Otomasyon tespitini engelle (standalone script ile uyumlu)
    - CFG üzerinden binary / profil / headless kontrolü
    - CDP ile navigator.webdriver gizleme
    """
    if not _SELENIUM:
        return None
    try:
        _sanitize_chromium_env()
        opts = ChromeOptions()

        # ── Headless ──────────────────────────────────────────────────────────
        if headless:
            opts.add_argument("--headless=new")

        # ── Otomasyon tespiti engelleme (standalone script ile örtüşen ayarlar) ─
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--start-maximized")
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--disable-notifications")
        opts.add_argument("--disable-popup-blocking")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--lang=tr-TR")

        # User-agent: gerçek Chromium Linux gibi görün
        ua = CFG.get("chromium_user_agent", "").strip()
        if not ua:
            ua = (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        opts.add_argument(f"--user-agent={ua}")

        # Experimental: otomasyon izlerini sil
        opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        opts.add_experimental_option("useAutomationExtension", False)

        # Profil tercihleri
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "credentials_enable_service":       False,
            "profile.password_manager_enabled": False,
        }
        opts.add_experimental_option("prefs", prefs)

        # ── Kalıcı profil (cookie/oturum kalıcılığı) ─────────────────────────
        user_data_dir = (CFG.get("chromium_user_data_dir") or "").strip()
        profile_dir   = (CFG.get("chromium_profile_directory") or "Default").strip()
        if user_data_dir:
            opts.add_argument(f"--user-data-dir={user_data_dir}")
            opts.add_argument(f"--profile-directory={profile_dir}")
            log.info("Chromium kalıcı profil: %s / %s", user_data_dir, profile_dir)

        # ── Binary çözümü ────────────────────────────────────────────────────
        chrome_bin = (CFG.get("chromium_binary") or os.environ.get("CHROMIUM_BIN") or "").strip()
        if chrome_bin and _is_chromium_binary(chrome_bin):
            opts.binary_location = chrome_bin
            log.info("Chromium binary (CFG/env): %s", chrome_bin)
        else:
            resolved = _resolve_chromium_binary()
            if resolved:
                opts.binary_location = resolved
                log.info("Chromium binary (otomatik): %s", resolved)
            else:
                log.warning("Chromium binary bulunamadı; Selenium Manager fallback.")

        drv = webdriver.Chrome(options=opts)
        drv.set_page_load_timeout(int(CFG.get("page_load_timeout", 60)))

        # ── CDP: navigator.webdriver = undefined ─────────────────────────────
        try:
            drv.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
                Object.defineProperty(navigator, 'webdriver',   {get: () => undefined});
                Object.defineProperty(navigator, 'plugins',     {get: () => [1,2,3,4,5]});
                Object.defineProperty(navigator, 'languages',   {get: () => ['tr-TR','tr','en-US','en']});
                window.chrome = {runtime: {}};
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({
                        query: p => Promise.resolve({
                            state: p.name === 'notifications' ? Notification.permission : 'granted'
                        })
                    })
                });
            """})
        except Exception as cdp_err:
            log.debug("CDP stealth script uygulanamadı: %s", cdp_err)

        log.info("✅ Chromium WebDriver başlatıldı (stealth mod)")
        return drv

    except Exception as e:
        log.error("Chromium başlatılamadı: %s", e)
        return None


def is_driver_alive(driver) -> bool:
    if not driver:
        return False
    try:
        _ = driver.current_url
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE/YOUTUBE GİRİŞ YARDIMCILARI
# Standalone script mantığı + çok-strateji güvenlik ağı.
# Hiçbir class adı / hard-coded değer YOK — tüm selector'lar
# DevTools'tan gözlemlenen jsname / id / name / type üzerinden çalışır.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_click(driver, element):
    """Normal click başarısız olursa JS fallback."""
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def _locate_first(wait, strategies, condition=EC.element_to_be_clickable):
    """
    Strateji listesini sırayla dene; ilk bulunan elementi döndür.
    condition: EC.element_to_be_clickable (default) veya EC.presence_of_element_located
    """
    for by, sel in strategies:
        try:
            el = wait.until(condition((by, sel)))
            if el:
                return el
        except Exception:
            continue
    return None


# E-posta alanı — DevTools: id="identifierId", jsname="YPqjbf", name="identifier"
_EMAIL_STRATEGIES = [
    (By.ID,          "identifierId"),           # ← standalone script'teki asıl ID (lowercase d)
    (By.CSS_SELECTOR, 'input[jsname="YPqjbf"]'),
    (By.CSS_SELECTOR, 'input[name="identifier"]'),
    (By.CSS_SELECTOR, 'input[type="email"]'),
    (By.CSS_SELECTOR, 'input[autocomplete="username webauthn"]'),
]

# "Sonraki" butonu — DevTools: jsname="LgbsSe", span jsname="V67aGc"
_NEXT_BTN_STRATEGIES = [
    (By.CSS_SELECTOR, 'button[jsname="LgbsSe"]'),
    (By.XPATH,        '//button[.//span[@jsname="V67aGc"]]'),
    (By.XPATH,        '//span[@jsname="V67aGc"]/ancestor::button'),
    (By.ID,           "identifierNext"),
    (By.CSS_SELECTOR, 'div#identifierNext button'),
    (By.XPATH, '//button[contains(@class,"VfPpkd") and .//*[contains(text(),"Sonraki") or contains(text(),"Next")]]'),
]

# Şifre alanı — DevTools + standalone: By.NAME "Passwd" en güvenilir
_PASS_STRATEGIES = [
    (By.NAME,         "Passwd"),                   # ← standalone script'teki selector
    (By.CSS_SELECTOR, 'input[type="password"][name="Passwd"]'),
    (By.CSS_SELECTOR, 'input[type="password"][jsname="YPqjbf"]'),
    (By.CSS_SELECTOR, 'input[type="password"][name="password"]'),
    (By.CSS_SELECTOR, 'input[type="password"]'),
    (By.XPATH,        '//input[@type="password"]'),
]

# Şifre "Sonraki" butonu
_PASS_NEXT_STRATEGIES = [
    (By.CSS_SELECTOR, 'button[jsname="LgbsSe"]'),
    (By.XPATH,        '//button[.//span[@jsname="V67aGc"]]'),
    (By.ID,           "passwordNext"),
    (By.CSS_SELECTOR, 'div#passwordNext button'),
    (By.XPATH,        '//button[@type="submit"]'),
]


def _human_type(element, text: str, driver=None):
    """
    Metni insan benzeri hızda yaz.
    Önce JS ile alanı temizle (visibility sorunundan kaçınmak için).
    """
    if driver:
        try:
            driver.execute_script("arguments[0].value = '';", element)
        except Exception:
            pass
    element.click()
    time.sleep(random.uniform(0.2, 0.5))
    for ch in text:
        element.send_keys(ch)
        time.sleep(random.uniform(0.04, 0.13))


def _save_screenshot(driver, name: str):
    """Debug ekran görüntüsü yt_data/ klasörüne kaydet."""
    try:
        p = str(Path(CFG.get("data_dir", "yt_data")) / name)
        driver.save_screenshot(p)
        log.info("📸 Ekran görüntüsü: %s", p)
    except Exception:
        pass


def yt_login(driver, email: str, password: str) -> bool:
    """
    Google/YouTube otomatik giriş.
    ─────────────────────────────
    Standalone script mantığı YT Guardian'a entegre edildi:
      • Aynı otomasyon-kaçınma seçenekleri (make_driver ile ortaklaşa)
      • E-posta: identifierId → Keys.ENTER (standalone gibi)
      • Buton yoksa Keys.ENTER fallback
      • Şifre: By.NAME "Passwd" öncelikli (standalone gibi) → Keys.ENTER
      • 180 saniyelik polling döngüsü (manuel tamamlama için)
      • Başarılı girişte cookie export (yt-dlp için)

    Kimlik bilgileri SADECE CFG veya parametre üzerinden gelir — hard-code yok.
    """
    if not driver:
        return False

    data_dir = Path(CFG.get("data_dir", "yt_data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    wait_sec = int(CFG.get("selenium_wait", 15))   # standalone default: 15s
    manual_timeout = int(CFG.get("manual_login_timeout_sec", 180) or 180)

    try:
        if not is_driver_alive(driver):
            log.error("yt_login: Chromium oturumu kapalı.")
            return False

        # ── Oturum zaten açık mı? ────────────────────────────────────────────
        driver.get("https://www.youtube.com")
        time.sleep(2)
        cur = driver.current_url or ""
        if "youtube.com" in cur and "accounts.google.com" not in cur:
            try:
                driver.find_element(By.CSS_SELECTOR, "button#avatar-btn,yt-img-shadow#avatar")
                log.info("✅ YouTube oturumu aktif — giriş atlandı")
                return True
            except Exception:
                pass  # avatar yoksa yeniden giriş yap

        # ── Manuel mod ───────────────────────────────────────────────────────
        if not email or not password:
            log.warning("E-posta/şifre sağlanmadı — manuel giriş bekleniyor (%ds)...", manual_timeout)
            driver.get("https://accounts.google.com/signin")
            deadline = time.time() + manual_timeout
            while time.time() < deadline:
                c = (driver.current_url or "").lower()
                if "youtube.com" in c and "accounts.google.com" not in c:
                    log.info("✅ Manuel giriş algılandı")
                    export_cookies_from_driver(driver)
                    return True
                time.sleep(2)
            log.error("⛔ Manuel giriş zaman aşımı")
            return False

        # ── Otomatik giriş ───────────────────────────────────────────────────
        # ServiceLogin: YouTube servisine yönelik — bot tespiti daha düşük
        login_url = (
            "https://accounts.google.com/ServiceLogin"
            "?service=youtube&uilel=3&passive=true"
            "&continue=https%3A%2F%2Fwww.youtube.com%2Fsignin"
            "%3Faction_handle_signin%3Dtrue%26app%3Ddesktop%26hl%3Dtr%26next%3D%252F"
            "&hl=tr"
        )
        driver.get(login_url)
        time.sleep(2)
        wait = WebDriverWait(driver, wait_sec)

        # ── ADIM 1 — E-posta ─────────────────────────────────────────────────
        log.info("[Login 1/4] E-posta giriliyor...")
        ef = _locate_first(wait, _EMAIL_STRATEGIES, EC.presence_of_element_located)
        if ef is None:
            # Fallback: doğrudan identifier sayfasına git
            driver.get("https://accounts.google.com/signin/v2/identifier?hl=tr&flowName=GlifWebSignIn")
            time.sleep(2)
            ef = _locate_first(wait, _EMAIL_STRATEGIES, EC.presence_of_element_located)
        if ef is None:
            _save_screenshot(driver, "login_no_email_field.png")
            log.error("E-posta alanı bulunamadı")
            return False

        _human_type(ef, email, driver)
        time.sleep(0.5)

        # ── ADIM 2 — E-posta Sonraki ─────────────────────────────────────────
        log.info("[Login 2/4] 'Sonraki' butonu...")
        next_btn = _locate_first(wait, _NEXT_BTN_STRATEGIES)
        if next_btn:
            time.sleep(0.3)
            _safe_click(driver, next_btn)
        else:
            # Standalone fallback: Keys.ENTER
            log.warning("'Sonraki' butonu bulunamadı — Keys.ENTER fallback")
            ef.send_keys(Keys.ENTER)
        log.info("  ↳ E-posta gönderildi")

        # ── ADIM 3 — Şifre ───────────────────────────────────────────────────
        log.info("[Login 3/4] Şifre alanı bekleniyor...")
        time.sleep(2.5)
        pf = _locate_first(wait, _PASS_STRATEGIES)
        if pf is None:
            _save_screenshot(driver, "login_no_pass_field.png")
            log.error("Şifre alanı bulunamadı — URL: %s", driver.current_url)
            # Son çare: manuel tamamlama (standalone'daki 180s loop)
            log.warning("180 saniyelik manuel tamamlama bekleniyor...")
            deadline = time.time() + manual_timeout
            while time.time() < deadline:
                c = (driver.current_url or "").lower()
                if "youtube.com" in c and "accounts.google.com" not in c:
                    export_cookies_from_driver(driver)
                    return True
                time.sleep(2)
            return False

        _human_type(pf, password, driver)
        time.sleep(0.5)

        # ── ADIM 4 — Şifre Sonraki ───────────────────────────────────────────
        log.info("[Login 4/4] Şifre 'Sonraki' butonu...")
        pw_btn = _locate_first(wait, _PASS_NEXT_STRATEGIES)
        if pw_btn:
            time.sleep(0.3)
            _safe_click(driver, pw_btn)
        else:
            log.warning("Şifre butonu bulunamadı — Keys.ENTER fallback")
            pf.send_keys(Keys.ENTER)

        # ── Oturum doğrulama + CAPTCHA/2FA bekleme ───────────────────────────
        time.sleep(4)
        cur = driver.current_url or ""
        if "accounts.google.com" in cur:
            _save_screenshot(driver, "login_challenge.png")
            log.warning(
                "⚠️ Google doğrulama ekranı (CAPTCHA / 2FA?). "
                "Ekran: yt_data/login_challenge.png — %ds içinde manuel tamamlayın.",
                manual_timeout
            )
            deadline = time.time() + manual_timeout
            while time.time() < deadline:
                if "accounts.google.com" not in (driver.current_url or ""):
                    log.info("✅ Doğrulama manuel tamamlandı")
                    break
                time.sleep(3)

        # YouTube'a yönlendir
        driver.get("https://www.youtube.com")
        time.sleep(2)
        ok = (
            "youtube.com"          in (driver.current_url or "") and
            "accounts.google.com" not in (driver.current_url or "")
        )

        if ok:
            export_cookies_from_driver(driver)
            log.info("✅ YouTube girişi başarılı: %s", email)
        else:
            _save_screenshot(driver, "login_failed.png")
            log.error("❌ YouTube girişi başarısız | URL: %s", driver.current_url)

        return ok

    except (InvalidSessionIdException, WebDriverException) as e:
        log.error("yt_login — Chromium oturumu düştü: %s", e)
        return False
    except Exception as e:
        log.error("yt_login — beklenmedik hata: %s", e)
        _save_screenshot(driver, "login_error.png")
        traceback.print_exc()
        return False


# ── TARİH ÇÖZÜCÜ ─────────────────────────────────────────────────────────────
def _parse_relative_date(text: str) -> str:
    """
    'Streamed 5 hours ago', '3 months ago', '2 years ago' gibi
    görece tarih metinlerini YYYYMMDD formatına çevirir.
    Günümüz referans alınarak geriye doğru hesaplanır.
    YouTube'un <span class="inline-metadata-item ..."> etiketi içindeki
    "Streamed X ago" metni bu fonksiyonla işlenir.
    """
    if not text:
        return ""
    now = datetime.now(tz=timezone.utc)
    t   = text.lower().strip()
    m   = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", t)
    if not m:
        return ""
    n, unit = int(m.group(1)), m.group(2)
    delta_map = {
        "second": timedelta(seconds=n),
        "minute": timedelta(minutes=n),
        "hour":   timedelta(hours=n),
        "day":    timedelta(days=n),
        "week":   timedelta(weeks=n),
        "month":  timedelta(days=n * 30),
        "year":   timedelta(days=n * 365),
    }
    dt = now - delta_map.get(unit, timedelta(0))
    return dt.strftime("%Y%m%d")


def _candidate_channel_urls(channel_url: str) -> List[str]:
    """
    HEM /videos HEM /streams sayfalarını döndür.
    Canlı yayın tekrarları (/streams) ve normal video yorumları (/videos)
    her ikisi de eksiksiz çekilsin. Erken çıkış yok.
    """
    url = (channel_url or "").strip().rstrip("/")
    if not url:
        return []

    if any(url.endswith(sfx) for sfx in ("/streams", "/videos", "/live", "/shorts")):
        base = url.rsplit("/", 1)[0]
    else:
        base = url

    # /videos önce (daha fazla genel yorum), sonra /streams (canlı yayın replay)
    candidates = []
    for sfx in ("/videos", "/streams", "/live", ""):
        cand = base + sfx
        if cand not in candidates:
            candidates.append(cand)
    return candidates


def _ytdlp_fetch_playlist(src_url: str, timeout: int = 360) -> List[Dict]:
    """Tek bir playlist URL'sinden video entry'lerini çek."""
    cmd = _yt_dlp_base_cmd() + [
        "--flat-playlist",
        "--dump-single-json",
        "--extractor-args", "youtube:skip=authcheck",
        src_url,
    ]
    try:
        res = _run_ytdlp(cmd, timeout=timeout)
        if res.returncode != 0 and res.stderr:
            log.warning("yt-dlp stderr (%s): %s", src_url, res.stderr.strip()[:800])
        payload = (res.stdout or "").strip()
        if not payload:
            return []
        return json.loads(payload).get("entries") or []
    except json.JSONDecodeError as e:
        log.warning("yt-dlp JSON parse hatası (%s): %s", src_url, e)
    except Exception as e:
        log.error("yt-dlp playlist hatası (%s): %s", src_url, e)
    return []


def ytdlp_list_videos(channel_url: str, date_from: str, date_to: str) -> List[Dict]:
    """
    Kanal video listesini çek — HEM /videos HEM /streams.
    Her giriş için tarih türetme hiyerarşisi:
      1. upload_date  (yt-dlp direkt verir)
      2. timestamp    → YYYYMMDD
      3. Göreceli metin alanları ("Streamed X hours ago")
      4. Boş kalır   (video yine eklenir, tarih alanı boş)
    """
    date_after  = (date_from or "").replace("-", "")
    date_before = (date_to   or "").replace("-", "")

    videos: List[Dict] = []
    seen_ids: set      = set()

    for src_url in _candidate_channel_urls(channel_url):
        entries    = _ytdlp_fetch_playlist(src_url)
        found_here = 0

        for e in entries:
            if not isinstance(e, dict):
                continue

            vid_id = (e.get("id") or "").strip()
            if not vid_id or len(vid_id) != 11 or vid_id in seen_ids:
                continue

            title       = (e.get("title") or "").strip()
            upload_date = (e.get("upload_date") or "").strip()
            ts          = int(e.get("timestamp") or e.get("release_timestamp") or 0)

            # ── Tarih türetme (hard-coding YOK) ──────────────────────────────
            if not upload_date:
                if ts:
                    upload_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")
                else:
                    # Herhangi bir metin alanında görece tarih varsa parse et
                    for field in ("description", "view_count_text", "duration_string",
                                  "live_status", "availability"):
                        val = str(e.get(field) or "")
                        rel = _parse_relative_date(val)
                        if rel:
                            upload_date = rel
                            break

            # ── Tarih filtresi ────────────────────────────────────────────────
            if upload_date:
                if date_after  and upload_date < date_after:  continue
                if date_before and upload_date > date_before: continue

            videos.append({
                "video_id":   vid_id,
                "title":      title,
                "video_date": upload_date,
                "source_url": src_url,
            })
            seen_ids.add(vid_id)
            found_here += 1

        log.info("yt-dlp kaynak: %s → %d video (%d toplam)", src_url, found_here, len(videos))
        # Erken çıkış YOK — tüm kaynaklar taranır

    log.info("yt-dlp: toplam %d benzersiz video", len(videos))
    return videos

def ytdlp_comments(video_id: str, title: str = "", video_date: str = "",
                    source_type: str = "comment") -> List[Dict]:
    """
    Video altındaki YORUMLARı çek (hem normal video hem canlı yayın replay yorumları).
    source_type parametresi dışarıdan gelir; hard-coded değil.
    """
    odir = Path(CFG["data_dir"]) / "comments"
    odir.mkdir(parents=True, exist_ok=True)
    cache = odir / f"{video_id}.json"
    if cache.exists():
        try:
            cached = json.load(open(cache, encoding="utf-8"))
            if cached:
                log.info("  %s yorumlar (cache): %d", video_id, len(cached))
                return cached
        except:
            pass

    # yt-dlp: yorumlar + tam metadata
    cmd = _yt_dlp_base_cmd() + [
        "--write-comments",
        "--write-info-json",
        "--skip-download",
        "--no-warnings",
        "--ignore-errors",
        "--extractor-args", "youtube:skip=authcheck",
        "-o", str(odir / f"{video_id}.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        res = _run_ytdlp(cmd, timeout=300)
        if res.returncode != 0 and res.stderr:
            log.warning("yt-dlp yorum stderr %s: %s", video_id, res.stderr.strip()[:600])
    except Exception as e:
        log.warning("yt-dlp yorum hatası %s: %s", video_id, e)

    # info.json dosyasını oku
    info_path = odir / f"{video_id}.info.json"
    msgs: List[Dict] = []

    if info_path.exists():
        try:
            data = json.load(open(info_path, encoding="utf-8"))

            # video_date boşsa info.json'dan türet
            vid_date = video_date
            if not vid_date:
                ud = data.get("upload_date") or ""
                if not ud:
                    ts_raw = data.get("timestamp") or data.get("release_timestamp") or 0
                    if ts_raw:
                        ud = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc).strftime("%Y%m%d")
                vid_date = ud

            # source_type: video türünü info.json'dan otomatik belirle
            live_status = data.get("live_status") or ""
            auto_type = source_type
            if "was_live" in live_status or "is_live" in live_status:
                auto_type = "stream"
            elif source_type == "comment":
                auto_type = "comment"

            comments = data.get("comments") or []
            for c in comments:
                author = (c.get("author") or "").strip()
                text   = (c.get("text")   or "").strip()
                if not author or not text:
                    continue
                ts = int(c.get("timestamp") or 0)
                # Yorum zaman damgası yoksa video tarihine göre tahmin et
                if ts == 0 and vid_date:
                    ts = _video_base_timestamp(vid_date)
                m = process_raw({
                    "video_id":          video_id,
                    "title":             title,
                    "video_date":        vid_date,
                    "author":            author,
                    "author_channel_id": c.get("author_id", ""),
                    "message":           text,
                    "timestamp_utc":     ts,
                    "source_type":       auto_type,
                })
                if m:
                    msgs.append(m)

        except Exception as e:
            log.warning("JSON parse hatası %s: %s", video_id, e)

    if msgs:
        json.dump(msgs, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    log.info("  %s yorumlar: %d", video_id, len(msgs))
    return msgs

def _video_base_timestamp(video_date: str) -> int:
    """video_date (YYYYMMDD veya YYYY-MM-DD) → Unix timestamp (gün başı UTC)"""
    if not video_date: return 0
    try:
        ds = video_date.replace("-","")
        dt = datetime(int(ds[:4]),int(ds[4:6]),int(ds[6:8]),tzinfo=timezone.utc)
        return int(dt.timestamp())
    except: return 0

def _parse_live_chat_json3(cd: dict, video_id: str, title: str,
                            video_date: str, base_ts: int) -> List[Dict]:
    """JSON3 formatı (.live_chat.json3): events[].segs + tOffsetMs"""
    msgs = []
    for ev in cd.get("events", []):
        segs = ev.get("segs", [])
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        author = ev.get("authorName", "")
        if not author:
            continue
        t_off_ms = int(ev.get("tOffsetMs", 0) or 0)
        abs_ts = base_ts + t_off_ms // 1000 if base_ts else t_off_ms // 1000
        m = process_raw({"video_id":video_id,"title":title,"video_date":video_date,
                         "author":author,"author_channel_id":ev.get("authorExternalChannelId",""),
                         "message":text,"timestamp_utc":abs_ts,
                         "source_type":"replay_chat","is_live":False})
        if m:
            msgs.append(m)

    # Bazı yt-dlp sürümleri json3 dosyasında actions/replayChatItemAction gömer.
    if msgs:
        return msgs

    actions = cd.get("actions", [])
    if not actions and isinstance(cd.get("continuationContents"), dict):
        live_cont = cd["continuationContents"].get("liveChatContinuation", {})
        actions = live_cont.get("actions", [])
    for act in actions:
        replay = act.get("replayChatItemAction", act)
        offset_ms = int(replay.get("videoOffsetTimeMsec", 0) or 0)
        abs_ts = base_ts + offset_ms // 1000 if base_ts else offset_ms // 1000
        for item in replay.get("actions", [replay]):
            renderer = (item.get("addChatItemAction", {})
                           .get("item", {})
                           .get("liveChatTextMessageRenderer", {}))
            if not renderer:
                renderer = (item.get("addChatItemAction", {})
                               .get("item", {})
                               .get("liveChatPaidMessageRenderer", {}))
            if not renderer:
                continue
            runs = renderer.get("message", {}).get("runs", [])
            text = "".join(r.get("text", "") for r in runs).strip()
            author = renderer.get("authorName", {}).get("simpleText", "")
            if not text or not author:
                continue
            ts_usec = int(renderer.get("timestampUsec", "0") or "0")
            if ts_usec > 0:
                abs_ts = ts_usec // 1_000_000
            m = process_raw({"video_id":video_id,"title":title,"video_date":video_date,
                             "author":author,"author_channel_id":renderer.get("authorExternalChannelId",""),
                             "message":text,"timestamp_utc":abs_ts,
                             "source_type":"replay_chat","is_live":False})
            if m:
                msgs.append(m)
    return msgs

def _parse_live_chat_jsonl(path: Path, video_id: str, title: str,
                            video_date: str, base_ts: int) -> List[Dict]:
    """Ham JSONL formatı (.live_chat.json): her satır bir JSON objesi"""
    msgs = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except: continue
                # Replay chat action sarmalayıcısını çöz
                action = obj.get("replayChatItemAction", obj)
                offset_ms = int(action.get("videoOffsetTimeMsec", 0) or 0)
                abs_ts = base_ts + offset_ms // 1000 if base_ts else offset_ms // 1000
                for act in action.get("actions", [action]):
                    renderer = (act.get("addChatItemAction",{})
                                   .get("item",{})
                                   .get("liveChatTextMessageRenderer",{}))
                    if not renderer:
                        renderer = (act.get("addChatItemAction",{})
                                       .get("item",{})
                                       .get("liveChatPaidMessageRenderer",{}))
                    if not renderer: continue
                    runs = renderer.get("message",{}).get("runs",[])
                    text = "".join(r.get("text","") for r in runs).strip()
                    if not text: continue
                    author = renderer.get("authorName",{}).get("simpleText","")
                    if not author: continue
                    # timestampUsec öncelikli (mikrosaniye → saniye)
                    ts_usec = int(renderer.get("timestampUsec","0") or "0")
                    if ts_usec > 0:
                        abs_ts = ts_usec // 1_000_000
                    cid = renderer.get("authorExternalChannelId","")
                    m = process_raw({"video_id":video_id,"title":title,"video_date":video_date,
                                      "author":author,"author_channel_id":cid,
                                      "message":text,"timestamp_utc":abs_ts,
                                      "source_type":"replay_chat","is_live":False})
                    if m: msgs.append(m)
    except Exception as e:
        log.warning("JSONL live chat parse hatası %s: %s", video_id, e)
    return msgs

def ytdlp_live_chat(video_id: str, title: str = "", video_date: str = "") -> List[Dict]:
    odir = Path(CFG["data_dir"]) / "chats"
    odir.mkdir(parents=True, exist_ok=True)
    cache = odir / f"{video_id}_chat.json"
    if cache.exists():
        try:
            data = json.load(open(cache, encoding="utf-8"))
            if data: return data
        except: pass

    # yt-dlp komutu: json3 formatını dene (hem json hem json3 uzantısını kontrol et)
    # 8) ytdlp_live_chat() içinde cmd satırını değiştir
    cmd = _yt_dlp_base_cmd() + [
    "--write-info-json",
    "--write-subs",
    "--write-auto-subs",
    "--sub-format", "json3",
    "--sub-langs", "live_chat",
    "-o", str(odir / f"{video_id}.%(ext)s"),
    f"https://www.youtube.com/watch?v={video_id}",]
    try:
        res = _run_ytdlp(cmd, timeout=480)
        if res.stderr and res.returncode != 0:
            log.warning("yt-dlp live chat stderr %s: %s", video_id, res.stderr.strip()[:1500])
    except Exception as e:
        log.warning("Live chat indir hatası %s: %s", video_id, e)

    base_ts = _video_base_timestamp(video_date)
    msgs: List[Dict] = []

    # 1) JSON3 formatı → .live_chat.json3
    chat_json3 = odir / f"{video_id}.live_chat.json3"
    if chat_json3.exists():
        try:
            cd = json.load(open(chat_json3, encoding="utf-8"))
            msgs = _parse_live_chat_json3(cd, video_id, title, video_date, base_ts)
            log.info("  %s live chat (json3): %d", video_id, len(msgs))
        except Exception as e:
            log.warning("JSON3 parse hatası %s: %s", video_id, e)

    # 2) Ham JSONL formatı → .live_chat.json
    if not msgs:
        chat_jsonl = odir / f"{video_id}.live_chat.json"
        if chat_jsonl.exists():
            msgs = _parse_live_chat_jsonl(chat_jsonl, video_id, title, video_date, base_ts)
            log.info("  %s live chat (jsonl): %d", video_id, len(msgs))

    # 3) Glob ile kalan dosyaları tara (bilinmeyen uzantılar)
    if not msgs:
        for f in odir.glob(f"{video_id}.live_chat*"):
            if f.suffix in (".json3",".json") and f != chat_json3:
                try:
                    content = open(f, encoding="utf-8").read().strip()
                    if content.startswith("{"):
                        cd = json.loads(content)
                        msgs = _parse_live_chat_json3(cd, video_id, title, video_date, base_ts)
                    else:
                        msgs = _parse_live_chat_jsonl(f, video_id, title, video_date, base_ts)
                    if msgs:
                        log.info("  %s live chat (glob %s): %d", video_id, f.name, len(msgs))
                        break
                except: pass

    if msgs:
        json.dump(msgs, open(cache,"w",encoding="utf-8"), ensure_ascii=False)
    log.info("  %s live chat toplam: %d", video_id, len(msgs))
    return msgs

def selenium_live_chat(driver, video_id: str, title: str = "") -> List[Dict]:
    """Selenium ile canlı yayın chat mesajlarını çek"""
    if not driver: return []
    if not is_driver_alive(driver):
        log.warning("Selenium live chat atlandı: Chromium oturumu kapalı/geçersiz.")
        return []
    msgs = []
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}")
        time.sleep(4)
        now_ts = int(time.time())
        items = driver.find_elements(By.CSS_SELECTOR,
            "yt-live-chat-text-message-renderer,yt-live-chat-paid-message-renderer")
        for item in items:
            try:
                a = item.find_element(By.ID,"author-name").text.strip()
                t = item.find_element(By.ID,"message").text.strip()
                if a and t:
                    m = process_raw({"video_id":video_id,"title":title,"author":a,
                                      "message":t,"timestamp_utc":now_ts,
                                      "source_type":"live","is_live":True})
                    if m: msgs.append(m)
            except: pass
    except (InvalidSessionIdException, WebDriverException) as e:
        log.warning("Selenium live chat oturum hatası: %s", e)
    except Exception as e:
        log.warning("Selenium live chat: %s", e)
    return msgs

def _scrape_one_video(vid: Dict, idx: int, total_count: int,
                       emit_fn=None) -> Tuple[int, int, int]:
    """
    Tek bir video için hem yorumları hem canlı chat'i çek.
    source_type:
      - /streams'ten gelen → "stream"  (yorum + replay chat)
      - /videos'tan gelen  → "comment" (sadece yorum; live chat genelde yok)
    Dönüş: (comment_count, chat_count, saved_count)
    """
    vid_id = vid["video_id"]
    title  = vid.get("title", "")
    date   = vid.get("video_date", "")
    src    = vid.get("source_url", "")

    # Kaynak URL'ye göre tip belirle (hard-coding yok)
    is_stream_source = "/streams" in src or "/live" in src

    if emit_fn:
        try:
            emit_fn({"step": idx, "total": total_count,
                     "video_id": vid_id, "title": title})
        except:
            pass

    # Yorum tipi: stream kaynağından geliyorsa "stream", değilse "comment"
    comment_src = "stream" if is_stream_source else "comment"

    comments: List[Dict] = []
    chats:    List[Dict] = []

    try:
        comments = ytdlp_comments(vid_id, title, date, comment_src)
    except Exception as e:
        log.warning("[%d/%d] %s yorum hatası: %s", idx, total_count, vid_id, e)

    # Canlı chat: yalnızca stream kaynağından gelen videoları dene
    if is_stream_source:
        try:
            chats = ytdlp_live_chat(vid_id, title, date)
        except Exception as e:
            log.warning("[%d/%d] %s live chat hatası: %s", idx, total_count, vid_id, e)

    all_msgs = comments + chats
    saved = 0
    for m in all_msgs:
        upsert_message(m)
        saved += 1

    db_exec(
        "INSERT OR REPLACE INTO scraped_videos"
        "(video_id,title,video_date,source_type,comment_count,chat_count)"
        " VALUES(?,?,?,?,?,?)",
        (vid_id, title, date, comment_src, len(comments), len(chats))
    )
    log.info("[%d/%d] %s — yorum:%d chat:%d (toplam kaydedilen:%d)",
             idx, total_count, vid_id, len(comments), len(chats), saved)
    return len(comments), len(chats), saved


def full_scrape(emit_fn=None) -> int:
    """
    Kanal taraması: HEM /videos HEM /streams — tüm yorumlar + canlı chat.
    Paralel işleme ile 10x hız (max 4 eş zamanlı yt-dlp işlemi).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Her iki kaynaktan video listesi al
    channel_base = CFG["channel_url"]
    videos = ytdlp_list_videos(
        channel_base,
        CFG.get("date_from", "2023-01-01"),
        CFG.get("date_to",   "2026-12-31"),
    )
    if not videos:
        log.warning("Video bulunamadı"); return 0

    log.info("Toplam %d video taranacak (paralel, max 4 iş parçacığı)", len(videos))

    total_saved = 0
    n = len(videos)

    # Paralel yorum/chat çekimi — yt-dlp süreçleri CPU/IO bound
    # max_workers=4: disk I/O + network doymadan önce ideal denge
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            pool.submit(_scrape_one_video, vid, i + 1, n, emit_fn): vid
            for i, vid in enumerate(videos)
        }
        for future in as_completed(future_map):
            vid = future_map[future]
            try:
                _, _, saved = future.result()
                total_saved += saved
            except Exception as e:
                log.error("Video tarama hatası %s: %s", vid.get("video_id"), e)

    log.info("✅ Tam tarama tamamlandı: %d mesaj kaydedildi", total_saved)
    return total_saved

# ═══════════════════════════════════════════════════════════════════════════════
# § 6b — NLP TABANLI OTOMATİK CANLI YAYIN TEKRAR SOHBETİ ÇEKME
# ═══════════════════════════════════════════════════════════════════════════════
_NLP_CHAT_CATEGORIES = [
    "toxic or hateful message",
    "spam or bot-generated message",
    "genuine fan interaction",
    "question to the streamer",
    "coordinated harassment",
    "neutral chat message",
]

def nlp_filter_messages(raw_msgs: List[Dict], batch_size: int = 50) -> List[Dict]:
    """
    NLP ile mesajları filtrele:
    - Spam/bot mesajları temizle
    - Tehlikeli içerikleri işaretle
    - Gerçek etkileşimleri önceliklendir
    """
    if not raw_msgs: return raw_msgs
    filtered = []
    for i in range(0, len(raw_msgs), batch_size):
        batch = raw_msgs[i:i+batch_size]
        for msg in batch:
            text = msg.get("message","")
            if not text or len(text.strip()) < 2: continue
            # Heuristik ön filtre (BART olmadan hızlı)
            tokens = text.lower().split()
            lex_d  = len(set(tokens)) / max(len(tokens), 1)
            if lex_d < 0.10 and len(tokens) > 5:  # %90+ tekrar → bot spam
                msg["_nlp_category"] = "spam"
                msg["_nlp_score"]    = 0.05
            else:
                msg["_nlp_category"] = "ok"
                msg["_nlp_score"]    = 1.0
            # BART varsa derin sınıflandırma (kritik mesajlarda)
            if _TRANS and len(text) > 10:
                try:
                    res = bart_classify(text[:300], _NLP_CHAT_CATEGORIES)
                    top_cat   = max(res, key=res.get)
                    top_score = res[top_cat]
                    msg["_nlp_category"] = top_cat
                    msg["_nlp_score"]    = round(top_score, 4)
                    if "spam" in top_cat or "bot" in top_cat:
                        msg["_nlp_filtered"] = True
                except: pass
            filtered.append(msg)
    kept = [m for m in filtered if not m.get("_nlp_filtered", False)]
    log.info("NLP filtre: %d → %d mesaj kaldı", len(raw_msgs), len(kept))
    return kept

def nlp_cluster_chat(msgs: List[Dict], eps: float = 0.35,
                      min_samples: int = 3) -> Dict[int, List[Dict]]:
    """
    Sohbet mesajlarını embedding ile kümelere ayır.
    Koordineli saldırı tespiti için kullanılır.
    """
    if len(msgs) < min_samples * 2: return {0: msgs}
    texts = [m.get("message","") for m in msgs]
    embs  = embed_batch(texts)
    if embs is None:
        # SBERT yoksa TF-IDF ile basit kümeleme
        if _tfidf_fitted:
            try:
                vecs = np.array([tfidf_vec(t) for t in texts])
                labels = DBSCAN(eps=0.5, min_samples=min_samples,
                                metric="cosine").fit_predict(vecs)
            except: labels = np.zeros(len(msgs), dtype=int)
        else:
            labels = np.zeros(len(msgs), dtype=int)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="cosine").fit_predict(embs)
    clusters: Dict[int, List[Dict]] = {}
    for label, msg in zip(labels, msgs):
        clusters.setdefault(int(label), []).append(msg)
    log.info("NLP kümeleme: %d küme bulundu", len(clusters))
    return clusters

def nlp_extract_key_topics(msgs: List[Dict], top_n: int = 10) -> List[Dict]:
    """TF-IDF + LDA ile sohbetten ana konuları çıkar"""
    texts = [m.get("message","") for m in msgs if m.get("message")]
    if len(texts) < 10: return []
    try:
        vec = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=2)
        X   = vec.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=min(5, len(texts)//3),
                                         random_state=42, max_iter=10)
        lda.fit(X)
        feat_names = vec.get_feature_names_out()
        topics = []
        for idx, comp in enumerate(lda.components_):
            top_words = [feat_names[i] for i in comp.argsort()[:-top_n-1:-1]]
            topics.append({"topic_id": idx, "keywords": top_words,
                           "weight": round(float(comp.max()), 4)})
        return topics
    except Exception as e:
        log.warning("Konu çıkarma hatası: %s", e); return []

def nlp_detect_coordinated(clusters: Dict[int, List[Dict]],
                             min_cluster_size: int = 3) -> List[Dict]:
    """
    Küme içindeki koordineli davranışı tespit et:
    - Aynı kümedeki çok sayıda farklı kullanıcı → koordineli saldırı
    - Zaman aralığı analizi
    """
    threats = []
    for cid, cmsg in clusters.items():
        if cid == -1: continue           # DBSCAN noise
        if len(cmsg) < min_cluster_size: continue
        authors    = list({m.get("author","") for m in cmsg})
        timestamps = sorted([m.get("timestamp_utc",0) for m in cmsg])
        if len(timestamps) > 1:
            span_sec = max(timestamps) - min(timestamps)
        else:
            span_sec = 0
        # Farklı kullanıcılardan kısa sürede benzer mesajlar → koordineli
        if len(authors) >= 2 and span_sec < 3600:
            sample_text = cmsg[0].get("message","")[:100]
            threats.append({
                "cluster_id":   cid,
                "member_count": len(cmsg),
                "author_count": len(authors),
                "authors":      authors[:10],
                "span_seconds": span_sec,
                "sample_text":  sample_text,
                "threat_type":  "COORDINATED_ATTACK",
                "confidence":   round(min(1.0, len(authors)/10 * (1/(1+span_sec/300))), 4),
            })
    threats.sort(key=lambda x: x["confidence"], reverse=True)
    return threats

def nlp_timeline_analysis(msgs: List[Dict], bin_minutes: int = 5) -> Dict:
    """
    Zaman çizelgesi analizi: mesaj yoğunluğu + ani artış tespiti
    """
    if not msgs: return {}
    ts_list = sorted([m.get("timestamp_utc",0) for m in msgs if m.get("timestamp_utc",0)>0])
    if not ts_list: return {}
    t0 = ts_list[0]; t1 = ts_list[-1]
    bin_sec = bin_minutes * 60
    n_bins  = max(1, (t1 - t0) // bin_sec + 1)
    bins    = np.zeros(int(n_bins), dtype=int)
    for t in ts_list:
        idx = min(int((t - t0) // bin_sec), len(bins)-1)
        bins[idx] += 1
    # Ani artış tespiti (2σ eşiği)
    mu, sg = float(bins.mean()), float(bins.std())
    spikes = [{"bin_index": i,
               "time_offset_min": i * bin_minutes,
               "count": int(bins[i]),
               "z_score": round((bins[i]-mu)/(sg+1e-9), 2)}
              for i in range(len(bins)) if bins[i] > mu + 2*sg]
    return {
        "total_messages": len(msgs),
        "duration_minutes": round((t1-t0)/60, 1),
        "peak_bin_count": int(bins.max()),
        "avg_per_bin": round(float(mu), 2),
        "spike_bins": spikes,
        "activity_bins": bins.tolist()[:200],  # max 200 bin
    }

def nlp_auto_replay_chat(video_id: str, title: str = "", video_date: str = "",
                          auto_analyze: bool = True,
                          filter_spam: bool = True) -> Dict:
    """
    NLP Tabanlı Otomatik Canlı Yayın Tekrar Sohbet Analizi
    ────────────────────────────────────────────────────────
    1. yt-dlp ile ham chat verisi çek
    2. NLP filtresi ile spam/bot mesajları temizle
    3. Embedding ile mesajları kümele
    4. Koordineli saldırı tespit et
    5. Zaman çizelgesi analizi yap
    6. Konuları çıkar
    7. Tehditkar kullanıcıları DB'ye kaydet
    """
    log.info("🤖 NLP Replay Chat Analizi başlıyor: %s", video_id)

    # 1. Ham veriyi çek
    raw_msgs = ytdlp_live_chat(video_id, title, video_date)
    if not raw_msgs:
        log.warning("NLP analiz: %s için sohbet verisi bulunamadı", video_id)
        return {"video_id":video_id,"status":"no_data","messages":0}

    # 2. NLP filtresi
    if filter_spam:
        filtered = nlp_filter_messages(raw_msgs)
    else:
        filtered = raw_msgs

    # 3. DB'ye kaydet
    saved = 0
    for m in filtered:
        upsert_message(m)
        saved += 1

    # 4. TF-IDF güncelle
    all_db_texts = db_exec("SELECT message FROM messages LIMIT 5000", fetch="all") or []
    if all_db_texts:
        fit_tfidf([r["message"] for r in all_db_texts])

    # 5. Kümeleme (koordineli saldırı tespiti)
    clusters    = nlp_cluster_chat(filtered)
    coordinated = nlp_detect_coordinated(clusters)

    # 6. Koordineli saldırı varsa identity_links'e kaydet
    for threat in coordinated:
        authors = threat.get("authors",[])
        conf    = threat.get("confidence", 0.5)
        for ai in range(len(authors)):
            for aj in range(ai+1, len(authors)):
                db_exec("INSERT OR IGNORE INTO identity_links"
                        "(user_a,user_b,sim_score,method,confidence)"
                        " VALUES(?,?,?,?,?)",
                        (authors[ai], authors[aj],
                         round(conf,4), "nlp_coordinated", round(conf,4)))

    # 7. Zaman çizelgesi
    timeline = nlp_timeline_analysis(filtered)

    # 8. Konu çıkarma
    topics = nlp_extract_key_topics(filtered)

    # 9. Otomatik kullanıcı analizi
    analyzed = []
    if auto_analyze:
        authors_in_video = list({m.get("author","") for m in filtered})
        for a in authors_in_video:
            try:
                res = analyze_user(a, run_ollama=False)
                if res.get("threat_score",0) > 0.3:
                    analyzed.append({"author":a,"threat_score":res["threat_score"],
                                     "threat_level":res["threat_level"]})
            except: pass
        analyzed.sort(key=lambda x: x.get("threat_score",0), reverse=True)
        log.info("NLP analiz: %d kullanıcıdan %d tehdit tespiti",
                 len(authors_in_video), len(analyzed))

    result = {
        "video_id":          video_id,
        "title":             title,
        "status":            "ok",
        "raw_messages":      len(raw_msgs),
        "filtered_messages": len(filtered),
        "saved_to_db":       saved,
        "clusters_found":    len(clusters),
        "coordinated_threats": coordinated,
        "timeline":          timeline,
        "topics":            topics,
        "threat_users":      analyzed[:20],
    }
    log.info("✅ NLP Replay Chat tamamlandı: %s → %d mesaj, %d tehdit",
             video_id, saved, len(coordinated))
    return result

def nlp_full_channel_scan(channel_url: str = None,
                           date_from: str = None,
                           date_to:   str = None) -> Dict:
    """
    NLP tabanlı tam kanal taraması.
    HEM /videos (normal video yorumları) HEM /streams (canlı yayın tekrarları)
    2023-2026 arası @ShmirchikArt varsayılan.
    """
    # Kanal URL'sini kanal kökünden başlat — _candidate_channel_urls her iki sayfayı açar
    channel_url = channel_url or CFG["channel_url"]
    date_from   = date_from   or CFG.get("date_from","2023-01-01")
    date_to     = date_to     or CFG.get("date_to","2026-12-31")

    log.info("🤖 NLP Tam Kanal Taraması: %s (%s → %s)",
             channel_url, date_from, date_to)

    videos = ytdlp_list_videos(channel_url, date_from, date_to)
    if not videos:
        return {"status":"no_videos","channel":channel_url}

    all_results = []
    global_coordinated: List[Dict] = []

    for i, vid in enumerate(videos):
        vid_id = vid["video_id"]
        title  = vid.get("title","")
        date   = vid.get("video_date","")
        log.info("[%d/%d] NLP analiz: %s — %s", i+1, len(videos), vid_id, title[:40])
        try:
            r = nlp_auto_replay_chat(vid_id, title, date, auto_analyze=True)
            all_results.append(r)
            global_coordinated.extend(r.get("coordinated_threats",[]))
        except Exception as e:
            log.warning("Video %s NLP hatası: %s", vid_id, e)
            all_results.append({"video_id":vid_id,"status":"error","error":str(e)})

    # Tüm kullanıcılar için final analiz
    all_authors = db_exec(
        "SELECT DISTINCT author FROM messages WHERE deleted=0", fetch="all") or []
    threat_summary = []
    for row in all_authors:
        try:
            p = db_exec("SELECT author,threat_score,threat_level FROM user_profiles"
                        " WHERE author=?", (row["author"],), fetch="one")
            if p and float(p["threat_score"] or 0) > 0.3:
                threat_summary.append(dict(p))
        except: pass
    threat_summary.sort(key=lambda x: x.get("threat_score",0), reverse=True)

    summary = {
        "channel":            channel_url,
        "date_range":         f"{date_from} → {date_to}",
        "videos_scanned":     len(videos),
        "videos_with_chat":   sum(1 for r in all_results if r.get("status")=="ok"),
        "total_messages":     sum(r.get("filtered_messages",0) for r in all_results),
        "coordinated_threats":len(global_coordinated),
        "top_threats":        threat_summary[:20],
        "video_results":      [{k:v for k,v in r.items() if k!="timeline"}
                               for r in all_results],
    }
    log.info("✅ NLP Tam Tarama tamamlandı: %d video, %d mesaj, %d koordineli tehdit",
             summary["videos_scanned"], summary["total_messages"],
             summary["coordinated_threats"])
    return summary

# ═══════════════════════════════════════════════════════════════════════════════
# § 7 — KULLANICI HESAP ANALİZİ (Selenium)
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_sub_count(text: str) -> int:
    t = text.lower().replace(",","").replace(".","").strip()
    m = re.search(r"([\d]+)\s*([kmb]?)", t)
    if not m: return 0
    n = int(m.group(1)); s = m.group(2)
    return n * (1000 if s=="k" else 1_000_000 if s=="m" else 1_000_000_000 if s=="b" else 1)

def _is_new_account(created_str: str, months: int = 6) -> bool:
    now = datetime.now()
    for pat, fmt in [
        (r"(\w{3,})\s+(\d{1,2}),?\s+(\d{4})", "%b %d %Y"),
        (r"(\d{1,2})\s+(\w{3,})\s+(\d{4})",   "%d %b %Y"),
        (r"(\d{4})-(\d{2})-(\d{2})",           "%Y-%m-%d"),
    ]:
        m = re.search(pat, created_str)
        if m:
            try:
                s = " ".join(m.groups())
                dt = datetime.strptime(s, fmt)
                diff = (now.year - dt.year)*12 + (now.month - dt.month)
                return diff <= months
            except: pass
    return False

def inspect_account(driver, channel_id: str) -> dict:
    if not driver or not channel_id: return {}
    if channel_id in _acct_cache: return _acct_cache[channel_id]
    r = {"channel_id":channel_id,"account_created":"","subscriber_count":0,
         "video_count":0,"is_new_account":False,"about_text":""}
    try:
        driver.get(f"https://www.youtube.com/channel/{channel_id}/about")
        time.sleep(3)
        # Katılma tarihi
        for xp in ["//*[contains(text(),'Joined')]","//*[contains(text(),'Katıldı')]",
                   "//*[@id='right-column']//yt-formatted-string[contains(.,'joined')]"]:
            try:
                r["account_created"] = driver.find_element(By.XPATH, xp).text.strip()
                break
            except: pass
        # Abone sayısı
        for sel in ["#subscriber-count","yt-formatted-string#subscribers"]:
            try:
                r["subscriber_count"] = _parse_sub_count(driver.find_element(By.CSS_SELECTOR,sel).text)
                break
            except: pass
        # Video sayısı
        try:
            vc = driver.find_element(By.XPATH,"//*[contains(text(),'video') or contains(text(),'Video')]")
            m = re.search(r"(\d[\d,]*)", vc.text)
            if m: r["video_count"] = int(m.group(1).replace(",",""))
        except: pass
        # Hakkında
        try: r["about_text"] = driver.find_element(By.CSS_SELECTOR,"#description-container").text[:500]
        except: pass
        if r["account_created"]:
            r["is_new_account"] = _is_new_account(r["account_created"], CFG.get("new_account_months",6))
        _acct_cache[channel_id] = r
        time.sleep(0.8)
    except Exception as e:
        log.debug("Hesap inceleme hatası %s: %s", channel_id, e)
    return r

def batch_inspect_accounts(driver, authors: List[str]) -> Dict[str,dict]:
    results = {}
    rows = db_exec(f"SELECT author,author_cid FROM user_profiles WHERE author IN ({','.join(['?']*len(authors))})",
                   tuple(authors), fetch="all") or []
    for row in rows:
        cid = row["author_cid"]
        if cid:
            info = inspect_account(driver, cid)
            if info:
                results[row["author"]] = info
                upsert_profile(row["author"], {
                    "account_created":   info.get("account_created",""),
                    "subscriber_count":  info.get("subscriber_count",0),
                    "video_count":       info.get("video_count",0),
                    "is_new_account":    int(info.get("is_new_account",False))
                })
    return results

def correlate_new_accounts(driver, threshold_months: int = 6) -> List[Tuple[str,str,float]]:
    """Yeni açılmış hesapları birbiriyle ilişkilendir"""
    rows = db_exec("SELECT author,author_cid,account_created,subscriber_count FROM user_profiles"
                   " WHERE is_new_account=1", fetch="all") or []
    if len(rows) < 2: return []
    pairs = []
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            a = dict(rows[i]); b = dict(rows[j])
            # Alt skor: her iki hesap da yeni
            score = 0.7
            # Aynı dönemde açılmış
            if a.get("account_created") and b.get("account_created"):
                score += 0.2
            # İkisi de düşük abone sayısı
            if int(a.get("subscriber_count",0)) < 100 and int(b.get("subscriber_count",0)) < 100:
                score += 0.1
            pairs.append((a["author"], b["author"], min(1.0, score)))
    return pairs

# ═══════════════════════════════════════════════════════════════════════════════
# § 8 — NLP PİPELİNE (Katman 1-2: TF-IDF, N-gram, Stilometri, Embedding)
# ═══════════════════════════════════════════════════════════════════════════════
_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3),
                          analyzer="word", sublinear_tf=True, min_df=2)
_tfidf_fitted = False
_tfidf_lock   = threading.Lock()

def fit_tfidf(texts: List[str]):
    global _tfidf_fitted
    with _tfidf_lock:
        if texts:
            try:
                _tfidf.fit(texts); _tfidf_fitted = True
            except Exception as e:
                log.warning("TF-IDF fit hatası: %s", e)

def tfidf_vec(text: str) -> np.ndarray:
    with _tfidf_lock:
        if not _tfidf_fitted: return np.zeros(100)
        try: return _tfidf.transform([text]).toarray()[0]
        except: return np.zeros(100)

def embed(text: str) -> Optional[List[float]]:
    m = get_sbert()
    if not m: return None
    try: return m.encode(text[:512], normalize_embeddings=True).tolist()
    except: return None

def embed_batch(texts: List[str]) -> Optional[np.ndarray]:
    m = get_sbert()
    if not m: return None
    try: return m.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    except: return None

def ngram_fp(text: str, n_range=(2,4)) -> Counter:
    text = text.lower()
    fp = Counter()
    for n in range(n_range[0], n_range[1]+1):
        for i in range(len(text)-n+1):
            fp[text[i:i+n]] += 1
    return fp

def jaccard(a: Counter, b: Counter) -> float:
    sa = set(a.keys()); sb = set(b.keys())
    u = len(sa | sb)
    return len(sa & sb)/u if u else 0.0

def pos_profile(text: str) -> dict:
    nlp = get_spacy()
    if not nlp or not text: return {}
    doc = nlp(text[:800])
    c = Counter(t.pos_ for t in doc)
    total = max(len(doc),1)
    return {p: cnt/total for p,cnt in c.items()}

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if c.isupper())/max(len(letters),1)

def punct_density(text: str) -> float:
    return sum(1 for c in text if c in ".,!?;:'\"()[]{}")/max(len(text),1)

def lexical_div(tokens: List[str]) -> float:
    return len(set(tokens))/max(len(tokens),1)

def shannon_H(text: str) -> float:
    freq = Counter(text); total = max(len(text),1)
    return -sum((cnt/total)*math.log2(cnt/total+1e-12) for cnt in freq.values())

def repetition_score(msgs: List[str]) -> float:
    if len(msgs) < 2: return 0.0
    sims = []
    for i in range(min(len(msgs)-1,50)):
        a = set(msgs[i].lower().split()); b = set(msgs[i+1].lower().split())
        sims.append(len(a&b)/max(len(a|b),1))
    return float(np.mean(sims)) if sims else 0.0

def typo_fp(msgs: List[str]) -> dict:
    text = " ".join(msgs)
    return {
        "double_letters":      len(re.findall(r"(\w)\1{2,}", text)),
        "uppercase_ratio":     round(uppercase_ratio(text),4),
        "punct_density":       round(punct_density(text),4),
        "ellipsis_rate":       text.count("...")/max(len(msgs),1),
        "avg_msg_len":         sum(len(m) for m in msgs)/max(len(msgs),1),
        "emoji_density":       len(extract_emojis(text))/max(len(text),1),
        "exclamation_rate":    text.count("!")/max(len(text),1),
        "question_rate":       text.count("?")/max(len(text),1),
        "capitalized_words":   len(re.findall(r'\b[A-Z]{2,}\b', text))/max(len(msgs),1),
    }

def burrows_delta(vecs: np.ndarray, i: int, j: int) -> float:
    mu = np.mean(vecs, axis=0); sigma = np.std(vecs, axis=0)+1e-12
    za = (vecs[i]-mu)/sigma; zb = (vecs[j]-mu)/sigma
    return float(np.mean(np.abs(za-zb)))

def cosine_delta(va: np.ndarray, vb: np.ndarray,
                  mu: np.ndarray, sigma: np.ndarray) -> float:
    za = (va-mu)/(sigma+1e-12); zb = (vb-mu)/(sigma+1e-12)
    na = np.linalg.norm(za); nb = np.linalg.norm(zb)
    if na<1e-9 or nb<1e-9: return 1.0
    return float(1.0 - np.dot(za,zb)/(na*nb))

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p,float)+1e-12; q = np.asarray(q,float)+1e-12
    p /= p.sum(); q /= q.sum(); m = (p+q)/2
    return float(0.5*scipy_entropy(p,m)+0.5*scipy_entropy(q,m))

def composite_sim(emb: float, ng: float, typo: float,
                   time_s: float, topic: float, w=None) -> float:
    if w is None: w = [0.35,0.25,0.15,0.15,0.10]
    return sum(wi*s for wi,s in zip(w,[emb,ng,typo,time_s,topic]))

# ═══════════════════════════════════════════════════════════════════════════════
# § 9 — BOT TESPİTİ (Katman 3: Burstiness, Hawkes, BART, Heuristik)
# ═══════════════════════════════════════════════════════════════════════════════
def burstiness(timestamps: List[int]) -> float:
    if len(timestamps) < 3: return 0.0
    diffs = np.diff(sorted(timestamps)).astype(float)
    mu = np.mean(diffs); sigma = np.std(diffs)
    return float((sigma-mu)/(sigma+mu)) if sigma+mu > 1e-9 else 0.0

def hawkes_intensity(t: float, history: List[float],
                      mu=0.1, alpha=0.5, beta=1.0) -> float:
    return mu + sum(alpha*math.exp(-beta*(t-ti)) for ti in history if ti < t)

def hawkes_stalker_score(user_ts: List[int], host_ts: List[int], delta=90) -> float:
    """Kullanıcının kanal sahibi konuştuktan sonra ne kadar hızlı yanıt verdiği"""
    if not user_ts or not host_ts: return 0.0
    resp = sum(1 for ut in user_ts
               if any(0 < ut-ht <= delta for ht in host_ts))
    return resp/max(len(user_ts),1)

def heuristic_bot(msgs: List[str], timestamps: List[int]) -> float:
    if not msgs: return 0.0
    text   = " ".join(msgs)
    tokens = text.lower().split()
    D = lexical_div(tokens)
    H = min(1.0, shannon_H(text)/4.5)
    L = min(1.0, sum(len(m) for m in msgs)/max(len(msgs),1)/80)
    Q = sum(1 for m in msgs if "?" in m)/max(len(msgs),1)
    P = punct_density(text)
    E = min(1.0, len(extract_emojis(text))/max(len(text),1)*200)
    U = uppercase_ratio(text)
    R = repetition_score(msgs)
    B = abs(burstiness(timestamps)) if timestamps else 0
    score = 1-(0.25*D+0.15*H+0.10*L+0.10*Q+0.08*P+0.07*E+0.05*(1-U)+0.10*(1-R)+0.10*(1-B))
    return max(0.0,min(1.0,float(score)))

def bart_classify(text: str, labels: List[str]) -> Dict[str,float]:
    b = get_bart()
    if not b: return {l:1/len(labels) for l in labels}
    try:
        r = b(text[:512], candidate_labels=labels,
               hypothesis_template="This text is {}.")
        return dict(zip(r["labels"],r["scores"]))
    except Exception as e:
        log.debug("BART hatası: %s", e)
        return {l:1/len(labels) for l in labels}

def bot_score(msgs: List[str], timestamps: List[int]) -> float:
    if not msgs: return 0.0
    h = heuristic_bot(msgs, timestamps)
    sample = " ".join(msgs[:5])[:400]
    bs = bart_classify(sample, BOT_LABELS)
    bv = bs.get("spam or bot-like message", 0.5)
    return round(0.55*bv + 0.45*h, 4)

def co_entry(events: List[Tuple[str,int]], delta=300) -> List[Tuple]:
    co = []
    for i,(ua,ta) in enumerate(events):
        for ub,tb in events[i+1:]:
            if abs(ta-tb) <= delta and ua != ub:
                co.append((ua,ub,abs(ta-tb)))
    return co

# ═══════════════════════════════════════════════════════════════════════════════
# § 10 — NEFRET SÖYLEMİ, KİMLİK ÖRTÜsü (Katman 4-5)
# ═══════════════════════════════════════════════════════════════════════════════
def hate_scores(text: str) -> Dict[str,float]:
    if not text.strip():
        return {"antisemitism":0,"hate_general":0,"groyper":0,
                "stalker_sig":0,"impersonation":0,"bot_signal":0,"neutral":1,"overall":0}
    s = bart_classify(text, THREAT_LABELS)
    antisem = max(s.get("antisemitic content",0), s.get("hate speech against Jewish people",0))
    groyper = s.get("groyper movement content",0)
    hate    = max(s.get("islamophobic content",0),s.get("white supremacist content",0)) + groyper*0.3
    overall = max(antisem, hate*0.8)
    return {
        "antisemitism": round(antisem,4),
        "hate_general": round(hate,4),
        "groyper":      round(groyper,4),
        "stalker_sig":  round(s.get("harassment and stalking behavior",0),4),
        "impersonation":round(s.get("identity impersonation",0),4),
        "bot_signal":   round(s.get("coordinated bot attack",0),4),
        "neutral":      round(s.get("neutral friendly message",0),4),
        "overall":      round(overall,4),
    }

def persona_masking(candidate: str, cand_msgs: List[str],
                     known_users: Dict[str,List[str]]) -> Tuple[float,str]:
    cand_norm = norm_username(candidate)
    best_sim  = 0.0; best_match = ""
    cand_text = " ".join(cand_msgs[:15])
    cand_emb  = embed(cand_text[:400]) if cand_text else None
    for ku, kmsgs in known_users.items():
        if ku == candidate: continue
        kn = norm_username(ku)
        # İsim Levenshtein benzeri
        max_len = max(len(cand_norm),len(kn),1)
        common  = sum(1 for c in cand_norm if c in kn)
        name_s  = common/max_len
        # Embedding
        emb_s = 0.0
        if cand_emb and kmsgs:
            ke = embed(" ".join(kmsgs[:15])[:400])
            if ke: emb_s = max(0.0, 1.0-cosine_dist(cand_emb,ke))
        combined = 0.35*name_s + 0.65*emb_s
        if combined > best_sim:
            best_sim = combined; best_match = ku
    return round(best_sim,4), best_match

# ═══════════════════════════════════════════════════════════════════════════════
# § 11 — KONU MODELLEMESİ (Katman 6: BERTopic + LDA)
# ═══════════════════════════════════════════════════════════════════════════════
_topic_model = None
_lda_model   = None
_lda_vec     = None
_N_TOPICS    = 20

def fit_topics(docs: List[str]):
    global _topic_model, _lda_model, _lda_vec
    if len(docs) < 30: return
    # BERTopic
    if _BERTOPIC:
        try:
            _topic_model = BERTopic(nr_topics=_N_TOPICS, language="multilingual",
                                     verbose=False, calculate_probabilities=True)
            _topic_model.fit(docs)
            log.info("✅ BERTopic eğitildi (%d döküman)", len(docs))
            return
        except Exception as e:
            log.warning("BERTopic başarısız: %s, LDA kullanılıyor", e)
    # LDA fallback
    try:
        _lda_vec = CountVectorizer(max_features=2000, ngram_range=(1,2), min_df=2)
        dtm = _lda_vec.fit_transform(docs)
        _lda_model = LatentDirichletAllocation(n_components=_N_TOPICS, random_state=42,
                                                max_iter=15)
        _lda_model.fit(dtm)
        log.info("✅ LDA eğitildi (%d döküman)", len(docs))
    except Exception as e:
        log.warning("LDA hatası: %s", e)

def user_topic_vec(msgs: List[str]) -> np.ndarray:
    text = " ".join(msgs[:30])
    if _topic_model:
        try:
            _, probs = _topic_model.transform([text])
            v = np.array(probs[0]) if hasattr(probs[0],"__len__") else np.zeros(_N_TOPICS)
            return v[:_N_TOPICS] if len(v) >= _N_TOPICS else np.pad(v,(_N_TOPICS-len(v),0))
        except: pass
    if _lda_model and _lda_vec:
        try:
            dtm = _lda_vec.transform([text])
            return _lda_model.transform(dtm)[0]
        except: pass
    return np.zeros(_N_TOPICS)

# ═══════════════════════════════════════════════════════════════════════════════
# § 12 — ZAMANSAL ANALİZ (Katman 7)
# ═══════════════════════════════════════════════════════════════════════════════
def temporal_fp(timestamps: List[int]) -> dict:
    if not timestamps: return {}
    ts = np.array(sorted(timestamps))
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]
    hours = [d.hour for d in dts]; days = [d.weekday() for d in dts]
    diffs = np.diff(ts).astype(float) if len(ts)>1 else np.array([0.0])
    return {
        "peak_hour":    int(Counter(hours).most_common(1)[0][0]) if hours else 0,
        "active_days":  list(set(days)),
        "mean_interval":round(float(np.mean(diffs)),2),
        "std_interval": round(float(np.std(diffs)),2),
        "burstiness":   round(burstiness(list(ts)),4),
        "min_interval": round(float(diffs.min()),2),
        "max_interval": round(float(diffs.max()),2),
        "total_span_days": round((max(timestamps)-min(timestamps))/86400,2) if len(timestamps)>1 else 0,
    }

def time_sim(tfp_a: dict, tfp_b: dict) -> float:
    if not tfp_a or not tfp_b: return 0.0
    hour_s = 1 - abs(tfp_a.get("peak_hour",0)-tfp_b.get("peak_hour",0))/24.0
    da = set(tfp_a.get("active_days",[])); db = set(tfp_b.get("active_days",[]))
    day_s  = len(da&db)/max(len(da|db),1)
    burst_s = 1 - min(1,abs(tfp_a.get("burstiness",0)-tfp_b.get("burstiness",0)))
    return float(0.40*hour_s + 0.35*day_s + 0.25*burst_s)

def ks_test(ts_a: List[int], ts_b: List[int]) -> Tuple[float,float]:
    if len(ts_a)<3 or len(ts_b)<3: return 0.0,1.0
    da = np.diff(sorted(ts_a)).astype(float)
    db = np.diff(sorted(ts_b)).astype(float)
    s,p = stats.ks_2samp(da,db)
    return float(s),float(p)

def pearson_activity(ts_a: List[int], ts_b: List[int], bin_sec=3600) -> float:
    if not ts_a or not ts_b: return 0.0
    all_ts = ts_a+ts_b; t0 = min(all_ts); t1 = max(all_ts)
    if t1==t0: return 0.0
    n = max(10,(t1-t0)//bin_sec+1)
    ba = np.zeros(n); bb = np.zeros(n)
    for t in ts_a: ba[min(int((t-t0)//bin_sec),n-1)]+=1
    for t in ts_b: bb[min(int((t-t0)//bin_sec),n-1)]+=1
    if ba.std()<1e-9 or bb.std()<1e-9: return 0.0
    c = np.corrcoef(ba,bb)[0,1]
    return float(c) if not np.isnan(c) else 0.0

def mutual_info(ts_a: List[int], ts_b: List[int], bin_sec=3600) -> float:
    from sklearn.metrics import mutual_info_score
    if not ts_a or not ts_b: return 0.0
    all_ts = ts_a+ts_b; t0 = min(all_ts); t1 = max(all_ts)
    if t1==t0: return 0.0
    n = max(10,(t1-t0)//bin_sec+1)
    ba = np.zeros(n,dtype=int); bb = np.zeros(n,dtype=int)
    for t in ts_a: ba[min(int((t-t0)//bin_sec),n-1)]+=1
    for t in ts_b: bb[min(int((t-t0)//bin_sec),n-1)]+=1
    try: return float(mutual_info_score((ba>0).astype(int),(bb>0).astype(int)))
    except: return 0.0

def changepoint(values: List[float]) -> List[int]:
    """CUSUM tabanlı davranış kırılma noktası tespiti"""
    if len(values)<4: return []
    a = np.array(values,float); mu = a.mean(); sg = a.std()+1e-9
    cusum = np.cumsum((a-mu)/sg); thr = 2.8; pts = []
    for i in range(1,len(cusum)-1):
        if (abs(cusum[i])>thr and abs(cusum[i])>abs(cusum[i-1])
                and abs(cusum[i])>abs(cusum[i+1])):
            pts.append(i)
    return pts

# ═══════════════════════════════════════════════════════════════════════════════
# § 13 — GRAF KÜMELEMESİ (Katman 8: DBSCAN, Spectral, Louvain, PageRank)
# ═══════════════════════════════════════════════════════════════════════════════
def build_graph(users: List[str], sim_mat: np.ndarray, thr: float = None) -> nx.Graph:
    if thr is None: thr = CFG.get("similarity_threshold",0.65)
    G = nx.Graph(); G.add_nodes_from(users)
    n = len(users)
    for i in range(n):
        for j in range(i+1,n):
            s = float(sim_mat[i,j])
            if s >= thr: G.add_edge(users[i],users[j],weight=s)
    return G

def dbscan_cluster(sim_mat: np.ndarray, thr: float = None) -> np.ndarray:
    if thr is None: thr = CFG.get("similarity_threshold",0.65)
    dist = np.clip(1-sim_mat, 0, 1)
    return DBSCAN(eps=1-thr, min_samples=2, metric="precomputed").fit_predict(dist)

def spectral_cluster(sim_mat: np.ndarray, k: int = 5) -> np.ndarray:
    n = sim_mat.shape[0]; k = min(k, n-1)
    if k < 2: return np.zeros(n,dtype=int)
    try:
        return SpectralClustering(n_clusters=k, affinity="precomputed",
                                   random_state=42).fit_predict(np.clip(sim_mat,0,1))
    except: return np.zeros(n,dtype=int)

def louvain_cluster(G: nx.Graph) -> Dict[str,int]:
    if community_louvain is None: return {n:0 for n in G.nodes()}
    try: return community_louvain.best_partition(G, weight="weight")
    except: return {n:0 for n in G.nodes()}

def pagerank(G: nx.Graph) -> Dict[str,float]:
    if not G.nodes(): return {}
    try: return nx.pagerank(G, weight="weight", alpha=0.85)
    except: return {n:1/max(len(G.nodes()),1) for n in G.nodes()}

def gmm_detect(embs: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray,np.ndarray]:
    """GMM ile anomali tespiti — Bot/Hater/Normal üç bileşen"""
    if len(embs) < n_components*2:
        return np.zeros(len(embs),dtype=int), np.ones(len(embs))
    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                               random_state=42, max_iter=100)
        gmm.fit(embs)
        labels = gmm.predict(embs)
        scores = gmm.score_samples(embs)
        return labels, scores
    except Exception as e:
        log.warning("GMM hatası: %s", e)
        return np.zeros(len(embs),dtype=int), np.ones(len(embs))

def build_sim_matrix(users: List[str]) -> Tuple[List[str],np.ndarray]:
    n = len(users)
    if n == 0: return [], np.zeros((0,0))
    # Kullanıcı verilerini topla
    user_data = {}
    for a in users:
        msgs_rows = get_user_msgs(a)
        msgs  = [r["message"] for r in msgs_rows]
        tss   = [int(r["timestamp"] or 0) for r in msgs_rows]
        text  = " ".join(msgs[:50])
        user_data[a] = {"msgs":msgs,"timestamps":tss,"text":text,
                         "ngram":ngram_fp(text),"tfp":temporal_fp(tss)}
    # Embeddings
    texts = [user_data[a]["text"] or "empty" for a in users]
    embs  = embed_batch(texts)
    sim   = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if i==j: sim[i][j]=1.0; continue
            ai = users[i]; aj = users[j]
            # Embedding sim
            emb_s = 0.0
            if embs is not None:
                es = 1-cosine_dist(embs[i],embs[j])
                emb_s = max(0.0,float(es))
            # N-gram Jaccard
            ng_s = jaccard(user_data[ai]["ngram"],user_data[aj]["ngram"])
            # Temporal sim
            ts_s = time_sim(user_data[ai]["tfp"],user_data[aj]["tfp"])
            # Typo sim (karşılaştır)
            tyi = typo_fp(user_data[ai]["msgs"])
            tyj = typo_fp(user_data[aj]["msgs"])
            typ_s = 0.0
            for k in ["uppercase_ratio","punct_density","question_rate","exclamation_rate"]:
                typ_s += 1-abs(tyi.get(k,0)-tyj.get(k,0))
            typ_s /= 4
            c = composite_sim(emb_s,ng_s,typ_s,ts_s,0.0)
            sim[i][j]=c; sim[j][i]=c
    return users,sim

def run_clustering(users: List[str] = None) -> dict:
    """
    Kullanıcı kümelemesi.
    DÜZELTME: Sadece gerçekten mesajı olan kullanıcıları kümele.
    Mesajsız kullanıcılar anlamsız "empty" embedding alır ve
    tüm benzerlik matrisini bozar — bunlar filtrelenir.
    Mesajsız profiller DB'de GREEN olarak işaretlenir.
    """
    # Mesajsız profillere varsayılan GREEN ata (UNKNOWN yerine)
    db_exec(
        "UPDATE user_profiles SET threat_level='GREEN' "
        "WHERE (threat_level IS NULL OR threat_level='') AND msg_count=0"
    )

    if users is None:
        # Sadece gerçek mesajı olan kullanıcılar — O(n^2) matris için kritik
        rows = db_exec(
            "SELECT DISTINCT up.author FROM user_profiles up "
            "INNER JOIN messages m ON m.author=up.author "
            "WHERE m.deleted=0 GROUP BY up.author HAVING COUNT(m.id)>0",
            fetch="all"
        ) or []
        users = [r["author"] for r in rows]

    if len(users) < 3:
        return {
            "error": f"Kümeleme için yeterli aktif kullanıcı yok (mevcut: {len(users)})",
            "clusters": {}, "graph_data": {"nodes": [], "links": []}
        }

    # Büyük kullanıcı setlerini örnekle (performans)
    MAX_CLUSTER_USERS = 500
    if len(users) > MAX_CLUSTER_USERS:
        log.warning(
            "Kümeleme: %d kullanıcı > MAX=%d, en yüksek tehdit skorlu %d alınıyor",
            len(users), MAX_CLUSTER_USERS, MAX_CLUSTER_USERS
        )
        scored = db_exec(
            f"SELECT author, threat_score FROM user_profiles "
            f"WHERE author IN ({','.join(['?']*len(users))}) "
            f"ORDER BY threat_score DESC LIMIT ?",
            tuple(users) + (MAX_CLUSTER_USERS,),
            fetch="all"
        ) or []
        users = [r["author"] for r in scored]

    log.info("Kümeleme başlıyor: %d aktif kullanıcı", len(users))
    user_list, sim_mat = build_sim_matrix(users)
    G         = build_graph(user_list, sim_mat)
    louvain   = louvain_cluster(G)
    pr        = pagerank(G)
    db_labels = dbscan_cluster(sim_mat)

    # Kimlik eşleşmeleri kaydet
    thr = CFG.get("similarity_threshold", 0.65)
    for i in range(len(user_list)):
        for j in range(i + 1, len(user_list)):
            s = float(sim_mat[i, j])
            if s >= thr:
                db_exec(
                    "INSERT OR IGNORE INTO identity_links"
                    "(user_a,user_b,sim_score,method,confidence)"
                    " VALUES(?,?,?,?,?)",
                    (user_list[i], user_list[j], round(s, 4), "combined", round(s, 4))
                )

    # Küme liderleri (PageRank)
    clusters: Dict[int, List[str]] = {}
    for u, cid in louvain.items():
        clusters.setdefault(cid, []).append(u)

    leaders = {
        cid: max(members, key=lambda x: pr.get(x, 0))
        for cid, members in clusters.items()
    }
    for cid, members in clusters.items():
        db_exec(
            "INSERT INTO graph_clusters(cluster_id,members,algorithm,pagerank_leaders)"
            " VALUES(?,?,?,?)",
            (cid, json.dumps(members), "louvain", json.dumps({cid: leaders.get(cid, "")}))
        )
        for m in members:
            upsert_profile(m, {"cluster_id": cid, "pagerank_score": round(pr.get(m, 0), 5)})

    # Tehdit seviyesini tek sorguda al (N+1 sorgu yerine)
    threat_map: Dict[str, str] = {}
    if user_list:
        rows_t = db_exec(
            f"SELECT author, threat_level FROM user_profiles "
            f"WHERE author IN ({','.join(['?']*len(user_list))})",
            tuple(user_list), fetch="all"
        ) or []
        threat_map = {r["author"]: (r["threat_level"] or "GREEN") for r in rows_t}

    graph_data = {
        "nodes": [
            {
                "id":       u,
                "group":    int(louvain.get(u, 0)),
                "pagerank": round(pr.get(u, 0), 5),
                "threat":   threat_map.get(u, "GREEN"),
            }
            for u in user_list
        ],
        "links": [
            {"source": u, "target": v, "value": round(float(G[u][v]["weight"]), 3)}
            for u, v in G.edges()
        ],
    }
    log.info("✅ Kümeleme tamamlandı: %d küme, %d kullanıcı", len(clusters), len(user_list))
    return {
        "clusters":   clusters,
        "graph_data": graph_data,
        "dbscan":     dict(zip(user_list, db_labels.tolist())),
        "pagerank":   {u: round(pr.get(u, 0), 5) for u in user_list},
        "leaders":    leaders,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# § 14 — Q-LEARNING & DQN (Katman 9)
# ═══════════════════════════════════════════════════════════════════════════════
class QTable:
    def __init__(self, dims=(10,10,10,10,10), n_actions=6,
                  alpha=0.15, gamma=0.90, eps=0.10):
        self.Q = np.zeros((*dims,n_actions)); self.alpha=alpha
        self.gamma=gamma; self.eps=eps; self.n_actions=n_actions; self.step=0
    def state(self,count,rep,div,hs,burst):
        return (min(9,count//5),min(9,int(rep*10)),min(9,int(div*10)),
                min(9,int(hs*10)),min(9,int((burst+1)/2*10)))
    def act(self, s):
        if random.random() < self.eps: return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[s]))
    def update(self, s, a, r, sn):
        self.step+=1; self.eps=max(0.01,self.eps*0.9999)
        self.Q[s][a]+=self.alpha*(r+self.gamma*np.max(self.Q[sn])-self.Q[s][a])
    def save(self,p="qtable.npy"): np.save(p,self.Q)
    def load(self,p="qtable.npy"):
        if Path(p).exists(): self.Q=np.load(p)

_qtable = QTable()

if _TORCH:
    class DQNet(nn.Module):
        def __init__(self,in_dim=64,n_act=6):
            super().__init__()
            self.net=nn.Sequential(
                nn.Linear(in_dim,256),nn.ReLU(),nn.Dropout(0.1),
                nn.Linear(256,128),nn.ReLU(),
                nn.Linear(128,64),nn.ReLU(),
                nn.Linear(64,n_act)
            )
        def forward(self,x): return self.net(x)

    class DQNAgent:
        def __init__(self,in_dim=64,n_act=6,lr=1e-4,gamma=0.90,eps=0.10):
            self.dev=torch.device("cuda" if DEVICE=="cuda" else "cpu")
            self.n_act=n_act; self.gamma=gamma; self.eps=eps; self.step=0
            self.online=DQNet(in_dim,n_act).to(self.dev)
            self.target=DQNet(in_dim,n_act).to(self.dev)
            self.target.load_state_dict(self.online.state_dict())
            self.opt=optim.Adam(self.online.parameters(),lr=lr)
            self.mem=deque(maxlen=10000); self.bs=64; self.tgt_upd=100
        def feat(self,p):
            f=[float(p.get("msg_count",0))/100,float(p.get("bot_prob",0)),
               float(p.get("hate_score",0)),float(p.get("stalker_score",0)),
               float(p.get("human_score",0.5)),float(p.get("impersonator_prob",0)),
               float(p.get("threat_score",0)),float(p.get("antisemitism_score",0))]
            f+=[0.0]*(64-len(f))
            return torch.tensor(f[:64],dtype=torch.float32,device=self.dev)
        def act(self,s):
            if random.random()<self.eps: return random.randrange(self.n_act)
            with torch.no_grad(): return int(self.online(s.unsqueeze(0)).argmax())
        def remember(self,s,a,r,sn,done): self.mem.append((s,a,r,sn,done))
        def train(self):
            if len(self.mem)<self.bs: return
            b=random.sample(self.mem,self.bs)
            ss=torch.stack([x[0] for x in b]); aa=torch.tensor([x[1] for x in b],device=self.dev)
            rr=torch.tensor([x[2] for x in b],dtype=torch.float32,device=self.dev)
            sns=torch.stack([x[3] for x in b]); ds=torch.tensor([x[4] for x in b],dtype=torch.float32,device=self.dev)
            cq=self.online(ss).gather(1,aa.unsqueeze(1)).squeeze()
            with torch.no_grad(): nq=self.target(sns).max(1)[0]
            tq=rr+self.gamma*nq*(1-ds)
            loss=nn.MSELoss()(cq,tq)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.step+=1; self.eps=max(0.01,self.eps*0.9999)
            if self.step%self.tgt_upd==0:
                self.target.load_state_dict(self.online.state_dict())
    _dqn=DQNAgent()
else:
    _dqn=None

def rl_act(profile: dict) -> Tuple[int,str]:
    s = _qtable.state(
        int(profile.get("msg_count",0)),
        float(profile.get("bot_prob",0)),
        float(profile.get("human_score",0.5)),
        float(profile.get("human_score",0.5)),
        float(profile.get("threat_score",0))
    )
    a = _qtable.act(s)
    return a, ACTION_NAMES.get(a,"UNKNOWN")

def rl_update(profile: dict, action: int, reward: float, next_profile: dict):
    s  = _qtable.state(int(profile.get("msg_count",0)),float(profile.get("bot_prob",0)),
                       float(profile.get("human_score",0.5)),float(profile.get("human_score",0.5)),
                       float(profile.get("threat_score",0)))
    sn = _qtable.state(int(next_profile.get("msg_count",0)),float(next_profile.get("bot_prob",0)),
                       float(next_profile.get("human_score",0.5)),float(next_profile.get("human_score",0.5)),
                       float(next_profile.get("threat_score",0)))
    _qtable.update(s, action, reward, sn)
    if _dqn:
        fs=_dqn.feat(profile); fsn=_dqn.feat(next_profile)
        _dqn.remember(fs,action,reward,fsn,False); _dqn.train()

# ═══════════════════════════════════════════════════════════════════════════════
# § 15 — OYUN KURAMI (Katman 10: Nash, Grim Trigger, Folk, Bayes)
# ═══════════════════════════════════════════════════════════════════════════════
def nash_equilibria() -> List[Tuple]:
    eq = []
    for i in range(len(MOD_ACTIONS)):
        for j in range(len(ACTOR_ACTIONS)):
            mp,ap = PAYOFF[i,j]
            if (mp==max(PAYOFF[k,j][0] for k in range(len(MOD_ACTIONS))) and
                ap==max(PAYOFF[i,l][1] for l in range(len(ACTOR_ACTIONS)))):
                eq.append((i,j,MOD_ACTIONS[i],ACTOR_ACTIONS[j],float(mp),float(ap)))
    return eq

def grim_trigger(history: List[str]) -> str:
    bad = {"TROLL","FLOOD","IMPERSONATE","COORDINATE","HATER","BOT"}
    return "BAN" if any(a in bad for a in history) else "MONITOR"

def tit_for_tat(prev_action: str) -> str:
    return "WARN" if prev_action in {"TROLL","FLOOD","IMPERSONATE"} else "IGNORE"

def bayes_update(prior: Dict[str,float], likelihood: Dict[str,float]) -> Dict[str,float]:
    denom = sum(likelihood.get(k,1e-9)*prior.get(k,0.25) for k in prior)+1e-12
    return {k:(likelihood.get(k,1e-9)*prior.get(k,0.25))/denom for k in prior}

def folk_theorem_check(ts: float, vc: float, vd: float, vp: float) -> bool:
    """Folk teoremi: delta yeterince büyükse işbirliği denge"""
    if vc - vd >= -1e-9: return True
    if vp >= vc: return False
    delta_min = (vd-vc)/(vd-vp) if abs(vd-vp) > 1e-9 else 1.0
    return ts >= delta_min

def game_score(profile: dict, history: List[str] = None) -> dict:
    if history is None: history = []
    ts  = float(profile.get("threat_score",0))
    prior = {"BOT":0.15,"HATER":0.15,"STALKER":0.10,"GROYPER":0.05,"NORMAL":0.55}
    likelihood = {"BOT":ts*0.8,"HATER":ts*0.9,"STALKER":ts*0.6,"GROYPER":ts*0.7,
                  "NORMAL":max(0.01,1-ts)}
    posterior = bayes_update(prior, likelihood)
    mod_action = grim_trigger(history)
    nash = nash_equilibria()
    dominant = max(posterior, key=posterior.get)
    folk_ok = folk_theorem_check(0.9, 2.0, -1.0, -3.0)
    return {"posterior":posterior,"mod_action":mod_action,"nash":nash,
            "dominant":dominant,"folk_theorem_stable":folk_ok}

# ═══════════════════════════════════════════════════════════════════════════════
# § 16 — BAYES/HMM/KALMAN/GMM (Katman 11)
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanFilter:
    def __init__(self, q=0.01, r=0.1):
        self.x=0.0; self.P=1.0; self.Q=q; self.R=r
    def step(self, z: float) -> float:
        self.x=self.x; self.P=self.P+self.Q
        K=self.P/(self.P+self.R)
        self.x+=K*(z-self.x); self.P=(1-K)*self.P
        return self.x

_kalmans: Dict[str,KalmanFilter] = {}

def kalman_update(author: str, score: float) -> float:
    if author not in _kalmans: _kalmans[author]=KalmanFilter()
    return _kalmans[author].step(score)

def hmm_states(scores: List[float]) -> List[str]:
    if not _HMM or len(scores) < 3:
        return ["NORMAL" if s<0.3 else "SUSPICIOUS" if s<0.6 else "ATTACKING" for s in scores]
    try:
        obs = np.array(scores).reshape(-1,1)
        m = hmmlearn_hmm.GaussianHMM(n_components=3, covariance_type="diag",
                                      n_iter=100, random_state=42)
        m.fit(obs); hidden = m.predict(obs)
        means = [float(m.means_[i][0]) for i in range(3)]
        order = sorted(range(3),key=lambda x:means[x])
        names = {order[0]:"NORMAL",order[1]:"LURKING",order[2]:"ATTACKING"}
        return [names.get(s,"NORMAL") for s in hidden]
    except: return ["NORMAL"]*len(scores)

def naive_bayes_classify(texts: List[str], labels: List[str], new_text: str) -> Dict[str,float]:
    if len(texts)<10 or len(set(labels))<2: return {"NORMAL":0.5,"THREAT":0.5}
    try:
        v = TfidfVectorizer(max_features=500)
        X = v.fit_transform(texts)
        nb = ComplementNB(); nb.fit(X,labels)
        return dict(zip(nb.classes_, nb.predict_proba(v.transform([new_text]))[0]))
    except: return {"NORMAL":0.5,"THREAT":0.5}

def wasserstein_sim(p: np.ndarray, q: np.ndarray) -> float:
    try:
        p = np.asarray(p,float)+1e-12; q = np.asarray(q,float)+1e-12
        p/=p.sum(); q/=q.sum()
        return float(1-min(1,wasserstein_distance(p,q)))
    except: return 0.0

def theorem_router(profile: dict) -> str:
    ts = float(profile.get("threat_score",0))
    mc = int(profile.get("msg_count",0))
    if mc>20 and ts>0.5: return "HMM"
    if ts>0.6: return "BayesianUpdate"
    if ts<0.2 and mc>5: return "MarkovChain"
    return "KalmanFilter"

# ═══════════════════════════════════════════════════════════════════════════════
# § 17 — OLLAMA (SADECE YORUM ANALİZİ İÇİN)
# ═══════════════════════════════════════════════════════════════════════════════
def ollama_analyze(author: str, msgs: List[str], task: str = "threat") -> dict:
    if not _OLLAMA:
        return {"summary":"Ollama yüklü değil","threat_indicators":[],
                "recommended_action":"MONITOR","confidence":0.0}
    model = CFG.get("ollama_model","phi4:14b")
    ctx   = "\n".join(f"- {m[:150]}" for m in msgs[:12])
    prompt = f"""YouTube kanal moderatörüsün. Kullanıcı @{author} yorumlarını analiz et.

Mesajlar:
{ctx}

Tehdit kategorileri: antisemitizm, groyper, nefret söylemi, stalker, bot, kimlik örtüsü, koordineli saldırı, normal.

SADECE JSON döndür (başka metin yok):
{{
  "summary": "kısa Türkçe özet",
  "threat_indicators": ["göstergeler"],
  "identity_clues": ["ipuçları"],
  "category": "ANTISEMITE|GROYPER|HATER|STALKER|BOT|IMPERSONATOR|COORDINATED|NORMAL",
  "recommended_action": "BAN|WARN|MONITOR|IGNORE",
  "confidence": 0.0
}}"""
    try:
        resp = ollama_sdk.chat(model=model,
                               messages=[{"role":"user","content":prompt}],
                               options={"temperature":0.05,"num_predict":400})
        raw = resp["message"]["content"].strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m: return json.loads(m.group())
        return {"summary":raw,"threat_indicators":[],"recommended_action":"MONITOR","confidence":0.3}
    except Exception as e:
        log.warning("Ollama hatası: %s", e)
        return {"summary":str(e),"threat_indicators":[],"recommended_action":"MONITOR","confidence":0.0}

def ollama_rag(query: str, context_msgs: List[Dict]) -> str:
    """RAG: ilgili mesajları bağlam olarak Ollama'ya ver"""
    if not _OLLAMA: return "Ollama mevcut değil"
    model  = CFG.get("ollama_model","phi4:14b")
    # Önbellekte var mı?
    qhash  = hashlib.md5(query.encode()).hexdigest()
    cached = db_exec("SELECT response FROM rag_cache WHERE query_hash=?", (qhash,), fetch="one")
    if cached: return cached["response"]
    ctx = "\n".join(f"@{m.get('author','?')}: {m.get('message','')[:100]}" for m in context_msgs[:10])
    prompt = f"Kanal moderatörü sorusu: {query}\n\nİlgili mesajlar:\n{ctx}\n\nKısa Türkçe cevap ver:"
    try:
        resp = ollama_sdk.chat(model=model,
                               messages=[{"role":"user","content":prompt}],
                               options={"temperature":0.1,"num_predict":300})
        ans = resp["message"]["content"].strip()
        db_exec("INSERT OR REPLACE INTO rag_cache(query_hash,query,response) VALUES(?,?,?)",
                (qhash,query,ans))
        return ans
    except Exception as e:
        return f"Ollama hatası: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# § 18 — YORUM SİLME (Selenium — Moderatör)
# ═══════════════════════════════════════════════════════════════════════════════
def delete_comment(driver, video_id: str, author: str,
                    msg_preview: str, max_scroll: int = 40) -> bool:
    if not driver: return False
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}"); time.sleep(3)
        for _ in range(5):
            driver.execute_script("window.scrollBy(0,600)"); time.sleep(0.8)
        for _ in range(max_scroll):
            cmts = driver.find_elements(By.CSS_SELECTOR,
                "ytd-comment-renderer,ytd-comment-thread-renderer")
            for cmt in cmts:
                try:
                    a = cmt.find_element(By.CSS_SELECTOR,"#author-text").text.strip()
                    t = cmt.find_element(By.CSS_SELECTOR,"#content-text").text.strip()
                    if author.lower() in a.lower() and msg_preview[:30].lower() in t.lower():
                        btn = cmt.find_element(By.CSS_SELECTOR,
                            "yt-icon-button#action-menu,button.dropdown-trigger")
                        driver.execute_script("arguments[0].click()",btn); time.sleep(0.8)
                        for sel in ["[aria-label*='Remove']","[aria-label*='Kaldır']",
                                    ".yt-simple-endpoint[role='menuitem']"]:
                            items = driver.find_elements(By.CSS_SELECTOR,sel)
                            for it in items:
                                txt = it.text.lower()
                                if "remove" in txt or "kaldır" in txt or "sil" in txt:
                                    it.click(); time.sleep(0.8)
                                    try:
                                        ok = WebDriverWait(driver,4).until(
                                            EC.element_to_be_clickable(
                                                (By.CSS_SELECTOR,"yt-button-renderer[dialog-confirm] button")))
                                        ok.click()
                                    except: pass
                                    db_exec("UPDATE messages SET deleted=1 WHERE"
                                            " author=? AND message LIKE ?",
                                            (author,f"%{msg_preview[:30]}%"))
                                    log.info("✅ Yorum silindi: @%s", author)
                                    return True
                except StaleElementReferenceException: pass
                except: pass
            driver.execute_script("window.scrollBy(0,1000)"); time.sleep(1)
        return False
    except Exception as e:
        log.error("Yorum silme hatası: %s", e); return False

def delete_live_msg(driver, video_id: str, author: str, msg_preview: str) -> bool:
    if not driver: return False
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}"); time.sleep(3)
        # Live chat iframe
        for fr in driver.find_elements(By.TAG_NAME,"iframe"):
            if "live_chat" in (fr.get_attribute("src") or ""):
                driver.switch_to.frame(fr); break
        time.sleep(2)
        items = driver.find_elements(By.CSS_SELECTOR,"yt-live-chat-text-message-renderer")
        for it in items:
            try:
                a = it.find_element(By.ID,"author-name").text
                t = it.find_element(By.ID,"message").text
                if a.lower()==author.lower() and msg_preview[:20].lower() in t.lower():
                    ActionChains(driver).context_click(it).perform(); time.sleep(0.5)
                    try:
                        rb = WebDriverWait(driver,3).until(EC.presence_of_element_located(
                            (By.XPATH,"//*[contains(text(),'Remove') or contains(text(),'Kaldır')]")))
                        rb.click()
                        log.info("✅ Canlı mesaj silindi: @%s", author)
                        driver.switch_to.default_content(); return True
                    except: pass
            except: pass
        driver.switch_to.default_content(); return False
    except Exception as e:
        log.error("Canlı silme hatası: %s", e)
        try: driver.switch_to.default_content()
        except: pass
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# § 19 — ANA ANALİZ PİPELİNE
# ═══════════════════════════════════════════════════════════════════════════════
def threat_level(profile: dict) -> dict:
    c = (0.30*float(profile.get("hate_score",0)) +
         0.25*float(profile.get("bot_prob",0)) +
         0.20*float(profile.get("stalker_score",0)) +
         0.15*float(profile.get("impersonator_prob",0)) +
         0.10*(1-float(profile.get("human_score",0.5))))
    c = max(0.0,min(1.0,c))
    # Anti-semitizm direkt CRIMSON'a çeker
    if float(profile.get("antisemitism_score",0)) >= 0.6: c = max(c, 0.85)
    if float(profile.get("groyper_score",0)) >= 0.5:      c = max(c, 0.75)
    lvl = ("CRIMSON" if c>=0.85 else "RED" if c>=0.70 else
           "ORANGE"  if c>=0.50 else "YELLOW" if c>=0.25 else "GREEN")
    return {"score":round(c,4),"level":lvl,"color":COLOR_MAP[lvl]}

def analyze_user(author: str, run_ollama: bool = True) -> dict:
    msgs_rows  = get_user_msgs(author)
    if not msgs_rows:
        return {"author":author,"error":"Mesaj yok","threat_level":"GREEN","threat_score":0}
    msgs   = [r["message"] for r in msgs_rows]
    tss    = [int(r["timestamp"] or 0) for r in msgs_rows]
    text   = " ".join(msgs)
    sample = " ".join(msgs[:8])[:800]

    # NLP
    tfidf_v = tfidf_vec(text)
    ngram_v = ngram_fp(text)
    typo_v  = typo_fp(msgs)
    pos_v   = pos_profile(text[:800])
    time_v  = temporal_fp(tss)
    emb     = embed(text[:600])

    # Bot
    b_score = bot_score(msgs, tss)
    h_score = max(0.0, 1.0-b_score)
    burst   = burstiness(tss)

    # Nefret
    h_res   = hate_scores(sample)
    antisem = h_res["antisemitism"]
    groyper = h_res["groyper"]
    hate    = h_res["overall"]

    # Stalker (Hawkes: kanal sahibinin mesajlarına tepki hızı)
    # Placeholder: host_ts boş, ilerleyen aşamada doldurulabilir
    stalker = min(1.0, abs(burst)*0.5 + h_res.get("stalker_sig",0)*0.5)

    # Kalman
    kal_score = kalman_update(author, hate*0.5+b_score*0.5)

    # Tehdit bileşik
    thr = threat_level({"hate_score":hate,"bot_prob":b_score,
                         "stalker_score":stalker,"impersonator_prob":0.0,
                         "human_score":h_score,"antisemitism_score":antisem,
                         "groyper_score":groyper})
    # HMM
    existing = db_exec("SELECT threat_score FROM user_profiles WHERE author=?",
                       (author,), fetch="one")
    hist_score = float(existing["threat_score"]) if existing else 0.0
    hmm_s = hmm_states([hist_score, thr["score"]])[-1]

    # Oyun kuramı
    game = game_score({"threat_score":thr["score"]}, [])
    mod_act = game["mod_action"]

    # Q-Learning
    rl_a, rl_name = rl_act({
        "msg_count":len(msgs),"bot_prob":b_score,
        "human_score":h_score,"threat_score":thr["score"]
    })

    # Güncelle
    upd = {
        "msg_count":      len(msgs),
        "human_score":    round(h_score,4),
        "bot_prob":       round(b_score,4),
        "hate_score":     round(hate,4),
        "stalker_score":  round(stalker,4),
        "antisemitism_score": round(antisem,4),
        "groyper_score":  round(groyper,4),
        "tfidf_json":     json.dumps(tfidf_v.tolist()[:30]),
        "ngram_json":     json.dumps(dict(ngram_v.most_common(20))),
        "typo_json":      json.dumps(typo_v),
        "pos_json":       json.dumps(pos_v),
        "temporal_json":  json.dumps(time_v),
        "identity_vector":json.dumps(h_res),
        "threat_level":   thr["level"],
        "threat_score":   thr["score"],
        "hmm_state":      hmm_s,
        "game_strategy":  mod_act,
        "kalman_score":   round(float(kal_score),4),
        "first_seen":     min(tss) if tss else 0,
        "last_seen":      max(tss) if tss else 0,
    }

    # Ollama (SADECE yorum analizi, kritik durumlarda)
    ollama_res = {}
    if run_ollama and _OLLAMA and (hate > 0.35 or b_score > 0.6 or antisem > 0.2):
        ollama_res = ollama_analyze(author, msgs, "threat")
        upd["ollama_summary"] = ollama_res.get("summary","")[:1000]
        upd["ollama_action"]  = ollama_res.get("recommended_action","MONITOR")

    upsert_profile(author, upd)

    # ChromaDB kaydet
    if emb:
        chroma_upsert(_ch_users, f"user_{author}", emb,
                      {"type":"user","author":author,"threat":thr["level"]})

    # Self-feeding dataset
    if thr["score"] >= 0.5:
        label = _infer_label(h_res, b_score)
        conf  = 1 if thr["score"] >= 0.85 else 0
        db_exec("INSERT OR IGNORE INTO dataset(msg_id,author,message,label,confirmed,source)"
                " VALUES(?,?,?,?,?,'auto')",
                (f"usr_{author}",author,sample[:500],label,conf))

    return {
        "author":author,"msg_count":len(msgs),
        "bot_prob":b_score,"hate_score":hate,"stalker_score":stalker,
        "human_score":h_score,"antisemitism_score":antisem,"groyper_score":groyper,
        "threat_level":thr["level"],"threat_score":thr["score"],"threat_color":thr["color"],
        "hmm_state":hmm_s,"game_strategy":mod_act,"rl_action":rl_name,
        "hate_breakdown":h_res,"temporal":time_v,"typo":typo_v,
        "ollama":ollama_res,"kalman_score":float(kal_score),
        "recommended_action":mod_act,
    }

def _infer_label(h: dict, b: float) -> str:
    if h.get("antisemitism",0)>0.4: return "ANTISEMITE"
    if h.get("groyper",0)>0.4:      return "GROYPER"
    if h.get("hate_general",0)>0.4: return "HATER"
    if b>0.65:                       return "BOT"
    if h.get("stalker_sig",0)>0.4:  return "STALKER"
    if h.get("impersonation",0)>0.4:return "IMPERSONATOR"
    return "SUSPICIOUS"

# Dataset ve yeniden eğitim
def check_retrain() -> bool:
    row = db_exec("SELECT MAX(trained_at) as t FROM training_log", fetch="one")
    last = row["t"] if row and row["t"] else 0
    row2 = db_exec("SELECT COUNT(*) as c FROM dataset WHERE confirmed=1 AND created_at>?",
                   (last,), fetch="one")
    return (row2["c"] if row2 else 0) >= CFG.get("retrain_threshold",500)

def retrain() -> dict:
    rows = db_exec("SELECT message,label FROM dataset WHERE confirmed=1 LIMIT 5000",fetch="all") or []
    if len(rows) < 30:
        return {"success":False,"error":f"Yetersiz veri ({len(rows)})","count":len(rows)}
    texts = [r["message"] for r in rows]; labels = [r["label"] for r in rows]
    v = TfidfVectorizer(max_features=2000); X = v.fit_transform(texts)
    nb = ComplementNB(); nb.fit(X,labels)
    try:
        sc = cross_val_score(nb,X,labels,cv=min(5,len(set(labels))),scoring="f1_macro")
        f1 = float(np.mean(sc))
    except: f1 = 0.0
    db_exec("INSERT INTO training_log(model_name,version,accuracy,f1_score,dataset_size)"
            " VALUES('nb_tfidf',1,?,?,?)",(f1,f1,len(texts)))
    log.info("✅ Yeniden eğitim: F1=%.3f (%d örnek)", f1, len(texts))
    return {"success":True,"f1":round(f1,4),"dataset_size":len(texts)}

# ═══════════════════════════════════════════════════════════════════════════════
# § 20 — GERÇEK ZAMANLI MONİTÖR (Katman 14)
# ═══════════════════════════════════════════════════════════════════════════════
_live_active  = False
_live_thread  = None
_live_vid_id  = None
_sio          = None

def start_live(video_id: str, drv, sio):
    global _live_active, _live_thread, _live_vid_id, _sio
    _live_vid_id = video_id; _sio = sio; _live_active = True
    _live_thread = threading.Thread(target=_live_loop,
                                     args=(video_id,drv,sio), daemon=True)
    _live_thread.start()
    log.info("⚡ Canlı monitör başladı: %s", video_id)

def stop_live():
    global _live_active; _live_active = False
    log.info("⚡ Canlı monitör durduruldu")

def _live_loop(video_id: str, drv, sio):
    seen = set(); interval = 5
    while _live_active:
        try:
            msgs = selenium_live_chat(drv, video_id)
            for m in msgs:
                if m["msg_id"] in seen: continue
                seen.add(m["msg_id"]); upsert_message(m)
                text = m["message"]; author = m["author"]
                hs = hate_scores(text[:400])
                bs = heuristic_bot([text],[m.get("timestamp_utc",0)])
                thr = threat_level({"hate_score":hs["overall"],"bot_prob":bs,
                                     "stalker_score":0,"impersonator_prob":0,
                                     "human_score":max(0,1-bs),
                                     "antisemitism_score":hs["antisemitism"],
                                     "groyper_score":hs["groyper"]})
                alert = {"type":"live","author":author,"message":text[:200],
                          "threat_level":thr["level"],"threat_score":thr["score"],
                          "threat_color":thr["color"],"video_id":video_id,
                          "msg_id":m["msg_id"],"timestamp":int(time.time())}
                if sio:
                    try: sio.emit("live_alert", alert, namespace="/ws")
                    except: pass
                if thr["level"] in ("RED","CRIMSON","ORANGE"):
                    log.warning("🚨 [%s] @%s: %s", thr["level"],author,text[:60])
        except Exception as e:
            log.debug("Canlı loop hata: %s", e)
        time.sleep(interval)

# ═══════════════════════════════════════════════════════════════════════════════
# § 21 — FLASK WEB PANELİ
# ═══════════════════════════════════════════════════════════════════════════════
_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YT Guardian v2.0 — @ShmirchikArt</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--bd:#30363d;--tx:#c9d1d9;
  --tx2:#8b949e;--acc:#58a6ff;--grn:#2ECC71;--ylw:#F1C40F;--org:#E67E22;
  --red:#E74C3C;--cri:#8B0000;--blu:#3498DB;--pur:#9B59B6}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',Tahoma,sans-serif;font-size:13px;overflow:hidden}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
#app{display:flex;height:100vh}
/* SIDEBAR */
#sb{width:195px;background:var(--bg2);border-right:1px solid var(--bd);
  display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
#sb-logo{padding:12px 14px;border-bottom:1px solid var(--bd)}
#sb-logo h1{font-size:13px;font-weight:700;color:var(--acc);line-height:1.4}
#sb-logo small{color:var(--tx2);font-weight:400;font-size:11px}
.nav{padding:8px 0}
.ni{display:flex;align-items:center;gap:9px;padding:9px 14px;cursor:pointer;
  color:var(--tx2);transition:.15s;font-size:12px;border-left:2px solid transparent}
.ni:hover{background:var(--bg3);color:var(--tx);border-left-color:var(--bd)}
.ni.act{background:var(--bg3);color:var(--tx);border-left-color:var(--acc)}
.ni span.ic{font-size:15px;width:18px;text-align:center}
/* MAIN */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#topbar{background:var(--bg2);border-bottom:1px solid var(--bd);
  padding:7px 14px;display:flex;align-items:center;gap:8px;flex-shrink:0;flex-wrap:wrap}
.inp{background:var(--bg3);border:1px solid var(--bd);color:var(--tx);
  padding:5px 10px;border-radius:6px;font-size:12px}
select.inp{cursor:pointer}
.btn{background:var(--acc);color:#000;border:none;padding:5px 12px;
  border-radius:6px;cursor:pointer;font-size:12px;font-weight:600}
.btn:hover{opacity:.85}.btn.red{background:var(--red);color:#fff}
.btn.grn{background:var(--grn);color:#000}.btn.ghost{background:var(--bg3);
  border:1px solid var(--bd);color:var(--tx)}
#content{flex:1;overflow-y:auto;padding:14px}
.tab{display:none}.tab.act{display:block}
/* CARDS */
.card{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;
  padding:14px;margin-bottom:12px}
.card h3{font-size:12px;font-weight:600;color:var(--tx);margin-bottom:10px;
  display:flex;align-items:center;gap:6px}
/* STATS */
.sgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.sbox{background:var(--bg3);border-radius:6px;padding:12px;text-align:center;border:1px solid var(--bd)}
.sbox .v{font-size:24px;font-weight:700}.sbox .l{font-size:10px;color:var(--tx2);margin-top:3px}
/* TABLE */
.tbl{width:100%;border-collapse:collapse}
.tbl th{background:var(--bg3);padding:7px 9px;text-align:left;font-size:11px;
  color:var(--tx2);border-bottom:1px solid var(--bd);font-weight:500}
.tbl td{padding:6px 9px;border-bottom:1px solid var(--bd);font-size:12px;vertical-align:middle}
.tbl tr:hover td{background:rgba(255,255,255,.03)}
.badge{padding:2px 7px;border-radius:20px;font-size:10px;font-weight:700}
.bg-G{background:#2ECC71;color:#000}.bg-Y{background:#F1C40F;color:#000}
.bg-O{background:#E67E22;color:#000}.bg-R{background:#E74C3C;color:#fff}
.bg-C{background:#8B0000;color:#fff}.bg-B{background:#3498DB;color:#fff}
.bg-P{background:#9B59B6;color:#fff}
/* MSG */
.msg{background:var(--bg2);border:1px solid var(--bd);border-radius:6px;
  padding:9px 12px;margin-bottom:7px;position:relative}
.msg .meta{font-size:11px;color:var(--tx2);margin-bottom:3px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.msg .txt{line-height:1.55;word-break:break-word}
.msg.hi{border-color:var(--acc);background:rgba(88,166,255,.06)}
.msg-acts{position:absolute;right:10px;top:9px;display:flex;gap:4px}
/* GRAPH */
#graph-svg{width:100%;height:480px;background:var(--bg2);border-radius:8px;border:1px solid var(--bd);display:block}
/* ALERTS */
#alerts{max-height:280px;overflow-y:auto}
.al{padding:7px 10px;border-radius:5px;margin-bottom:5px;font-size:12px;
  display:flex;align-items:center;gap:7px;border-left:3px solid transparent}
.al-R,.al-C{border-left-color:var(--red);background:rgba(231,76,60,.08)}
.al-O{border-left-color:var(--org);background:rgba(230,126,34,.08)}
.al-Y{border-left-color:var(--ylw);background:rgba(241,196,15,.08)}
.al-G{border-left-color:var(--grn)}
/* MODAL */
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.72);
  z-index:999;align-items:center;justify-content:center}
.modal.open{display:flex}
.modal-box{background:var(--bg2);border:1px solid var(--bd);border-radius:10px;
  padding:20px;width:640px;max-height:82vh;overflow-y:auto;position:relative}
.modal-box h2{font-size:14px;margin-bottom:14px;color:var(--acc)}
.modal-close{position:absolute;top:12px;right:15px;cursor:pointer;color:var(--tx2);font-size:18px;line-height:1}
.dr{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-size:12px}
.dr label{width:150px;color:var(--tx2);flex-shrink:0}
.bar{height:7px;background:var(--bg3);border-radius:4px;overflow:hidden;flex:1}
.bar-fill{height:100%;border-radius:4px;transition:.3s}
/* PAGINATION */
.pager{display:flex;gap:5px;align-items:center;margin-top:12px;flex-wrap:wrap}
.pager button{padding:4px 9px;background:var(--bg3);border:1px solid var(--bd);
  color:var(--tx);border-radius:4px;cursor:pointer;font-size:11px}
.pager button.cur{background:var(--acc);color:#000;border-color:var(--acc)}
/* MISC */
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--bd);
  border-top-color:var(--acc);border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
#live-dot{width:8px;height:8px;border-radius:50%;background:var(--grn);
  display:inline-block;animation:pulse 1.2s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
#status{font-size:11px;color:var(--tx2);max-width:280px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
mark{background:rgba(88,166,255,.25);color:var(--tx);border-radius:2px;padding:0 1px}
.prog-row{display:flex;align-items:center;gap:6px;font-size:11px;padding:3px 0}
.prog-bar{flex:1;height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}
.prog-fill{height:100%;border-radius:3px}
</style>
</head>
<body>
<div id="app">
<div id="sb">
  <div id="sb-logo"><h1>🛡️ YT Guardian<br><small>@ShmirchikArt v2.0</small></h1></div>
  <div class="nav">
    <div class="ni act" onclick="nav('dashboard',this)"><span class="ic">📊</span>Dashboard</div>
    <div class="ni" onclick="nav('users',this)"><span class="ic">👥</span>Kullanıcılar</div>
    <div class="ni" onclick="nav('messages',this)"><span class="ic">💬</span>Mesajlar</div>
    <div class="ni" onclick="nav('graph',this)"><span class="ic">🔗</span>İlişki Ağı</div>
    <div class="ni" onclick="nav('live',this)"><span class="ic">⚡</span>Canlı Yayın</div>
    <div class="ni" onclick="nav('search',this)"><span class="ic">🔍</span>Arama</div>
    <div class="ni" onclick="nav('stats',this)"><span class="ic">📈</span>İstatistikler</div>
    <div class="ni" onclick="nav('dataset',this)"><span class="ic">🗃️</span>Dataset</div>
    <div class="ni" onclick="nav('nlp',this)"><span class="ic">🤖</span>NLP Otomasyon</div>
    <div class="ni" onclick="nav('settings',this)"><span class="ic">⚙️</span>Ayarlar</div>
  </div>
</div>

<div id="main">
<div id="topbar">
  <input class="inp" id="gs" placeholder="🔍 Kullanıcı veya mesaj ara..." style="width:240px" oninput="gs_input(this.value)">
  <select class="inp" id="gs-mode">
    <option value="text">Metin</option><option value="user">Kullanıcı</option><option value="semantic">Semantik</option>
  </select>
  <span id="live-ind" style="display:none;align-items:center;gap:5px;font-size:11px">
    <span id="live-dot"></span> Canlı
  </span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
    <span id="status"></span>
    <button class="btn ghost" onclick="doLogin()">🔑 Giriş</button>
    <button class="btn" onclick="doScrape()">▶ Tara</button>
  </div>
</div>

<div id="content">
<!-- DASHBOARD -->
<div id="tab-dashboard" class="tab act">
  <div class="sgrid" id="sgrid"></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
    <div class="card"><h3>Tehdit Dağılımı</h3><canvas id="threat-chart" height="160"></canvas></div>
    <div class="card"><h3>Son Uyarılar
      <button class="btn ghost" style="font-size:10px;padding:2px 7px;margin-left:auto" onclick="$('#alerts').empty()">Temizle</button>
    </h3><div id="alerts"></div></div>
  </div>
  <div class="card"><h3>Tarama Durumu <span id="scrape-prog" style="color:var(--tx2);font-size:11px;margin-left:8px"></span></h3>
    <div id="scrape-status" style="font-size:12px;color:var(--tx2)">Henüz tarama yapılmadı</div>
  </div>
</div>

<!-- KULLANICILAR -->
<div id="tab-users" class="tab">
  <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center">
    <input class="inp" id="uf" placeholder="Kullanıcı filtrele..." oninput="loadUsers(1)" style="width:180px">
    <select class="inp" id="tf" onchange="loadUsers(1)">
      <option value="">Tüm Seviyeler</option>
      <option value="CRIMSON">⬛ CRIMSON</option><option value="RED">🔴 RED</option>
      <option value="ORANGE">🟠 ORANGE</option><option value="YELLOW">🟡 YELLOW</option>
      <option value="GREEN">🟢 GREEN</option>
    </select>
    <button class="btn" onclick="analyzeAll()">⚡ Tümünü Analiz Et</button>
    <button class="btn ghost" onclick="doClustering()">🕸️ Kümeleme</button>
    <button class="btn ghost" onclick="inspectNewAccounts()">🆕 Yeni Hesaplar</button>
    <span id="ucnt" style="color:var(--tx2);font-size:11px;margin-left:auto"></span>
  </div>
  <table class="tbl">
    <thead><tr><th>Kullanıcı</th><th>Msg</th><th>Tehdit</th><th>Bot%</th>
    <th>Nefret%</th><th>AntiSem%</th><th>Stalker%</th><th>HMM</th><th>Skor</th><th>İşlem</th></tr></thead>
    <tbody id="utbody"></tbody>
  </table>
  <div class="pager" id="upager"></div>
</div>

<!-- MESAJLAR -->
<div id="tab-messages" class="tab">
  <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center">
    <input class="inp" id="mq" placeholder="Mesajda ara..." oninput="debMsg()" style="flex:1;min-width:160px">
    <input class="inp" id="mauth" placeholder="@kullanıcı..." oninput="debMsg()" style="width:140px">
    <select class="inp" id="msrc" onchange="loadMsgs(1)">
      <option value="">Tüm Kaynaklar</option>
      <option value="stream">Stream</option><option value="replay_chat">Replay Chat</option>
      <option value="live">Canlı</option><option value="comment">Yorum</option>
    </select>
    <span id="mcnt" style="color:var(--tx2);font-size:11px"></span>
  </div>
  <div id="mlist"></div>
  <div class="pager" id="mpager"></div>
</div>

<!-- GRAF -->
<div id="tab-graph" class="tab">
  <div style="display:flex;gap:8px;margin-bottom:10px">
    <button class="btn" onclick="loadGraph()">🔄 Grafiği Yükle</button>
    <button class="btn ghost" onclick="doClustering()">⚙️ Kümeleri Yenile</button>
  </div>
  <svg id="graph-svg"></svg>
  <div class="card" style="margin-top:12px"><h3>Kimlik Eşleşmeleri & Kümeler</h3>
    <div id="cluster-list"></div></div>
</div>

<!-- CANLI YAYIN -->
<div id="tab-live" class="tab">
  <div class="card">
    <h3>⚡ Canlı Yayın Moderasyonu</h3>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap">
      <input class="inp" id="live-vid" placeholder="Video ID (11 karakter)" style="width:180px">
      <button class="btn grn" onclick="startLive()">▶ Başlat</button>
      <button class="btn red" onclick="stopLive()">⏹ Durdur</button>
    </div>
    <div id="live-msgs" style="max-height:420px;overflow-y:auto"></div>
  </div>
</div>

<!-- ARAMA -->
<div id="tab-search" class="tab">
  <div class="card">
    <h3>🔍 Gelişmiş Arama (Ajax/jQuery)</h3>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
      <input class="inp" id="aq" placeholder="Arama terimi..." style="flex:1;min-width:200px">
      <select class="inp" id="am">
        <option value="text">Tam Metin (FTS5)</option>
        <option value="user">Kullanıcı Adı</option>
        <option value="semantic">Semantik Benzerlik</option>
        <option value="pattern">N-gram Pattern</option>
      </select>
      <button class="btn" onclick="advSearch()">🔍 Ara</button>
    </div>
    <div id="sresults"></div>
  </div>
  <div class="card">
    <h3>RAG — AI Sorgulama (Ollama)</h3>
    <div style="display:flex;gap:8px">
      <input class="inp" id="rq" placeholder="Kanal hakkında soru sor..." style="flex:1">
      <button class="btn" onclick="doRag()">💬 Sor</button>
    </div>
    <div id="rag-ans" style="margin-top:10px;font-size:12px;line-height:1.6;color:var(--tx2)"></div>
  </div>
</div>

<!-- İSTATİSTİKLER -->
<div id="tab-stats" class="tab">
  <div class="card"><h3>Kimlik Eşleşmeleri</h3><div id="ilinks"></div></div>
  <div class="card"><h3>Nash Dengesi — Moderatör & Aktör Stratejileri</h3><div id="nash-tbl"></div></div>
  <div class="card"><h3>Küme Liderleri (PageRank)</h3><div id="pr-list"></div></div>
</div>

<!-- DATASET -->
<div id="tab-dataset" class="tab">
  <div style="display:flex;gap:8px;margin-bottom:12px">
    <button class="btn" onclick="loadPending()">⏳ Onay Bekleyenleri Göster</button>
    <button class="btn red" onclick="doRetrain()">🔄 Modeli Yeniden Eğit</button>
  </div>
  <div id="ds-items"></div>
</div>

<!-- NLP OTOMASYOn -->
<div id="tab-nlp" class="tab">
  <div class="card">
    <h3>🤖 NLP Tabanlı Canlı Yayın Tekrar Sohbet Analizi</h3>
    <p style="font-size:11px;color:var(--tx2);margin-bottom:12px">
      Kanal: <b style="color:var(--acc)">@ShmirchikArt</b> · 2023–2026 · BART + Embedding + Kümeleme + Koordineli Saldırı Tespiti
    </p>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">
      <button class="btn" onclick="nlpChannelScan()" id="nlp-scan-btn">🚀 Tam Kanal Taraması (2023-2026)</button>
      <button class="btn ghost" onclick="nav('nlp-video',this)" id="nlp-video-btn">📹 Tek Video Analizi</button>
      <button class="btn ghost" onclick="nlpClusterCurrent()">🔗 Mevcut Mesajları Kümele</button>
      <button class="btn ghost" onclick="nlpTimeline()">📈 Zaman Çizelgesi</button>
    </div>
    <div id="nlp-status" style="font-size:11px;color:var(--tx2);margin-bottom:8px"></div>
  </div>
  <div class="card" id="nlp-video-card" style="display:none">
    <h3>📹 Tek Video NLP Analizi</h3>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
      <input class="inp" id="nlp-vid" placeholder="Video ID (ör. dQw4w9WgXcQ)" style="width:180px">
      <input class="inp" id="nlp-title" placeholder="Başlık (opsiyonel)" style="flex:1;min-width:150px">
      <input class="inp" id="nlp-date" placeholder="Tarih (YYYYMMDD)" style="width:130px">
      <label style="display:flex;align-items:center;gap:4px;font-size:11px">
        <input type="checkbox" id="nlp-filter" checked> Spam Filtrele
      </label>
      <button class="btn" onclick="nlpSingleVideo()">▶ Analiz Et</button>
    </div>
  </div>
  <div id="nlp-results" style="margin-top:8px"></div>
  <div class="card" id="nlp-timeline-card" style="display:none">
    <h3>📈 Mesaj Yoğunluğu Zaman Çizelgesi</h3>
    <canvas id="nlp-chart" style="max-height:200px"></canvas>
  </div>
</div>

<!-- AYARLAR -->
<div id="tab-settings" class="tab">
  <div class="card"><h3>Sistem Durumu</h3><div id="sys-status"></div></div>
  <div class="card"><h3>YouTube Giriş (Firefox)</h3>
    <div style="display:flex;flex-direction:column;gap:8px;max-width:360px">
      <input class="inp" id="yt-em" placeholder="E-posta" style="width:100%">
      <input class="inp" type="password" id="yt-pw" placeholder="Şifre" style="width:100%">
      <button class="btn" onclick="doLogin()">🔑 Firefox ile Giriş Yap</button>
      <div id="login-msg" style="font-size:11px;color:var(--tx2)"></div>
    </div>
  </div>
  <div class="card"><h3>Hesap İnceleme (Selenium)</h3>
    <div style="display:flex;gap:8px">
      <input class="inp" id="insp-author" placeholder="@kullanıcı adı" style="width:200px">
      <button class="btn" onclick="inspectUser()">🔎 Hesabı İncele</button>
    </div>
    <div id="insp-result" style="margin-top:10px;font-size:12px;color:var(--tx2)"></div>
  </div>
</div>
</div><!-- /content -->
</div><!-- /main -->
</div><!-- /app -->

<!-- Modal -->
<div class="modal" id="modal">
  <div class="modal-box">
    <span class="modal-close" onclick="closeModal()">✕</span>
    <h2 id="modal-title"></h2>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const socket = io('/ws', {transports:['websocket','polling']});
let page = {users:1,msgs:1}, pgSize = 50;
let threatChart = null, graphLoaded = false;
const CLR = {G:'#2ECC71',Y:'#F1C40F',O:'#E67E22',R:'#E74C3C',C:'#8B0000',B:'#3498DB',P:'#9B59B6'};
const LVL2CLS = {GREEN:'G',YELLOW:'Y',ORANGE:'O',RED:'R',CRIMSON:'C',BLUE:'B',PURPLE:'P'};
let msgTimer = null, gsTimer = null;

function status(msg,ms=0){ $('#status').text(msg); if(ms) setTimeout(()=>$('#status').text(''),ms); }
function nav(name,el){
  $('.tab').removeClass('act'); $('#tab-'+name).addClass('act');
  $('.ni').removeClass('act'); $(el).addClass('act');
  if(name==='dashboard') loadDash();
  else if(name==='users') loadUsers(1);
  else if(name==='messages') loadMsgs(1);
  else if(name==='graph') { if(!graphLoaded) loadGraph(); }
  else if(name==='stats') loadStats();
  else if(name==='settings') loadSysStatus();
}

// ── DASHBOARD ─────────────────────────────────────────────────────────────────
function loadDash(){
  $.get('/api/stats',function(d){
    const items=[
      {v:d.total_messages,l:'Mesaj',c:'var(--acc)'},
      {v:d.total_users,l:'Kullanıcı',c:'var(--acc)'},
      {v:d.crimson,l:'CRIMSON',c:'#8B0000'},
      {v:d.red,l:'RED',c:'var(--red)'},
      {v:d.orange,l:'ORANGE',c:'var(--org)'},
      {v:d.bots,l:'BOT',c:'var(--blu)'},
      {v:d.antisemites,l:'ANTİSEM.',c:'#8B0000'},
      {v:d.videos,l:'Video',c:'var(--tx2)'},
    ];
    $('#sgrid').html(items.map(x=>`<div class="sbox"><div class="v" style="color:${x.c}">${x.v}</div><div class="l">${x.l}</div></div>`).join(''));
    renderThreatChart(d);
  });
}

function renderThreatChart(d){
  const ctx = document.getElementById('threat-chart').getContext('2d');
  if(threatChart) threatChart.destroy();
  threatChart = new Chart(ctx,{
    type:'doughnut',
    data:{labels:['CRIMSON','RED','ORANGE','YELLOW','GREEN'],
      datasets:[{data:[d.crimson,d.red,d.orange,d.yellow,d.green],
        backgroundColor:['#8B0000','#E74C3C','#E67E22','#F1C40F','#2ECC71'],
        borderWidth:0}]},
    options:{plugins:{legend:{labels:{color:'#c9d1d9',boxWidth:12,font:{size:11}}}},
      cutout:'68%',maintainAspectRatio:false}
  });
}

function addAlert(d){
  const t = new Date(d.timestamp*1000).toLocaleTimeString();
  const cls = 'al-'+(LVL2CLS[d.threat_level]||'G');
  const bg = CLR[LVL2CLS[d.threat_level]||'G'];
  const html=`<div class="al ${cls}">
    <span class="badge bg-${LVL2CLS[d.threat_level]||'G'}">${d.threat_level}</span>
    <a href="#" onclick="showUser('${d.author}')">@${d.author}</a>
    <span style="color:var(--tx2)">${t}</span>
    <span>${(d.message||'').substring(0,70)}</span>
    ${d.video_id?`<button class="btn red" style="font-size:10px;padding:2px 7px;margin-left:auto"
      onclick="delComment('${d.video_id}','${d.author}','${(d.message||'').substring(0,25).replace(/'/g,"\\'")}')">🗑️</button>`:''}
  </div>`;
  $('#alerts').prepend(html);
  if($('#alerts .al').length>60) $('#alerts .al:last').remove();
}

// ── KULLANICILAR ──────────────────────────────────────────────────────────────
function loadUsers(p){
  if(p) page.users=p;
  $.get('/api/users',{page:page.users,size:pgSize,
    filter:$('#uf').val(),threat:$('#tf').val()},function(d){
    $('#ucnt').text(d.total+' kullanıcı');
    let h='';
    (d.users||[]).forEach(u=>{
      const cls=LVL2CLS[u.threat_level]||'G';
      const sp=((u.threat_score||0)*100).toFixed(0);
      h+=`<tr>
        <td><a href="#" onclick="showUser('${u.author}')">${u.author}</a>
          ${u.is_new_account?'<sup style="background:var(--pur);color:#fff;padding:1px 4px;border-radius:3px;font-size:9px">YENİ</sup>':''}</td>
        <td>${u.msg_count||0}</td>
        <td><span class="badge bg-${cls}">${u.threat_level}</span></td>
        <td style="color:var(--blu)">${((u.bot_prob||0)*100).toFixed(0)}%</td>
        <td style="color:var(--red)">${((u.hate_score||0)*100).toFixed(0)}%</td>
        <td style="color:#8B0000">${((u.antisemitism_score||0)*100).toFixed(0)}%</td>
        <td style="color:var(--pur)">${((u.stalker_score||0)*100).toFixed(0)}%</td>
        <td style="color:var(--tx2);font-size:10px">${u.hmm_state||'NORMAL'}</td>
        <td><div class="bar"><div class="bar-fill" style="width:${sp}%;background:${CLR[cls]||'#2ECC71'}"></div></div></td>
        <td style="display:flex;gap:3px;flex-wrap:wrap">
          <button class="btn" style="font-size:10px;padding:2px 6px" onclick="analyzeUser('${u.author}')">⚡</button>
          <button class="btn red" style="font-size:10px;padding:2px 6px" onclick="banUser('${u.author}')">🚫</button>
        </td>
      </tr>`;
    });
    $('#utbody').html(h);
    pager('upager',d.total,page.users,'loadUsers');
  });
}

function analyzeUser(a){
  status('Analiz: @'+a+'...');
  $.post('/api/analyze/user',{author:a},function(d){
    status('✅ @'+a+' → '+d.threat_level,4000); loadUsers();
  }).fail(()=>status('❌ Analiz hatası',3000));
}

function analyzeAll(){
  status('Tüm kullanıcılar analiz ediliyor...');
  $.post('/api/analyze/all',{},function(d){
    status('✅ '+d.analyzed+' kullanıcı analiz edildi',4000); loadUsers();
  });
}

function banUser(a){
  if(!confirm('@'+a+' kullanıcısını işaretle (BAN)?')) return;
  $.post('/api/user/'+encodeURIComponent(a)+'/ban',{},function(d){
    status(d.message||'✅ Tamamlandı',3000); loadUsers();
  });
}

function inspectNewAccounts(){
  status('Yeni hesaplar inceleniyor...');
  $.post('/api/inspect/new-accounts',{},function(d){
    status('✅ '+d.count+' hesap incelendi',4000); loadUsers();
  });
}

function doClustering(){
  status('Kümeleme çalışıyor...');
  $.post('/api/cluster',{},function(d){
    status('✅ Kümeleme tamamlandı',3000); if(graphLoaded) loadGraph();
  }).fail(()=>status('❌ Kümeleme hatası',3000));
}

function showUser(author){
  $('#modal-title').html('👤 @'+author+' <span style="font-size:11px;color:var(--tx2)">Detay</span>');
  $('#modal-body').html('<div class="spin"></div> Yükleniyor...');
  $('#modal').addClass('open');
  $.get('/api/user/'+encodeURIComponent(author),function(d){
    if(d.error){$('#modal-body').html('<p style="color:var(--red)">'+d.error+'</p>');return;}
    const bars=[
      {l:'Bot Olasılığı',v:d.bot_prob||0,c:'var(--blu)'},
      {l:'Nefret Söylemi',v:d.hate_score||0,c:'var(--red)'},
      {l:'Anti-Semitizm',v:d.antisemitism_score||0,c:'#8B0000'},
      {l:'Groyper',v:d.groyper_score||0,c:'#555'},
      {l:'Stalker',v:d.stalker_score||0,c:'var(--pur)'},
      {l:'Bot Sinyali (Hawkes)',v:(d.identity_vector?.bot_signal||0),c:'var(--blu)'},
      {l:'İnsanlık Skoru',v:d.human_score||0.5,c:'var(--grn)'},
    ];
    let h=`<div class="dr"><label>Tehdit Seviyesi</label>
      <span class="badge bg-${LVL2CLS[d.threat_level]||'G'}" style="font-size:12px;padding:3px 10px">${d.threat_level} (${((d.threat_score||0)*100).toFixed(1)}%)</span>
    </div>`;
    bars.forEach(b=>{const p=((b.v||0)*100).toFixed(1);
      h+=`<div class="dr"><label>${b.l}</label>
        <div class="bar"><div class="bar-fill" style="width:${p}%;background:${b.c}"></div></div>
        <span style="width:42px;text-align:right;color:var(--tx2)">${p}%</span></div>`;
    });
    if(d.account_created) h+=`<div class="dr"><label>Hesap Oluşturma</label><span>${d.account_created}</span>
      ${d.is_new_account?'<span class="badge bg-P" style="margin-left:6px">YENİ HESAP</span>':''}</div>`;
    if(d.subscriber_count) h+=`<div class="dr"><label>Abone</label><span>${d.subscriber_count.toLocaleString()}</span></div>`;
    if(d.hmm_state) h+=`<div class="dr"><label>HMM Durumu</label><span style="color:var(--acc)">${d.hmm_state}</span></div>`;
    if(d.game_strategy) h+=`<div class="dr"><label>Önerilen Mod Aksiyonu</label><span style="color:var(--ylw)">${d.game_strategy}</span></div>`;
    if(d.ollama_summary) h+=`<div class="card" style="margin-top:10px"><h3>🤖 AI Analizi (Ollama ${d.ollama_action||''})</h3>
      <p style="font-size:12px;color:var(--tx2);line-height:1.6">${d.ollama_summary}</p></div>`;
    if(d.identity_links&&d.identity_links.length){
      h+=`<div style="margin-top:12px"><h4 style="font-size:12px;color:var(--tx2);margin-bottom:6px">🔗 Kimlik Eşleşmeleri</h4>`;
      d.identity_links.forEach(l=>{
        const other=l.user_a===author?l.user_b:l.user_a;
        h+=`<div style="font-size:12px;padding:4px 0;border-bottom:1px solid var(--bd)">
          <a href="#" onclick="showUser('${other}')">${other}</a>
          <span style="color:var(--ylw);margin-left:8px">${((l.sim_score||0)*100).toFixed(0)}% benzerlik</span>
          <span style="color:var(--tx2);font-size:10px;margin-left:6px">[${l.method||'combined'}]</span>
        </div>`;
      });
      h+=`</div>`;
    }
    h+=`<div style="display:flex;gap:7px;margin-top:14px;flex-wrap:wrap">
      <button class="btn" onclick="analyzeUser('${author}');closeModal()">⚡ Yeniden Analiz</button>
      <button class="btn ghost" onclick="closeModal();nav('messages',document.querySelector('.ni:nth-child(3)'));$('#mauth').val('${author}');debMsg()">💬 Mesajlar</button>
      <button class="btn ghost" onclick="inspectAccount('${author}')">🔎 Hesap Detayı</button>
      <button class="btn red" onclick="banUser('${author}')">🚫 Yasakla</button>
    </div>`;
    $('#modal-body').html(h);
  });
}

function inspectAccount(author){
  $('#insp-result').html('<span class="spin"></span>');
  $.get('/api/user/'+encodeURIComponent(author)+'/account',function(d){
    if(d.error){$('#insp-result').html('<span style="color:var(--red)">'+d.error+'</span>');return;}
    $('#insp-result').html(`Oluşturma: <b>${d.account_created||'?'}</b> | Abone: <b>${d.subscriber_count}</b>
      | Video: <b>${d.video_count}</b> | Yeni Hesap: <b style="color:${d.is_new_account?'var(--ylw)':'var(--grn)'}">${d.is_new_account?'EVET':'HAYIR'}</b>`);
    loadUsers();
  });
}

function inspectUser(){
  const a=$('#insp-author').val().replace('@','');
  if(!a) return;
  inspectAccount(a);
}

function closeModal(){ $('#modal').removeClass('open'); }

// ── MESAJLAR ──────────────────────────────────────────────────────────────────
function debMsg(){ clearTimeout(msgTimer); msgTimer=setTimeout(()=>loadMsgs(1),300); }

function loadMsgs(p){
  if(p) page.msgs=p;
  $.get('/api/messages',{page:page.msgs,size:pgSize,
    q:$('#mq').val(),author:$('#mauth').val(),source:$('#msrc').val()},function(d){
    $('#mcnt').text(d.total+' mesaj');
    let h=''; const q=$('#mq').val();
    (d.messages||[]).forEach(m=>{
      const cls=LVL2CLS[m.threat_level||'GREEN']||'G';
      const ts=m.timestamp?new Date(m.timestamp*1000).toLocaleString():'';
      h+=`<div class="msg ${['R','C','O'].includes(cls)?'hi':''}">
        <div class="meta">
          <a href="#" onclick="showUser('${m.author}')">@${m.author}</a>
          <span class="badge bg-${cls}" style="font-size:9px">${m.threat_level||'GREEN'}</span>
          <span style="font-size:10px">${m.source_type||''}</span>
          <span>${ts}</span>
          <span style="font-size:10px;color:var(--tx2)">${m.lang||''}</span>
        </div>
        <div class="txt">${hl(m.message,q)}</div>
        <div class="msg-acts">
          <button class="btn red" style="font-size:10px;padding:2px 6px"
            onclick="delComment('${m.video_id}','${m.author}','${(m.message||'').substring(0,25).replace(/'/g,"\\'")}')">🗑️</button>
        </div>
      </div>`;
    });
    $('#mlist').html(h||'<p style="color:var(--tx2)">Mesaj bulunamadı</p>');
    pager('mpager',d.total,page.msgs,'loadMsgs');
  });
}

function hl(text,q){
  if(!q||q.length<2) return text;
  const re=new RegExp('('+q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','gi');
  return text.replace(re,'<mark>$1</mark>');
}

function delComment(vid,author,prev){
  if(!confirm('Yorum silinsin mi?\n@'+author+': '+prev+'...')) return;
  status('Yorum siliniyor...');
  $.post('/api/delete/comment',{video_id:vid,author:author,message:prev},function(d){
    status(d.success?'✅ Silindi':'❌ '+d.error,3000); loadMsgs();
  });
}

// ── GRAF ──────────────────────────────────────────────────────────────────────
function loadGraph(){
  status('Graf yükleniyor...');
  $.get('/api/graph',function(d){
    graphLoaded=true; renderGraph(d.graph_data||{}); renderClusters(d.clusters||{},d.leaders||{});
    status('✅ Graf hazır',2000);
  }).fail(()=>status('❌ Graf hatası',3000));
}

function renderGraph(data){
  const el=document.getElementById('graph-svg');
  el.innerHTML='';
  if(!data.nodes||!data.nodes.length){el.innerHTML='<text x="20" y="30" fill="#8b949e">Veri yok — önce kümeleme çalıştırın</text>';return;}
  const W=el.clientWidth||800,H=480;
  const svg=d3.select('#graph-svg').attr('viewBox',`0 0 ${W} ${H}`)
    .call(d3.zoom().on('zoom',e=>g.attr('transform',e.transform)));
  const g=svg.append('g');
  const C=d3.schemeTableau10;
  const sim=d3.forceSimulation(data.nodes)
    .force('link',d3.forceLink(data.links).id(d=>d.id).distance(90))
    .force('charge',d3.forceManyBody().strength(-220))
    .force('center',d3.forceCenter(W/2,H/2))
    .force('collision',d3.forceCollide(14));
  const link=g.append('g').selectAll('line').data(data.links).enter()
    .append('line').attr('stroke','#30363d').attr('stroke-opacity',.7)
    .attr('stroke-width',d=>Math.max(1,d.value*3));
  const threatClr={GREEN:'#2ECC71',YELLOW:'#F1C40F',ORANGE:'#E67E22',
    RED:'#E74C3C',CRIMSON:'#8B0000',BLUE:'#3498DB',PURPLE:'#9B59B6'};
  const node=g.append('g').selectAll('circle').data(data.nodes).enter()
    .append('circle').attr('r',7).attr('fill',d=>threatClr[d.threat]||C[d.group%10])
    .attr('stroke','#21262d').attr('stroke-width',1.5).attr('cursor','pointer')
    .on('click',(_,d)=>showUser(d.id))
    .call(d3.drag()
      .on('start',e=>{if(!e.active)sim.alphaTarget(.3).restart();e.subject.fx=e.subject.x;e.subject.fy=e.subject.y})
      .on('drag',e=>{e.subject.fx=e.x;e.subject.fy=e.y})
      .on('end',e=>{if(!e.active)sim.alphaTarget(0);e.subject.fx=null;e.subject.fy=null}));
  node.append('title').text(d=>d.id+' ['+d.threat+'] PR:'+d.pagerank);
  const lbl=g.append('g').selectAll('text').data(data.nodes).enter()
    .append('text').attr('font-size',9).attr('fill','#8b949e').attr('dy',18)
    .attr('text-anchor','middle').text(d=>d.id.substring(0,12));
  sim.on('tick',()=>{
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('cx',d=>d.x).attr('cy',d=>d.y);
    lbl.attr('x',d=>d.x).attr('y',d=>d.y);
  });
}

function renderClusters(clusters,leaders){
  let h='';
  Object.entries(clusters).forEach(([id,members])=>{
    if(!Array.isArray(members)||!members.length) return;
    const leader=leaders[id]||'';
    h+=`<div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid var(--bd)">
      <b>Küme ${id}</b> (${members.length} üye)${leader?` — Lider: <a href="#" onclick="showUser('${leader}')" style="color:var(--ylw)">${leader}</a>`:''}:
      <span style="color:var(--tx2)"> ${members.map(m=>`<a href="#" onclick="showUser('${m}')">${m}</a>`).join(', ')}</span>
    </div>`;
  });
  $('#cluster-list').html(h||'<span style="color:var(--tx2)">Küme bulunamadı</span>');
}

// ── CANLI YAYIN ───────────────────────────────────────────────────────────────
function startLive(){
  const v=$('#live-vid').val().trim();
  if(v.length!==11){alert('Geçerli Video ID: 11 karakter');return;}
  $.post('/api/live/start',{video_id:v},function(d){
    status('⚡ Canlı monitör: '+v);
    $('#live-ind').css('display','flex');
  });
}
function stopLive(){
  $.post('/api/live/stop',{},function(){status('Monitör durduruldu',2000);$('#live-ind').hide();});
}
function addLiveMsg(d){
  const t=new Date(d.timestamp*1000).toLocaleTimeString();
  const cls=LVL2CLS[d.threat_level]||'G';
  const h=`<div class="msg ${['R','C','O'].includes(cls)?'hi':''}">
    <div class="meta">
      <span class="badge bg-${cls}">${d.threat_level}</span>
      <a href="#" onclick="showUser('${d.author}')">@${d.author}</a>
      <span style="color:var(--tx2)">${t}</span>
    </div>
    <div class="txt">${d.message||''}</div>
    <div class="msg-acts">
      <button class="btn red" style="font-size:10px;padding:2px 6px"
        onclick="delLiveMsg('${d.video_id}','${d.author}','${(d.message||'').substring(0,20).replace(/'/g,"\\'")}')">🗑️</button>
    </div>
  </div>`;
  $('#live-msgs').prepend(h);
  if($('#live-msgs .msg').length>80) $('#live-msgs .msg:last').remove();
  addAlert(d);
}
function delLiveMsg(vid,author,prev){
  $.post('/api/delete/live',{video_id:vid,author:author,message:prev},function(d){
    status(d.success?'✅ Canlı mesaj silindi':'❌ '+d.error,3000);
  });
}

// ── ARAMA ─────────────────────────────────────────────────────────────────────
function gs_input(v){
  clearTimeout(gsTimer);
  if(v.length<2) return;
  gsTimer=setTimeout(()=>{
    $.get('/api/search',{q:v,mode:$('#gs-mode').val()},function(d){
      // quick preview
    });
  },300);
}

function advSearch(){
  const q=$('#aq').val().trim(); if(!q) return;
  $('#sresults').html('<div class="spin"></div>');
  $.get('/api/search',{q:q,mode:$('#am').val()},function(d){
    let h='';
    if(d.users&&d.users.length){
      h+=`<h4 style="font-size:12px;color:var(--tx2);margin-bottom:7px">Kullanıcılar (${d.users.length})</h4>`;
      d.users.forEach(u=>{
        h+=`<div class="msg"><div class="meta">
          <a href="#" onclick="showUser('${u.author}')">@${u.author}</a>
          <span class="badge bg-${LVL2CLS[u.threat_level]||'G'}">${u.threat_level}</span>
          <span>${u.msg_count||0} mesaj</span></div></div>`;
      });
    }
    if(d.messages&&d.messages.length){
      h+=`<h4 style="font-size:12px;color:var(--tx2);margin:10px 0 7px">Mesajlar (${d.messages.length})</h4>`;
      d.messages.forEach(m=>{
        h+=`<div class="msg"><div class="meta">
          <a href="#" onclick="showUser('${m.author}')">@${m.author}</a>
          <span style="font-size:10px;color:var(--tx2)">${m.video_id||''}</span></div>
          <div class="txt">${hl(m.message||'',q)}</div></div>`;
      });
    }
    $('#sresults').html(h||'<p style="color:var(--tx2)">Sonuç bulunamadı</p>');
  });
}

function doRag(){
  const q=$('#rq').val().trim(); if(!q) return;
  $('#rag-ans').html('<span class="spin"></span>');
  $.post('/api/rag',{query:q},function(d){
    $('#rag-ans').html(d.response||d.error||'—');
  }).fail(()=>$('#rag-ans').html('Hata'));
}

// ── İSTATİSTİKLER ─────────────────────────────────────────────────────────────
function loadStats(){
  $.get('/api/identity-links',function(d){
    let h='';
    (d.links||[]).slice(0,60).forEach(l=>{
      h+=`<div class="prog-row">
        <a href="#" onclick="showUser('${l.user_a}')">${l.user_a}</a>
        <span style="color:var(--tx2)"> ↔ </span>
        <a href="#" onclick="showUser('${l.user_b}')">${l.user_b}</a>
        <span class="prog-bar" style="max-width:80px"><span class="prog-fill" style="width:${((l.sim_score||0)*100).toFixed(0)}%;background:var(--ylw)"></span></span>
        <span style="color:var(--ylw)">${((l.sim_score||0)*100).toFixed(0)}%</span>
        <span style="color:var(--tx2);font-size:10px">[${l.method||''}]</span>
      </div>`;
    });
    $('#ilinks').html(h||'<span style="color:var(--tx2)">Bağlantı yok</span>');
  });
  $.get('/api/nash',function(d){
    let h=`<table class="tbl"><thead><tr><th>Mod Eylemi</th><th>Aktör Eylemi</th><th>Mod</th><th>Aktör</th></tr></thead><tbody>`;
    (d.equilibria||[]).forEach(e=>{
      h+=`<tr><td style="color:var(--acc)">${e[2]}</td><td>${e[3]}</td>
        <td style="color:var(--grn)">${e[4]}</td><td style="color:var(--red)">${e[5]}</td></tr>`;
    });
    h+=`</tbody></table>`;
    $('#nash-tbl').html(h);
  });
  $.get('/api/pagerank',function(d){
    let h=''; const items=Object.entries(d.scores||{}).sort((a,b)=>b[1]-a[1]).slice(0,20);
    items.forEach(([u,v])=>{
      h+=`<div class="prog-row"><a href="#" onclick="showUser('${u}')">${u}</a>
        <span class="prog-bar"><span class="prog-fill" style="width:${(v*1000).toFixed(0)}%;background:var(--acc)"></span></span>
        <span style="color:var(--tx2);font-size:10px">${v.toFixed(4)}</span></div>`;
    });
    $('#pr-list').html(h||'<span style="color:var(--tx2)">PageRank yok</span>');
  });
}

// ── DATASET ────────────────────────────────────────────────────────────────────
function loadPending(){
  $.get('/api/dataset/pending',function(d){
    let h='';
    (d.items||[]).forEach(i=>{
      h+=`<div class="msg" style="display:flex;gap:10px;align-items:flex-start">
        <div style="flex:1">
          <div class="meta"><a href="#" onclick="showUser('${i.author}')">@${i.author}</a>
            <span class="badge bg-${LVL2CLS[i.label]||'Y'}">${i.label}</span></div>
          <div class="txt">${i.message.substring(0,200)}</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px;flex-shrink:0">
          <button class="btn grn" style="font-size:10px;padding:2px 8px" onclick="approveDs(${i.id})">✓ Onayla</button>
          <select id="ds-lbl-${i.id}" class="inp" style="font-size:10px;padding:2px">
            <option>ANTISEMITE</option><option>GROYPER</option><option>HATER</option>
            <option>BOT</option><option>STALKER</option><option>IMPERSONATOR</option>
            <option>COORDINATED</option><option>NORMAL</option>
          </select>
          <button class="btn ghost" style="font-size:10px;padding:2px 8px" onclick="approveDsLabel(${i.id})">Etiketle</button>
        </div>
      </div>`;
    });
    $('#ds-items').html(h||'<p style="color:var(--tx2)">Onay bekleyen öğe yok</p>');
  });
}

function approveDs(id){
  $.post('/api/dataset/approve',{id:id},function(){ status('✅ Onaylandı',2000); loadPending(); });
}
function approveDsLabel(id){
  $.post('/api/dataset/approve',{id:id,label:$('#ds-lbl-'+id).val()},function(){
    status('✅ Etiketlendi',2000); loadPending();
  });
}
function doRetrain(){
  if(!confirm('Modeli yeniden eğitmek istediğinizden emin misiniz?')) return;
  status('Eğitim başlıyor...');
  $.post('/api/retrain',{},function(d){
    status(d.success?'✅ Eğitim tamamlandı — F1:'+d.f1:'❌ '+d.error,5000);
  });
}

// ── AYARLAR ────────────────────────────────────────────────────────────────────
function loadSysStatus(){
  $.get('/api/status',function(d){
    let h='';
    Object.entries(d).forEach(([k,v])=>{
      const c=v===true||v==='OK'?'var(--grn)':v===false?'var(--red)':'var(--tx2)';
      h+=`<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--bd)">
        <span>${k}</span><span style="color:${c}">${v===true?'✅ Aktif':v===false?'❌ Pasif':v}</span></div>`;
    });
    $('#sys-status').html(h);
  });
}
function doLogin(){
  const em=$('#yt-em').val()||'physicus93@hotmail.com';
  const pw=$('#yt-pw').val()||'';
  if(!em){alert('Email gerekli');return;}
  $('#login-msg').html('<span class="spin"></span> Giriş yapılıyor...');
  $.post('/api/yt/login',{email:em,password:pw},function(d){
    $('#login-msg').html(d.message||(d.success?'✅ Başarılı':'❌ Başarısız'));
  }).fail(()=>$('#login-msg').html('❌ Sunucu hatası'));
}

// ── SCRAPING ───────────────────────────────────────────────────────────────────
function doScrape(){
  if(!confirm('Kanal taraması başlatılsın? (Uzun sürebilir)')) return;
  status('Tarama başlatıldı...');
  $('#scrape-status').html('<span class="spin"></span> Taranıyor...');
  $.post('/api/scrape',{},function(d){ status('✅ '+d.message,3000); });
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
socket.on('connect', ()=>{ console.log('WS bağlandı'); });
socket.on('live_alert', d=>{ addLiveMsg(d); });
socket.on('scrape_progress', d=>{
  $('#scrape-prog').text(`[${d.step}/${d.total}] ${d.title.substring(0,30)}`);
  $('#scrape-status').html(`⚙️ ${d.video_id} — ${d.title.substring(0,50)}`);
});
socket.on('scrape_done', d=>{
  $('#scrape-prog').text('');
  $('#scrape-status').html(`✅ Tarama tamamlandı — ${d.total_messages} mesaj, ${d.analyzed_users||0} kullanıcı analiz edildi`);
  status('✅ Tarama tamamlandı',4000); loadDash();
});
socket.on('login_result', d=>{
  $('#login-msg').html(d.success?'✅ Giriş başarılı: '+d.email:'❌ Giriş başarısız');
});

// ── YARDIMCI ──────────────────────────────────────────────────────────────────
function pager(id,total,cur,fn){
  const pages=Math.ceil(total/pgSize); let h='';
  for(let p=1;p<=Math.min(pages,12);p++)
    h+=`<button ${p===cur?'class="cur"':''} onclick="${fn}(${p})">${p}</button>`;
  $('#'+id).html(h);
}

// ── NLP OTOMASYOn ──────────────────────────────────────────────────────────────
let nlpChart = null;

function nlpChannelScan(){
  if(!confirm('NLP tam kanal taraması başlatılsın?\n@ShmirchikArt · 2023-2026\nBu işlem uzun sürebilir.')) return;
  $('#nlp-status').html('<span class="spin"></span> NLP kanal taraması başlatıldı...');
  // channel_url backend config'den alınır (hard-code yok)
  $.post('/api/nlp/channel-scan',{
    channel_url:'',
    date_from:'2023-01-01', date_to:'2026-12-31'
  },function(d){
    status('✅ '+d.message, 5000);
    $('#nlp-status').html('⚙️ '+d.message+' — Sonuçlar WebSocket ile gelecek...');
  });
}

function nlpSingleVideo(){
  const vid=$('#nlp-vid').val().trim();
  if(!vid){alert('Video ID girin'); return;}
  $('#nlp-status').html('<span class="spin"></span> Analiz ediliyor: '+vid);
  $.post('/api/nlp/replay-chat',{
    video_id:vid,
    title:$('#nlp-title').val(),
    video_date:$('#nlp-date').val(),
    filter_spam:$('#nlp-filter').is(':checked')?'1':'0',
    auto_analyze:'1'
  },function(d){ status('✅ '+d.message, 4000); });
}

function nlpClusterCurrent(){
  $('#nlp-status').html('<span class="spin"></span> Kümeleniyor...');
  $.post('/api/nlp/cluster-chat',{},function(d){
    renderNlpResults(d);
    $('#nlp-status').html(`✅ ${d.clusters} küme · ${d.coordinated_threats.length} koordineli tehdit · ${d.total_messages} mesaj`);
  });
}

function nlpTimeline(){
  $.get('/api/nlp/timeline',{bin_minutes:5},function(d){
    $('#nlp-timeline-card').show();
    const bins = d.activity_bins||[];
    const labels = bins.map((_,i)=>(i*5)+'dk');
    if(nlpChart) nlpChart.destroy();
    nlpChart = new Chart(document.getElementById('nlp-chart').getContext('2d'),{
      type:'bar',
      data:{labels:labels.slice(0,bins.length),
            datasets:[{data:bins,backgroundColor:'rgba(88,166,255,0.5)',
                       borderColor:'rgba(88,166,255,1)',borderWidth:1,label:'Mesaj/5dk'}]},
      options:{plugins:{legend:{labels:{color:'#c9d1d9',font:{size:11}}}},
               scales:{x:{ticks:{color:'#8b949e',font:{size:9},maxTicksLimit:20}},
                       y:{ticks:{color:'#8b949e',font:{size:10}}}},
               maintainAspectRatio:false}
    });
    $('#nlp-status').html(`📈 ${d.total_messages} mesaj · ${d.duration_minutes} dk · ${(d.spike_bins||[]).length} ani artış`);
  });
}

function renderNlpResults(d){
  let h='';
  if(d.coordinated_threats && d.coordinated_threats.length){
    h+='<div class="card"><h3>🚨 Koordineli Saldırı Tespiti</h3>';
    d.coordinated_threats.forEach(t=>{
      h+=`<div style="border-left:3px solid var(--red);padding:8px 12px;margin-bottom:6px;background:rgba(231,76,60,.06)">
        <div style="font-size:11px;color:var(--red);font-weight:700">KÜMe #${t.cluster_id} — Güven: ${(t.confidence*100).toFixed(0)}%</div>
        <div style="font-size:11px;margin-top:4px">
          👥 ${t.author_count} kullanıcı · ${t.member_count} mesaj · ⏱ ${t.span_seconds}sn
        </div>
        <div style="font-size:11px;color:var(--tx2);margin-top:2px">${t.authors.slice(0,5).map(a=>'@'+a).join(', ')}</div>
        <div style="font-size:11px;color:var(--tx);margin-top:3px;font-style:italic">"${(t.sample_text||'').substring(0,80)}"</div>
      </div>`;
    });
    h+='</div>';
  }
  if(d.topics && d.topics.length){
    h+='<div class="card"><h3>🏷️ Sohbet Konuları</h3>';
    d.topics.forEach(t=>{
      h+=`<div style="margin-bottom:6px;font-size:11px">
        <span style="color:var(--acc)">Konu ${t.topic_id+1}:</span>
        ${t.keywords.slice(0,6).join(', ')}
      </div>`;
    });
    h+='</div>';
  }
  if(d.threat_users && d.threat_users.length){
    h+='<div class="card"><h3>⚠️ Tehdit Kullanıcılar</h3>';
    h+='<table class="tbl"><thead><tr><th>Kullanıcı</th><th>Tehdit</th><th>Seviye</th></tr></thead><tbody>';
    d.threat_users.forEach(u=>{
      const cls=LVL2CLS[u.threat_level]||'G';
      h+=`<tr><td><a href="#" onclick="showUser('${u.author}')">@${u.author}</a></td>
        <td>${((u.threat_score||0)*100).toFixed(0)}%</td>
        <td><span class="badge bg-${cls}">${u.threat_level}</span></td></tr>`;
    });
    h+='</tbody></table></div>';
  }
  if(h) $('#nlp-results').html(h);
}

socket.on('nlp_replay_done', d=>{
  renderNlpResults(d);
  $('#nlp-status').html(`✅ ${d.video_id}: ${d.filtered_messages} mesaj, ${(d.coordinated_threats||[]).length} tehdit`);
  loadDash();
});
socket.on('nlp_scan_done', d=>{
  renderNlpResults({threat_users:d.top_threats||[], coordinated_threats:d.coordinated_threats||0});
  $('#nlp-status').html(`✅ Tarama tamamlandı: ${d.videos_scanned} video · ${d.total_messages} mesaj · ${d.coordinated_threats} tehdit`);
  loadDash();
});

$(document).ready(loadDash);
$(document).on('keydown',e=>{ if(e.key==='Escape') closeModal(); });
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# § 22 — FLASK API ROTALARI
# ═══════════════════════════════════════════════════════════════════════════════
def create_app():
    print(">>> create_app START")

    if not _FLASK or not _FLASK_SIO:
        raise RuntimeError("Flask veya Flask-SocketIO eksik")

    app = Flask(__name__)
    app.config["SECRET_KEY"] = CFG.get("flask_secret","secret")

    if _FLASK_CORS:
        CORS(app)

    # eventlet varsa kullan; bad file descriptor hataları loglama düzeyinde bastır
    _ev_mod, _HAS_EVENTLET = _try_import("eventlet")
    async_mode = "eventlet" if _HAS_EVENTLET else "threading"

    import logging as _logging
    # eventlet socket shutdown hatalarını INFO yerine DEBUG'a bastır
    _logging.getLogger("socketio").setLevel(_logging.WARNING)
    _logging.getLogger("engineio").setLevel(_logging.WARNING)
    _logging.getLogger("engineio.server").setLevel(_logging.WARNING)

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode=async_mode,
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25,
    )

    global _sio
    _sio = socketio

    @app.route("/")
    def index(): return render_template_string(_HTML)

    # ── Stats ─────────────────────────────────────────────────────────────────
    @app.route("/api/stats")
    def api_stats():
        try:
            tm = (db_exec("SELECT COUNT(*) c FROM messages WHERE deleted=0",fetch="one") or {}).get("c",0)
            tu = (db_exec("SELECT COUNT(*) c FROM user_profiles",fetch="one") or {}).get("c",0)
            vd = (db_exec("SELECT COUNT(*) c FROM scraped_videos",fetch="one") or {}).get("c",0)
            lvls= db_exec("SELECT threat_level,COUNT(*) c FROM user_profiles GROUP BY threat_level",fetch="all") or []
            lm = {r["threat_level"]:r["c"] for r in lvls}
            bots=(db_exec("SELECT COUNT(*) c FROM user_profiles WHERE bot_prob>=?",
                          (CFG["bot_threshold"],),fetch="one") or {}).get("c",0)
            anti=(db_exec("SELECT COUNT(*) c FROM user_profiles WHERE antisemitism_score>=0.4",
                          fetch="one") or {}).get("c",0)
            return jsonify({"total_messages":tm,"total_users":tu,"videos":vd,"bots":bots,
                            "antisemites":anti,
                            "crimson":lm.get("CRIMSON",0),"red":lm.get("RED",0),
                            "orange":lm.get("ORANGE",0),"yellow":lm.get("YELLOW",0),
                            "green":lm.get("GREEN",0)})
        except Exception as e: return jsonify({"error":str(e)})

    # ── Users ─────────────────────────────────────────────────────────────────
    @app.route("/api/users")
    def api_users():
        p=int(request.args.get("page",1)); sz=int(request.args.get("size",50))
        flt=request.args.get("filter",""); thr=request.args.get("threat","")
        off=(p-1)*sz; wh="WHERE 1=1"; prms=[]
        if flt: wh+=" AND author LIKE ?"; prms.append(f"%{flt}%")
        if thr: wh+=" AND threat_level=?"; prms.append(thr)
        tot=(db_exec(f"SELECT COUNT(*) c FROM user_profiles {wh}",tuple(prms),fetch="one") or {}).get("c",0)
        rows=db_exec(f"SELECT * FROM user_profiles {wh} ORDER BY threat_score DESC LIMIT ? OFFSET ?",
                     tuple(prms)+(sz,off),fetch="all") or []
        return jsonify({"users":[dict(r) for r in rows],"total":tot})

    @app.route("/api/user/<path:author>")
    def api_user(author):
        row=db_exec("SELECT * FROM user_profiles WHERE author=?",(author,),fetch="one")
        if not row: return jsonify({"error":"Kullanıcı bulunamadı"})
        d=dict(row)
        for f in ["identity_vector","tfidf_json","ngram_json","typo_json","pos_json","temporal_json"]:
            if d.get(f):
                try: d[f]=json.loads(d[f])
                except: pass
        links=db_exec("SELECT * FROM identity_links WHERE user_a=? OR user_b=? ORDER BY sim_score DESC LIMIT 20",
                      (author,author),fetch="all") or []
        d["identity_links"]=[dict(r) for r in links]
        if isinstance(d.get("identity_vector"),dict):
            d["hate_breakdown"]=d["identity_vector"]
        return jsonify(d)

    @app.route("/api/user/<path:author>/ban", methods=["POST"])
    def api_ban(author):
        db_exec("UPDATE user_profiles SET game_strategy='BAN' WHERE author=?",(author,))
        return jsonify({"success":True,"message":f"@{author} BAN işaretlendi"})

    @app.route("/api/user/<path:author>/account")
    def api_user_account(author):
        row=db_exec("SELECT author_cid FROM user_profiles WHERE author=?",(author,),fetch="one")
        if not row or not row["author_cid"]:
            return jsonify({"error":"Channel ID bulunamadı — yorumları önce çekin"})
        info=inspect_account(_driver, row["author_cid"])
        if info:
            upsert_profile(author,{
                "account_created":  info.get("account_created",""),
                "subscriber_count": info.get("subscriber_count",0),
                "video_count":      info.get("video_count",0),
                "is_new_account":   int(info.get("is_new_account",False))
            })
        return jsonify(info or {"error":"Bilgi alınamadı"})

    @app.route("/api/user/<path:author>/links")
    def api_user_links(author):
        rows=db_exec("SELECT * FROM identity_links WHERE user_a=? OR user_b=? ORDER BY sim_score DESC",
                     (author,author),fetch="all") or []
        return jsonify({"links":[dict(r) for r in rows]})

    @app.route("/api/user/<path:author>/messages")
    def api_user_messages(author):
        rows=get_user_msgs(author)
        return jsonify({"messages":rows[:200]})

    # ── Messages ──────────────────────────────────────────────────────────────
    @app.route("/api/messages")
    def api_messages():
        p=int(request.args.get("page",1)); sz=int(request.args.get("size",50))
        q=request.args.get("q",""); auth=request.args.get("author","")
        src=request.args.get("source",""); off=(p-1)*sz
        wh="WHERE m.deleted=0"; prms=[]
        if q:
            try:
                fts=db_exec("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT 500",
                            (q,),fetch="all") or []
                if fts:
                    rids=tuple(r["rowid"] for r in fts)
                    wh+=f" AND m.rowid IN ({','.join(['?']*len(rids))})"; prms.extend(rids)
                else: wh+=" AND m.message LIKE ?"; prms.append(f"%{q}%")
            except: wh+=" AND m.message LIKE ?"; prms.append(f"%{q}%")
        if auth: wh+=" AND m.author LIKE ?"; prms.append(f"%{auth}%")
        if src:  wh+=" AND m.source_type=?"; prms.append(src)
        tot=(db_exec(f"SELECT COUNT(*) c FROM messages m {wh}",tuple(prms),fetch="one") or {}).get("c",0)
        rows=db_exec(
            f"SELECT m.*,up.threat_level,up.threat_score FROM messages m"
            f" LEFT JOIN user_profiles up ON m.author=up.author {wh}"
            f" ORDER BY m.timestamp DESC LIMIT ? OFFSET ?",
            tuple(prms)+(sz,off),fetch="all") or []
        return jsonify({"messages":[dict(r) for r in rows],"total":tot})

    # ── Analysis ──────────────────────────────────────────────────────────────
    @app.route("/api/analyze/user", methods=["POST"])
    def api_analyze_user():
        author=request.form.get("author","")
        if not author: return jsonify({"error":"author gerekli"})
        try: return jsonify(analyze_user(author, run_ollama=True))
        except Exception as e: return jsonify({"error":str(e)})

    @app.route("/api/analyze/all", methods=["POST"])
    def api_analyze_all():
        from concurrent.futures import ThreadPoolExecutor
        rows = db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0", fetch="all") or []
        authors = [r["author"] for r in rows]
        n = 0

        def _analyze_one(author):
            try:
                analyze_user(author, run_ollama=False)
                return 1
            except Exception as e:
                log.debug("Analiz hatası @%s: %s", author, e)
                return 0

        # GPU paylaşımı: max 8 thread (BART/SBERT GPU sıralaması için)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_analyze_one, authors))
        n = sum(results)
        _qtable.save()
        return jsonify({"analyzed": n})

    @app.route("/api/analyze/message", methods=["POST"])
    def api_analyze_msg():
        text=request.form.get("message","")
        if not text: return jsonify({"error":"message gerekli"})
        hs=hate_scores(text[:500]); bs=heuristic_bot([text],[0])
        thr=threat_level({"hate_score":hs["overall"],"bot_prob":bs,
                           "stalker_score":0,"impersonator_prob":0,"human_score":max(0,1-bs),
                           "antisemitism_score":hs["antisemitism"],"groyper_score":hs["groyper"]})
        return jsonify({"hate":hs,"bot_prob":bs,"threat":thr})

    # ── Cluster ───────────────────────────────────────────────────────────────
    @app.route("/api/cluster", methods=["POST"])
    def api_cluster():
        try:
            rows=db_exec("SELECT author FROM user_profiles",fetch="all") or []
            users=[r["author"] for r in rows]
            r=run_clustering(users); return jsonify(r)
        except Exception as e: return jsonify({"error":str(e)})

    @app.route("/api/graph")
    def api_graph():
        try:
            rows=db_exec("SELECT author FROM user_profiles",fetch="all") or []
            users=[r["author"] for r in rows]
            r=run_clustering(users); return jsonify(r)
        except Exception as e: return jsonify({"error":str(e),"graph_data":{"nodes":[],"links":[]},"clusters":{}})

    @app.route("/api/clusters")
    def api_clusters():
        rows=db_exec("SELECT * FROM graph_clusters ORDER BY created_at DESC LIMIT 50",fetch="all") or []
        return jsonify({"clusters":[dict(r) for r in rows]})

    @app.route("/api/cluster/<int:cid>/members")
    def api_cluster_members(cid):
        row=db_exec("SELECT * FROM graph_clusters WHERE cluster_id=? ORDER BY created_at DESC LIMIT 1",
                    (cid,),fetch="one")
        if not row: return jsonify({"members":[]})
        return jsonify({"cluster_id":cid,"members":json.loads(row["members"] or "[]")})

    # ── Search ────────────────────────────────────────────────────────────────
    @app.route("/api/search")
    def api_search():
        q=request.args.get("q",""); mode=request.args.get("mode","text")
        if not q: return jsonify({"messages":[],"users":[]})
        users_out=[]; msgs_out=[]
        if mode in ("text","user"):
            ur=db_exec("SELECT * FROM user_profiles WHERE author LIKE ? LIMIT 20",
                       (f"%{q}%",),fetch="all") or []
            users_out=[dict(r) for r in ur]
        if mode in ("text","semantic"):
            try:
                fr=db_exec("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? LIMIT 300",
                           (q,),fetch="all") or []
                if fr:
                    rids=tuple(r["rowid"] for r in fr)
                    mr=db_exec(f"SELECT * FROM messages WHERE rowid IN ({','.join(['?']*len(rids))}) AND deleted=0 LIMIT 100",
                               rids,fetch="all") or []
                    msgs_out=[dict(r) for r in mr]
            except:
                mr=db_exec("SELECT * FROM messages WHERE message LIKE ? AND deleted=0 LIMIT 100",
                           (f"%{q}%",),fetch="all") or []
                msgs_out=[dict(r) for r in mr]
        if mode=="semantic" and _SBERT:
            e=embed(q)
            if e:
                cr=chroma_query(_ch_msgs, e, 20)
                for c in cr:
                    mid=c.get("id","")
                    if not mid.startswith("user_"):
                        r=db_exec("SELECT * FROM messages WHERE id=? AND deleted=0",(mid,),fetch="one")
                        if r: msgs_out.append(dict(r))
        return jsonify({"messages":msgs_out[:100],"users":users_out})

    # ── RAG ───────────────────────────────────────────────────────────────────
    @app.route("/api/rag", methods=["POST"])
    def api_rag():
        q=request.form.get("query","")
        if not q: return jsonify({"error":"query gerekli"})
        # Semantik arama ile bağlam oluştur
        emb=embed(q)
        ctx_msgs=[]
        if emb:
            cr=chroma_query(_ch_msgs, emb, 8)
            for c in cr:
                mid=c.get("id","")
                if not mid.startswith("user_"):
                    r=db_exec("SELECT * FROM messages WHERE id=?",(mid,),fetch="one")
                    if r: ctx_msgs.append(dict(r))
        if not ctx_msgs:
            mr=db_exec("SELECT * FROM messages WHERE deleted=0 ORDER BY RANDOM() LIMIT 8",fetch="all") or []
            ctx_msgs=[dict(r) for r in mr]
        ans=ollama_rag(q, ctx_msgs)
        return jsonify({"response":ans})

    # ── Identity Links ────────────────────────────────────────────────────────
    @app.route("/api/identity-links")
    def api_identity_links():
        rows=db_exec("SELECT * FROM identity_links ORDER BY sim_score DESC LIMIT 200",fetch="all") or []
        return jsonify({"links":[dict(r) for r in rows]})

    # ── Nash & PageRank ───────────────────────────────────────────────────────
    @app.route("/api/nash")
    def api_nash():
        eq=nash_equilibria()
        return jsonify({"equilibria":[[e[0],e[1],e[2],e[3],e[4],e[5]] for e in eq]})

    @app.route("/api/pagerank")
    def api_pagerank():
        with _get_conn() as c:
            cols = _table_columns(c, "user_profiles")
        author_col = "author" if "author" in cols else ("username" if "username" in cols else "author")
        rows=db_exec(f"SELECT {author_col} AS author,pagerank_score FROM user_profiles WHERE pagerank_score>0"
                     " ORDER BY pagerank_score DESC LIMIT 50",fetch="all") or []
        return jsonify({"scores":{r["author"]:r["pagerank_score"] for r in rows}})

    # ── Delete Comment ────────────────────────────────────────────────────────
    @app.route("/api/delete/comment", methods=["POST"])
    def api_del_comment():
        vid=request.form.get("video_id",""); auth=request.form.get("author","")
        msg=request.form.get("message","")
        if not vid or not auth: return jsonify({"success":False,"error":"video_id ve author gerekli"})
        if not _driver: return jsonify({"success":False,"error":"Selenium bağlantısı yok — önce giriş yapın"})
        def _bg():
            ok=delete_comment(_driver,vid,auth,msg)
            if _sio:
                try: _sio.emit("delete_result",{"success":ok,"author":auth},namespace="/ws")
                except: pass
        threading.Thread(target=_bg,daemon=True).start()
        return jsonify({"success":True,"message":"Silme arka planda başlatıldı"})

    @app.route("/api/delete/live", methods=["POST"])
    def api_del_live():
        vid=request.form.get("video_id",""); auth=request.form.get("author","")
        msg=request.form.get("message","")
        if not _driver: return jsonify({"success":False,"error":"Selenium bağlantısı yok"})
        ok=delete_live_msg(_driver,vid,auth,msg)
        return jsonify({"success":ok})

    # ── NLP Replay Chat ───────────────────────────────────────────────────────
    @app.route("/api/nlp/replay-chat", methods=["POST"])
    def api_nlp_replay_chat():
        """NLP tabanlı tek video canlı yayın sohbet analizi"""
        vid_id     = request.form.get("video_id","")
        title      = request.form.get("title","")
        video_date = request.form.get("video_date","")
        filter_sp  = request.form.get("filter_spam","1") == "1"
        auto_an    = request.form.get("auto_analyze","1") == "1"
        if not vid_id:
            return jsonify({"success":False,"error":"video_id gerekli"})
        def _bg():
            try:
                r = nlp_auto_replay_chat(vid_id, title, video_date,
                                          auto_analyze=auto_an,
                                          filter_spam=filter_sp)
                if _sio:
                    try: _sio.emit("nlp_replay_done", r, namespace="/ws")
                    except: pass
            except Exception as e:
                log.error("NLP replay chat API hatası: %s", e)
        threading.Thread(target=_bg, daemon=True).start()
        return jsonify({"success":True,
                        "message":f"NLP analizi başlatıldı: {vid_id}"})

    @app.route("/api/nlp/channel-scan", methods=["POST"])
    def api_nlp_channel_scan():
        """
        NLP tabanlı tam kanal taraması.
        HEM /videos HEM /streams — tüm yorumlar + canlı chat.
        channel_url boş gelirse CFG'den alınır (hard-code yok).
        """
        channel_url = (request.form.get("channel_url") or "").strip() or CFG["channel_url"]
        date_from   = (request.form.get("date_from")   or "").strip() or CFG.get("date_from","2023-01-01")
        date_to     = (request.form.get("date_to")     or "").strip() or CFG.get("date_to","2026-12-31")
        def _bg():
            try:
                result = nlp_full_channel_scan(channel_url, date_from, date_to)
                if _sio:
                    try:
                        # Büyük payload'ı küçült
                        summary = {k:v for k,v in result.items() if k!="video_results"}
                        _sio.emit("nlp_scan_done", summary, namespace="/ws")
                    except: pass
            except Exception as e:
                log.error("NLP kanal tarama hatası: %s", e)
        threading.Thread(target=_bg, daemon=True).start()
        return jsonify({"success":True,
                        "message":f"NLP kanal taraması başlatıldı: {channel_url}",
                        "channel": channel_url,
                        "date_from": date_from,
                        "date_to": date_to})

    @app.route("/api/nlp/cluster-chat", methods=["POST"])
    def api_nlp_cluster():
        """Mevcut DB mesajlarını kümelere ayır, koordineli saldırıları bul"""
        video_id = request.form.get("video_id","")
        wh = "WHERE deleted=0"
        if video_id: wh += f" AND video_id='{video_id}'"
        rows = db_exec(f"SELECT * FROM messages {wh} ORDER BY timestamp LIMIT 2000",
                       fetch="all") or []
        msgs = [dict(r) for r in rows]
        clusters    = nlp_cluster_chat(msgs)
        coordinated = nlp_detect_coordinated(clusters)
        timeline    = nlp_timeline_analysis(msgs)
        topics      = nlp_extract_key_topics(msgs)
        return jsonify({"clusters": len(clusters),
                        "coordinated_threats": coordinated,
                        "timeline": timeline,
                        "topics":   topics,
                        "total_messages": len(msgs)})

    @app.route("/api/nlp/timeline")
    def api_nlp_timeline():
        """Belirli video veya tüm DB için zaman çizelgesi"""
        video_id  = request.args.get("video_id","")
        bin_min   = int(request.args.get("bin_minutes","5"))
        wh = "WHERE deleted=0"
        if video_id: wh += f" AND video_id=?"
        params = (video_id,) if video_id else ()
        rows = db_exec(f"SELECT timestamp,author,message FROM messages {wh}"
                       f" ORDER BY timestamp LIMIT 5000", params, fetch="all") or []
        msgs = [dict(r) for r in rows]
        return jsonify(nlp_timeline_analysis(msgs, bin_minutes=bin_min))

    # ── Scrape ────────────────────────────────────────────────────────────────
    @app.route("/api/scrape", methods=["POST"])
    def api_scrape():
        def _run():
            from concurrent.futures import ThreadPoolExecutor
            def em(d):
                if _sio:
                    try: _sio.emit("scrape_progress", d, namespace="/ws")
                    except: pass

            total = full_scrape(em)

            # TF-IDF güncelle
            rows = db_exec("SELECT message FROM messages LIMIT 10000", fetch="all") or []
            if rows:
                fit_tfidf([r["message"] for r in rows])

            # ── Scrape sonrası paralel kullanıcı analizi ──────────────────────
            authors_rows = db_exec(
                "SELECT DISTINCT author FROM messages WHERE deleted=0", fetch="all") or []
            authors = [ar["author"] for ar in authors_rows]

            def _analyze_one(a):
                try:
                    analyze_user(a, run_ollama=False)
                    return 1
                except Exception as e:
                    log.debug("Scrape sonrası analiz @%s: %s", a, e)
                    return 0

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(_analyze_one, authors))
            analyzed = sum(results)

            _qtable.save()
            log.info("✅ Scrape sonrası %d kullanıcı analiz edildi", analyzed)

            # Konu modeli
            if len(rows) >= 30:
                try: fit_topics([r["message"] for r in rows])
                except: pass

            if _sio:
                try:
                    _sio.emit("scrape_done",
                              {"total_messages": total, "analyzed_users": analyzed},
                              namespace="/ws")
                except: pass

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"success": True, "message": "Tarama ve paralel analiz başlatıldı"})

    # ── Inspect Accounts ──────────────────────────────────────────────────────
    @app.route("/api/inspect/new-accounts", methods=["POST"])
    def api_inspect_new():
        if not _driver: return jsonify({"success":False,"error":"Selenium yok"})
        rows=db_exec("SELECT author FROM user_profiles ORDER BY RANDOM() LIMIT 30",fetch="all") or []
        authors=[r["author"] for r in rows]
        res=batch_inspect_accounts(_driver,authors)
        pairs=correlate_new_accounts(_driver)
        for a,b,s in pairs:
            db_exec("INSERT OR IGNORE INTO identity_links(user_a,user_b,sim_score,method,confidence)"
                    " VALUES(?,?,?,?,?)",(a,b,s,"new_account_correlation",s))
        return jsonify({"count":len(res),"new_pairs":len(pairs)})

    # ── YouTube Login ─────────────────────────────────────────────────────────
    @app.route("/api/yt/login", methods=["POST"])
    def api_yt_login():
        global _driver
        em=request.form.get("email",CFG["yt_email"])
        pw=request.form.get("password",CFG["yt_password"])
        if not em: return jsonify({"success":False,"message":"Email gerekli"})
        def _bg():
            global _driver
            if _driver:
                try:
                    _driver.quit()
                except:
                    pass
            _driver = make_driver(headless=False)
            if not _driver:
                if _sio:
                    try:
                        _sio.emit("login_result", {"success": False, "email": em}, namespace="/ws")
                    except:
                        pass
                return
            ok = yt_login(_driver, em, pw)
            if _sio:
                try:
                    _sio.emit("login_result", {"success": ok, "email": em}, namespace="/ws")
                except:
                    pass

        threading.Thread(target=_bg,daemon=True).start()
        return jsonify({"success":True,"message":"Giriş Chromium'da başlatıldı..."})

    # ── Live Monitor ──────────────────────────────────────────────────────────
    @app.route("/api/live/start", methods=["POST"])
    def api_live_start():
        vid=request.form.get("video_id","")
        if not vid: return jsonify({"success":False,"error":"video_id gerekli"})
        if not _driver: return jsonify({"success":False,"error":"Selenium bağlantısı yok"})
        start_live(vid,_driver,socketio)
        return jsonify({"success":True,"video_id":vid})

    @app.route("/api/live/stop", methods=["POST"])
    def api_live_stop():
        stop_live(); return jsonify({"success":True})

    # ── Dataset ───────────────────────────────────────────────────────────────
    @app.route("/api/dataset/pending")
    def api_ds_pending():
        rows=db_exec("SELECT * FROM dataset WHERE confirmed=0 ORDER BY created_at DESC LIMIT 100",
                     fetch="all") or []
        return jsonify({"items":[dict(r) for r in rows]})

    @app.route("/api/dataset/approve", methods=["POST"])
    def api_ds_approve():
        iid=request.form.get("id"); label=request.form.get("label")
        if not iid: return jsonify({"success":False})
        if label:
            db_exec("UPDATE dataset SET confirmed=1,label=? WHERE id=?",(label,int(iid)))
        else:
            db_exec("UPDATE dataset SET confirmed=1 WHERE id=?",(int(iid),))
        return jsonify({"success":True})

    @app.route("/api/retrain", methods=["POST"])
    def api_retrain():
        return jsonify(retrain())

    @app.route("/api/retrain/approve", methods=["POST"])
    def api_retrain_approve():
        if check_retrain(): return jsonify(retrain())
        return jsonify({"message":"Henüz yeterli yeni veri yok"})

    # ── Status ────────────────────────────────────────────────────────────────
    @app.route("/api/status")
    def api_status():
        return jsonify({
            "Selenium":        _SELENIUM,
            "Flask":           _FLASK,
            "BART Zeroshot":   _TRANS,
            "Sentence-BERT":   _SBERT,
            "PyTorch":         _TORCH,
            "spaCy":           _SPACY,
            "LangDetect":      _LANGDETECT,
            "fasttext":        _FASTTEXT,
            "BERTopic":        _BERTOPIC,
            "ChromaDB":        _CHROMA,
            "HMMlearn":        _HMM,
            "Louvain":         _LOUVAIN,
            "Ollama":          _OLLAMA,
            "Device":          DEVICE,
            "Driver Aktif":    _driver is not None,
            "Canlı Monitör":   _live_active,
            "TF-IDF Hazır":    _tfidf_fitted,
            "Ollama Model":    CFG.get("ollama_model","phi4:14b"),
            "DB":              CFG.get("db_path","yt_guardian.db"),
        })

    @app.route("/api/stats/realtime")
    def api_realtime():
        return api_stats()

    # ── WebSocket ─────────────────────────────────────────────────────────────
    @socketio.on("connect", namespace="/ws")
    def ws_connect(): emit("connected",{"status":"OK"})

    @socketio.on("ping", namespace="/ws")
    def ws_ping(): emit("pong",{"ts":int(time.time())})
    print(">>> create_app END")
    return app, socketio

# ═══════════════════════════════════════════════════════════════════════════════
# § 23 — BOOTSTRAP & MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap():
    log.info("═"*60)
    log.info("  YT GUARDIAN v2.0 — Başlatılıyor")
    log.info("  Kanal  : %s", CFG.get("channel_url",""))
    log.info("  Cihaz  : %s | Ollama: %s", DEVICE, CFG.get("ollama_model",""))
    log.info("  DB     : %s", CFG.get("db_path",""))
    log.info("═"*60)
    Path(CFG["data_dir"]).mkdir(parents=True,exist_ok=True)
    init_db()
    init_chroma()
    # TF-IDF ile mevcut mesajları yükle
    rows=db_exec("SELECT message FROM messages LIMIT 10000",fetch="all") or []
    if rows:
        fit_tfidf([r["message"] for r in rows])
        log.info("✅ TF-IDF: %d mesajla güncellendi", len(rows))
    # Konu modelleme
    if len(rows) >= 30:
        try:
            fit_topics([r["message"] for r in rows])
        except Exception as e:
            log.warning("Konu modelleme başlatılamadı: %s", e)
    # Q-table yükle
    _qtable.load()
    log.info("✅ Bootstrap tamamlandı")

def main():
    global CFG
    parser = argparse.ArgumentParser(description="YT Guardian v2.0 — Tek Dosya Moderasyon Sistemi")
    parser.add_argument("--scrape",      action="store_true", help="Sadece kanal tarama yap")
    parser.add_argument("--analyze-all", action="store_true", help="Tüm kullanıcıları analiz et")
    parser.add_argument("--port",  type=int, default=CFG.get("flask_port",5000))
    parser.add_argument("--config",type=str, default="yt_guardian_config.json")
    parser.add_argument("--headless",    action="store_true", help="Firefox headless modda aç")
    parser.add_argument("--login",       action="store_true", help="Başlarken YouTube'a giriş yap")
    args = parser.parse_args()

    # Config yeniden yükle
    CFG = load_config(args.config)
    bootstrap()

    if args.scrape:
        log.info("▶ Kanal taraması: %s", CFG["channel_url"])
        total=full_scrape()
        log.info("✅ %d mesaj çekildi", total)
        rows=db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0",fetch="all") or []
        for r in rows:
            try: analyze_user(r["author"], run_ollama=False)
            except Exception as e: log.warning("@%s analiz hatası: %s",r["author"],e)
        _qtable.save()
        log.info("✅ Analiz tamamlandı")
        return

    if args.analyze_all:
        rows=db_exec("SELECT DISTINCT author FROM messages WHERE deleted=0",fetch="all") or []
        for r in rows:
            try: analyze_user(r["author"], run_ollama=False)
            except: pass
        _qtable.save()
        log.info("✅ Tüm kullanıcı analizi tamamlandı")
        return

    if not _FLASK:
        log.error("Flask yüklü değil: pip install flask flask-socketio flask-cors eventlet")
        sys.exit(1)

    # Başlarken YouTube girişi (opsiyonel)
    if args.login or (CFG.get("yt_email") and CFG.get("yt_password")):
        def _auto_login():
            global _driver
            _driver = make_driver(headless=args.headless)
            if not _driver:
                log.error("Otomatik login için driver oluşturulamadı.")
                return
            yt_login(_driver, CFG["yt_email"], CFG["yt_password"])
        threading.Thread(target=_auto_login, daemon=True).start()

    result = create_app()
    if not result or not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("create_app() (app, socketio) döndürmedi")
    app, socketio = result
    log.info("🌐 Web panel: http://localhost:%d", args.port)
    log.info("   Ctrl+C ile durdur")
    try:
        socketio.run(app, host="0.0.0.0", port=args.port,
                     debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        log.info("Durduruldu — kaydediliyor...")
        stop_live(); _qtable.save()
        if _driver:
            try: _driver.quit()
            except: pass
        log.info("✅ Çıkış")

if __name__ == "__main__":
    main()
