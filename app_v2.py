#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  YouTube RAG + NLP + RL Pipeline                                 ║
║  Fully local • No API keys • Ollama phi4:14b                     ║
║  BART Emotion • HDBSCAN Intent • Q-Learning • ChromaDB           ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ───────────────────────────────────────────────────────────────────
# IMPORTS
# ───────────────────────────────────────────────────────────────────
import os, json, math, time, random, sqlite3, asyncio, threading
import traceback, re, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ───────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────
CHANNEL_URLS = [
    "https://www.youtube.com/@ShmirchikArt/videos",
    "https://www.youtube.com/@ShmirchikArt/streams",
]
MAX_VIDEOS        = 100
DB_PATH           = "yt_rag.db"
CHROMA_PATH       = "./chroma_store"
COLLECTION_NAME   = "yt_rag_v3"
OLLAMA_MODEL      = "phi4:14b"
OLLAMA_URL        = "http://localhost:11434/api/generate"
EMBED_MODEL       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMOTION_MODEL     = "j-hartmann/emotion-english-distilroberta-base"
CHUNK_SIZE        = 250   # words
CHUNK_OVERLAP     = 40

# RL hyperparameters
RL_STATE_DIM    = 64
RL_ACTION_DIM   = 8
RL_LR           = 0.001
RL_GAMMA        = 0.95
RL_EPS          = 1.0
RL_EPS_MIN      = 0.05
RL_EPS_DECAY    = 0.995
RL_MEM          = 10_000
RL_BATCH        = 32

# ───────────────────────────────────────────────────────────────────
# SQLITE  ──  schema
# ───────────────────────────────────────────────────────────────────
def init_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS videos (
        id          TEXT PRIMARY KEY,
        title       TEXT,
        url         TEXT,
        transcript  TEXT DEFAULT '',
        processed   INTEGER DEFAULT 0,
        fetched_at  TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id      TEXT,
        chunk_text    TEXT,
        chunk_index   INTEGER,
        embedding_id  TEXT UNIQUE,
        cluster_id    INTEGER DEFAULT -1,
        intent_label  TEXT DEFAULT '',
        emotion       TEXT DEFAULT 'neutral',
        emotion_score REAL  DEFAULT 0.5,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS relations (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id      TEXT,
        chunk_id      INTEGER,
        subject       TEXT,
        relation      TEXT,
        object        TEXT,
        confidence    REAL DEFAULT 0,
        emotion       TEXT DEFAULT 'neutral',
        rl_action     INTEGER DEFAULT 0,
        reward        REAL   DEFAULT 0,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS rl_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        step       INTEGER,
        epsilon    REAL,
        reward     REAL,
        action     INTEGER,
        ts         TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS kg_nodes (
        entity     TEXT PRIMARY KEY,
        etype      TEXT,
        frequency  INTEGER DEFAULT 1
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS kg_edges (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        subject    TEXT,
        relation   TEXT,
        object     TEXT,
        emotion    TEXT,
        confidence REAL,
        frequency  INTEGER DEFAULT 1,
        UNIQUE(subject, relation, object)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS user_messages (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        username       TEXT NOT NULL,
        message_text   TEXT NOT NULL,
        video_id       TEXT DEFAULT '',
        video_title    TEXT DEFAULT '',
        video_second   INTEGER DEFAULT 0,
        created_at     TEXT NOT NULL
    )""")

    conn.commit()
    return conn


# ───────────────────────────────────────────────────────────────────
# YOUTUBE FETCHER
# ───────────────────────────────────────────────────────────────────
class YouTubeFetcher:
    def __init__(self, max_videos: int = MAX_VIDEOS):
        self.max_videos = max_videos

    def fetch_video_ids(self, channel_url: str) -> List[Dict]:
        from yt_dlp import YoutubeDL
        opts = {"quiet": True, "extract_flat": True,
                "skip_download": True, "playlistend": self.max_videos}
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
            entries = (info.get("entries") or [])[:self.max_videos]
            return [{"id": e["id"], "title": e.get("title", ""),
                     "url": f"https://www.youtube.com/watch?v={e['id']}"}
                    for e in entries if e.get("id")]
        except Exception as ex:
            return []

    def get_transcript(self, video_id: str) -> str:
        from youtube_transcript_api import YouTubeTranscriptApi
        for lang in (["tr"], ["en"], None):
            try:
                kw = {"languages": lang} if lang else {}
                segs = YouTubeTranscriptApi.get_transcript(video_id, **kw)
                return " ".join(s["text"] for s in segs)
            except Exception:
                pass
        # last resort: any available transcript
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            tl = YouTubeTranscriptApi.list_transcripts(video_id)
            t  = next(iter(tl))
            return " ".join(s["text"] for s in t.fetch())
        except Exception:
            return ""


# ───────────────────────────────────────────────────────────────────
# PREPROCESSOR
# ───────────────────────────────────────────────────────────────────
class Preprocessor:
    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            for model in ("xx_core_web_sm", "en_core_web_sm", "en_core_web_md"):
                try:
                    self._nlp = spacy.load(model)
                    break
                except Exception:
                    pass
        return self._nlp

    def clean(self, text: str) -> str:
        text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def sentence_split(self, text: str) -> List[str]:
        if self.nlp:
            doc = self.nlp(text[:80_000])
            return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]

    def semantic_chunk(self, text: str,
                       size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP) -> List[str]:
        words  = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + size]))
            i += size - overlap
        return [c for c in chunks if len(c.split()) >= 20]


# ───────────────────────────────────────────────────────────────────
# EMBEDDING MODEL  (multilingual, local)
# ───────────────────────────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[Embed] Loading {model_name} …")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim   = self.model.get_sentence_embedding_dimension()
        print(f"[Embed] dim={self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False,
                                 normalize_embeddings=True)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ───────────────────────────────────────────────────────────────────
# NER EXTRACTOR  (transformer-based, no hand-crafted rules)
# ───────────────────────────────────────────────────────────────────
class NERExtractor:
    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            for model in ("xx_core_web_sm", "en_core_web_sm"):
                try:
                    self._nlp = spacy.load(model)
                    break
                except Exception:
                    pass
        return self._nlp

    def extract(self, text: str) -> List[Dict]:
        if not self.nlp:
            return []
        doc = self.nlp(text[:4000])
        return [{"text": e.text, "label": e.label_,
                 "start": e.start_char, "end": e.end_char}
                for e in doc.ents]


# ───────────────────────────────────────────────────────────────────
# EMOTION CLASSIFIER  (j-hartmann DistilRoBERTa, fully local)
# ───────────────────────────────────────────────────────────────────
class EmotionClassifier:
    LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    def __init__(self):
        self._pipe = None
        self._load()

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            print(f"[Emotion] Loading {EMOTION_MODEL} …")
            self._pipe = hf_pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                top_k=None,
                device=-1,          # CPU
                truncation=True,
                max_length=512,
            )
            print("[Emotion] Ready.")
        except Exception as ex:
            print(f"[Emotion] Load failed: {ex}")

    def classify(self, text: str) -> Dict:
        default = {"label": "neutral", "score": 0.5, "all": {}}
        if not self._pipe:
            return default
        try:
            out    = self._pipe(text[:512])
            scores = {r["label"]: r["score"] for r in out[0]}
            top    = max(scores, key=scores.get)
            return {"label": top, "score": scores[top], "all": scores}
        except Exception:
            return default

    def valence(self, emotion: str) -> float:
        """−1 (negative) … +1 (positive)"""
        return {"joy": 1.0, "surprise": 0.3, "neutral": 0.0,
                "fear": -0.5, "sadness": -0.6,
                "disgust": -0.8, "anger": -0.9}.get(emotion, 0.0)


# ───────────────────────────────────────────────────────────────────
# INTENT DISCOVERY  (HDBSCAN – fully unsupervised, no labels needed)
# ───────────────────────────────────────────────────────────────────
class IntentDiscovery:
    def __init__(self):
        self._pca        = None
        self._clusterer  = None
        self.fitted      = False
        self.n_clusters  = 0

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Discover latent intent clusters. Returns cluster id per row (-1 = noise)."""
        if len(embeddings) < 5:
            return np.zeros(len(embeddings), dtype=int)

        import hdbscan
        from sklearn.decomposition import PCA

        n_comp = min(32, len(embeddings) - 1, embeddings.shape[1])
        self._pca = PCA(n_components=n_comp)
        reduced   = self._pca.fit_transform(embeddings)

        min_sz = max(2, len(embeddings) // 25)
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_sz,
            min_samples=1,
            metric="euclidean",
            prediction_data=True,
        )
        labels = self._clusterer.fit_predict(reduced)
        self.fitted    = True
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels

    def predict_one(self, embedding: np.ndarray) -> int:
        if not self.fitted:
            return 0
        try:
            import hdbscan as hdb
            reduced = self._pca.transform(embedding.reshape(1, -1))
            labels, _  = hdb.approximate_predict(self._clusterer, reduced)
            return int(labels[0])
        except Exception:
            return 0


# ───────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT  (linear DQN + replay buffer)
# ───────────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, state_dim=RL_STATE_DIM, action_dim=RL_ACTION_DIM):
        self.sd    = state_dim
        self.ad    = action_dim
        self.lr    = RL_LR
        self.gamma = RL_GAMMA
        self.eps   = RL_EPS
        self.eps_min   = RL_EPS_MIN
        self.eps_decay = RL_EPS_DECAY

        # Weight matrices  Q(s,a) ≈ s @ W[:,a]
        self.W        = np.random.randn(state_dim, action_dim) * 0.01
        self.W_target = self.W.copy()

        self.memory = deque(maxlen=RL_MEM)
        self.steps  = 0
        self.total_reward = 0.0

        # Discovered action labels (populated by data, not by code)
        self.action_labels: Dict[int, str] = {i: f"rel_{i}" for i in range(action_dim)}
        # Cluster-to-example mapping for label discovery
        self._cluster_examples: Dict[int, List[str]] = defaultdict(list)

    # ── state builder ────────────────────────────────────────────
    def build_state(self,
                    emb:      np.ndarray,
                    cluster:  int,
                    emo_all:  Dict[str, float]) -> np.ndarray:
        emb_part  = emb[:self.sd - 12]
        if len(emb_part) < self.sd - 12:
            emb_part = np.pad(emb_part, (0, self.sd - 12 - len(emb_part)))

        # Cluster encoding (4 dims, soft one-hot)
        c_vec = np.zeros(4)
        idx   = abs(cluster) % 4
        c_vec[idx] = 1.0

        # Emotion vector (8 dims, covers all 7 labels + 1 valence)
        emo_labels = ["anger","disgust","fear","joy","neutral","sadness","surprise","valence"]
        e_vec = np.array([
            emo_all.get("anger",    0.0),
            emo_all.get("disgust",  0.0),
            emo_all.get("fear",     0.0),
            emo_all.get("joy",      0.0),
            emo_all.get("neutral",  0.5),
            emo_all.get("sadness",  0.0),
            emo_all.get("surprise", 0.0),
            max(emo_all.values()) if emo_all else 0.5,
        ])

        state = np.concatenate([emb_part, c_vec, e_vec]).astype(np.float32)
        return state

    # ── action selection (ε-greedy) ──────────────────────────────
    def act(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.ad)
        return int(np.argmax(state @ self.W))

    # ── memory + replay ──────────────────────────────────────────
    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))
        self.total_reward += r

    def replay(self):
        if len(self.memory) < RL_BATCH:
            return
        batch = random.sample(self.memory, RL_BATCH)
        for s, a, r, ns, done in batch:
            tgt = r if done else r + self.gamma * np.max(ns @ self.W_target)
            err = tgt - (s @ self.W)[a]
            self.W[:, a] += self.lr * err * s

        self.steps += 1
        if self.steps % 100 == 0:
            self.W_target = self.W.copy()
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    # ── reward model (no hard rules — signal from data quality) ──
    def reward(self,
               action:      int,
               retrieved:   List[str],
               emb:         np.ndarray,
               ret_embs:    List[np.ndarray],
               emo_score:   float,
               consistency: float) -> float:
        r = 0.0
        # Retrieval coherence
        if retrieved and ret_embs:
            sims = [float(np.dot(emb, re)) for re in ret_embs]
            r += float(np.mean(sims)) * 0.6
        # Emotion intensity bonus
        r += (emo_score - 0.5) * 0.2
        # Temporal / cross-chunk consistency
        r += consistency * 0.4
        return float(np.clip(r, -1.0, 2.0))

    # ── emergent label discovery ──────────────────────────────────
    def register_example(self, action: int, text: str):
        self._cluster_examples[action].append(text)
        if len(self._cluster_examples[action]) % 20 == 0:
            self._rediscover_label(action)

    def _rediscover_label(self, action: int):
        texts = self._cluster_examples[action][-50:]
        freq  = defaultdict(int)
        for t in texts:
            for w in t.lower().split():
                if len(w) > 4:
                    freq[w] += 1
        top = sorted(freq, key=freq.get, reverse=True)[:3]
        self.action_labels[action] = "+".join(top) if top else f"rel_{action}"

    def stats(self) -> Dict:
        return {"epsilon":       round(self.eps,  4),
                "memory_size":   len(self.memory),
                "total_reward":  round(self.total_reward, 3),
                "steps":         self.steps,
                "action_labels": self.action_labels}


# ───────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH  (in-memory + persisted to SQLite)
# ───────────────────────────────────────────────────────────────────
class KnowledgeGraph:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        # in-mem
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict]      = []

    def add(self, subject: str, relation: str, obj: str,
            emotion: str = "neutral", confidence: float = 0.5):
        s, o = subject.lower().strip(), obj.lower().strip()
        if not s or not o or len(o) < 2:
            return

        # in-mem
        self._nodes.setdefault(s, {"freq": 0})["freq"] += 1
        self._nodes.setdefault(o, {"freq": 0})["freq"] += 1
        existing = next((e for e in self._edges
                         if e["s"] == s and e["r"] == relation and e["o"] == o), None)
        if existing:
            existing["freq"]   += 1
            existing["conf"]    = max(existing["conf"], confidence)
        else:
            self._edges.append({"s": s, "r": relation, "o": o,
                                 "e": emotion, "conf": confidence, "freq": 1})

        # persist
        c = self.db.cursor()
        c.execute("INSERT OR IGNORE INTO kg_nodes(entity, etype) VALUES (?, 'unk')", (s,))
        c.execute("UPDATE kg_nodes SET frequency = frequency + 1 WHERE entity = ?", (s,))
        c.execute("INSERT OR IGNORE INTO kg_nodes(entity, etype) VALUES (?, 'unk')", (o,))
        c.execute("UPDATE kg_nodes SET frequency = frequency + 1 WHERE entity = ?", (o,))
        try:
            c.execute("""INSERT INTO kg_edges(subject,relation,object,emotion,confidence)
                         VALUES(?,?,?,?,?)""", (s, relation, o, emotion, confidence))
        except sqlite3.IntegrityError:
            c.execute("""UPDATE kg_edges
                         SET frequency=frequency+1, confidence=MAX(confidence,?)
                         WHERE subject=? AND relation=? AND object=?""",
                      (confidence, s, relation, o))
        self.db.commit()

    def summary(self) -> Dict:
        top_nodes = sorted(self._nodes.items(), key=lambda x: x[1]["freq"], reverse=True)[:15]
        top_edges = sorted(self._edges, key=lambda x: x["freq"], reverse=True)[:30]
        return {"nodes":     len(self._nodes),
                "edges":     len(self._edges),
                "top_nodes": [{"entity": k, "freq": v["freq"]} for k, v in top_nodes],
                "top_edges": [{"s": e["s"], "r": e["r"], "o": e["o"],
                               "emotion": e["e"], "freq": e["freq"]} for e in top_edges]}


# ───────────────────────────────────────────────────────────────────
# CHROMA STORE
# ───────────────────────────────────────────────────────────────────
class ChromaStore:
    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL)
        self.col = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=ef)

    def add(self, eid: str, text: str, meta: Dict):
        try:
            self.col.add(documents=[text], metadatas=[meta], ids=[eid])
        except Exception:
            pass  # already exists

    def query(self, q: str, n: int = 5) -> List[Dict]:
        cnt = self.col.count()
        if cnt == 0:
            return []
        try:
            res  = self.col.query(query_texts=[q], n_results=min(n, cnt))
            docs = res.get("documents", [[]])[0]
            mets = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[1.0]*len(docs)])[0]
            return [{"text": d, "meta": m, "score": 1 - dist}
                    for d, m, dist in zip(docs, mets, dists)]
        except Exception:
            return []

    def count(self) -> int:
        try:
            return self.col.count()
        except Exception:
            return 0


# ───────────────────────────────────────────────────────────────────
# OLLAMA CLIENT
# ───────────────────────────────────────────────────────────────────
class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url   = url

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 2048) -> str:
        payload: Dict[str, Any] = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        try:
            r = requests.post(self.url, json=payload, timeout=180)
            return r.json().get("response", "")
        except Exception as ex:
            return f"[Ollama error] {ex}"

    def alive(self) -> bool:
        try:
            return requests.get("http://localhost:11434/api/tags",
                                timeout=4).status_code == 200
        except Exception:
            return False

    def models(self) -> List[str]:
        try:
            data = requests.get("http://localhost:11434/api/tags", timeout=4).json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ───────────────────────────────────────────────────────────────────
# PIPELINE  ──  orchestrates everything
# ───────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self):
        self.status: Dict = {"state": "idle", "progress": 0,
                             "message": "Waiting…", "logs": []}
        print("[Pipeline] Init DB …")
        self.db   = init_db()
        self.prep = Preprocessor()
        self.ner  = NERExtractor()

        print("[Pipeline] Loading embedding model …")
        self.emb   = EmbeddingModel()
        self.emo   = EmotionClassifier()
        self.intent= IntentDiscovery()
        self.rl    = QLearningAgent(RL_STATE_DIM, RL_ACTION_DIM)
        self.kg    = KnowledgeGraph(self.db)
        self.chroma: Optional[ChromaStore] = None
        self.ollama = OllamaClient()
        self.fetcher = YouTubeFetcher(MAX_VIDEOS)
        self._log("Pipeline initialised — ready.")

    # ── logging ──────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.status["logs"].append(line)
        if len(self.status["logs"]) > 300:
            self.status["logs"] = self.status["logs"][-150:]
        print(line)

    def _chroma(self) -> ChromaStore:
        if self.chroma is None:
            self.chroma = ChromaStore()
        return self.chroma

    # ════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ════════════════════════════════════════════════════════════
    def run(self):
        self.status.update(state="running", progress=0, message="Starting …")
        try:
            # ── 1. fetch videos ────────────────────────────────
            self._log("▶ Phase 1 — Fetching video list …")
            self.status["message"] = "Fetching video metadata from YouTube …"

            all_vids: List[Dict] = []
            for url in CHANNEL_URLS:
                batch = self.fetcher.fetch_video_ids(url)
                all_vids.extend(batch)
                self._log(f"  Found {len(batch)} videos: {url}")

            seen: set = set()
            unique = [v for v in all_vids
                      if v["id"] not in seen and not seen.add(v["id"])]
            self._log(f"  Total unique: {len(unique)}")
            self.status["progress"] = 8

            # ── 2. fetch transcripts ───────────────────────────
            self._log("▶ Phase 2 — Downloading transcripts …")
            cur = self.db.cursor()
            done_ids = {r[0] for r in
                        cur.execute("SELECT id FROM videos WHERE processed=1").fetchall()}

            for i, v in enumerate(unique):
                pct = 8 + int(32 * i / max(len(unique), 1))
                self.status.update(
                    message=f"Transcript [{i+1}/{len(unique)}]: {v['title'][:55]}…",
                    progress=pct)

                if v["id"] in done_ids:
                    continue

                tx = self.fetcher.get_transcript(v["id"])
                cur.execute("""INSERT OR REPLACE INTO videos
                    (id,title,url,transcript,processed,fetched_at)
                    VALUES (?,?,?,?,0,?)""",
                    (v["id"], v["title"], v["url"], tx,
                     datetime.now().isoformat()))
                self.db.commit()
                self._log(f"  {'✓' if tx else '✗'} {v['title'][:60]}")
                time.sleep(0.3)

            self.status["progress"] = 40

            # ── 3. NLP feature extraction ──────────────────────
            self._log("▶ Phase 3 — NLP feature extraction …")
            self.status["message"] = "Chunking, embedding, NER, emotion …"

            cur.execute("""SELECT id,title,transcript FROM videos
                           WHERE transcript!='' AND processed=0""")
            rows = cur.fetchall()
            self._log(f"  Videos to process: {len(rows)}")

            all_embs: List[np.ndarray]   = []
            all_meta: List[Tuple]        = []  # (chunk_db_id, vid_id, chunk_text, emo_res)

            for vi, (vid_id, title, transcript) in enumerate(rows):
                pct = 40 + int(25 * vi / max(len(rows), 1))
                self.status.update(
                    message=f"NLP [{vi+1}/{len(rows)}]: {title[:50]}",
                    progress=pct)

                cleaned = self.prep.clean(transcript)
                chunks  = self.prep.semantic_chunk(cleaned)
                entities_per_chunk = []

                for ci, chunk in enumerate(chunks):
                    eid     = hashlib.md5(f"{vid_id}_{ci}".encode()).hexdigest()
                    emb_vec = self.emb.encode_one(chunk)
                    emo_res = self.emo.classify(chunk)
                    ents    = self.ner.extract(chunk)

                    all_embs.append(emb_vec)
                    entities_per_chunk.append(ents)

                    c2 = self.db.cursor()
                    c2.execute("""INSERT OR IGNORE INTO chunks
                        (video_id,chunk_text,chunk_index,embedding_id,emotion,emotion_score)
                        VALUES (?,?,?,?,?,?)""",
                        (vid_id, chunk, ci, eid,
                         emo_res["label"], emo_res["score"]))
                    chunk_db_id = c2.lastrowid
                    self.db.commit()

                    # index into Chroma
                    self._chroma().add(eid, chunk, {
                        "video_id": vid_id,
                        "title":    title,
                        "chunk_idx": ci,
                        "emotion":  emo_res["label"],
                        "entities": json.dumps([e["text"] for e in ents[:6]]),
                    })
                    all_meta.append((chunk_db_id, vid_id, chunk, emo_res, ents, emb_vec))

            self._log(f"  Total chunks indexed: {len(all_embs)}")
            self.status["progress"] = 65

            # ── 4. Intent discovery (HDBSCAN) ──────────────────
            self._log("▶ Phase 4 — Unsupervised intent discovery (HDBSCAN) …")
            self.status["message"] = "Discovering latent intents via HDBSCAN …"

            cluster_labels = np.zeros(len(all_embs), dtype=int)
            if len(all_embs) >= 5:
                cluster_labels = self.intent.fit(np.array(all_embs))
                n_cl = self.intent.n_clusters
                self._log(f"  Discovered {n_cl} intent clusters")

                c3 = self.db.cursor()
                for idx, (chunk_db_id, *_rest) in enumerate(all_meta):
                    if idx < len(cluster_labels):
                        c3.execute("UPDATE chunks SET cluster_id=? WHERE id=?",
                                   (int(cluster_labels[idx]), chunk_db_id))
                self.db.commit()

            self.status["progress"] = 75

            # ── 5. RL training ─────────────────────────────────
            self._log("▶ Phase 5 — Q-Learning agent training …")
            self.status["message"] = "Training RL agent (reward-shaped pattern learning) …"

            c4 = self.db.cursor()
            limit = min(len(all_meta), 800)

            for idx in range(limit):
                chunk_db_id, vid_id, chunk_text, emo_res, ents, emb_vec = all_meta[idx]
                cl = int(cluster_labels[idx]) if idx < len(cluster_labels) else 0

                state = self.rl.build_state(emb_vec, max(0, cl), emo_res.get("all", {}))
                action = self.rl.act(state)

                # Retrieval coherence reward component
                ret    = self._chroma().query(chunk_text, n=3)
                ret_embs = [self.emb.encode_one(r["text"]) for r in ret[:2]]
                consistency = 1.0 - (abs(cl) % 4) / 4.0

                r = self.rl.reward(action, [x["text"] for x in ret],
                                   emb_vec, ret_embs,
                                   emo_res["score"], consistency)

                # next state
                ni = idx + 1
                if ni < len(all_meta):
                    ni_cl  = int(cluster_labels[ni]) if ni < len(cluster_labels) else 0
                    ns     = self.rl.build_state(all_meta[ni][5], max(0, ni_cl), {})
                else:
                    ns = state

                self.rl.remember(state, action, r, ns, ni >= len(all_meta))
                self.rl.replay()
                self.rl.register_example(action, chunk_text)

                # Extract relations → KG
                action_label = self.rl.action_labels.get(action, f"rel_{action}")
                for ent in ents[:4]:
                    self.kg.add(subject=vid_id[:10],
                                relation=action_label,
                                obj=ent["text"],
                                emotion=emo_res["label"],
                                confidence=float(r))
                    c4.execute("""INSERT INTO relations
                        (video_id,chunk_id,subject,relation,object,
                         confidence,emotion,rl_action,reward)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        (vid_id, chunk_db_id, vid_id[:10], action_label,
                         ent["text"], float(r), emo_res["label"], action, float(r)))

                c4.execute("INSERT INTO rl_log(step,epsilon,reward,action,ts) VALUES(?,?,?,?,?)",
                           (self.rl.steps, self.rl.eps, r, action, datetime.now().isoformat()))

                if idx % 100 == 0:
                    self._log(f"  RL step {idx}/{limit} | ε={self.rl.eps:.3f} | R={r:.3f}")

            self.db.commit()

            # Mark processed
            c5 = self.db.cursor()
            c5.execute("UPDATE videos SET processed=1 WHERE transcript!=''")
            self.db.commit()

            n_chunks = len(all_meta)
            n_rels   = self.db.cursor().execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            self.status.update(state="complete", progress=100,
                               message=f"Done ✓  {n_chunks} chunks · {n_rels} relations")
            self._log(f"✅ Pipeline complete — {n_chunks} chunks, {n_rels} relations")

        except Exception as ex:
            tb = traceback.format_exc()
            self.status.update(state="error",
                               message=f"Error: {ex}")
            self._log(f"❌ {ex}\n{tb}")

    # ════════════════════════════════════════════════════════════
    # RAG QUERY  (RL-enhanced, no hard-coded patterns)
    # ════════════════════════════════════════════════════════════
    def query(self, question: str) -> Dict:
        ch = self._chroma()
        if ch.count() == 0:
            return {"answer": "No data indexed yet. Run the pipeline first.",
                    "sources": [], "patterns": [], "rl": {}}

        # Encode query
        q_emb  = self.emb.encode_one(question)
        q_cl   = self.intent.predict_one(q_emb)
        q_emo  = self.emo.classify(question)

        # RL action for this query
        q_state  = self.rl.build_state(q_emb, max(0, q_cl), q_emo.get("all", {}))
        q_action = self.rl.act(q_state)
        a_label  = self.rl.action_labels.get(q_action, f"rel_{q_action}")

        # Vector retrieval
        retrieved = ch.query(question, n=6)

        # KG context — entities mentioned in query
        q_words   = set(question.lower().split())
        kg_hits   = [e for e in self.kg._edges
                     if any(w in e["o"] for w in q_words)][:5]

        # Assemble context
        ctx_parts = [r["text"] for r in retrieved]
        context   = "\n\n---\n\n".join(ctx_parts)

        kg_ctx = ""
        if kg_hits:
            kg_ctx = "\n\nLearned patterns:\n" + "\n".join(
                f"• {e['s']} —[{e['r']}]→ {e['o']}  (emotion: {e['e']}, freq: {e['freq']})"
                for e in kg_hits)

        system = (
            "You are an expert research assistant analysing YouTube transcripts "
            "from a specific creator. You have access to transcript chunks, "
            "emotion analysis, and a learned knowledge graph. "
            "Answer questions precisely. Name specific people, books, topics "
            "mentioned by the creator. If unsure, say so."
        )
        prompt = (
            f"Transcript context:\n{context}"
            f"{kg_ctx}\n\n"
            f"Question: {question}\n\n"
            "Detailed answer:"
        )
        answer = self.ollama.generate(prompt, system=system)

        sources = []
        for r in retrieved[:4]:
            m = r.get("meta", {})
            try:
                ents = json.loads(m.get("entities", "[]"))
            except Exception:
                ents = []
            sources.append({"title":    m.get("title", "?"),
                             "video_id": m.get("video_id", ""),
                             "emotion":  m.get("emotion", ""),
                             "score":    round(r.get("score", 0), 3),
                             "entities": ents[:4]})

        return {
            "answer":   answer,
            "sources":  sources,
            "patterns": [{"s": e["s"], "r": e["r"], "o": e["o"],
                          "e": e["e"], "freq": e["freq"]} for e in kg_hits],
            "rl":       {"action": q_action, "label": a_label,
                         "cluster": q_cl,    "emotion": q_emo["label"]},
        }

    # ── helpers ──────────────────────────────────────────────────
    def get_videos(self) -> List[Dict]:
        c = self.db.cursor()
        rows = c.execute("""
            SELECT v.id, v.title, v.url, v.processed, v.fetched_at,
                   COUNT(ch.id) as chunks
            FROM videos v
            LEFT JOIN chunks ch ON ch.video_id = v.id
            GROUP BY v.id ORDER BY v.rowid DESC LIMIT 100
        """).fetchall()
        return [{"id":r[0],"title":r[1],"url":r[2],
                 "processed":bool(r[3]),"fetched_at":r[4],"chunks":r[5]}
                for r in rows]

    def get_patterns(self) -> List[Dict]:
        c = self.db.cursor()
        rows = c.execute("""
            SELECT r.relation, r.object, r.emotion,
                   AVG(r.confidence), COUNT(*) as cnt, v.title
            FROM relations r
            JOIN videos v ON v.id = r.video_id
            GROUP BY r.relation, r.object
            ORDER BY cnt DESC LIMIT 60
        """).fetchall()
        return [{"relation":r[0],"entity":r[1],"emotion":r[2],
                 "confidence":round(r[3],3),"frequency":r[4],"video":r[5]}
                for r in rows]

    def get_rl_log(self) -> List[Dict]:
        c = self.db.cursor()
        rows = c.execute("""
            SELECT step,epsilon,reward,action,ts FROM rl_log
            ORDER BY id DESC LIMIT 200
        """).fetchall()
        return [{"step":r[0],"epsilon":r[1],"reward":r[2],
                 "action":r[3],"ts":r[4]} for r in reversed(rows)]

    def get_clusters(self) -> List[Dict]:
        c = self.db.cursor()
        rows = c.execute("""
            SELECT cluster_id, COUNT(*) as cnt, emotion,
                   GROUP_CONCAT(SUBSTR(chunk_text,1,60), ' | ')
            FROM chunks
            WHERE cluster_id >= 0
            GROUP BY cluster_id
            ORDER BY cnt DESC LIMIT 20
        """).fetchall()
        return [{"cluster_id":r[0],"count":r[1],"emotion":r[2],
                 "examples": (r[3] or "")[:200]} for r in rows]

    def save_user_message(self, username: str, message_text: str,
                          video_id: str = "", video_title: str = "",
                          video_second: int = 0) -> None:
        u = (username or "").strip()
        m = (message_text or "").strip()
        if not u or not m:
            return
        try:
            sec = max(0, int(video_second or 0))
        except Exception:
            sec = 0
        c = self.db.cursor()
        c.execute("""INSERT INTO user_messages
            (username, message_text, video_id, video_title, video_second, created_at)
            VALUES (?,?,?,?,?,?)""",
            (u, m, (video_id or "").strip(), (video_title or "").strip(),
             sec, datetime.now().isoformat()))
        self.db.commit()

    def get_message_users(self) -> List[str]:
        c = self.db.cursor()
        rows = c.execute("""
            SELECT username FROM user_messages
            GROUP BY username
            ORDER BY MAX(created_at) DESC
        """).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_user_messages(self, username: str) -> List[Dict]:
        u = (username or "").strip()
        if not u:
            return []
        c = self.db.cursor()
        rows = c.execute("""
            SELECT id, username, message_text, video_id, video_title,
                   video_second, created_at
            FROM user_messages
            WHERE username=?
            ORDER BY datetime(created_at) DESC, id DESC
        """, (u,)).fetchall()
        out = []
        for r in rows:
            sec = int(r[5] or 0)
            vid = r[3] or ""
            ct_link = f"https://www.youtube.com/watch?v={vid}&t={sec}s" if vid else ""
            out.append({
                "id": r[0],
                "username": r[1],
                "message_text": r[2],
                "video_id": vid,
                "video_title": r[4] or "",
                "video_second": sec,
                "youtube_link": ct_link,
                "created_at": r[6],
            })
        return out

    # ════════════════════════════════════════════════════════════
    # PROCESS EXISTING TRANSCRIPTS
    # Videolar zaten DB'de; sadece NLP+Embed+ChromaDB+RL+KG çalıştır
    # ════════════════════════════════════════════════════════════
    def process_existing(self):
        """
        YouTube'a hiç bağlanmadan, SQLite'daki mevcut transcript'leri
        alıp tam NLP pipeline'ını (chunk→embed→NER→emotion→HDBSCAN→RL→KG→Chroma)
        çalıştırır.  processed=0 olan veya hiç chunk'ı olmayan tüm videolar hedef.
        """
        self.status.update(state="running", progress=0,
                           message="Mevcut transcript'ler işleniyor…")
        try:
            cur = self.db.cursor()

            # ── Hedef videoları bul ────────────────────────────────
            # processed=0 VEYA chunk'ı henüz olmayan tüm videolar
            self._log("▶ [process_existing] Hedef videolar seçiliyor…")
            rows = cur.execute("""
                SELECT v.id, v.title, v.transcript
                FROM videos v
                WHERE v.transcript != ''
                  AND (v.processed = 0
                       OR v.id NOT IN (SELECT DISTINCT video_id FROM chunks))
            """).fetchall()

            total = len(rows)
            self._log(f"  İşlenecek video sayısı: {total}")
            if total == 0:
                self.status.update(state="complete", progress=100,
                                   message="Zaten tüm videolar işlenmiş.")
                self._log("✅ Yapılacak iş yok — tüm videolar zaten işlenmiş.")
                return

            self.status["progress"] = 5

            # ── Phase A: Chunk + Embed + NER + Emotion ─────────────
            self._log("▶ Phase A — Chunking · Embedding · NER · Emotion …")
            all_embs: List[np.ndarray] = []
            all_meta: List[Tuple]      = []

            for vi, (vid_id, title, transcript) in enumerate(rows):
                pct = 5 + int(55 * vi / max(total, 1))
                self.status.update(
                    message=f"NLP [{vi+1}/{total}]: {title[:50]}",
                    progress=pct)

                cleaned = self.prep.clean(transcript)
                chunks  = self.prep.semantic_chunk(cleaned)
                self._log(f"  [{vi+1}/{total}] {title[:55]} → {len(chunks)} chunk")

                for ci, chunk in enumerate(chunks):
                    eid     = hashlib.md5(f"{vid_id}_{ci}".encode()).hexdigest()
                    emb_vec = self.emb.encode_one(chunk)
                    emo_res = self.emo.classify(chunk)
                    ents    = self.ner.extract(chunk)

                    all_embs.append(emb_vec)

                    c2 = self.db.cursor()
                    c2.execute("""INSERT OR IGNORE INTO chunks
                        (video_id, chunk_text, chunk_index, embedding_id,
                         emotion, emotion_score)
                        VALUES (?,?,?,?,?,?)""",
                        (vid_id, chunk, ci, eid,
                         emo_res["label"], emo_res["score"]))
                    chunk_db_id = c2.lastrowid or 0
                    self.db.commit()

                    # ChromaDB'ye ekle
                    self._chroma().add(eid, chunk, {
                        "video_id":  vid_id,
                        "title":     title,
                        "chunk_idx": ci,
                        "emotion":   emo_res["label"],
                        "entities":  json.dumps([e["text"] for e in ents[:6]]),
                    })
                    all_meta.append((chunk_db_id, vid_id, chunk,
                                     emo_res, ents, emb_vec))

            n_chunks = len(all_embs)
            self._log(f"  Toplam chunk: {n_chunks}")
            self.status["progress"] = 60

            # ── Phase B: HDBSCAN intent clustering ─────────────────
            self._log("▶ Phase B — HDBSCAN intent clustering …")
            cluster_labels = np.zeros(n_chunks, dtype=int)
            if n_chunks >= 5:
                cluster_labels = self.intent.fit(np.array(all_embs))
                self._log(f"  Keşfedilen küme sayısı: {self.intent.n_clusters}")

                c3 = self.db.cursor()
                for idx, (chunk_db_id, *_rest) in enumerate(all_meta):
                    if idx < len(cluster_labels):
                        c3.execute("UPDATE chunks SET cluster_id=? WHERE id=?",
                                   (int(cluster_labels[idx]), chunk_db_id))
                self.db.commit()

            self.status["progress"] = 75

            # ── Phase C: RL training + KG extraction ───────────────
            self._log("▶ Phase C — Q-Learning agent training + KG …")
            c4   = self.db.cursor()
            limit = min(n_chunks, 800)

            for idx in range(limit):
                chunk_db_id, vid_id, chunk_text, emo_res, ents, emb_vec = all_meta[idx]
                cl = int(cluster_labels[idx]) if idx < len(cluster_labels) else 0

                state  = self.rl.build_state(emb_vec, max(0, cl),
                                             emo_res.get("all", {}))
                action = self.rl.act(state)

                ret      = self._chroma().query(chunk_text, n=3)
                ret_embs = [self.emb.encode_one(r["text"]) for r in ret[:2]]
                consistency = 1.0 - (abs(cl) % 4) / 4.0

                r = self.rl.reward(action, [x["text"] for x in ret],
                                   emb_vec, ret_embs,
                                   emo_res["score"], consistency)

                ni = idx + 1
                if ni < len(all_meta):
                    ni_cl = int(cluster_labels[ni]) if ni < len(cluster_labels) else 0
                    ns    = self.rl.build_state(all_meta[ni][5], max(0, ni_cl), {})
                else:
                    ns = state

                self.rl.remember(state, action, r, ns, ni >= len(all_meta))
                self.rl.replay()
                self.rl.register_example(action, chunk_text)

                # İlişki → KG
                action_label = self.rl.action_labels.get(action, f"rel_{action}")
                for ent in ents[:4]:
                    self.kg.add(subject=vid_id[:10],
                                relation=action_label,
                                obj=ent["text"],
                                emotion=emo_res["label"],
                                confidence=float(r))
                    c4.execute("""INSERT INTO relations
                        (video_id, chunk_id, subject, relation, object,
                         confidence, emotion, rl_action, reward)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        (vid_id, chunk_db_id, vid_id[:10], action_label,
                         ent["text"], float(r), emo_res["label"],
                         action, float(r)))

                c4.execute(
                    "INSERT INTO rl_log(step,epsilon,reward,action,ts) VALUES(?,?,?,?,?)",
                    (self.rl.steps, self.rl.eps, r, action,
                     datetime.now().isoformat()))

                if idx % 100 == 0:
                    pct = 75 + int(20 * idx / max(limit, 1))
                    self.status["progress"] = pct
                    self._log(
                        f"  RL adım {idx}/{limit} | ε={self.rl.eps:.3f} | R={r:.3f}")

            self.db.commit()

            # ── Tüm işlenenleri işaretli olarak güncelle ───────────
            c5 = self.db.cursor()
            c5.execute("UPDATE videos SET processed=1 WHERE transcript!=''")
            self.db.commit()

            n_rels = self.db.cursor().execute(
                "SELECT COUNT(*) FROM relations").fetchone()[0]

            self.status.update(
                state="complete", progress=100,
                message=f"✓ Tamamlandı — {n_chunks} chunk · {n_rels} ilişki")
            self._log(
                f"✅ process_existing tamamlandı — {n_chunks} chunk, {n_rels} ilişki")

        except Exception as ex:
            tb = traceback.format_exc()
            self.status.update(state="error", message=f"Hata: {ex}")
            self._log(f"❌ {ex}\\n{tb}")


# ───────────────────────────────────────────────────────────────────
# FASTAPI  APP
# ───────────────────────────────────────────────────────────────────
app = FastAPI(title="YT-RAG-RL")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_pipeline: Optional[Pipeline] = None
_lock = threading.Lock()


def P() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = Pipeline()
    return _pipeline


# ── serve frontend ────────────────────────────────────────────────
# ── embedded HTML (merged from index.html) ───────────────────────
EMBEDDED_HTML = """\
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>YT · RAG · RL — Neural Pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
/* ── RESET & ROOT ───────────────────────────────────────────── */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:       #060810;
  --bg1:      #0b0f1a;
  --bg2:      #111827;
  --bg3:      #1a2235;
  --green:    #00e5a0;
  --green2:   #00b476;
  --cyan:     #22d3ee;
  --amber:    #f59e0b;
  --red:      #f87171;
  --purple:   #a78bfa;
  --text:     #cbd5e1;
  --text2:    #64748b;
  --text3:    #94a3b8;
  --border:   #1e2d3d;
  --glow:     0 0 20px rgba(0,229,160,0.15);
  --glowc:    0 0 20px rgba(34,211,238,0.15);
}
html,body{height:100%;background:var(--bg);color:var(--text);
          font-family:'JetBrains Mono',monospace;font-size:13px;overflow:hidden}

/* ── GRID BACKGROUND ────────────────────────────────────────── */
body::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(0,229,160,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,229,160,.03) 1px,transparent 1px);
  background-size:40px 40px;
}
body::after{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 80% 60% at 50% 0%,
    rgba(0,229,160,.06) 0%, transparent 60%);
}

/* ── LAYOUT ─────────────────────────────────────────────────── */
#app{position:relative;z-index:1;display:flex;flex-direction:column;
     height:100vh;overflow:hidden}

/* ── TOP BAR ────────────────────────────────────────────────── */
.topbar{
  display:flex;align-items:center;gap:16px;
  padding:0 20px;height:52px;
  background:rgba(6,8,16,.95);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(12px);
  flex-shrink:0;
}
.logo{
  font-family:'Syne',sans-serif;font-weight:800;font-size:15px;
  letter-spacing:.06em;color:var(--green);text-transform:uppercase;
  display:flex;align-items:center;gap:8px;
}
.logo-dot{
  width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.7)}}

.topbar-spacer{flex:1}
.stat-pill{
  display:flex;align-items:center;gap:6px;
  padding:4px 10px;border-radius:20px;
  background:var(--bg2);border:1px solid var(--border);
  color:var(--text3);font-size:11px;
}
.stat-pill .val{color:var(--cyan);font-weight:600}
.status-badge{
  padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;transition:all .3s;
}
.status-badge.idle    {background:rgba(100,116,139,.15);color:var(--text2);border:1px solid var(--border)}
.status-badge.running {background:rgba(245,158,11,.12);color:var(--amber);border:1px solid rgba(245,158,11,.3);animation:pulsebg 1.5s ease-in-out infinite}
.status-badge.complete{background:rgba(0,229,160,.1); color:var(--green);border:1px solid rgba(0,229,160,.3)}
.status-badge.error   {background:rgba(248,113,113,.1);color:var(--red);border:1px solid rgba(248,113,113,.3)}
@keyframes pulsebg{0%,100%{opacity:1}50%{opacity:.6}}

.ollama-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--text2);transition:background .3s;
}
.ollama-dot.alive{background:var(--green);box-shadow:0 0 6px var(--green)}

/* ── TABS ────────────────────────────────────────────────────── */
.tabs{
  display:flex;gap:2px;padding:0 20px;
  background:var(--bg1);border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.tab{
  padding:10px 16px;font-size:11px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;
  color:var(--text2);cursor:pointer;border-bottom:2px solid transparent;
  transition:all .2s;white-space:nowrap;
}
.tab:hover{color:var(--text3)}
.tab.active{color:var(--green);border-bottom-color:var(--green)}

/* ── MAIN BODY ───────────────────────────────────────────────── */
.main{flex:1;overflow:hidden;display:flex;flex-direction:column}
.panel{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:12px}
.panel::-webkit-scrollbar{width:4px}
.panel::-webkit-scrollbar-track{background:transparent}
.panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── CARDS ───────────────────────────────────────────────────── */
.card{
  background:var(--bg1);border:1px solid var(--border);
  border-radius:8px;padding:14px 16px;
}
.card-title{
  font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--text2);margin-bottom:10px;
  display:flex;align-items:center;gap:8px;
}
.card-title::before{
  content:'';width:3px;height:12px;border-radius:2px;
  background:var(--green);box-shadow:0 0 6px var(--green);
}

/* ── PROGRESS BAR ────────────────────────────────────────────── */
.prog-wrap{
  height:3px;background:var(--bg3);border-radius:2px;overflow:hidden;margin:8px 0;
}
.prog-bar{
  height:100%;border-radius:2px;transition:width .4s ease;
  background:linear-gradient(90deg,var(--green2),var(--green),var(--cyan));
  box-shadow:0 0 8px var(--green);
}
.prog-shimmer{
  width:100%;background:linear-gradient(90deg,transparent,rgba(0,229,160,.3),transparent);
  animation:shimmer 1.5s ease-in-out infinite;
}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(200%)}}

/* ── BTN ─────────────────────────────────────────────────────── */
.btn{
  padding:8px 18px;border:none;border-radius:6px;
  font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;
  cursor:pointer;transition:all .2s;letter-spacing:.05em;
}
.btn-primary{
  background:var(--green);color:#050710;
  box-shadow:0 0 16px rgba(0,229,160,.3);
}
.btn-primary:hover{background:#1fffa8;box-shadow:0 0 24px rgba(0,229,160,.5)}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-ghost{background:transparent;color:var(--text3);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--green);color:var(--green)}
.btn-warning{
  background:rgba(245,158,11,.15);color:var(--amber);
  border:1px solid rgba(245,158,11,.4);
}
.btn-warning:hover{background:rgba(245,158,11,.28);border-color:var(--amber)}
.btn-warning:disabled{opacity:.4;cursor:not-allowed}

/* ── CHAT ────────────────────────────────────────────────────── */
.chat-wrap{display:flex;flex-direction:column;height:100%;gap:10px}
.chat-messages{
  flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
  padding:4px 0;
}
.chat-messages::-webkit-scrollbar{width:4px}
.chat-messages::-webkit-scrollbar-thumb{background:var(--border)}
.msg{display:flex;flex-direction:column;gap:4px}
.msg-user .bubble{
  background:var(--bg3);border:1px solid var(--border);
  border-radius:8px 8px 2px 8px;padding:10px 14px;
  color:var(--cyan);align-self:flex-end;max-width:80%;
  margin-left:auto;
}
.msg-ai .bubble{
  background:rgba(0,229,160,.04);border:1px solid rgba(0,229,160,.15);
  border-radius:2px 8px 8px 8px;padding:12px 14px;
  color:var(--text);max-width:90%;line-height:1.7;
}
.msg-ai .bubble pre{
  background:var(--bg);border:1px solid var(--border);
  border-radius:4px;padding:8px;margin:8px 0;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word;
  font-size:12px;color:var(--green);
}
.msg-meta{
  font-size:10px;color:var(--text2);display:flex;gap:10px;flex-wrap:wrap;
  padding:2px 4px;
}
.meta-chip{
  padding:1px 6px;border-radius:3px;
  background:var(--bg2);border:1px solid var(--border);
}
.meta-chip.emotion-joy    {color:var(--green) ;border-color:rgba(0,229,160,.3)}
.meta-chip.emotion-anger  {color:var(--red)   ;border-color:rgba(248,113,113,.3)}
.meta-chip.emotion-neutral{color:var(--text2) }
.meta-chip.emotion-sadness{color:var(--cyan)  ;border-color:rgba(34,211,238,.3)}
.meta-chip.emotion-surprise{color:var(--purple);border-color:rgba(167,139,250,.3)}
.meta-chip.emotion-fear   {color:var(--amber) ;border-color:rgba(245,158,11,.3)}

.source-list{margin-top:6px;display:flex;flex-direction:column;gap:4px}
.source-item{
  font-size:11px;padding:6px 10px;border-radius:5px;
  background:var(--bg2);border:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;gap:8px;
}
.source-item a{color:var(--cyan);text-decoration:none}
.source-item a:hover{text-decoration:underline}
.score-bar{
  height:2px;border-radius:1px;background:var(--bg3);width:50px;flex-shrink:0;
}
.score-fill{height:100%;border-radius:1px;background:var(--green);transition:width .3s}

.chat-input-row{display:flex;gap:8px;flex-shrink:0}
.chat-input{
  flex:1;background:var(--bg2);border:1px solid var(--border);
  border-radius:6px;padding:10px 14px;
  font-family:'JetBrains Mono',monospace;font-size:13px;
  color:var(--text);outline:none;transition:border .2s;resize:none;height:44px;
}
.chat-input:focus{border-color:var(--green);box-shadow:0 0 0 2px rgba(0,229,160,.08)}
.typing-indicator{
  display:flex;gap:4px;align-items:center;padding:10px 14px;
  background:rgba(0,229,160,.04);border:1px solid rgba(0,229,160,.15);
  border-radius:2px 8px 8px 8px;width:60px;
}
.typing-dot{
  width:5px;height:5px;border-radius:50%;background:var(--green);
  animation:bounce .8s ease-in-out infinite;
}
.typing-dot:nth-child(2){animation-delay:.15s}
.typing-dot:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,100%{transform:translateY(0);opacity:.4}
                  50%{transform:translateY(-4px);opacity:1}}

/* ── LOGS ────────────────────────────────────────────────────── */
.log-box{
  background:var(--bg);border:1px solid var(--border);
  border-radius:6px;padding:10px 12px;
  height:320px;overflow-y:auto;
  font-size:11px;line-height:1.8;color:var(--text2);
}
.log-box::-webkit-scrollbar{width:4px}
.log-box::-webkit-scrollbar-thumb{background:var(--border)}
.log-line{padding:1px 0}
.log-line .ts{color:var(--text2);margin-right:6px}
.log-line .ok {color:var(--green)}
.log-line .err{color:var(--red)}
.log-line .warn{color:var(--amber)}
.log-line .phase{color:var(--cyan)}

/* ── TABLES ──────────────────────────────────────────────────── */
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th{
  text-align:left;padding:8px 10px;
  font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:var(--text2);
  border-bottom:1px solid var(--border);
}
.tbl td{
  padding:8px 10px;border-bottom:1px solid rgba(30,45,61,.5);
  vertical-align:middle;
}
.tbl tr:hover td{background:rgba(0,229,160,.025)}
.entity-tag{
  display:inline-block;padding:1px 6px;border-radius:3px;
  background:var(--bg3);border:1px solid var(--border);
  color:var(--cyan);font-size:10px;margin:1px;
}
.emo-tag{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-size:10px;font-weight:600;
}
.emo-joy    {background:rgba(0,229,160,.1);color:var(--green)}
.emo-anger  {background:rgba(248,113,113,.1);color:var(--red)}
.emo-neutral{background:var(--bg3);color:var(--text2)}
.emo-sadness{background:rgba(34,211,238,.1);color:var(--cyan)}
.emo-surprise{background:rgba(167,139,250,.1);color:var(--purple)}
.emo-fear   {background:rgba(245,158,11,.1);color:var(--amber)}
.emo-disgust{background:rgba(248,113,113,.08);color:#fca5a5}

/* ── RL STATS ────────────────────────────────────────────────── */
.rl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.rl-card{
  background:var(--bg);border:1px solid var(--border);border-radius:6px;
  padding:12px;text-align:center;
}
.rl-card .rl-val{
  font-size:22px;font-weight:700;color:var(--green);margin:4px 0;
  font-family:'Syne',sans-serif;
}
.rl-card .rl-lbl{font-size:9px;text-transform:uppercase;
                  letter-spacing:.1em;color:var(--text2)}

.action-label{
  display:inline-flex;align-items:center;gap:4px;
  padding:3px 8px;border-radius:4px;
  background:var(--bg3);border:1px solid var(--border);
  color:var(--purple);font-size:11px;
}
.action-label .idx{
  width:16px;height:16px;border-radius:3px;
  background:var(--purple);color:var(--bg);
  display:flex;align-items:center;justify-content:center;
  font-size:9px;font-weight:700;flex-shrink:0;
}

/* ── MINI SPARK CHART ────────────────────────────────────────── */
.spark{display:flex;align-items:flex-end;gap:1px;height:40px;padding:0}
.spark-bar{
  flex:1;min-width:2px;border-radius:1px 1px 0 0;
  background:var(--green);opacity:.7;transition:opacity .2s;
}
.spark-bar:hover{opacity:1}
.spark-bar.neg{background:var(--red)}

/* ── KG VIZ ──────────────────────────────────────────────────── */
.kg-container{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.kg-node{
  display:flex;align-items:center;justify-content:space-between;
  padding:6px 10px;border-radius:4px;
  background:var(--bg);border:1px solid var(--border);
  font-size:12px;
}
.kg-node .nfreq{
  font-size:10px;padding:1px 6px;border-radius:3px;
  background:rgba(0,229,160,.1);color:var(--green);
}
.kg-edge{
  padding:6px 10px;border-radius:4px;
  background:var(--bg);border:1px solid var(--border);
  font-size:11px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;
}
.kg-edge .subj{color:var(--cyan)}
.kg-edge .rel{
  padding:1px 6px;border-radius:3px;
  background:rgba(167,139,250,.1);color:var(--purple);
  font-size:10px;font-weight:600;
}
.kg-edge .obj{color:var(--text)}
.kg-edge .freq{
  margin-left:auto;font-size:10px;
  padding:1px 5px;border-radius:3px;
  background:var(--bg3);color:var(--text2);
}

/* ── PIPELINE VISUAL ─────────────────────────────────────────── */
.pipeline-steps{display:flex;flex-direction:column;gap:6px;padding:4px 0}
.pipe-step{
  display:flex;align-items:center;gap:10px;
  padding:8px 12px;border-radius:5px;
  background:var(--bg);border:1px solid var(--border);
  font-size:11px;transition:all .3s;
}
.pipe-step.active{
  background:rgba(0,229,160,.05);border-color:rgba(0,229,160,.3);
}
.pipe-step.done{border-color:rgba(0,229,160,.2)}
.pipe-icon{
  width:24px;height:24px;border-radius:5px;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;flex-shrink:0;background:var(--bg2);
}
.pipe-step.done .pipe-icon{background:rgba(0,229,160,.15)}
.pipe-name{flex:1;color:var(--text3)}
.pipe-step.active .pipe-name{color:var(--green)}
.pipe-step.done  .pipe-name{color:var(--text)}
.pipe-check{font-size:12px;color:var(--text2)}
.pipe-step.done .pipe-check{color:var(--green)}
.pipe-step.active .pipe-check{color:var(--amber)}

/* ── CLUSTER GRID ────────────────────────────────────────────── */
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.cluster-card{
  background:var(--bg);border:1px solid var(--border);
  border-radius:6px;padding:10px 12px;
}
.cluster-id{
  font-size:11px;font-weight:700;color:var(--cyan);margin-bottom:6px;
}
.cluster-examples{font-size:10px;color:var(--text2);line-height:1.6}

/* ── SCROLLABLE TABLE WRAP ───────────────────────────────────── */
.table-wrap{overflow-x:auto;border-radius:6px;border:1px solid var(--border)}

/* ── GRID ROWS ───────────────────────────────────────────────── */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}

/* ── EMPTY STATE ─────────────────────────────────────────────── */
.empty{
  text-align:center;padding:40px 20px;color:var(--text2);
  font-size:12px;
}
.empty .empty-icon{font-size:36px;margin-bottom:10px;opacity:.5}

/* ── TOOLTIP ─────────────────────────────────────────────────── */
[title]{cursor:help}

/* ── ANIMATIONS ──────────────────────────────────────────────── */
.fade-in{animation:fadeIn .3s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
</style>
</head>
<body>
<div id="app">

<!-- ── TOP BAR ─────────────────────────────────────────────────── -->
<div class="topbar">
  <div class="logo">
    <div class="logo-dot" :class="{running: pipelineStatus.state==='running'}"></div>
    YT·RAG·RL
  </div>
  <span style="font-size:10px;color:var(--text2);letter-spacing:.04em">
    phi4:14b · HDBSCAN · Q-Learning · BART·Emotion
  </span>
  <div class="topbar-spacer"></div>
  <div class="stat-pill" title="Videos in DB">
    📹 <span class="val">{{dbStats.videos||0}}</span>
  </div>
  <div class="stat-pill" title="Indexed chunks">
    🔷 <span class="val">{{dbStats.chroma||0}}</span>
  </div>
  <div class="stat-pill" title="Relations extracted">
    🔗 <span class="val">{{dbStats.relations||0}}</span>
  </div>
  <div class="stat-pill" :title="ollamaAlive?'Ollama running':'Ollama offline'">
    <div class="ollama-dot" :class="{alive:ollamaAlive}"></div>
    <span :style="{color: ollamaAlive?'var(--green)':'var(--text2)'}">
      {{ollamaAlive?'Ollama ✓':'Ollama ✗'}}
    </span>
  </div>
  <div class="status-badge" :class="pipelineStatus.state">
    {{pipelineStatus.state}}
  </div>
</div>

<!-- ── TABS ────────────────────────────────────────────────────── -->
<div class="tabs">
  <div v-for="t in tabs" :key="t.id"
       class="tab" :class="{active:activeTab===t.id}"
       @click="activeTab=t.id">
    {{t.icon}} {{t.label}}
  </div>
</div>

<!-- ── MAIN ────────────────────────────────────────────────────── -->
<div class="main">

  <!-- ═══ PIPELINE TAB ════════════════════════════════════════ -->
  <div v-if="activeTab==='pipeline'" class="panel fade-in">
    <div class="two-col">
      <!-- left: controls + status -->
      <div style="display:flex;flex-direction:column;gap:10px">
        <div class="card">
          <div class="card-title">Pipeline Control</div>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <button class="btn btn-primary"
                    :disabled="pipelineStatus.state==='running'"
                    @click="startPipeline">
              {{pipelineStatus.state==='running'?'⚙ Running…':'▶ Start Full Pipeline'}}
            </button>
            <button class="btn btn-warning"
                    :disabled="pipelineStatus.state==='running'"
                    @click="processExisting"
                    title="YouTube'a bağlanmadan DB'deki mevcut transcript'leri chunk+embed+ChromaDB+RL+KG olarak işle">
              {{pipelineStatus.state==='running'?'⚙ Running…':'⚡ Process Existing Transcripts'}}
            </button>
            <button class="btn btn-ghost" @click="refreshAll">↻ Refresh</button>
          </div>
          <div style="margin-top:8px;font-size:10px;color:var(--amber);opacity:.8">
            ⚡ butonu: YouTube'a bağlanmadan, DB'deki transcript'leri NLP·Embed·ChromaDB·RL·KG pipeline'ına sokar.
          </div>
          <div style="margin-top:10px;font-size:11px;color:var(--text2)">
            Channels: <span style="color:var(--cyan)">@ShmirchikArt/videos</span>
            + <span style="color:var(--cyan)">@ShmirchikArt/streams</span>
          </div>
          <div style="margin-top:6px;font-size:11px;color:var(--text2)">
            Max videos: <span style="color:var(--green)">100</span> ·
            Model: <span style="color:var(--amber)">phi4:14b</span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Progress</div>
          <div style="margin-bottom:6px;font-size:11px;color:var(--text3)">
            {{pipelineStatus.message}}
          </div>
          <div class="prog-wrap">
            <div class="prog-bar" :style="{width: pipelineStatus.progress+'%'}">
              <div v-if="pipelineStatus.state==='running'" class="prog-shimmer"></div>
            </div>
          </div>
          <div style="font-size:10px;color:var(--text2);text-align:right">
            {{pipelineStatus.progress}}%
          </div>
        </div>

        <div class="card">
          <div class="card-title">Database Stats</div>
          <div class="rl-grid">
            <div class="rl-card">
              <div class="rl-lbl">Videos</div>
              <div class="rl-val">{{dbStats.videos||0}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Processed</div>
              <div class="rl-val" style="color:var(--cyan)">{{dbStats.processed||0}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Chunks</div>
              <div class="rl-val" style="color:var(--amber)">{{dbStats.chunks||0}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Chroma</div>
              <div class="rl-val" style="color:var(--purple)">{{dbStats.chroma||0}}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- right: pipeline architecture steps -->
      <div class="card">
        <div class="card-title">Architecture Layers</div>
        <div class="pipeline-steps">
          <div v-for="(s,i) in pipeSteps" :key="i"
               class="pipe-step"
               :class="{active: isActiveStep(i), done: isDoneStep(i)}">
            <div class="pipe-icon">{{s.icon}}</div>
            <div style="flex:1">
              <div class="pipe-name">{{s.name}}</div>
              <div style="font-size:9px;color:var(--text2);margin-top:1px">{{s.desc}}</div>
            </div>
            <div class="pipe-check">
              <span v-if="isDoneStep(i)">✓</span>
              <span v-else-if="isActiveStep(i)" style="animation:pulse 1s infinite">●</span>
              <span v-else style="color:var(--bg3)">○</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- logs -->
    <div class="card">
      <div class="card-title" style="justify-content:space-between">
        <span>Live Logs</span>
        <span style="font-size:10px;color:var(--text2)">{{pipelineStatus.logs.length}} lines</span>
      </div>
      <div class="log-box" ref="logBox">
        <div v-for="(line,i) in pipelineStatus.logs" :key="i" class="log-line">
          <span :class="logClass(line)">{{line}}</span>
        </div>
        <div v-if="pipelineStatus.state==='running'" style="color:var(--amber);animation:pulse 1s infinite">
          ● processing…
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ CHAT TAB ════════════════════════════════════════════ -->
  <div v-if="activeTab==='chat'" class="panel fade-in">
    <div class="chat-wrap" style="height:calc(100vh - 130px)">
      <div class="chat-messages" ref="chatBox">
        <!-- welcome -->
        <div v-if="messages.length===0" class="empty">
          <div class="empty-icon">🧠</div>
          <div style="color:var(--text3);font-size:13px;margin-bottom:6px">
            Ask anything about the creator's content
          </div>
          <div style="color:var(--text2);font-size:11px">
            The system answers using vector retrieval + RL-learned patterns + Ollama phi4:14b
          </div>
          <div style="margin-top:16px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center">
            <span class="entity-tag" style="cursor:pointer"
                  v-for="q in suggestedQuestions" :key="q"
                  @click="sendSuggestion(q)">{{q}}</span>
          </div>
        </div>

        <div v-for="(m,i) in messages" :key="i"
             class="msg fade-in"
             :class="'msg-'+m.role">
          <div class="bubble" v-if="m.role==='user'">{{m.text}}</div>
          <template v-else>
            <div class="bubble" v-html="formatAnswer(m.text)"></div>
            <div class="msg-meta" v-if="m.rl || m.emotion">
              <span class="meta-chip"
                    :class="'emotion-'+(m.rl?.emotion||m.emotion||'neutral')">
                {{m.rl?.emotion||m.emotion||'neutral'}}
              </span>
              <span class="meta-chip" v-if="m.rl?.label">
                🔗 {{m.rl.label}}
              </span>
              <span class="meta-chip" v-if="m.rl?.cluster!==undefined">
                cluster {{m.rl.cluster}}
              </span>
            </div>
            <div class="source-list" v-if="m.sources && m.sources.length">
              <div style="font-size:10px;color:var(--text2);margin-bottom:2px">
                Sources:
              </div>
              <div v-for="(s,si) in m.sources" :key="si" class="source-item">
                <div style="flex:1;min-width:0">
                  <a :href="'https://youtube.com/watch?v='+s.video_id"
                     target="_blank">{{s.title||'Video'}}</a>
                  <div v-if="s.entities && s.entities.length" style="margin-top:2px">
                    <span class="entity-tag" v-for="e in s.entities.slice(0,3)" :key="e">
                      {{e}}
                    </span>
                  </div>
                </div>
                <div>
                  <span :class="'emo-tag emo-'+(s.emotion||'neutral')">
                    {{s.emotion||'?'}}
                  </span>
                  <div class="score-bar" style="margin-top:3px">
                    <div class="score-fill" :style="{width:(s.score*100)+'%'}"></div>
                  </div>
                </div>
              </div>
            </div>
          </template>
        </div>

        <div v-if="isQuerying" class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:180px 180px 1fr 120px;gap:8px">
        <input v-model="chatUserName"
               style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                      color:var(--text);padding:6px 8px;font-family:inherit"
               placeholder="Kullanıcı adı"/>
        <input v-model="chatVideoId"
               style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                      color:var(--text);padding:6px 8px;font-family:inherit"
               placeholder="Video ID"/>
        <input v-model="chatVideoTitle"
               style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                      color:var(--text);padding:6px 8px;font-family:inherit"
               placeholder="Video başlığı"/>
        <input v-model.number="chatVideoSecond" type="number" min="0"
               style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                      color:var(--text);padding:6px 8px;font-family:inherit"
               placeholder="Saniye"/>
      </div>

      <div class="chat-input-row">
        <textarea class="chat-input"
                  v-model="chatInput"
                  placeholder="Ask about the creator's recommendations, criticisms, patterns…"
                  @keydown.enter.exact.prevent="sendChat"
                  :disabled="isQuerying"></textarea>
        <button class="btn btn-primary"
                :disabled="isQuerying || !chatInput.trim()"
                @click="sendChat">
          {{isQuerying?'…':'Send'}}
        </button>
      </div>
    </div>
  </div>

  <!-- ═══ MESSAGES TAB ═══════════════════════════════════════ -->
  <div v-if="activeTab==='messages'" class="panel fade-in">
    <div class="card" style="display:flex;flex-direction:column;gap:10px">
      <div class="card-title">Mesajlar</div>
      <div style="display:flex;gap:8px;align-items:end;flex-wrap:wrap">
        <div style="display:flex;flex-direction:column;gap:4px;min-width:220px">
          <label style="font-size:10px;color:var(--text2)">Kullanıcı</label>
          <select v-model="selectedMessageUser"
                  style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;
                         color:var(--text);padding:7px;font-family:inherit">
            <option value="">Kullanıcı seç</option>
            <option v-for="u in messageUsers" :key="u" :value="u">{{u}}</option>
          </select>
        </div>
        <button class="btn btn-ghost" @click="loadMessageUsers">↻ Kullanıcıları Yenile</button>
        <button class="btn btn-ghost" :disabled="!selectedMessageUser" @click="loadSelectedUserMessages">↻ Mesajları Yenile</button>
        <button class="btn btn-primary" :disabled="!selectedMessageUser" @click="exportMessagesPdf">🧾 Mesajlar-Pdf</button>
      </div>

      <div class="table-wrap" style="max-height:58vh;overflow-y:auto">
        <table class="tbl">
          <thead>
            <tr>
              <th>Mesaj</th>
              <th>Video Başlığı</th>
              <th>Tarih-Saat</th>
              <th>Zaman</th>
              <th>YouTube Current Time</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in selectedUserMessages" :key="m.id">
              <td style="max-width:300px">{{m.message_text}}</td>
              <td style="max-width:220px">{{m.video_title || '—'}}</td>
              <td style="color:var(--text3)">{{formatLocalDate(m.created_at)}}</td>
              <td style="color:var(--amber)">{{formatSeconds(m.video_second)}}</td>
              <td>
                <a v-if="m.youtube_link" :href="m.youtube_link" target="_blank">open</a>
                <span v-else style="color:var(--text2)">—</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-if="!selectedMessageUser" class="empty">
        <div class="empty-icon">👤</div>
        Mesajları görmek için kullanıcı seç.
      </div>
      <div v-else-if="selectedUserMessages.length===0" class="empty">
        <div class="empty-icon">📭</div>
        Bu kullanıcı için mesaj bulunamadı.
      </div>
    </div>
  </div>

  <!-- ═══ VIDEOS TAB ══════════════════════════════════════════ -->
  <div v-if="activeTab==='videos'" class="panel fade-in">
    <div class="card" style="padding:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding:0 4px">
        <div class="card-title" style="margin:0">Video Index</div>
        <input v-model="videoSearch"
               style="background:var(--bg2);border:1px solid var(--border);
                      border-radius:4px;padding:5px 10px;color:var(--text);
                      font-family:inherit;font-size:11px;width:200px;outline:none"
               placeholder="Search titles…"/>
      </div>
      <div class="table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th>Title</th>
              <th>ID</th>
              <th style="text-align:center">Chunks</th>
              <th style="text-align:center">Status</th>
              <th>Fetched</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="v in filteredVideos" :key="v.id">
              <td>
                <a :href="v.url" target="_blank"
                   style="color:var(--cyan);text-decoration:none;
                          font-size:12px;display:block;max-width:420px;
                          overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                   :title="v.title">
                  {{v.title||'(no title)'}}
                </a>
              </td>
              <td style="color:var(--text2);font-size:10px">{{v.id}}</td>
              <td style="text-align:center">
                <span style="color:var(--amber)">{{v.chunks||0}}</span>
              </td>
              <td style="text-align:center">
                <span v-if="v.processed"
                      style="color:var(--green);font-size:10px">✓ done</span>
                <span v-else style="color:var(--text2);font-size:10px">○ pending</span>
              </td>
              <td style="color:var(--text2);font-size:10px">
                {{v.fetched_at ? v.fetched_at.slice(0,16).replace('T',' ') : '—'}}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="filteredVideos.length===0" class="empty">
        <div class="empty-icon">📭</div>
        No videos yet — run the pipeline first.
      </div>
    </div>
  </div>

  <!-- ═══ PATTERNS TAB ════════════════════════════════════════ -->
  <div v-if="activeTab==='patterns'" class="panel fade-in">
    <div class="two-col">
      <div class="card">
        <div class="card-title">Learned Relations (RL-Discovered)</div>
        <div class="table-wrap">
          <table class="tbl">
            <thead>
              <tr>
                <th>Relation</th>
                <th>Entity</th>
                <th>Emotion</th>
                <th style="text-align:right">Freq</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(p,i) in patterns.slice(0,30)" :key="i">
                <td>
                  <span class="action-label">
                    <span>{{p.relation||'—'}}</span>
                  </span>
                </td>
                <td style="color:var(--text);font-size:12px;max-width:150px;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                    :title="p.entity">
                  {{p.entity||'—'}}
                </td>
                <td>
                  <span :class="'emo-tag emo-'+(p.emotion||'neutral')">
                    {{p.emotion||'?'}}
                  </span>
                </td>
                <td style="text-align:right;color:var(--amber)">{{p.frequency}}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="patterns.length===0" class="empty">
          <div class="empty-icon">🔬</div>
          No patterns yet — run pipeline first.
        </div>
      </div>

      <!-- clusters -->
      <div class="card">
        <div class="card-title">Intent Clusters (HDBSCAN)</div>
        <div class="cluster-grid">
          <div v-for="c in clusters" :key="c.cluster_id" class="cluster-card">
            <div class="cluster-id">Cluster #{{c.cluster_id}}
              <span style="color:var(--text2);font-weight:400"> · {{c.count}} chunks</span>
            </div>
            <div style="margin-bottom:4px">
              <span :class="'emo-tag emo-'+(c.emotion||'neutral')">
                {{c.emotion||'?'}}
              </span>
            </div>
            <div class="cluster-examples">{{c.examples}}</div>
          </div>
        </div>
        <div v-if="clusters.length===0" class="empty">
          <div class="empty-icon">🌀</div>
          No clusters yet.
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ KNOWLEDGE GRAPH TAB ═════════════════════════════════ -->
  <div v-if="activeTab==='kg'" class="panel fade-in">
    <div style="display:grid;grid-template-columns:1fr 2fr;gap:12px">
      <!-- top nodes -->
      <div class="card">
        <div class="card-title">Top Entities</div>
        <div style="display:flex;flex-direction:column;gap:4px">
          <div v-for="(n,i) in (kgData.top_nodes||[]).slice(0,12)"
               :key="i" class="kg-node">
            <span :title="n.entity" style="overflow:hidden;text-overflow:ellipsis;
                  white-space:nowrap;max-width:140px;font-size:12px">
              {{n.entity}}
            </span>
            <span class="kg-node nfreq">×{{n.freq}}</span>
          </div>
        </div>
        <div v-if="!(kgData.top_nodes||[]).length" class="empty">
          No nodes yet.
        </div>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);
                    display:flex;gap:16px;font-size:10px;color:var(--text2)">
          <span>Nodes: <strong style="color:var(--cyan)">{{kgData.nodes||0}}</strong></span>
          <span>Edges: <strong style="color:var(--purple)">{{kgData.edges||0}}</strong></span>
        </div>
      </div>

      <!-- top edges -->
      <div class="card">
        <div class="card-title">Top Relations</div>
        <div style="display:flex;flex-direction:column;gap:4px;overflow-y:auto;max-height:400px">
          <div v-for="(e,i) in (kgData.top_edges||[]).slice(0,20)"
               :key="i" class="kg-edge">
            <span class="subj" :title="e.s"
                  style="max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
              {{e.s}}
            </span>
            <span>—</span>
            <span class="rel">{{e.r}}</span>
            <span>→</span>
            <span class="obj" :title="e.o"
                  style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
              {{e.o}}
            </span>
            <span :class="'emo-tag emo-'+(e.emotion||'neutral')">{{e.emotion||'?'}}</span>
            <span class="freq">×{{e.freq}}</span>
          </div>
        </div>
        <div v-if="!(kgData.top_edges||[]).length" class="empty">
          No edges yet — run pipeline.
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ RL STATS TAB ════════════════════════════════════════ -->
  <div v-if="activeTab==='rl'" class="panel fade-in">
    <div class="two-col">
      <!-- metrics -->
      <div style="display:flex;flex-direction:column;gap:10px">
        <div class="card">
          <div class="card-title">Agent Metrics</div>
          <div class="rl-grid">
            <div class="rl-card">
              <div class="rl-lbl">Epsilon (ε)</div>
              <div class="rl-val">{{rlStats.epsilon||'1.0'}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Memory</div>
              <div class="rl-val" style="color:var(--cyan)">{{rlStats.memory_size||0}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Steps</div>
              <div class="rl-val" style="color:var(--amber)">{{rlStats.steps||0}}</div>
            </div>
            <div class="rl-card">
              <div class="rl-lbl">Total Reward</div>
              <div class="rl-val" style="color:var(--purple)">
                {{rlStats.total_reward||'0'}}
              </div>
            </div>
          </div>
        </div>

        <!-- action labels -->
        <div class="card">
          <div class="card-title">Discovered Action Labels</div>
          <div style="display:flex;flex-direction:column;gap:5px">
            <div v-for="(label,idx) in (rlStats.action_labels||{})"
                 :key="idx" class="action-label">
              <span class="idx">{{idx}}</span>
              {{label}}
            </div>
          </div>
        </div>

        <!-- Epsilon decay progress -->
        <div class="card">
          <div class="card-title">Exploration Rate (ε)</div>
          <div style="position:relative;height:6px;background:var(--bg3);
                      border-radius:3px;overflow:hidden;margin:6px 0">
            <div style="height:100%;border-radius:3px;transition:width .4s;
                        background:linear-gradient(90deg,var(--red),var(--amber),var(--green))"
                 :style="{width: ((rlStats.epsilon||1.0)*100)+'%'}">
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text2)">
            <span>Exploit (0)</span>
            <span>Explore (1)</span>
          </div>
        </div>
      </div>

      <!-- reward chart -->
      <div style="display:flex;flex-direction:column;gap:10px">
        <div class="card">
          <div class="card-title">Reward History (last 100 steps)</div>
          <div class="spark" style="height:60px">
            <div v-for="(r,i) in sparkRewards" :key="i"
                 class="spark-bar"
                 :class="{neg: r < 0}"
                 :style="{height: Math.abs(r)*100+'%', opacity: 0.5 + (i/sparkRewards.length)*0.5}"
                 :title="r.toFixed(3)">
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-size:10px;color:var(--text2);margin-top:4px">
            <span>oldest</span>
            <span v-if="sparkRewards.length">
              avg: {{(sparkRewards.reduce((a,b)=>a+b,0)/sparkRewards.length).toFixed(3)}}
            </span>
            <span>latest</span>
          </div>
        </div>

        <!-- RL log -->
        <div class="card">
          <div class="card-title">Recent RL Steps</div>
          <div class="table-wrap" style="max-height:220px;overflow-y:auto">
            <table class="tbl">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>ε</th>
                  <th>Reward</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row,i) in rlLog.slice(-20).reverse()" :key="i">
                  <td style="color:var(--text2)">{{row.step}}</td>
                  <td style="color:var(--cyan)">{{(row.epsilon||0).toFixed(3)}}</td>
                  <td :style="{color: row.reward>0?'var(--green)':'var(--red)'}">
                    {{(row.reward||0).toFixed(3)}}
                  </td>
                  <td>
                    <span class="action-label" style="font-size:10px">
                      <span class="idx">{{row.action}}</span>
                      {{(rlStats.action_labels||{})[row.action]||'?'}}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- end main -->
</div><!-- end app -->

<script>
const { createApp, ref, computed, onMounted, onUnmounted, nextTick, watch } = Vue;

createApp({
  setup() {
    // ── state ───────────────────────────────────────────────
    const activeTab       = ref('pipeline');
    const pipelineStatus  = ref({state:'idle', progress:0, message:'Ready', logs:[]});
    const dbStats         = ref({});
    const ollamaAlive     = ref(false);
    const videos          = ref([]);
    const patterns        = ref([]);
    const clusters        = ref([]);
    const kgData          = ref({});
    const rlStats         = ref({});
    const rlLog           = ref([]);
    const messages        = ref([]);
    const chatInput       = ref('');
    const chatUserName    = ref('default-user');
    const chatVideoId     = ref('');
    const chatVideoTitle  = ref('');
    const chatVideoSecond = ref(0);
    const messageUsers    = ref([]);
    const selectedMessageUser = ref('');
    const selectedUserMessages = ref([]);
    const isQuerying      = ref(false);
    const videoSearch     = ref('');
    const chatBox         = ref(null);
    const logBox          = ref(null);
    let   pollTimer       = null;

    // ── tabs ────────────────────────────────────────────────
    const tabs = [
      {id:'pipeline', icon:'⚙', label:'Pipeline'},
      {id:'chat',     icon:'💬', label:'Ask'},
      {id:'messages', icon:'🧾', label:'Mesajlar'},
      {id:'videos',   icon:'📹', label:'Videos'},
      {id:'patterns', icon:'🔬', label:'Patterns'},
      {id:'kg',       icon:'🕸',  label:'KG'},
      {id:'rl',       icon:'🤖', label:'RL Stats'},
    ];

    // ── pipeline steps ──────────────────────────────────────
    const pipeSteps = [
      {icon:'📥', name:'YouTube Fetch',       desc:'yt-dlp · metadata · 100 videos'},
      {icon:'📝', name:'Transcript Download', desc:'youtube-transcript-api · multilingual'},
      {icon:'✂️', name:'Preprocessing',       desc:'clean · sentence-split · chunk'},
      {icon:'🧩', name:'Embeddings',          desc:'paraphrase-multilingual-MiniLM-L12'},
      {icon:'🏷',  name:'NER Extraction',     desc:'spaCy transformer · no rules'},
      {icon:'😐', name:'Emotion Analysis',    desc:'j-hartmann DistilRoBERTa'},
      {icon:'🌀', name:'Intent Discovery',    desc:'HDBSCAN unsupervised clustering'},
      {icon:'🤖', name:'Q-Learning Agent',    desc:'linear DQN · replay buffer'},
      {icon:'🕸',  name:'Knowledge Graph',    desc:'entity + relation graph'},
      {icon:'🔷', name:'Chroma Index',        desc:'vector store · semantic search'},
    ];

    const progressToStep = computed(() => {
      const p = pipelineStatus.value.progress;
      if (p <  8) return 0;
      if (p < 40) return 1;
      if (p < 50) return 2;
      if (p < 60) return 3;
      if (p < 62) return 4;
      if (p < 64) return 5;
      if (p < 75) return 6;
      if (p < 90) return 7;
      if (p < 95) return 8;
      return 9;
    });

    function isActiveStep(i) {
      return pipelineStatus.value.state==='running' && i===progressToStep.value;
    }
    function isDoneStep(i) {
      if (pipelineStatus.value.state==='complete') return true;
      return pipelineStatus.value.state==='running' && i < progressToStep.value;
    }

    // ── suggested questions ─────────────────────────────────
    const suggestedQuestions = [
      'What books has the creator recommended?',
      'Which rabbis does the creator mention positively?',
      'Who does the creator criticize?',
      'What are the creator\\'s main topics?',
      'Which streamers or creators are recommended?',
      'What emotions are most common in the content?',
    ];

    // ── API helpers ─────────────────────────────────────────
    async function api(path, opts={}) {
      try {
        const r = await fetch('/api' + path, opts);
        return await r.json();
      } catch(e) {
        return null;
      }
    }

    function formatSeconds(seconds) {
      const s = Math.max(0, Number(seconds||0));
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = Math.floor(s % 60);
      return [h, m, sec].map(v => String(v).padStart(2,'0')).join(':');
    }

    function formatLocalDate(iso) {
      if (!iso) return '—';
      const d = new Date(iso);
      if (isNaN(d)) return iso;
      return d.toLocaleString('tr-TR');
    }

    async function loadMessageUsers() {
      const res = await api('/messages/users');
      if (res?.users) {
        messageUsers.value = res.users;
        if (!selectedMessageUser.value && res.users.length) {
          selectedMessageUser.value = res.users[0];
        }
      }
    }

    async function loadSelectedUserMessages() {
      if (!selectedMessageUser.value) {
        selectedUserMessages.value = [];
        return;
      }
      const res = await api('/messages?user=' + encodeURIComponent(selectedMessageUser.value));
      if (res?.messages) selectedUserMessages.value = res.messages;
    }

    async function exportMessagesPdf() {
      if (!selectedMessageUser.value) return;
      const res = await api('/messages/export?user=' + encodeURIComponent(selectedMessageUser.value));
      const rows = res?.messages || [];
      const htmlRows = rows.map((m, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td>${(m.message_text||'').replaceAll('<','&lt;')}</td>
          <td>${(m.video_title||'').replaceAll('<','&lt;')}</td>
          <td>${formatLocalDate(m.created_at)}</td>
          <td>${formatSeconds(m.video_second)}</td>
          <td>${m.youtube_link ? `<a href="${m.youtube_link}">${m.youtube_link}</a>` : '—'}</td>
        </tr>
      `).join('');

      const w = window.open('', '_blank');
      if (!w) return;
      w.document.write(`
        <html><head><title>Mesajlar PDF</title>
        <style>
          body{font-family:Arial,sans-serif;padding:16px}
          table{width:100%;border-collapse:collapse;font-size:12px}
          th,td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}
          th{background:#f5f5f5}
          h2{margin-bottom:4px}
        </style></head><body>
          <h2>Kullanıcı Mesajları</h2>
          <div><strong>Kullanıcı:</strong> ${selectedMessageUser.value}</div>
          <div><strong>Oluşturma:</strong> ${new Date().toLocaleString('tr-TR')}</div>
          <table>
            <thead>
              <tr><th>#</th><th>Mesaj</th><th>Video Başlığı</th><th>Tarih-Saat</th><th>Zaman Damgası</th><th>YouTube Link</th></tr>
            </thead>
            <tbody>${htmlRows || '<tr><td colspan="6">Kayıt yok</td></tr>'}</tbody>
          </table>
        </body></html>
      `);
      w.document.close();
      w.focus();
      w.print();
    }

    async function startPipeline() {
      await api('/pipeline/start', {method:'POST'});
      setTimeout(pollStatus, 500);
    }

    async function processExisting() {
      if (pipelineStatus.value.state === 'running') return;
      const res = await api('/pipeline/process-existing', {method:'POST'});
      if (res && res.ok === false) {
        alert('Pipeline zaten çalışıyor: ' + (res.reason || ''));
        return;
      }
      setTimeout(pollStatus, 500);
    }

    async function pollStatus() {
      const s = await api('/pipeline/status');
      if (s) {
        pipelineStatus.value = s;
        // auto-scroll logs
        nextTick(() => {
          if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight;
        });
      }
      if (s?.state==='running') {
        clearTimeout(pollTimer);
        pollTimer = setTimeout(pollStatus, 1200);
      } else if (s?.state==='complete') {
        refreshAll();
      }
    }

    async function refreshAll() {
      const [db, ol, vids, pats, cl, kg, rl, rlog] = await Promise.all([
        api('/db/stats'),
        api('/ollama'),
        api('/videos'),
        api('/patterns'),
        api('/clusters'),
        api('/kg'),
        api('/rl/stats'),
        api('/rl/log'),
      ]);
      if (db)   dbStats.value   = db;
      if (ol)   ollamaAlive.value = ol.alive;
      if (vids) videos.value    = vids;
      if (pats) patterns.value  = pats;
      if (cl)   clusters.value  = cl;
      if (kg)   kgData.value    = kg;
      if (rl)   rlStats.value   = rl;
      if (rlog) rlLog.value     = rlog;
    }

    // ── chat ────────────────────────────────────────────────
    async function sendChat() {
      const q = chatInput.value.trim();
      if (!q || isQuerying.value) return;
      chatInput.value = '';
      messages.value.push({role:'user', text:q});
      await api('/messages', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          username: chatUserName.value || 'default-user',
          message_text: q,
          video_id: chatVideoId.value || '',
          video_title: chatVideoTitle.value || '',
          video_second: Number(chatVideoSecond.value || 0),
        })
      });
      isQuerying.value = true;
      await nextTick();
      scrollChat();

      const res = await api('/query', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question:q})
      });

      isQuerying.value = false;
      if (res && res.answer) {
        messages.value.push({
          role:'ai',
          text:    res.answer,
          sources: res.sources||[],
          rl:      res.rl||null,
          patterns:res.patterns||[],
        });
      } else {
        messages.value.push({role:'ai', text:'Error — no response from server.'});
      }
      await nextTick();
      scrollChat();
    }

    function sendSuggestion(q) {
      chatInput.value = q;
      sendChat();
    }

    function scrollChat() {
      if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight;
    }

    // ── format answer ────────────────────────────────────────
    function formatAnswer(text) {
      if (!text) return '';
      return text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/```([\\s\\S]*?)```/g, '<pre>$1</pre>')
        .replace(/`([^`]+)`/g,'<code style="color:var(--green);background:var(--bg2);padding:1px 4px;border-radius:3px">$1</code>')
        .replace(/\\*\\*(.+?)\\*\\*/g,'<strong style="color:var(--cyan)">$1</strong>')
        .replace(/\\n/g,'<br>');
    }

    // ── log coloring ─────────────────────────────────────────
    function logClass(line) {
      if (line.includes('✅') || line.includes('✓') || line.includes('complete'))
        return 'ok';
      if (line.includes('❌') || line.includes('Error') || line.includes('error'))
        return 'err';
      if (line.includes('Phase') || line.includes('▶'))
        return 'phase';
      if (line.includes('⚠') || line.includes('warn'))
        return 'warn';
      return '';
    }

    // ── computed ──────────────────────────────────────────────
    const filteredVideos = computed(() => {
      const q = videoSearch.value.toLowerCase();
      if (!q) return videos.value;
      return videos.value.filter(v => (v.title||'').toLowerCase().includes(q));
    });

    const sparkRewards = computed(() => {
      const data = rlLog.value.slice(-100).map(r => r.reward||0);
      if (!data.length) return [];
      const mx = Math.max(...data.map(Math.abs), 0.001);
      return data.map(v => v / mx);
    });

    // ── watchers ──────────────────────────────────────────────
    watch(activeTab, (tab) => {
      if (tab === 'videos')   api('/videos').then(r=>r&&(videos.value=r));
      if (tab === 'patterns') {
        api('/patterns').then(r=>r&&(patterns.value=r));
        api('/clusters').then(r=>r&&(clusters.value=r));
      }
      if (tab === 'kg')       api('/kg').then(r=>r&&(kgData.value=r));
      if (tab === 'messages') {
        loadMessageUsers().then(loadSelectedUserMessages);
      }
      if (tab === 'rl')  {
        api('/rl/stats').then(r=>r&&(rlStats.value=r));
        api('/rl/log').then(r=>r&&(rlLog.value=r));
      }
    });

    // ── lifecycle ────────────────────────────────────────────
    onMounted(async () => {
      await pollStatus();
      await refreshAll();
      await loadMessageUsers();
      // poll status every 3s
      const intervalId = setInterval(async () => {
        const s = await api('/pipeline/status');
        if (s) pipelineStatus.value = s;
        const db = await api('/db/stats');
        if (db) dbStats.value = db;
        const ol = await api('/ollama');
        if (ol) ollamaAlive.value = ol.alive;
      }, 3000);
      onUnmounted(() => clearInterval(intervalId));
    });

    return {
      activeTab, tabs, pipelineStatus, dbStats, ollamaAlive,
      videos, patterns, clusters, kgData, rlStats, rlLog,
      messages, chatInput, chatUserName, chatVideoId, chatVideoTitle, chatVideoSecond,
      messageUsers, selectedMessageUser, selectedUserMessages,
      isQuerying, videoSearch, chatBox, logBox,
      pipeSteps, suggestedQuestions, filteredVideos, sparkRewards,
      startPipeline, processExisting, pollStatus, refreshAll, sendChat, sendSuggestion,
      loadMessageUsers, loadSelectedUserMessages, exportMessagesPdf,
      formatAnswer, logClass, isActiveStep, isDoneStep, formatSeconds, formatLocalDate,
    };
  }
}).mount('#app');
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return EMBEDDED_HTML


# ── API endpoints ─────────────────────────────────────────────────
@app.post("/api/pipeline/start")
async def start_pipeline(bg: BackgroundTasks):
    p = P()
    if p.status["state"] == "running":
        return {"ok": False, "reason": "already_running"}
    bg.add_task(p.run)
    return {"ok": True}

@app.post("/api/pipeline/process-existing")
async def process_existing_endpoint(bg: BackgroundTasks):
    """Mevcut DB transcript'lerini işle — YouTube'a bağlanmadan."""
    p = P()
    if p.status["state"] == "running":
        return {"ok": False, "reason": "already_running"}
    bg.add_task(p.process_existing)
    return {"ok": True}

@app.get("/api/pipeline/status")
async def pipeline_status():
    return P().status

@app.post("/api/query")
async def query_api(req: Request):
    body = await req.json()
    q    = (body.get("question") or "").strip()
    if not q:
        return {"error": "empty question"}
    return P().query(q)

@app.post("/api/messages")
async def save_message(req: Request):
    body = await req.json()
    P().save_user_message(
        username=body.get("username") or "",
        message_text=body.get("message_text") or "",
        video_id=body.get("video_id") or "",
        video_title=body.get("video_title") or "",
        video_second=body.get("video_second") or 0,
    )
    return {"ok": True}

@app.get("/api/messages/users")
async def message_users():
    return {"users": P().get_message_users()}

@app.get("/api/messages")
async def user_messages(user: str = ""):
    return {"messages": P().get_user_messages(user)}

@app.get("/api/messages/export")
async def export_messages(user: str = ""):
    return {"messages": P().get_user_messages(user)}

@app.get("/api/videos")
async def videos():
    return P().get_videos()

@app.get("/api/patterns")
async def patterns():
    return P().get_patterns()

@app.get("/api/rl/stats")
async def rl_stats():
    p = P()
    return {**p.rl.stats(), "kg": p.kg.summary()}

@app.get("/api/rl/log")
async def rl_log():
    return P().get_rl_log()

@app.get("/api/clusters")
async def clusters():
    return P().get_clusters()

@app.get("/api/kg")
async def kg():
    return P().kg.summary()

@app.get("/api/ollama")
async def ollama_info():
    ol = P().ollama
    return {"alive": ol.alive(), "models": ol.models()}

@app.get("/api/db/stats")
async def db_stats():
    p  = P()
    c  = p.db.cursor()
    return {
        "videos":    c.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        "processed": c.execute("SELECT COUNT(*) FROM videos WHERE processed=1").fetchone()[0],
        "chunks":    c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "relations": c.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
        "chroma":    p._chroma().count(),
    }


# ───────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  YouTube RAG · NLP · RL  — starting server  ║")
    print("║  http://localhost:8000                       ║")
    print("╚══════════════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


# ═══════════════════════════════════════════════════════════════════
# REQUIREMENTS (pip install):
#
#   fastapi uvicorn[standard]
#   yt-dlp youtube-transcript-api
#   chromadb sentence-transformers
#   transformers torch
#   spacy hdbscan scikit-learn numpy
#   requests
#
# spaCy model (one of):
#   python -m spacy download xx_core_web_sm
#   python -m spacy download en_core_web_sm
#
# Ollama must be running:
#   ollama serve
#   ollama pull phi4:14b
# ═══════════════════════════════════════════════════════════════════
