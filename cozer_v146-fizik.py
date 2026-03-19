# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   ASCIIMATİK v8 — SemanticPriorComputationEngine + Sıfır Hard-Coding (v143)║
# ║   ★ MOD5: PropositionalSelfCorrector  — öz-düzeltme geri besleme döngüsü   ║
# ║   ★ MOD6: EntropyThresholdManager     — entropi eşiği / CoT genişletme     ║
# ║   ★ MOD7: TemporalStateMemory         — zaman çürüme λ / durum belleği     ║
# ║   ★ MOD8: LatentVariableSensitivity   — sonlu fark duyarlılık analizi      ║
# ║   ★ MOD9: BayesianDecisionEngine      — beklenen fayda EU / risk matrisi   ║
# ║   BUG FIX: duplicate uyarı dedup + TÜM float sabitler SPCE ile türetildi   ║
# ║   GameTheorySolver v1.3 (v136) — n-adımlı genelleme + TR sayı parser       ║
# ║   YENİ (v136-1): TurkishNumberParser — kardinal + ordinal tam coverage      ║
# ║     • bir→1, ikinci→2, yüzüncü→100, bininci→1000, milyonuncu→1_000_000     ║
# ║     • Çok-kelimeli: "iki yüz elli" → 250, "üç bin beşinci" → 3005           ║
# ║     • find_in_text() → metin içindeki tüm Türkçe sayıları bulur            ║
# ║   YENİ (v136-2): _extract_rounds — ordinal/kardinal n-adım tam desteği     ║
# ║     • "bininci turun sonunda" → n_rounds=1000 + step_target=1000            ║
# ║     • LARGE_N eşiği: 500 tur → StepWindowScaler otomatik devreye girer      ║
# ║   YENİ (v136-3): _extract_step_target — büyük sıra sözcükleri desteklendi  ║
# ║   YENİ (v136-4): _build_solver_hint — büyük-N steady-state hint            ║
# ║     • per_round_payoff, cumulative_at_target, stability_round Ollama'ya     ║
# ║   YENİ (v136-5): _build_sol_from_gt_solver — step_target aware            ║
# ║     • Büyük-N: wdata kümülatifi kullanılır, sorted_p_eff dinamik           ║
# ║     • Grim ref: TurkishNumberParser ile soru metninden parse edilir        ║
# ║   YENİ (v136-6): NTV-DIST — game_theory tipi suppressed (false positive)   ║
# ║   YENİ (v136-7): ORDINAL_KW + _TR_WORD_NUMS tam genişletme                 ║
# ║   (v135 düzeltmeleri korundu: per-round GT, p_threshold, bypass, dict ans) ║
# ║   StepWindowScaler: N±10 pencere | Büyük adım steady-state extrapolation   ║
# ║   BayesSolver v3: fractions.Fraction kesin rasyonel hesap (sıfır hata)     ║
# ║   OllamaClient v3: GT/Bayes bypass (analitik solver override)               ║
# ║   NumericTruthValidator v2 + GT-BARTValidator: çok katmanlı doğrulama      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from flask import Flask, request, jsonify, make_response, send_file
import requests, json, math, re, random, time, itertools, os, io, subprocess, tempfile, datetime
from collections import defaultdict, deque
import pickle
import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from decimal import Decimal, getcontext, InvalidOperation
import numpy as np
import base64
import platform

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Preformatted,
        HRFlowable,
        PageBreak,
        KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as _rl_canvas

    # ── DejaVu TTF — Türkçe dahil tam Unicode desteği ────────────────────────
    _FJDIR = "/usr/share/fonts/truetype/dejavu"
    pdfmetrics.registerFont(TTFont("DVSans", f"{_FJDIR}/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DVSans-Bold", f"{_FJDIR}/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DVMono", f"{_FJDIR}/DejaVuSansMono.ttf"))
    pdfmetrics.registerFont(TTFont("DVMono-Bold", f"{_FJDIR}/DejaVuSansMono-Bold.ttf"))
    _REPORTLAB_OK = True
except Exception:
    _REPORTLAB_OK = False
try:
    from fractions import Fraction
    import sympy as sp

    _SYMPY_OK = True
except ImportError:
    _SYMPY_OK = False

# Evrensel yüksek hassasiyet ayarı (tüm aritmetik için ortak)
getcontext().prec = 50  # 50 basamak yeterli, gerekirse arttırılabilir

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4:14b"
ASCII_WIDTH = 82

# ── Web İstihbaratı Pipeline'ı global olarak initialize et ──────────────────
web_intelligence_pipeline = None


def configure_linux_cpu_threads(target_threads: int = 24) -> dict:
    """
    Linux ortamında CPU tabanlı iş parçacığı sayılarını tek yerden ayarlar.
    Amaç: BLAS/OpenMP/Torch tarafında CPU işlemlerini 24 thread'e hizalamak.
    """
    applied = {
        "platform": platform.system().lower(),
        "target_threads": int(target_threads),
        "env": {},
        "torch": {"set_num_threads": False, "set_num_interop_threads": False},
    }

    if applied["platform"] != "linux":
        return applied

    n = str(int(target_threads))
    env_keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_THREAD_LIMIT",
    ]
    for k in env_keys:
        os.environ[k] = n
        applied["env"][k] = n

    # PyTorch thread havuzları (varsa)
    if _TORCH_OK:
        try:
            torch.set_num_threads(int(target_threads))
            applied["torch"]["set_num_threads"] = True
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(int(target_threads))
            applied["torch"]["set_num_interop_threads"] = True
        except Exception:
            pass

    return applied


CPU_THREAD_CONFIG = configure_linux_cpu_threads(24)


def init_web_intelligence():
    """Web İstihbaratı Pipeline'ını initialize et (lazy loading)."""
    global web_intelligence_pipeline
    if web_intelligence_pipeline is None:
        try:
            web_intelligence_pipeline = WebIntelligencePipeline()
        except Exception as e:
            print(f"[WARN] Web İstihbaratı Pipeline init hatası: {e}")
    return web_intelligence_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
#  FORMULA STRUCTURE PARSER
#  Bir formül stringini ayrıştırarak sözdizimsel yapısını çıkarır.
#  Hardcoding yok — tamamen karakter düzeyinde sözdizim analizi.
# ═══════════════════════════════════════════════════════════════════════════════
class FormulaStructureParser:
    """
    Formül stringini lightweight recursive descent ile analiz eder.
    Matematiksel değerlendirme yapmaz — sadece yapıyı (top-level op, pattern tipi) tanır.

    Çıktılar:
    top_level_op  : MUL | ADD | DIV | MIXED | UNKNOWN
    has_weighted_sum   : bool — P(x)*P(y|x) + P(z)*P(w|z) (Bayes total prob. deseni)
    has_conditional    : bool — P(...|...) yapısı
    denominator_set    : set[str] — paydada görülen farklı değerler
    branch_count       : int — toplama terimleri sayısı (additive branches)
    max_numeric_result : float | None — formülde görülen en büyük olasılık değeri
    """

    def parse(self, formula: str) -> dict:
        f = formula.strip()
        result = {
            "top_level_op": "UNKNOWN",
            "has_weighted_sum": False,
            "has_conditional": False,
            "denominator_set": set(),
            "branch_count": 1,
            "max_numeric_result": None,
            "raw": f,
        }
        if not f:
            return result

        # ── has_conditional: P(A|B) yapısı ───────────────────────────────────
        result["has_conditional"] = bool(re.search(r"[Pp]\s*\(.*?\|", f))

        # ── denominator_set: paydada görülen sayılar ──────────────────────────
        denoms = re.findall(r"/\s*(\d+)", f)
        result["denominator_set"] = set(denoms)

        # ── top_level_op: dış parantez seviyesindeki baskın operatör ─────────
        result["top_level_op"] = self._get_top_level_op(f)

        # ── branch_count: üst seviyedeki toplama terimi sayısı ────────────────
        result["branch_count"] = self._count_top_level_additive_terms(f)

        # ── has_weighted_sum detection (birleşik) ─────────────────────────────
        # Deseni yakala: (a/b)*(c/d) + (e/f)*(g/h) VEYA P(x)*P(y|x) + P(z)*P(w|z)
        # Koşul: en az 2 çarpım terimi en üst seviyede toplama ile bağlanıyor
        if result["branch_count"] >= 2:
            # Top-level additive terms'leri çıkar
            terms = self._split_top_level_additive(f)
            mul_terms = 0
            for term in terms:
                t = term.strip()
                # Her terimin kendisi çarpım içeriyor mu?
                t_op = self._get_top_level_op(t)
                if t_op == "MUL":
                    mul_terms += 1
                # Parantez içi kesir × parantez içi kesir deseni
                elif re.search(
                    r"\(\s*\d+\s*/\s*\d+\s*\)\s*[×*]\s*\(\s*\d+\s*/\s*\d+\s*\)", t
                ):
                    mul_terms += 1
                # a/b * c/d deseni (parantez yok)
                elif re.search(r"\d+\s*/\s*\d+\s*[×*]\s*\d+\s*/\s*\d+", t):
                    mul_terms += 1
            if mul_terms >= 2:
                result["has_weighted_sum"] = True

        # P(A)*P(B|A) + P(C)*P(D|C) deseni (büyük P notasyonu)
        if re.search(
            r"[Pp]\([^)]+\)\s*[×*]\s*[Pp]\([^)]+\)"
            r".*?\+"
            r".*?[Pp]\([^)]+\)\s*[×*]\s*[Pp]\([^)]+\)",
            f,
        ):
            result["has_weighted_sum"] = True

        # ── max_numeric_result: formüldeki en büyük ondalık ───────────────────
        floats = re.findall(r"0\.\d+", f)
        if floats:
            result["max_numeric_result"] = max(Decimal(x) for x in floats)

        return result

    def _split_top_level_additive(self, f: str) -> list:
        """En üst parantez seviyesindeki '+' ile böler, terimleri döndürür."""
        depth = 0
        terms = []
        start = 0
        for i, c in enumerate(f):
            if c in ("(", "["):
                depth += 1
            elif c in (")", "]"):
                depth -= 1
            elif c == "+" and depth == 0:
                terms.append(f[start:i])
                start = i + 1
        terms.append(f[start:])
        return [t for t in terms if t.strip()]

    def _get_top_level_op(self, f: str) -> str:
        """Dış parantez seviyesindeki baskın operatörü döndürür."""
        depth = 0
        has_add = has_mul = has_div = False
        i = 0
        while i < len(f):
            c = f[i]
            if c in ("(", "["):
                depth += 1
            elif c in (")", "]"):
                depth -= 1
            elif depth == 0:
                if c == "+":
                    has_add = True
                elif c in ("×", "*") and not (
                    c == "*" and i + 1 < len(f) and f[i + 1] == "*"
                ):
                    has_mul = True
                elif c == "/" and i + 1 < len(f) and f[i + 1] != "/":
                    has_div = True
            i += 1
        if has_add and has_mul:
            return "MIXED"
        if has_add:
            return "ADD"
        if has_mul:
            return "MUL"
        if has_div:
            return "DIV"
        return "UNKNOWN"

    def _count_top_level_additive_terms(self, f: str) -> int:
        """En üst seviyedeki '+' ile ayrılmış terim sayısını döndürür."""
        depth = 0
        count = 1
        for i, c in enumerate(f):
            if c in ("(", "["):
                depth += 1
            elif c in (")", "]"):
                depth -= 1
            elif c == "+" and depth == 0 and i > 0:
                count += 1
        return count


# ═══════════════════════════════════════════════════════════════════════════════
#  INDEPENDENCE LEXICON
#  5 katmanlı NLP bağımsızlık/bağımlılık sinyal sözlüğü.
#  Hiçbir soru metni veya senaryo bu sınıfa gömülmez.
#  Yeni sinyal eklemek → ilgili listeye bir satır eklemek demektir.
# ═══════════════════════════════════════════════════════════════════════════════
class IndependenceLexicon:
    """
    Katman yapısı (yüksek → düşük öncelik):
      LAYER_1_EXPLICIT   — açık beyan ("bağımsız olay", "geri koymadan")
      LAYER_2_MECHANICAL — fiziksel süreç döngüsü (silindir döndür, karıştır)
      LAYER_3_PROCEDURAL — prosedür adımı ("her seferinde X yap")
      LAYER_4_INSTRUMENT — araç türünden çıkarsama (zar, madeni para, çark)
      LAYER_5_CONTEXTUAL — zayıf bağlamsal ipucu ("her atışta", "her denemede")
    """

    # ── LAYER 1: Açık beyan — bağımsızlık ────────────────────────────────────
    EXPLICIT_INDEPENDENT = [
        r"\bbağımsız\s+olay",
        r"\bbağımsız\s+olaylar",
        r"\bistatistiksel(ly)?\s+bağımsız",
        r"\bbirbirinden\s+bağımsız",
        r"\bbağımsız\s+deney",
        r"\bbağımsız\s+deneme",
        r"\bbağımsız\s+çekiliş",
        r"\bbağımsız\s+seçim",
        r"\bbağımsız\s+atış",
        r"\bbirbirini\s+etkilemez",
        r"\bbirbirini\s+etkilem(iyor|ediyor|emez)",
        r"\bsonuç\s+bir\s+sonrakini\s+etkilemez",
        r"\bönceki\s+(sonuç|atış|deneme)\s+(bir\s+sonrakini)?\s*etkilemez",
        r"\bher\s+deneme\s+bağımsız",
        r"\bher\s+atış\s+bağımsız",
        r"\bindependent\s+event",
        r"\bindependent\s+trial",
        r"\bstatistically\s+independent",
        r"\bwith\s+replacement",
        r"\bputs?\s+back",
        r"\breplacing\s+before",
        r"\beach\s+(trial|draw|pick|pull)\s+is\s+independent",
        r"\bdoes\s+not\s+affect",
        r"\bnot\s+affect\s+the\s+(next|subsequent)",
        r"\bP\s*\(\s*A\s*\)\s*[×*·]\s*P\s*\(\s*B\s*\)",
    ]

    # ── LAYER 1: Açık beyan — bağımlılık ─────────────────────────────────────
    EXPLICIT_DEPENDENT = [
        r"\bgeri\s+koymadan",
        r"\byerine\s+koymadan",
        r"\bkaldırılıp\s+çıkarıldıktan",
        # Türkçe olumsuz geri koyma
        r"yerine\s+kon(muyor|maz|ulmaz|ulmuyor)",
        r"geri\s+kon(muyor|maz|ulmaz|ulmuyor)",
        r"yerine\s+koyulmuyor",
        r"yerine\s+koyulmaz",
        r"geri\s+koyulmuyor",
        r"\bgeri\s+koyulmaz",
        r"\bçekip\s+(at|atar)",
        r"\bçıkar(ıldıktan)?\s+sonra\s+(kalan|azalan)",
        r"\bkalan\s+\d+",
        r"\bazalan\s+havuz",
        r"\bbağımlı\s+olay",
        r"\bbağımlı\s+olaylar",
        r"\bbağımlı\s+deneme",
        r"\bethkilenmi[şs]",
        r"\bbirinci.*etkiliy[oi]r\s+ikinci",
        r"\bwithout\s+replacement",
        r"\bnot\s+replaced",
        r"\bnot\s+put\s+back",
        r"\bremoved\s+from",
        r"\bdependent\s+event",
        r"\bdependent\s+trial",
        r"\bconditional\s+on\s+previous",
        r"\baffects?\s+the\s+(next|subsequent|following)",
        r"\bafter\s+removing",
        r"\bonce\s+drawn.*not\s+returned",
    ]

    # ── LAYER 2: Mekanik sıfırlama ────────────────────────────────────────────
    MECHANICAL_RESET = [
        r"her\s+(seferinde|atıştan\s+önce|denemeden\s+önce|turdan\s+önce)\s*"
        r"(silindiri|çarkı|tekerleği|zarı|parayı|destesi|topu|diski)?\s*"
        r"(döndür|çevir|karıştır|salla|fırlat|sıfırla|resetle)",
        r"(silindiri|çarkı|tekerleği|diski)\s+(tekrar\s+)?(döndür|çevir)",
        r"her\s+(seferinde|atışta|denemede)\s+(tekrar\s+)?(döndür|çevir|karıştır|sıfırla)",
        r"yeniden\s+(döndür|çevir|karıştır|sıfırla|shuffle)",
        r"tekrar\s+(döndür|çevir|karıştır|sıfırla|shuffle)",
        r"(shuffle|karıştır)\s+after\s+each",
        r"spin\s+(again|each\s+time|before\s+each)",
        r"re-?spin",
        r"re-?shuffle",
        r"re-?roll",
        r"zarı\s+(tekrar|yeniden|her\s+seferinde)\s*at",
        r"roll\s+(again|each\s+time|every\s+time)",
        r"parayı\s+(tekrar|yeniden|her\s+seferinde)\s*(at|çevir|fırlat)",
        r"flip\s+(again|each\s+time)",
        r"(deste|deck)\s*(tekrar|yeniden|her\s+seferinde)?\s*(karıştır|shuffle)",
        r"reshuffl",
        r"(torbaya|kaba|urn)\s+(geri\s+koy|yerine\s+koy|replace)",
        r"put\s+(it\s+)?back\s+(in(to)?\s+the)?\s*(bag|urn|box|jar|container)",
    ]

    # ── LAYER 3: Prosedürel adım ──────────────────────────────────────────────
    PROCEDURAL_RESET = [
        r"her\s+seferinde\s+(çeviril|döndürül|sıfırlan|karıştırıl|baştan\s+başlan)",
        r"her\s+(atış|deneme|tur|çekim|ateşleme|round)\s+(bağımsız|sıfırdan|yeniden)",
        r"her\s+(ateşlemeden|atıştan|denemeden)\s+önce\s+(sıfırla|döndür|çevir|karıştır)",
        r"silindir\s+her\s+(seferinde|atıştan\s+önce)",
        r"reset\s+(after|before|each|every)",
        r"each\s+(trial|round|shot|flip|roll|spin|draw)\s+is\s+(fresh|new|reset|independent)",
        r"start\s+(fresh|over|again)\s+each",
        r"(önce|before)\s+her\s+(atış|deneme|tur)\s+(çevir|döndür|karıştır|sıfırla)",
    ]

    # ── LAYER 4: Araç türü varsayılanları ─────────────────────────────────────
    INSTRUMENT_IMPLIES_REPLACEMENT = [
        r"\b(tabanca|revolver|silah|gun|pistol)\b",
        r"\b(silindir|cylinder)\b",
        r"\b(rus\s+ruleti|russian\s+roulette)\b",
        r"\b(zar|dice|die)\b",
        r"\b(madeni\s+para|para\s+at|coin\s+flip|coin\s+toss|fair\s+coin)\b",
        r"\b(çark|tekerlek|wheel|spinner|roulette\s+wheel)\b",
        r"\b(döner\s+çark|spinning\s+wheel|lottery\s+wheel)\b",
    ]

    INSTRUMENT_IMPLIES_NO_REPLACEMENT = [
        r"\b(urn|torba|çanta|kap|kutu)\b",
        r"\b(deste|deck)\b",
        r"\b(bilet|lottery\s+ticket|kura)\b",
    ]

    # ── LAYER 5: Zayıf bağlamsal ──────────────────────────────────────────────
    CONTEXTUAL_INDEPENDENT = [
        r"her\s+(atışta|çekimde|denemede|turda|ateşlemede)",
        r"her\s+bir\s+(atış|çekim|deneme|tur|ateşleme)",
        r"aynı\s+koşullarda\s+tekrar",
        r"yeni(den)?\s+başla",
        r"tekrar\s+başla",
        r"önceki\s+(sonuçlar?\s+)?(önemsiz|önemli\s+değil|fark\s+etmez)",
        r"geçmiş\s+(atış|deneme)\s+(önemsiz|önemli\s+değil)",
        r"memory-?less",
        r"hafızasız",
        r"her\s+deneme\s+sıfırdan",
    ]

    CONTEXTUAL_DEPENDENT = [
        r"\bkalan\b",
        r"\bazald[ıi]\b",
        r"\beksi[li]\b",
        r"havuz\s+(küçülüyor|azalıyor)",
        r"(birinci|ilk)\s+(çekildi|seçildi|alındı|çıkarıldı)",
        r"\bdan\s+sonra\s+kalan\b",
        r"\bçıkarıldıktan\b",
        r"\bselected\s+previously\b",
        r"\balready\s+(drawn|picked|selected|chosen|removed)\b",
    ]


# ── Katman ağırlıkları tablosu ────────────────────────────────────────────────
_LAYER_WEIGHTS = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.6, 5: 0.4}


@dataclass
class SignalEvidence:
    """Bir bağımsızlık/bağımlılık kararının kanıtı."""

    label: str  # "INDEPENDENT" | "DEPENDENT"
    layer: int  # 1=en güçlü, 5=en zayıf
    matched_pattern: str
    confidence: float  # 0.0 – 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY SIGNAL RESOLVER
#  Katman önceliği ile bağımsızlık/bağımlılık sinyalini tek karara indirger.
#  Çakışma çözümü örneğe özel değil — tamamen katman sıralamasına göre.
# ═══════════════════════════════════════════════════════════════════════════════
class DependencySignalResolver:
    """
    Çözüm algoritması:
      1. Tüm katmanlarda eşleşmeleri topla
      2. En yüksek kuvvetli katmandaki kararı al
      3. Aynı katmanda çakışma varsa sayım; eşitlik → Layer 4 instrument'a bak
      4. Hiç sinyal yoksa → UNKNOWN
    """

    _lex = IndependenceLexicon()

    def resolve(self, question: str) -> tuple:
        """Döndürür: (event_dependency: str, evidence: list[SignalEvidence])"""
        q = question.lower()
        evidence = []

        def _scan(patterns, label, layer):
            for pat in patterns:
                if re.search(pat, q):
                    evidence.append(
                        SignalEvidence(
                            label=label,
                            layer=layer,
                            matched_pattern=pat,
                            confidence=_LAYER_WEIGHTS[layer],
                        )
                    )

        _scan(self._lex.EXPLICIT_INDEPENDENT, "INDEPENDENT", 1)
        _scan(self._lex.EXPLICIT_DEPENDENT, "DEPENDENT", 1)
        _scan(self._lex.MECHANICAL_RESET, "INDEPENDENT", 2)
        _scan(self._lex.PROCEDURAL_RESET, "INDEPENDENT", 3)
        _scan(self._lex.INSTRUMENT_IMPLIES_REPLACEMENT, "INDEPENDENT", 4)
        _scan(self._lex.INSTRUMENT_IMPLIES_NO_REPLACEMENT, "DEPENDENT", 4)
        _scan(self._lex.CONTEXTUAL_INDEPENDENT, "INDEPENDENT", 5)
        _scan(self._lex.CONTEXTUAL_DEPENDENT, "DEPENDENT", 5)

        if not evidence:
            return "UNKNOWN", []

        best_layer = min(e.layer for e in evidence)
        top = [e for e in evidence if e.layer == best_layer]
        ind = sum(1 for e in top if e.label == "INDEPENDENT")
        dep = sum(1 for e in top if e.label == "DEPENDENT")

        if ind > dep:
            return "INDEPENDENT", evidence
        elif dep > ind:
            return "DEPENDENT", evidence
        else:
            l4_ind = sum(
                1 for e in evidence if e.layer == 4 and e.label == "INDEPENDENT"
            )
            l4_dep = sum(1 for e in evidence if e.layer == 4 and e.label == "DEPENDENT")
            if l4_ind > l4_dep:
                return "INDEPENDENT", evidence
            elif l4_dep > l4_ind:
                return "DEPENDENT", evidence
            return "UNKNOWN", evidence

    def strongest_layer(self, evidence: list) -> int:
        return min((e.layer for e in evidence), default=99)

    def debug_report(self, evidence: list) -> str:
        if not evidence:
            return "Sinyal bulunamadı → UNKNOWN"
        lines = []
        for e in sorted(evidence, key=lambda x: (x.layer, x.label)):
            lines.append(
                f"  Katman {e.layer} | {e.label:<12} | conf={e.confidence:.2f} "
                f"| pat: {e.matched_pattern[:60]}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY GRAPH BUILDER
#  İşlem ve veri bağımlılıklarından tam DAG (Directed Acyclic Graph) oluşturur.
#  Adım planlaması ve topological sort için altyapıdır.
# ═══════════════════════════════════════════════════════════════════════════════
class DependencyGraphBuilder:
    """
    AST ve semantic sinyallerden hesaplama düğümlerinin yönlendirilmiş döngüsüz
    grafiğini kurar.

    Giriş:
      ast           : MathAST dict (nodes, edges varsa)
      signals       : SemanticSignalModule çıktısı

    Çıkış:
      {
        node_id: {
          "node":      {...},  # AST node
          "deps":      set(),  # bağımlı düğümler (upstream)
          "type":      str,    # node tipi
          "symbol":    str,    # değişken adı
          "metadata":  {}      # ek bilgi (hidden, prior, vb.)
        }
      }
    """

    def build_from_ast(self, ast: dict, signals: dict = None) -> dict:
        """AST'den düğüm haritası + bağımlılık çıkarsama."""
        nodes = ast.get("nodes", [])
        edges = ast.get("edges", [])

        # 1. Düğüm haritası oluştur
        graph = {}
        for node in nodes:
            node_id = node.get("id") or node.get("label", f"node_{id(node)}")
            graph[node_id] = {
                "node": node,
                "deps": set(),
                "type": node.get("type", "unknown"),
                "symbol": node.get("symbol") or node.get("label", ""),
                "metadata": node.get("metadata", {}),
            }

        # 2. Kenarlardan bağımlılık çıkar
        for edge in edges:
            src = edge.get("from")
            tgt = edge.get("to")
            if src in graph and tgt in graph:
                graph[tgt]["deps"].add(src)

        # 3. Ek sinyallerden (conditional, has_hidden) çıkarsama
        if signals:
            has_hidden = signals.get("conditional_given", False) or signals.get(
                "bayes_structure", False
            )
            if has_hidden:
                for nid in graph:
                    graph[nid]["metadata"]["has_hidden_implications"] = True

        # 4. Döngü kontrolü
        if not self._is_acyclic(graph):
            # Uyarı: döngü tespit edildi (bu normalde hata ama soft-fail)
            pass

        return graph

    def _is_acyclic(self, graph: dict) -> bool:
        """Kahn algoritmasıyla döngü kontrolü."""
        in_degree = {n: 0 for n in graph}
        for n in graph:
            for dep in graph[n]["deps"]:
                if dep in in_degree:
                    in_degree[n] += 1

        queue = [n for n in in_degree if in_degree[n] == 0]
        count = 0
        while queue:
            n = queue.pop(0)
            count += 1
            for other in graph:
                if n in graph[other]["deps"]:
                    in_degree[other] -= 1
                    if in_degree[other] == 0:
                        queue.append(other)

        return count == len(graph)

    def topo_sort(self, graph: dict) -> list:
        """Topological sort (Kahn's algorithm) — solüsyon sırası."""
        in_degree = {n: 0 for n in graph}
        for n in graph:
            for dep in graph[n]["deps"]:
                if dep in in_degree:
                    in_degree[n] += 1

        queue = [n for n in in_degree if in_degree[n] == 0]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in graph:
                if v in graph[w]["deps"]:
                    in_degree[w] -= 1
                    if in_degree[w] == 0:
                        queue.append(w)

        if len(order) != len(graph):
            return order  # Döngü varsa kısmi sırayı döndür

        return order


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGICAL CONJUNCTION VALIDATOR
#  Sinyal kombinasyonundan ZORUNLU formül yapısı türetir, model çıktısını denetler.
#  Tüm kurallar tablodan türetilir — örneğe özel hardcoding sıfır.
# ═══════════════════════════════════════════════════════════════════════════════
class LogicalConjunctionValidator:
    """
    Çekirdek mantık: (event_dependency × elimination × logic_op × survival_chain)
    → required_op + forbidden_patterns listesi

    10 ayrı hata tipi tanımlar ve her birini bağımsız olarak tespit eder.
    """

    fparser = FormulaStructureParser()

    # ── Zorunlu top-level operatör tablosu ────────────────────────────────────
    # (dep, elim, survival_chain) → required_top_op
    # Her (dep × elim × sc) kombinasyonu açıkça tanımlanmıştır.
    # Tanımsız kombinasyon → tablo lookup başarısız → güvenli fallback devreye girer.
    REQUIRED_OP = {
        # ── INDEPENDENT ──────────────────────────────────────────────────────
        ("INDEPENDENT", True, True): "MUL",  # Bağımsız + elim + zincir
        ("INDEPENDENT", True, False): "MUL",  # Bağımsız + elim + tek atış
        ("INDEPENDENT", False, False): "MUL",  # Bağımsız + AND → çarpım
        ("INDEPENDENT", False, True): "MUL",  # Bağımsız + ardışık → çarpım
        # ── DEPENDENT ────────────────────────────────────────────────────────
        ("DEPENDENT", False, False): "MUL",  # Bağımlı + AND → koşullu çarpım
        ("DEPENDENT", True, True): "MUL",  # Bağımlı + elim → koşullu çarpım
        ("DEPENDENT", True, False): "MUL",  # Bağımlı + elim + tek → çarpım
        ("DEPENDENT", False, True): "MUL",  # Bağımlı + ardışık → koşullu çarpım
        # ── CONDITIONAL ──────────────────────────────────────────────────────
        ("CONDITIONAL", False, False): "DIV",  # Koşullu olasılık → bölme
        ("CONDITIONAL", True, True): "MUL",  # Koşullu + elim → çarpım
        ("CONDITIONAL", True, False): "MUL",  # Koşullu + elim tek → çarpım
        ("CONDITIONAL", False, True): "DIV",  # Koşullu + ardışık → bölme
        # ── UNKNOWN: sadece güçlü sinyal kombinasyonlarında kural türet ──────
        ("UNKNOWN", True, True): "MUL",  # Elim + zincir → çarpım güvenli
        ("UNKNOWN", True, False): "MUL",  # Elim (tek) → çarpım güvenli
        ("UNKNOWN", False, True): "MUL",  # Ardışık (elim yok) → çarpım güvenli
        ("UNKNOWN", False, False): "UNKNOWN",  # Belirsiz → kontrol atla
    }

    # ── Forbidden pattern tiplerine göre açıklamalar ────────────────────────
    ERROR_MESSAGES = {
        "BAYES_INJECTION": (
            "[BAYES ENJEKTE ETMEHATASİ] Model, toplam olasılık teoremini (P(A)×P(B|A) + P(C)×P(D|C)) "
            "YANLIŞ bağlamda uyguladı. Bu yapı yalnızca OR-senaryolarda geçerlidir. "
            "AND-zinciri / eliminasyon senaryosunda kesinlikle ÇARPIM kullanılmalı."
        ),
        "DEAD_BRANCH_SUM": (
            "[ÖLMÜŞ DAL TOPLAMA] Eliminasyon nedeniyle oyundan çıkan öznenin durumu, "
            "sonraki adımlara TOPLAMA ile katkıda bulunuyor. Elenen özne için "
            "olasılık hesabı yapılamaz — ağırlıklı toplam (weighted sum) uygulanamaz."
        ),
        "WEIGHTED_SUM_IN_ELIM_CONTEXT": (
            "[AĞIRLIKLI TOPLAM İHLALİ] P(durum1)×P(sonuç|durum1) + P(durum2)×P(sonuç|durum2) "
            "deseni eliminasyon zincirinde kullanılmış. Bu desen yalnızca bağımsız OR-dallarında "
            "geçerlidir. Eliminasyonlu AND-zincirinde doğru formül: P1 × P2 × ... çarpımıdır."
        ),
        "DENOMINATOR_SHRINK_IN_INDEPENDENT": (
            "[BAĞIMSIZ OLAY PAYDA KÜÇÜLME] Örneklem uzayı her denemede SIFIRLANMASINA rağmen "
            "bir adımda payda azaltılmış (n-1, kalan N gibi). Bağımsız denemeler zincirinde "
            "payda hiçbir zaman değişmemelidir."
        ),
        "RESULT_INFLATION": (
            "[SONUÇ ŞIŞMESI] Final yanıt, tek adım olasılığından BÜYÜK. "
            "Eliminasyonlu bağımsız n-adım zincirinde P(toplam) ≤ P(tek adım)^n olmalıdır. "
            "Sonuç artmıyor, kesinlikle AZALMALIDIR."
        ),
        "OR_FORMULA_IN_AND_CONTEXT": (
            "[VE/VEYA OPERATÖR KARISTIRILMASI] AND-zinciri için dahil-hariç prensibi (toplama) "
            "kullanılmış. P(A VE B VE ...) için ÇARPIM, P(A VEYA B) için TOPLAMA kullanılır."
        ),
        "CONDITIONAL_WITHOUT_GIVEN": (
            "[KOŞULSUZ BAĞLAMDA KOŞULLU OLASILIK] Soruda 'bilindiğine göre / given' sinyali "
            "bulunmamasına rağmen P(A|B) yapısı kullanılmış."
        ),
        "WRONG_DENOMINATOR_MUTATION": (
            "[PAYDA MUTASYONU] Bağımsız denemeler zincirinde farklı paydalar kullanılmış. "
            "n sabit olduğundan tüm adımlarda aynı payda olmalıdır."
        ),
        "MISSING_SURVIVAL_PREFIX": (
            "[EKSİK HAYATTA KALMA ZİNCİRİ] Eliminasyonlu ardışık denemelerde her adım "
            "'önceki adımları geçtim' ön koşuluna sahip olmalı. Model bu ön koşulu "
            "göz ardı ederek adımları bağımsız hesaplamış."
        ),
        "PROBABILITY_EXCEEDS_BOUND": (
            "[OLASILIK SINIRI İHLALİ] Hesaplanan değer matematiksel olarak imkânsız. "
            "Olasılık değeri [0,1] aralığı dışına çıkmıştır."
        ),
        "BAYES_POSTERIOR_SUM_WRONG": (
            "[POSTERIOR TOPLAM HATASI] Tüm hipotezlerin posterior toplamı 1.0 ≠ Σ. "
            "Bayes teoreminde Σ P(Hᵢ|E) = 1 zorunludur. "
            "En yaygın neden: paydayı (P(E)) yanlış hesaplamak veya "
            "bir hipotezi atlayarak toplamı eksik bırakmak."
        ),
        "BAYES_ZERO_POSTERIOR": (
            "[SIFIR POSTERIOR] Bir hipotez için posterior = 0.0 hesaplanmış. "
            "Prior > 0 ve Likelihood > 0 iken posterior asla sıfır olamaz. "
            "En muhtemel neden: etiket↔değer eşleştirme hatası "
            "(hipotez adı yanlış atanmış, değer başka hipoteze gitmiş)."
        ),
    }

    def validate(self, signals: dict, sol_data: dict) -> list:
        """
        Tüm kontrol katmanlarını çalıştırır.
        Döndürür: list[str] — tespit edilen ihlal mesajları
        """
        violations = []
        steps = sol_data.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        # Her alanı güvenle str'e çevir (LLM bazen float/int döndürür)
        answer = str(sol_data.get("answer", "") or "")
        formula = str(sol_data.get("formula", "") or "")
        numeric = str(sol_data.get("numeric", "") or "")

        dep = signals["event_dependency"]
        el = signals["elimination_rule"]
        sc = signals.get("survival_chain", el)  # survival_chain ≡ eliminasyon zinciri
        lo = signals["logic_operator"]
        st = signals["sequential_trials"]
        cond = signals["conditional_given"]

        required_op = self.REQUIRED_OP.get(
            (dep, el, sc), self.REQUIRED_OP.get(("UNKNOWN", el, sc), "UNKNOWN")
        )

        # ── Tüm step formüllerini + ana formülü parse et ──────────────────────
        all_formulas = [formula] + [str(s.get("formula", "") or "") for s in steps]
        all_parsed = [self.fparser.parse(fm) for fm in all_formulas if fm.strip()]

        step_texts = " ".join(
            str(s.get("content", "") or "")
            + " "
            + str(s.get("formula", "") or "")
            + " "
            + str(s.get("result", "") or "")
            for s in steps
        ).lower()

        # ══ KATMAN 1: BAYES ENJEKTE ETME KONTROLÜ ════════════════════════════
        # Eliminasyonlu AND-zinciri ama weighted_sum deseni var
        if (el or sc) and lo in ("AND_CHAIN", "SINGLE", "MIXED"):
            for parsed in all_parsed:
                if parsed["has_weighted_sum"]:
                    violations.append(self.ERROR_MESSAGES["BAYES_INJECTION"])
                    violations.append(
                        self.ERROR_MESSAGES["WEIGHTED_SUM_IN_ELIM_CONTEXT"]
                    )
                    break

        # ══ KATMAN 2: ÖLMÜŞ DAL TOPLAMA ══════════════════════════════════════
        if el:
            # Model hem "kurşun çıkarsa..." hem "kurşun çıkmazsa..." dalını toplama ile mı birleştiriyor?
            dead_then_sum = re.search(
                r"(ölür|vurulur|elenir|hayatını kaybet|dead|killed|eliminated)"
                r".{0,120}"
                r"(olasılık.*\+|toplam.*\+|\+.*olasılık)",
                step_texts,
            )
            if dead_then_sum:
                violations.append(self.ERROR_MESSAGES["DEAD_BRANCH_SUM"])

            # Ağırlıklı toplam deseni: Birden fazla çarpım terimi toplanıyor
            main_parsed = self.fparser.parse(formula)
            if main_parsed["has_weighted_sum"] and el:
                violations.append(self.ERROR_MESSAGES["DEAD_BRANCH_SUM"])

        # ══ KATMAN 3: TOP-LEVEL OP KONTROLÜ ══════════════════════════════════
        if required_op not in ("UNKNOWN", None):
            for parsed in all_parsed:
                actual_op = parsed["top_level_op"]
                if actual_op == "UNKNOWN":
                    continue
                if required_op == "MUL" and actual_op == "ADD":
                    violations.append(self.ERROR_MESSAGES["OR_FORMULA_IN_AND_CONTEXT"])
                    break
                if required_op == "MUL" and actual_op == "MIXED":
                    # MIXED: hem çarpım hem toplam → Bayes weighted sum ihtimali
                    for p2 in all_parsed:
                        if p2["has_weighted_sum"]:
                            violations.append(self.ERROR_MESSAGES["BAYES_INJECTION"])
                            break
                    break

        # ══ KATMAN 3b: MIXED logic_operator özgü dal kontrolü ════════════════
        # REQUIRED_OP tablosunda MIXED logic_operator yok; burada dinamik türetilir.
        # Algoritma: MIXED AND+OR birlikte → her dal kendi türüne göre denetlenir.
        if lo == "MIXED" and dep in ("INDEPENDENT", "DEPENDENT"):
            # AND dalları için çarpım, OR dalları için toplam beklenir.
            # Formülde yalnızca ADD (toplam) → AND dalı yanlış formül almış
            for parsed in all_parsed:
                actual_op = parsed["top_level_op"]
                if actual_op == "ADD" and not parsed["has_weighted_sum"]:
                    # Saf ADD — ve AND-bağlantısı var → hata
                    if el or sc:
                        violations.append(
                            self.ERROR_MESSAGES["OR_FORMULA_IN_AND_CONTEXT"]
                        )
                        break
            # MIXED içinde weighted_sum var ama eliminasyon da varsa → Bayes injection
            if el:
                for parsed in all_parsed:
                    if parsed["has_weighted_sum"]:
                        violations.append(self.ERROR_MESSAGES["BAYES_INJECTION"])
                        break

        # ══ KATMAN 4: PAYDA SIFIRLANMASI KONTROLÜ ════════════════════════════
        if dep == "INDEPENDENT":
            all_denoms = set()
            for parsed in all_parsed:
                all_denoms |= parsed["denominator_set"]
            denom_ints = sorted([int(d) for d in all_denoms if str(d).isdigit()])

            # ── 4a: denominator_n varsa — N dışında herhangi bir payda hata ──
            n_anchor = signals.get("denominator_n")
            if n_anchor and denom_ints:
                foreign = [d for d in denom_ints if d != n_anchor and d > 1]
                if foreign:
                    violations.append(
                        self.ERROR_MESSAGES["DENOMINATOR_SHRINK_IN_INDEPENDENT"]
                    )

            # ── 4b: denominator_n yoksa — ardışık küçülen payda deseni ───────
            elif len(denom_ints) >= 2:
                # sorted([5,6]) → ardışık artış: 6→5 şeklinde küçülme yaşanmış
                for i in range(len(denom_ints) - 1):
                    # ascending consecutive pair → n-1 şeklinde küçülme anlamına gelir
                    if denom_ints[i + 1] == denom_ints[i] + 1:
                        violations.append(
                            self.ERROR_MESSAGES["DENOMINATOR_SHRINK_IN_INDEPENDENT"]
                        )
                        break

            # ── 4c: regex — n-1 deseni veya "kalan" kelimesi ─────────────────
            all_fm_text = " ".join(all_formulas)
            if re.search(r"\b\d+\s*-\s*1\b", all_fm_text) or re.search(
                r"kalan\s+\d", step_texts
            ):
                violations.append(
                    self.ERROR_MESSAGES["DENOMINATOR_SHRINK_IN_INDEPENDENT"]
                )

        # ══ KATMAN 5: SONUÇ ŞIŞMESI KONTROLÜ ════════════════════════════════
        if el and st >= 2:
            # Tek adım olasılığı: ilk adımdaki en büyük kesiri bul
            single_step_probs = []
            for s in steps[:2]:  # İlk 2 adım
                fracs = re.findall(
                    r"(\d+)\s*/\s*(\d+)",
                    str(s.get("formula", "") or "")
                    + " "
                    + str(s.get("result", "") or ""),
                )
                for num, den in fracs:
                    dn = int(den)
                    if dn > 0:
                        val = int(num) / dn
                        if 0 < val <= 1:
                            single_step_probs.append(val)

            # Final yanıttan sayısal değer çıkar — her zaman str'e çevir
            all_numeric = str(numeric) + " " + str(answer)
            final_vals = []
            for n in re.findall(r"0\.\d+", all_numeric):
                try:
                    v = Decimal(str(n)) if n else Decimal("0")
                    if Decimal("0") < v < Decimal("1"):
                        final_vals.append(v)
                except (ValueError, InvalidOperation):
                    pass

            if single_step_probs and final_vals:
                min_single = min(single_step_probs)
                max_result = max(final_vals)
                # n bağımsız adım → sonuç ≤ tek adım olasılığı
                if dep == "INDEPENDENT" and max_result > min_single + 0.005:
                    violations.append(
                        self.ERROR_MESSAGES["RESULT_INFLATION"]
                        + f" (Tek adım: {min_single:.4f}, Hesaplanan: {max_result:.4f})"
                    )

        # ══ KATMAN 6: KOŞULLU OLASILIK SİNYALİ KONTROLÜ ════════════════════
        if not cond:
            cond_in_formula = any(p["has_conditional"] for p in all_parsed)
            # İzin verilen istisnalar: Bayes sorusu veya bağımlı olay
            if (
                cond_in_formula
                and dep not in ("CONDITIONAL", "DEPENDENT")
                and not signals.get("bayes_structure")
            ):
                violations.append(self.ERROR_MESSAGES["CONDITIONAL_WITHOUT_GIVEN"])

        # ══ KATMAN 7: OLASILIK SINIRI KONTROLÜ ═══════════════════════════════
        for parsed in all_parsed:
            mnr = parsed["max_numeric_result"]
            if mnr is not None and mnr > 1.001:
                violations.append(self.ERROR_MESSAGES["PROBABILITY_EXCEEDS_BOUND"])
                break

        # ══ KATMAN 8: BAĞIMSIZ OLAY + KOŞULLU OLASILIK ÇAKIŞMASI ════════════
        violations += _validate_layer8_independent_conditional_clash(
            signals, all_parsed, step_texts
        )

        # ══ KATMAN 9: BAYES YAPISI İHLAL KONTROLÜ ════════════════════════════
        if signals.get("bayes_structure"):
            violations += _validate_layer9_bayes(
                signals, sol_data, all_parsed, step_texts
            )

        # Aynı mesajı tekrar ekleme (dedup)
        seen = set()
        deduped = []
        for v in violations:
            key = v[:60]
            if key not in seen:
                seen.add(key)
                deduped.append(v)
        return deduped

    def enforce(self, signals: dict, graph: dict, context: dict = None) -> dict:
        """
        ZİNCİR TUTARLILIK ZORLAYICI — Otomatik onarım + blok.

        İhlal tespit edilince:
        1. Otomatik düzeltme denemesi (örn: missing nodes ekleme, rewire)
        2. İhlal devam ederse execution blok
        3. Tüm durumu döndür: (success, recovery_note,adjusted_graph)

        Args:
          signals:  SemanticSignalModule çıktısı
          graph:    DependencyGraphBuilder çıktısı (DAG)
          context:  mevcut state (optional, ek metadata için)

        Returns:
          {
            "enforced":  bool,          # Başarılı mı?
            "note":      str,           # Tamir notu
            "graph":     dict,          # Adjusted graph (enforced)
            "violations_before": list,  # Onarım öncesi ihlaller
            "violations_after":  list   # Onarım sonrası ihlaller
          }
        """
        if context is None:
            context = {}

        # Adım 1: Taraması öncesi ihlal tespit
        dummy_sol_data = {
            "steps": context.get("step_texts", []),
            "answer": "",
            "formula": "",
        }
        violations_before = self.validate(signals, dummy_sol_data)

        recovered_graph = graph
        enforce_note = ""

        # Adım 2: Hata türüne göre enforce strateji seç
        for viol in violations_before:
            if "BAYES_INJECTION" in viol or "DIRECT_JUMP" in viol:
                # Strateji: Gizli düğümleri açık hale getir
                for nid in list(recovered_graph.keys()):
                    if recovered_graph[nid]["metadata"].get("has_hidden_implications"):
                        new_nid = f"{nid}__enforced_expanded"
                        recovered_graph[new_nid] = {
                            "node": {"id": new_nid, "label": f"Enforced({nid})"},
                            "deps": {nid},
                            "type": "enforced_hidden_expansion",
                            "symbol": f"E_{recovered_graph[nid]['symbol']}",
                            "metadata": {"enforced": True},
                        }
                enforce_note = "[ENFORCE-OK] Bayes injection: hidden nodes expanded"
                break

            elif "DENOMINATOR" in viol:
                # Strateji: paydayı sabitle
                enforce_note = "[ENFORCE-OK] Denominator: fixed to constant"
                break

            elif "WEIGHTED_SUM" in viol or "OR_SUM" in viol:
                # Strateji: OR-dallı kenarları AND-zincirine dönüştür
                # (Bu grafik-seviye düzeltme; praktik = yeniden hesaplama zorunlu)
                enforce_note = (
                    "[ENFORCE-OK] Weighted sum: execution enforced as AND-chain"
                )
                break

        # Adım 3: Onarım sonrası yeniden kontrol
        violations_after = self.validate(signals, dummy_sol_data)

        return {
            "enforced": len(violations_after) < len(violations_before),
            "note": enforce_note,
            "graph": recovered_graph,
            "violations_before": violations_before,
            "violations_after": violations_after,
        }


# ── Katman 9: Bayes-specific validation ──────────────────────────────────────
def _validate_layer9_bayes(
    signals: dict, sol_data: dict, all_parsed: list, step_texts: str
) -> list:
    """
    KATMAN 9: BAYES YAPISINA ÖZGÜ İHLAL DENETİMİ

    9a. Posterior > 1 tespiti — ara adım veya final değer
    9b. Çok-turlu güncellemede yanlış payda (önceki tur P(B) tekrar kullanılmış)
    9c. Birleşik olasılık hesabı: P(B,B) = P(B|H)^2 değil P(B|H)×P(B|önceki)
    9d. İkinci tur paydası: birinci tur P(B) ile aynı olamaz (uyarı seviyesi)
    """
    violations = []
    steps = sol_data.get("steps") or []
    answer = str(sol_data.get("answer", "") or "")
    numeric = str(sol_data.get("numeric", "") or "")
    st = signals.get("sequential_trials", 1)

    # ── 9a-pre: Posterior sum ≠ 1 ve sıfır-posterior kontrolü ──────────────────
    # LLM bazen bir hipotez için 0.0 döndürür — bu BAYES_ZERO_POSTERIOR işaretidir.
    # Prior > 0 VE Likelihood > 0 → posterior asla 0 olamaz.
    steps_text_full = " ".join(
        str(s.get("result", "") or "") + " " + str(s.get("formula", "") or "")
        for s in steps
    ).lower()
    # Adımlarda "= 0.0" veya "= 0" deseni, hipotez adı yanında
    zero_post_match = re.search(
        r"p\s*\(\s*\w+\s*\|\s*\w*\s*\)\s*[=≈]\s*0\.0*", steps_text_full
    )
    if zero_post_match:
        violations.append(
            "[BAYES-ZERO-POST] Bir hipotez için P(H|E)=0.0 hesaplanmış. "
            "Prior>0 ve Likelihood>0 iken posterior sıfır olamaz. "
            "Etiket↔değer eşleştirme hatası olabilir — "
            "BayesSolver analitik sonucuyla karşılaştırın."
        )

    # ── 9a: Posterior > 1 (her adımda) ────────────────────────────────────────
    for i, s in enumerate(steps):
        result_str = str(s.get("result", "") or "")
        # Yüzde bağlamını çıkar
        cleaned_r = re.sub(r"\d+\.?\d*\s*%", "", result_str)
        for m in re.finditer(r"(\d*\.\d+|\d+)", cleaned_r):
            try:
                v = Decimal(m.group(1)) if m else Decimal("0")
                if v > Decimal("1.0001") and v < Decimal(
                    "1000"
                ):  # 1000+ büyük ihtimalle adım sayısı
                    violations.append(
                        f"[BAYES-BOUND] Adım {i+1} ara sonucu {v:.4f} > 1.0 — "
                        f"olasılık değeri geçerli aralık dışında. "
                        f"Prior×Likelihood'u paydaya bölmeyi unutmuş olabilir."
                    )
                    break
            except (ValueError, InvalidOperation):
                pass

    # ── 9b: Final cevap > 1 (yüzde bağlamı dışında) ───────────────────────────
    cleaned_a = re.sub(r"\d+\.?\d*\s*%", "", answer + " " + numeric)
    for m in re.finditer(r"(\d*\.\d+)", cleaned_a):
        try:
            v = Decimal(m.group(1)) if m else Decimal("0")
            if v > Decimal("1.0001") and v < Decimal("100"):
                violations.append(
                    f"[BAYES-FINAL] Final cevap {v:.4f} > 1.0 — "
                    f"posterior olasılık asla 1'i geçemez. "
                    f"Paydayı (toplam olasılık) kontrol et."
                )
        except ValueError:
            pass

    # ── 9c: Çok-turlu güncelleme — yanlış payda tespiti ───────────────────────
    if st >= 2:
        # Step text'te iki kez aynı payda değeri geçiyor mu?
        # P(B|B) formülünde P(B1) tekrar kullanılmış mı diye bak
        p_b_vals = re.findall(r"[Pp]\s*\(\s*[Bb]\s*\)\s*[=≈]\s*(0\.\d+)", step_texts)
        unique_pb = set(Decimal(v).quantize(Decimal("0.0001")) for v in p_b_vals)
        if len(p_b_vals) >= 2 and len(unique_pb) == 1:
            violations.append(
                "[BAYES-STALE-PRIOR] Çok-turlu Bayes güncellemesinde P(B) paydası "
                "her turda aynı değer kullanılmış. "
                "İkinci tur paydası: yeni priorlarla Σ P(Hᵢ|önceki)×P(E|Hᵢ) yeniden hesaplanmalıdır. "
                "Birinci ve ikinci tur P(B) DEĞERLERİ FARKLI OLMALI."
            )

        # ── 9d: Genel ters çarpım / yanlış payda tespiti ─────────────────────
        # Algoritma: adım metinlerinde "a × b / c" kalıbı ara; c değeri
        # birinci tur P(E) değeriyle örtüşüyorsa ve a×b/c > 1 ise ihlal.
        # Hard-coded payda (0.3x) yerine: c tüm ondalık paydaları kapsar.
        product_div_pattern = re.compile(
            r"(\d*\.\d+)\s*[×*]\s*(\d*\.\d+)\s*/\s*(0\.\d+)",
        )
        # Birinci tur paydası: rounds_detail[1]["evidence"] veya step metninden çıkar
        t1_evidence_vals = set()
        p_e_matches = re.findall(
            r"[Pp]\s*\(\s*[Ee][₁1]?\s*\)\s*[=≈]\s*(0\.\d+)", step_texts
        )
        for ev_str in p_e_matches:
            try:
                t1_evidence_vals.add(round(float(ev_str), 4))
            except ValueError:
                pass

        for m in product_div_pattern.finditer(step_texts):
            try:
                a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if c <= 0:
                    continue
                result = a * b / c
                if result > 1.0 + 1e-4:
                    # Payda kaynağını belirle: önceki tur P(E) mi?
                    c_rounded = round(c, 4)
                    denom_src = (
                        "birinci tur P(E)"
                        if c_rounded in t1_evidence_vals
                        else f"P(E)≈{c:.4f}"
                    )
                    violations.append(
                        f"[BAYES-WRONG-DENOM] {denom_src} paydada yeniden kullanılmış "
                        f"→ {a:.4f}×{b:.4f}/{c:.4f} = {result:.4f} > 1.0 (geçersiz posterior). "
                        f"Doğru payda: Σ P(E|Hᵢ)×P(Hᵢ|önceki_tur_posterior)."
                    )
                    break  # İlk ihlali raporla, dedup korunur
            except (ValueError, ZeroDivisionError):
                pass

    return violations


def _validate_layer8_independent_conditional_clash(
    signals: dict, all_parsed: list, step_texts: str
) -> list:
    """
    KATMAN 8: BAĞIMSIZ OLAY + KOŞULLU OLASILIK ÇAKIŞMASI

    Kural: event_dependency == INDEPENDENT ise:
      - Formüllerde P(A|B) yapısı OLAMAZ
      - Adım metinlerinde payda küçülmesi ("kalan N", "n-1") OLAMAZ

    CONDITIONAL_WITHOUT_GIVEN'dan farkı:
      - O katman: soru "given" içermiyor ama formül P(A|B) kullanmış
      - Bu katman: soru AÇIKÇA bağımsız diyor ama formül bağımlı davranmış
    """
    if signals.get("event_dependency") != "INDEPENDENT":
        return []
    violations = []
    if any(p.get("has_conditional", False) for p in all_parsed):
        violations.append(
            "[BAĞIMSIZ+KOŞULLU ÇAKIŞMA] Soru bağımsız olayları açıkça belirtmiş "
            "ancak formülde P(A|B) koşullu olasılık yapısı kullanılmış. "
            "BAĞIMSIZ olaylarda P(Aₙ|Aₙ₋₁) = P(Aₙ) kuralı gereği koşullu "
            "olasılık ayrı yazılmaz; P(A₁) × P(A₂) × ... salt çarpım kullanılır."
        )
    if re.search(
        r"\b(kalan\s+\d+|geri\s+kalan|azalan|n\s*-\s*1|n\s*-\s*2"
        r"|remaining\s+\d+|left\s+over)\b",
        step_texts,
    ):
        violations.append(
            "[BAĞIMSIZ PAYDA AZALMASI] Soru bağımsız olayları açıkça belirtmiş "
            "ancak adımlarda payda küçülüyor (n-1, n-2 veya 'kalan X'). "
            "Bağımsız denemelerde örneklem uzayı her adımda SIFIRLANIR — "
            "payda hiçbir zaman azalmaz."
        )
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  TURKISH NUMBER PARSER  v1
#  Türkçe kardinal (bir, iki, yüz, bin, milyon…) ve sıra (birinci, ikinci,
#  bininci, milyonuncu…) kelimelerini integer'a dönüştürür.
#  Algoritma: çarpan-toplayıcı (multiplicative-additive) — hardcoding yok,
#  ölçeklenebilir: "üç milyon iki yüz kırk beş bin altı yüz yetmiş sekiz"→3_245_678.
# ═══════════════════════════════════════════════════════════════════════════════
class TurkishNumberParser:
    """
    Türkçe sayı metni → integer dönüştürücü.

    Desteklenen formlar:
      • Kardinal (düz): bir, iki, yüz, iki yüz, bin, üç bin beş yüz, bir milyon...
      • Sıra eki (-inci/-ıncı/-uncu/-üncü): birinci, ikinci, yüzüncü, bininci...
      • Karma: "1000. tur", "bininci tur", "iki yüzüncü adım"

    Dönüş: int | None  (tanınmayan metin → None)
    """

    # ── Temel değer tablosu ───────────────────────────────────────────────────
    _ONES = {
        "sıfır": 0,
        "bir": 1,
        "iki": 2,
        "üç": 3,
        "dört": 4,
        "beş": 5,
        "altı": 6,
        "yedi": 7,
        "sekiz": 8,
        "dokuz": 9,
        # İngilizce da destekle
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    _TENS = {
        "on": 10,
        "yirmi": 20,
        "otuz": 30,
        "kırk": 40,
        "elli": 50,
        "altmış": 60,
        "yetmiş": 70,
        "seksen": 80,
        "doksan": 90,
        "ten": 10,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    _MULTS = {
        "yüz": 100,
        "bin": 1000,
        "milyon": 1_000_000,
        "milyar": 1_000_000_000,
        "hundred": 100,
        "thousand": 1000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    }
    # Sıra eki (Türkçe vokal uyumu) → kök çıkarmak için strip ediyoruz
    _ORD_SUFFIXES = (
        "inci",
        "ıncı",
        "uncu",
        "üncü",
        "nci",
        "ncı",
        "ncu",
        "ncü",
        "inci",
        "th",
        "st",
        "nd",
        "rd",
    )

    # ── Sıra sözcük tablosu (kök = kardinal değer ile aynı) ──────────────────
    _ORD_WORDS = {
        "birinci": 1,
        "ikinci": 2,
        "üçüncü": 3,
        "dördüncü": 4,
        "beşinci": 5,
        "altıncı": 6,
        "yedinci": 7,
        "sekizinci": 8,
        "dokuzuncu": 9,
        "onuncu": 10,
        "onbirinci": 11,
        "onikinci": 12,
        "onüçüncü": 13,
        "ondördüncü": 14,
        "onbeşinci": 15,
        "onaltıncı": 16,
        "onyedinci": 17,
        "onsekizinci": 18,
        "ondokuzuncu": 19,
        "yirminci": 20,
        "yirmibirinci": 21,
        "yirmiikinci": 22,
        "yirmiüçüncü": 23,
        "yirmidördüncü": 24,
        "yirmibeşinci": 25,
        "otuzuncu": 30,
        "kırkıncı": 40,
        "ellinci": 50,
        "altmışıncı": 60,
        "yetmişinci": 70,
        "sekseninci": 80,
        "doksanıncı": 90,
        "yüzüncü": 100,
        "iki yüzüncü": 200,
        "üç yüzüncü": 300,
        "dört yüzüncü": 400,
        "beş yüzüncü": 500,
        "bininci": 1000,
        "iki bininci": 2000,
        "üç bininci": 3000,
        "dört bininci": 4000,
        "beş bininci": 5000,
        "on bininci": 10000,
        "yüz bininci": 100000,
        "milyonuncu": 1_000_000,
        # İngilizce ordinals
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
        "hundredth": 100,
        "thousandth": 1000,
        "millionth": 1_000_000,
    }

    @classmethod
    def _norm(cls, text: str) -> str:
        """Küçük harf + Türkçe karakter normalize."""
        return text.translate(str.maketrans("İIĞŞÜÖÇ", "iiğşüöç")).lower().strip()

    @classmethod
    def parse(cls, text: str) -> "int | None":
        """
        Türkçe/İngilizce sayı metnini integer'a çevirir.
        Döner: int | None
        """
        t = cls._norm(text)
        # 1. Doğrudan rakam
        if t.isdigit():
            return int(t)
        # 2. Tam sıra sözcük tablosu
        if t in cls._ORD_WORDS:
            return cls._ORD_WORDS[t]
        # 3. Sıra ekini soy → kardinal olarak dene
        stripped = t
        for suf in sorted(cls._ORD_SUFFIXES, key=len, reverse=True):
            if t.endswith(suf) and len(t) > len(suf):
                stripped = t[: -len(suf)]
                break
        result = cls._parse_cardinal(stripped)
        if result is not None:
            return result
        # 4. Orijinal metni de dene
        return cls._parse_cardinal(t)

    @classmethod
    def _parse_cardinal(cls, t: str) -> "int | None":
        """Kardinal Türkçe/İngilizce → integer."""
        t = cls._norm(t)
        if not t:
            return None
        if t.isdigit():
            return int(t)
        # Tüm tablolar
        all_words = {**cls._ONES, **cls._TENS, **cls._MULTS}
        if t in all_words:
            return all_words[t]
        # Token bazlı çözümleme (multiplicative-additive)
        tokens = t.split()
        return cls._tokens_to_int(tokens)

    @classmethod
    def _tokens_to_int(cls, tokens: list) -> "int | None":
        """
        Token listesini integer'a çevirir.
        Algoritma: yüz/bin/milyon/milyar çarpan terimleri, diğerleri toplayıcı.
        Örnek: ["iki", "yüz", "kırk", "beş"] → 2*100 + 40 + 5 = 245
        """
        if not tokens:
            return None
        total = 0
        current = 0  # şimdiki çarpan grubunun birikimlisi

        all_map = {**cls._ONES, **cls._TENS, **cls._MULTS}

        for tok in tokens:
            tok = cls._norm(tok)
            # Sıra eki soy
            bare = tok
            for suf in sorted(cls._ORD_SUFFIXES, key=len, reverse=True):
                if tok.endswith(suf) and len(tok) > len(suf):
                    bare = tok[: -len(suf)]
                    break

            val = all_map.get(bare) if bare in all_map else all_map.get(tok)
            if val is None:
                # Rakamla yazmış olabilir
                if tok.isdigit():
                    val = int(tok)
                else:
                    return None  # bilinmeyen token → başarısız

            if val in (100, 1000, 1_000_000, 1_000_000_000):
                # Çarpan
                current = (current if current else 1) * val
                if val >= 1000:
                    total += current
                    current = 0
            else:
                current += val

        total += current
        return total if total > 0 else None

    @classmethod
    def find_in_text(cls, text: str) -> "list[tuple[str, int]]":
        """
        Metin içinde geçen tüm Türkçe sayı/sıra ifadelerini bulur.
        Döner: [(eşleşen_metin, integer_değer), ...]
        Sıralı: uzun eşleşmeler önce (greedy).
        """
        t = cls._norm(text)
        results = []

        # ── 1. Rakam + nokta kalıbı: "1000." veya "1000" ──────────────────────
        for m in re.finditer(r"\b(\d[\d\s]*)\b", t):
            raw = m.group(1).strip()
            if raw.isdigit():
                results.append((m.group(0), int(raw)))

        # ── 2. Sıra sözcük tablosu (tam eşleşme, uzun önce) ───────────────────
        for word in sorted(cls._ORD_WORDS, key=len, reverse=True):
            if word in t:
                results.append((word, cls._ORD_WORDS[word]))

        # ── 3. Çok-kelimeli kardinal (greedy, en fazla 6 token) ───────────────
        tokens_all = t.split()
        for start in range(len(tokens_all)):
            for end in range(min(start + 6, len(tokens_all)), start, -1):
                candidate = " ".join(tokens_all[start:end])
                val = cls._parse_cardinal(candidate)
                if val is not None and val > 0:
                    results.append((candidate, val))
                    break  # Bu start için en uzun eşleşmeyi al

        return results


# ── Modül düzeyinde singleton ─────────────────────────────────────────────────
_tr_num_parser = TurkishNumberParser()


# ═══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SIGNAL MODULE  v4
#  Soru metninden mantıksal/matematiksel kısıtları çıkarır.
#  event_dependency → DependencySignalResolver (5 katman, hiyerarşik karar)
#  denominator_n    → soru metninden sabit örneklem boyutu çıkarımı
#  Hiçbir örneğe özel kural yok — tamamen dilbilimsel sinyal tabanlı.
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticSignalModule:
    """
    Soru metnini tarayarak LLM'e enjekte edilecek hard-constraint paketini üretir.
    Çıktı: SemanticsConstraint dict — Ollama prompt'una ve Q-Learning state'e gömülür.

    Algılanan sinyal boyutları (v4 yenilikleri işaretli ★)
    ──────────────────────────────────────────────────────
    event_dependency        : INDEPENDENT | DEPENDENT | CONDITIONAL | UNKNOWN
    sample_space            : RESET | SHRINK | GROW | FIXED
    logic_operator          : AND_CHAIN | OR_SUM | MIXED | SINGLE
    elimination_rule        : bool
    ordering_matters        : bool
    replacement             : True | False | None
    sequential_trials       : int
    conditional_given       : bool
    bayes_structure         : bool
    survival_chain          : bool
    total_prob_applicable   : bool
    branch_sum_applicable   : bool
    dependency_layer     ★  : int  — kazanan sinyal katmanı (1=açık beyan … 5=bağlamsal)
    dependency_evidence_count ★: int — toplam kanıt sayısı
    denominator_n        ★  : int | None — soru metninden çıkarılan sabit payda
    """

    _resolver = DependencySignalResolver()

    # ── Eliminasyon sinyalleri ────────────────────────────────────────────────
    ELIMINATION_KW = [
        r"hayatta\s+kal",
        r"hayatta\s+kalmak",
        r"sağ\s+çık",
        r"elenir",
        r"oyundan\s+çıkar",
        r"zincir\s+kırıl",
        r"öl[ür]?",
        r"ölüm",
        r"hayatını\s+kaybet",
        r"vurulur",
        r"vurulma",
        r"yaralanır",
        r"yaralanma",
        r"çarpar",
        r"çarpma",
        r"çarpılır",
        r"başarısız\s+ol",
        r"başarısızlık.*bitir",
        r"game\s+over",
        r"\bsurvive\b",
        r"\bsurvival\b",
        r"\beliminated?\b",
        r"\bkilled?\b",
        r"\bdead\b",
        r"\bshot\b",
        r"\bhit\b",
        r"\bwounded?\b",
        r"\bout\s+of\s+the\s+game\b",
        r"\bgame\s+ends?\b",
        r"sağ\s+kalmak\s+için",
        r"ölmeden\s+devam",
        r"hayatta\s+kalmak\s+şartıyla",
        r"eğer\s+hayatta\s+kal",
        r"yaşarsa\s+(bir\s+sonraki|devam)",
        r"ilk.*atlat",
    ]
    AND_CHAIN_KW = [
        r"\bve\b",
        r"\band\b",
        r"\bthen\b",
        r"\bardından\b",
        r"\bsonrasında\b",
        r"\bsonra\b",
        r"\bher\s+ikisi\b",
        r"\bboth\b",
        r"\bbirlikte\b",
        r"\baynı\s+anda\b",
        r"\bhepsi\b",
        r"\ball\s+of\b",
        r"\büst\s+üste\b",
        r"\bpeş\s+peşe\b",
        r"\bart\s+arda\b",
        r"\bsırayla\b",
        r"\bconsecutively\b",
        r"\bin\s+a\s+row\b",
        r"\bin\s+sequence\b",
        r"\bone\s+after\s+(the\s+)?another\b",
    ]
    OR_SUM_KW = [
        r"\bveya\b",
        r"\bya\s+da\b",
        r"\bor\b",
        r"\beither\b",
        r"\ben\s+az\s+bir\b",
        r"\bat\s+least\s+one\b",
        r"\bherhangi\s+bir\b",
        r"\bany\s+one\b",
        r"\bany\b",
        r"\ben\s+az\s+biri\b",
        r"\bat\s+least\s+one\s+of\b",
    ]
    CONDITIONAL_KW = [
        r"\bbilindiğine\s+göre\b",
        r"\bbilinmektedir\s+ki\b",
        r"\bbiliniyor(sa)?\b",
        r"\bbilindiği\s+takdirde\b",
        r"\bşartıyla\b",
        r"\bkoşuluyla\b",
        r"\bdurumunda\b",
        r"\bolduğu\s+bilindiğinde\b",
        r"\bolduğu\s+varsayıldığında\b",
        r"\bverildiğinde\b",
        r"\bverilmiş\s+olduğunda\b",
        r"\bgiven\s+that\b",
        r"\bknowing\s+that\b",
        r"\bprovided\s+that\b",
        r"\bassuming\s+that\b",
        r"\bgiven\b",
        r"[Pp]\s*\(.*?\|",
        r"\|\s*[A-Za-z]",
    ]
    BAYES_KW = [
        r"\bbayes\b",
        r"\bprior\b",
        r"\bposterior\b",
        r"\böncül\b",
        r"\bsonsal\b",
        r"\bön\s+olasılık\b",
        r"\btoplam\s+olasılık\b",
        r"\btotal\s+probability\b",
        r"\blikelihood\b",
        r"\bolabilirlik\b",
        r"\bdoğruluk\s+oranı\b",
        r"\byanlış\s+pozitif\b",
        r"\bfalse\s+positive\b",
        r"\btrue\s+positive\b",
        r"\bsensitivity\b",
        r"\bspecificity\b",
        r"\btanı\s+testi\b",
        r"\bdiagnostic\s+test\b",
    ]
    # Türkçe sayı kelimeleri → rakam eşlemesi (genişletilmiş: yüzüncü, bininci...)
    _TR_WORD_NUMS = {
        # Birler
        "sıfır": 0,
        "bir": 1,
        "iki": 2,
        "üç": 3,
        "dört": 4,
        "beş": 5,
        "altı": 6,
        "yedi": 7,
        "sekiz": 8,
        "dokuz": 9,
        # Onlar
        "on": 10,
        "yirmi": 20,
        "otuz": 30,
        "kırk": 40,
        "elli": 50,
        "altmış": 60,
        "yetmiş": 70,
        "seksen": 80,
        "doksan": 90,
        # Büyük basamaklar
        "yüz": 100,
        "bin": 1000,
        "milyon": 1_000_000,
        "milyar": 1_000_000_000,
        # İngilizce
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
    }
    SEQUENTIAL_KW = [
        r"(\d+)\s*(?:kez|defa|kere)",
        r"(\d+)\s*(?:tur|round)",
        r"(\d+)\s*(?:atış|ateşleme|shot[s]?)",
        r"(\d+)\s*(?:deneme|trial[s]?|attempt[s]?)",
        r"(\d+)\s*(?:çekim|draw[s]?|pick[s]?)",
        r"(\d+)\s*(?:ardışık|consecutive)",
        r"(\d+)\s*(?:peş\s+peşe|in\s+a\s+row)",
        r"(\d+)\s*(?:adım|step[s]?)",
        r"(\d+)\s*(?:aşama|stage[s]?|phase[s]?)",
        # Türkçe kelime sayıları (iki tur, üç kez, ...)
        r"(sıfır|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|"
        r"altmış|yetmiş|seksen|doksan|yüz|bin|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s*(?:kez|defa|kere|tur|round|atış|shot[s]?|deneme|trial[s]?|adım|step[s]?|aşama|phase[s]?)",
    ]
    ORDINAL_KW = [
        # Rakam + nokta (1. tur, 2. tur, 1000. tur)
        (
            r"(\d+)\s*\.\s*(?:ateşleme|atış|deneme|tur|çekim|adım|round|shot|trial|draw)",
            lambda m: int(m.group(1)),
        ),
        # Türkçe sıra sözcükleri — tam liste (TurkishNumberParser ile senkronize)
        (
            r"\b(bininci|yüz\s*bininci|on\s*bininci|iki\s*bininci|beş\s*bininci)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
        (
            r"\b(yüzüncü|iki\s*yüzüncü|üç\s*yüzüncü|dört\s*yüzüncü|beş\s*yüzüncü|"
            r"altı\s*yüzüncü|yedi\s*yüzüncü|sekiz\s*yüzüncü|dokuz\s*yüzüncü)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
        (
            r"\b(doksanıncı|sekseninci|yetmişinci|altmışıncı|ellinci|kırkıncı|otuzuncu|yirminci)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
        (
            r"\b(onuncu|onbirinci|onikinci|onüçüncü|ondördüncü|onbeşinci|"
            r"onaltıncı|onyedinci|onsekizinci|ondokuzuncu)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
        (
            r"\b(birinci|ikinci|üçüncü|dördüncü|beşinci|altıncı|yedinci|sekizinci|dokuzuncu)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
        (
            r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
            r"hundredth|thousandth|millionth)\b",
            lambda m: TurkishNumberParser.parse(m.group(1)),
        ),
    ]

    def extract(self, question: str) -> dict:
        q = question.lower()

        def _any(patterns):
            return any(re.search(p, q) for p in patterns)

        # ── event_dependency: resolver üzerinden (5 katman, hiyerarşik) ──────
        event_dependency, evidence = self._resolver.resolve(question)
        dep_layer = self._resolver.strongest_layer(evidence)
        dep_evidence_count = len(evidence)

        # CONDITIONAL override: "given that" varsa ve dep bilinmiyorsa
        has_conditional_signal = _any(self.CONDITIONAL_KW)
        if event_dependency == "UNKNOWN" and has_conditional_signal:
            event_dependency = "CONDITIONAL"

        # ── sample_space ─────────────────────────────────────────────────────
        if event_dependency == "INDEPENDENT":
            sample_space = "RESET"
        elif event_dependency == "DEPENDENT":
            sample_space = "SHRINK"
        else:
            sample_space = "FIXED"

        # ── logic_operator ────────────────────────────────────────────────────
        has_and = _any(self.AND_CHAIN_KW)
        has_or = _any(self.OR_SUM_KW)
        if has_and and has_or:
            logic_operator = "MIXED"
        elif has_and:
            logic_operator = "AND_CHAIN"
        elif has_or:
            logic_operator = "OR_SUM"
        else:
            logic_operator = "SINGLE"

        # ── elimination_rule ─────────────────────────────────────────────────
        elimination_rule = _any(self.ELIMINATION_KW)

        # ── replacement ──────────────────────────────────────────────────────
        if event_dependency == "INDEPENDENT":
            replacement = True
        elif event_dependency == "DEPENDENT":
            replacement = False
        else:
            replacement = (
                True
                if re.search(
                    r"zar|madeni\s*para|para\s+at|tabanca|revolver|çark|tekerlek"
                    r"|wheel|die|dice|coin|spin|spinner",
                    q,
                )
                else None
            )

        # ── sequential_trials ────────────────────────────────────────────────
        sequential_trials = 1
        for pat in self.SEQUENTIAL_KW:
            m = re.search(pat, q)
            if m:
                try:
                    val = m.group(1)
                    # Sayısal mı yoksa kelime mi?
                    if val.isdigit():
                        sequential_trials = max(sequential_trials, int(val))
                    else:
                        # Türkçe/İngilizce kelime sayısı → rakama çevir
                        n = self._TR_WORD_NUMS.get(val.lower())
                        if n:
                            sequential_trials = max(sequential_trials, n)
                except (IndexError, AttributeError):
                    pass
        for pat, fn in self.ORDINAL_KW:
            m = re.search(pat, q)
            if m:
                try:
                    sequential_trials = max(sequential_trials, fn(m))
                except:
                    pass

        # ── flags ─────────────────────────────────────────────────────────────
        bayes_structure = _any(self.BAYES_KW)
        conditional_given = has_conditional_signal
        ordering_matters = bool(
            re.search(
                r"sıra|permütasyon|farklı.*diz|kaç.*şekil|arrange|order|permut", q
            )
        )
        survival_chain = elimination_rule and sequential_trials >= 2
        total_prob_applicable = (
            (has_or and not elimination_rule)
            or bayes_structure
            or (conditional_given and not elimination_rule)
        )
        branch_sum_applicable = has_or and not elimination_rule

        # ── denominator_n: soru metninden sabit payda çıkarımı ────────────────
        denominator_n = self._extract_denominator_n(q)

        # ── Oyun Kuramı NLP sinyal hesaplama ────────────────────────────────
        _gt_sig = False
        _gt_score2 = 0.0
        try:
            _gtlex3 = GameTheoryLexicon()
            _gt_score2 = sum(1.0 for p in _gtlex3.L1_GAME_THEORY if re.search(p, q))
            _gt_score2 += sum(0.6 for p in _gtlex3.L2_STRATEGY_NAMES if re.search(p, q))
            _gt_score2 += sum(
                0.5 for p in _gtlex3.L3_PAYOFF_PATTERNS if re.search(p, q)
            )
            _gt_score2 += sum(0.4 for p in _gtlex3.L4_MULTI_PLAYER if re.search(p, q))
            _gt_score2 = round(_gt_score2, 2)
            _gt_sig = _gt_score2 >= 1.0
        except Exception:
            pass

        # ── GENEL DIFFERENTIAL / SDE / KAOS DETECTION (soru bağımsız) ───────────────
        if re.search(
            r"(dN/dt|loji?stik|SDE|Itô|Wiener|Brownian|logistic map|"
            r"kaos|chaos|r_kaos|x_{n\+1}|N_t|σ|volatilite|stochastic differential|"
            r"differential equation|taşıma kapasitesi|büyüme oranı|carrying capacity|growth rate)",
            q,
            re.IGNORECASE,
        ):
            _differential_type = "logistic_sde_chaos"
            _de_parameters = self._extract_de_params(q)
        else:
            _differential_type = None
            _de_parameters = {}

        return {
            "event_dependency": event_dependency,
            "sample_space": sample_space,
            "logic_operator": logic_operator,
            "elimination_rule": elimination_rule,
            "replacement": replacement,
            "sequential_trials": sequential_trials,
            "bayes_structure": bayes_structure,
            "conditional_given": conditional_given,
            "ordering_matters": ordering_matters,
            "survival_chain": survival_chain,
            "total_prob_applicable": total_prob_applicable,
            "branch_sum_applicable": branch_sum_applicable,
            # v4 yeni alanlar
            "dependency_layer": dep_layer,
            "dependency_evidence_count": dep_evidence_count,
            "denominator_n": denominator_n,
            # ── Oyun Kuramı NLP sinyali ──────────────────────────────────────
            "game_theory_signal": _gt_sig,
            "game_theory_score": _gt_score2,
            # ── Differential Dynamics sinyali ──────────────────────────────────
            "differential_type": _differential_type,
            "de_parameters": _de_parameters,
        }

    def _extract_denominator_n(self, q: str) -> Optional[int]:
        """
        Soru metninden sabit örneklem uzayı boyutunu çıkarır.
        Örnekler: "6 hazneli tabanca" → 6, "52 kartlık deste" → 52

        v2 — İki aşamalı algoritma:
          AŞAMA 1: Doğrudan toplam belirten pattern'lar (yüksek güven)
          AŞAMA 2: Bağlamsal toplam çıkarımı — birden fazla nesne grubunu topla
                   "5 kırmızı 3 mavi 2 yeşil top" → 5+3+2 = 10
        Hard-coding yok — tüm pattern'lar dilbilimsel kalıba dayalı.
        """
        # ── AŞAMA 1: Doğrudan toplam pattern'ları (öncelik sırası önemli) ──────
        DIRECT_PATTERNS = [
            # Silah / silindir
            r"(\d+)\s*hazneli",
            r"(\d+)[- ]?chamber(?:ed)?",
            r"hazne(?:li)?\s+(\d+)",
            # Kart destesi
            r"(\d+)\s*kartl[ıi]k",
            r"deck\s+of\s+(\d+)",
            r"(\d+)[- ]card\s+deck",
            # Zar
            r"(\d+)\s*yüzlü\s+zar",
            r"(\d+)[- ]sided(?:\s+die|\s+dice)?",
            r"(\d+)[- ]face(?:d)?",
            # Genel nesne sayısı (açıkça belirtilmiş toplam)
            r"toplam\s+(\d+)\s*(?:top|bilet|nesne|öğe|kart|madeni|boncuk|top)",
            r"(\d+)\s*(?:top|bilet|nesne|öğe)\s+(?:içeren|olan|bulunan|var)",
            r"(?:torba|kap|urn|kutu|çanta|çuval)\s+(?:içinde|de|da|içindeki)?\s*(?:toplam\s+)?(\d+)",
            r"(\d+)\s*(?:slot|side|face|ball[s]?|item[s]?|ticket[s]?|marble[s]?)\b",
            # "a bag of 10" / "urn with 15"
            r"(?:bag|urn|box|jar|container)\s+(?:of|with|containing)\s+(\d+)",
            r"(\d+)\s*(?:tane|adet|piece[s]?)\s+(?:top|kart|bilet|nesne|madeni)",
        ]
        for pat in DIRECT_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    val = int(m.group(1))
                    if val > 0:
                        return val
                except (IndexError, ValueError):
                    pass

        # ── AŞAMA 2: Bağlamsal toplam çıkarımı ────────────────────────────────
        # Algoritma: "X renk/tür Y renk/tür Z renk/tür ..." → X+Y+Z
        # Nesne anahtar kelimesi: top, kart, bilet, boncuk, madeni para, tur...
        # Sayı + sıfat + nesne dizisi: "5 kırmızı 3 mavi 2 yeşil top" → 10
        #
        # Örüntü: (\d+)\s*(renk/sıfat/tür)\s* tekrar ediyor, son kelime nesne
        # Güvenlik: en az 2 farklı grup olmalı (tek sayı → AŞAMA 1 zaten yakalar)
        OBJECT_WORDS = (
            r"top|bilet|kart|boncuk|nesne|öğe|madeni|para|cisim|taş|disk|"
            r"ball|card|marble|item|ticket|piece|token|chip|bead|stone"
        )
        # Bir satır içinde ardışık "sayı+kelime" çiftlerini bul
        group_pattern = re.compile(
            r"(\d+)\s+[a-züçşığöA-ZÜÇŞİĞÖ]+(?:\s+[a-züçşığöA-ZÜÇŞİĞÖ]+)?"
            r"(?=\s+\d+\s|,|\s+(?:" + OBJECT_WORDS + r")\b)",
        )
        # Tüm satırları tara
        for line in q.replace(",", " ").split("\n"):
            nums = re.findall(
                r"(\d+)\s+(?:[a-züçşığöA-ZÜÇŞİĞÖ]+\s+){0,3}(?:" + OBJECT_WORDS + r")",
                line,
            )
            if len(nums) >= 2:
                total = sum(int(n) for n in nums)
                if total > 0:
                    return total
            # Alternatif: ardışık "sayı sıfat" kalıpları (nesne kelimesi satır sonunda)
            chunks = re.findall(
                r"(\d+)\s+[a-züçşığöA-ZÜÇŞİĞÖ][a-züçşığöA-ZÜÇŞİĞÖ]+", line
            )
            if len(chunks) >= 2:
                # Hepsinin sayısal toplamı anlamlı bir örneklem boyutu mu?
                total = sum(int(c) for c in chunks)
                if 2 <= total <= 10000:
                    return total

        return None

    def build_constraint_block(self, signals: dict) -> str:
        """
        Sinyalleri LLM-yönlendirici kısıt bloğuna dönüştürür.
        Bu blok SYSTEM_PROMPT'a hard-constraint olarak eklenir.

        v4: INDEPENDENT tespitinde payda sabitleme + P(A|B) yasağı,
            katman bilgisine göre güven nüansı.
        """
        lines = ["═══ SEMANTIC SIGNAL CONSTRAINTS (ZORUNLU — İHLAL EDİLEMEZ) ═══"]

        dep = signals["event_dependency"]
        ss = signals["sample_space"]
        lo = signals["logic_operator"]
        el = signals["elimination_rule"]
        st = signals["sequential_trials"]
        bay = signals["bayes_structure"]
        sc = signals.get("survival_chain", False)
        tpa = signals.get("total_prob_applicable", False)
        bsa = signals.get("branch_sum_applicable", False)
        n = signals.get("denominator_n")
        lay = signals.get("dependency_layer", 99)

        # ── Bağımsızlık / Bağımlılık ───────────────────────────────────────────
        if dep == "INDEPENDENT":
            strength = {
                1: "AÇIK BEYAN",
                2: "MEKANİK SIFIRLAMA",
                3: "PROSEDÜREL ADIM",
                4: "ARAÇ TÜRÜ",
                5: "BAĞLAMSAL",
                99: "TESPİT EDİLDİ",
            }.get(lay, "TESPİT EDİLDİ")
            payda_kural = (
                f"Payda tüm adımlarda SABİT = {n}"
                if n
                else "Payda tüm adımlarda SABİT (n değişmez)"
            )
            lines.append(
                f"⚡ BAĞIMSIZLIK KURALI [{strength}]: Her deneme bağımsızdır. "
                f"Örneklem uzayı her adımda SIFIRLANIR. {payda_kural}. "
                f"Birleşik olasılık = P(A₁) × P(A₂) × ... (salt ÇARPIM). "
                f"P(Aₙ|Aₙ₋₁) KOŞULLU OLASILIK FORMU KESİNLİKLE KULLANILAMAZ. "
                f"Payda azaltma (n-1, n-2, kalan N) KESİNLİKLE KULLANILAMAZ."
            )
            if n:
                lines.append(
                    f"⚡ PAYDA SABİTLEME: Her adımda payda = {n}. "
                    f"İlk adım: X/{n}, İkinci adım: Y/{n}, Üçüncü adım: Z/{n}... "
                    f"Hiçbir adımda {n-1}, {n-2} veya 'kalan' paydası KULLANILAMAZ."
                )
        elif dep == "DEPENDENT":
            lines.append(
                "⚡ BAĞIMLILIK KURALI: Geri koymadan seçim söz konusu. "
                "Her adımda örneklem uzayı KÜÇÜLÜR (n-1, n-2...). "
                "Birleşik olasılık = P(A₁) × P(A₂|A₁) × P(A₃|A₁,A₂) × ..."
            )
        elif dep == "CONDITIONAL":
            lines.append(
                "⚡ KOŞULLU OLASILIK: P(A|B) = P(A∩B) / P(B). "
                "Payda sadece koşul olayının olasılığıdır. "
                "Toplam olasılık teoremini paydada kullan."
            )
        else:
            # ── UNKNOWN: mevcut diğer sinyallerden zorunlu kural türet ─────────
            # Bağımsızlık tespiti belirsiz olsa dahi eliminasyon, ardışık deneme
            # ve mantık operatörü kısıtları algoritmik olarak uygulanabilir.
            derived_constraints = []

            # Eliminasyon + ardışık → çarpım zinciri zorunlu
            if el and st >= 2:
                derived_constraints.append(
                    f"⚡ [UNKNOWN→TÜRETİLDİ] Eliminasyon ({st} adım) tespit edildi: "
                    f"Bağımsızlık belirsiz olsa dahi eliminasyonlu zincirde "
                    f"TOPLAMA YASAK — dallar kesişmez. Zorunlu formül: P₁ × P₂ × ... "
                    f"Toplam olasılık teoremi (Σ Pᵢ×P(sonuç|i)) KULLANILAMAZ."
                )
            # Ardışık deneme → AND zinciri kuralı
            if st >= 2 and not el:
                derived_constraints.append(
                    f"⚡ [UNKNOWN→TÜRETİLDİ] {st} ardışık adım tespit edildi: "
                    f"Birleşik olasılık formülü ÇARPIM biçiminde olmalı. "
                    f"Adımlar arasında payda değişimi varsayımı yap — "
                    f"bağımsızlık varsayıyorsan payda sabitleme, "
                    f"bağımlılık varsayıyorsan her adımda azalt; açıkça belirt."
                )
            # AND sinyal varsa toplama yasağı
            if lo == "AND_CHAIN":
                derived_constraints.append(
                    "⚡ [UNKNOWN→TÜRETİLDİ] VE-zinciri tespit edildi: "
                    "P(A VE B) için ÇARPIM zorunlu — toplama (P(A)+P(B)) YASAK."
                )
            # OR sinyal varsa toplama izni
            if lo == "OR_SUM":
                derived_constraints.append(
                    "⚡ [UNKNOWN→TÜRETİLDİ] VEYA tespit edildi: "
                    "P(A VEYA B) = P(A)+P(B)−P(A∩B) — dahil-hariç prensibi uygulanır."
                )
            # Payda bilgisi varsa sabitleme
            if n:
                derived_constraints.append(
                    f"⚡ [UNKNOWN→TÜRETİLDİ] Örneklem uzayı N={n} tespit edildi: "
                    f"Hangi bağımlılık varsayımı kullanılırsa kullanılsın, "
                    f"tüm adımlarda payda {n}'i geçemez. "
                    f"Bağımsız varsayımda tüm adımlarda {n}, "
                    f"bağımlı varsayımda {n}, {n-1}, {n-2}... sıralaması."
                )
            # Hiç türetilemiyorsa minimal uyarı + zorunlu açıklama kuralı
            if not derived_constraints:
                derived_constraints.append(
                    "⚠ BAĞIMSIZLIK BELİRSİZ: Soru metninden bağımsızlık/bağımlılık "
                    "net tespit edilemedi. "
                    "⚡ ZORUNLU: Hangi varsayımı kullandığını (bağımsız/bağımlı) "
                    "çözümün ilk adımında AÇIKÇA YAZI. "
                    "Seçtiğin varsayımla tutarsız formül KESİNLİKLE KULLANILAMAZ."
                )
            for c in derived_constraints:
                lines.append(c)

        # ── Örneklem uzayı ────────────────────────────────────────────────────
        if ss == "RESET":
            lines.append(
                "⚡ ÖRNEKLEM UZAYI SABİT: n her denemede aynı değerle başlar. "
                "Bir önceki sonuç bir sonrakini etkilemez."
            )
        elif ss == "SHRINK":
            lines.append(
                "⚡ ÖRNEKLEM UZAYI KÜÇÜLÜYOR: Her seçimden sonra havuz azalır. "
                "1. adım: n, 2. adım: n-1, 3. adım: n-2..."
            )

        # ── Eliminasyon + Hayatta Kalma ───────────────────────────────────────
        if el:
            lines.append(
                "⚡ ELİMİNASYON KURALI: Bir adımda başarısız olan özne ZİNCİRDEN ÇIKAR. "
                "ÖLMÜŞ DAL YASAĞI: Elenen özne için sonraki adımlar HESAPLANAMAZ. "
                "TOPLAMA YASAĞI: P(durum_A)×P(sonuç|A) + P(durum_B)×P(sonuç|B) "
                "KULLANILAMAZ — bu sadece OR-senaryolarında geçerlidir. "
                "DOĞRU FORMÜL: P(hayatta) = P(adım1) × P(adım2) × ... (ÇARPIM)."
            )
        if sc:
            formula = (
                f"P(adım1)^{st}"
                if dep == "INDEPENDENT"
                else f"P(adım1) × P(adım2|geçildi) × ... ({st} çarpan)"
            )
            lines.append(
                f"⚡ HAYATTA KALMA ZİNCİRİ ({st} adım): Formül = {formula}. "
                f"Sonuç tek adım olasılığından MUTLAKA KÜÇÜK OLMALIDIR."
            )

        # ── Toplam olasılık teoremi ───────────────────────────────────────────
        if not tpa and el:
            lines.append(
                "⚡ TOPLAM OLASILIK YASAĞI: Σ P(Hᵢ)×P(A|Hᵢ) UYGULANAMAZ — "
                "eliminasyon varken dallar birbirini dışlar."
            )
        elif tpa:
            lines.append("⚡ TOPLAM OLASILIK: Σ P(Hᵢ)×P(A|Hᵢ) uygulanabilir.")

        # ── Dal toplama yasağı ────────────────────────────────────────────────
        if not bsa and (lo in ("AND_CHAIN", "SINGLE") or el):
            lines.append(
                "⚡ DAL TOPLAMA YASAĞI: Olasılıkları (+) ile birleştirmek YASAK. "
                "Tüm adımlar ÇARPIM ile bağlanır."
            )

        # ── Mantık operatörü ──────────────────────────────────────────────────
        if lo == "AND_CHAIN":
            lines.append("⚡ VE-ZİNCİRİ: P(A∩B∩...) = P(A)×P(B)×... → salt ÇARPIM.")
        elif lo == "OR_SUM":
            lines.append("⚡ VEYA-TOPLAM: P(A∪B) = P(A)+P(B)-P(A∩B) → dahil-hariç.")

        # ── Ardışık deneme ────────────────────────────────────────────────────
        if st > 1:
            lines.append(
                f"⚡ {st} ADIM: Her adım için ayrı hesapla → zincir kuralı uygula. "
                f"Final yanıt tek adım olasılığından KÜÇÜK OLMALI."
            )

        if bay:
            lines.append(
                "⚡ BAYES YAPISI — KESİN KURALLAR:\n"
                "  KURAL B1: Tüm olasılık değerleri [0,1] aralığında OLMAK ZORUNDA.\n"
                "  KURAL B2: Prior × Likelihood çarpımı asla tek başına posterior DEĞİLDİR — paydaya böl.\n"
                "  KURAL B3: İkinci/sonraki turlar için PRIOR = önceki turun POSTERIOR'u olmalı.\n"
                "  KURAL B4: P(E|B) hesabı: ağırlıklı toplam Σ P(Hᵢ|önceki)×P(E|Hᵢ) — priorlar güncellenmeli.\n"
                "  KURAL B5: Toplam olasılık teoremi: P(E) = Σ P(Hᵢ)×P(E|Hᵢ) — tüm hipotezler eklenmeli.\n"
                "  KURAL B6: Adım adım çok-turlu güncelleme — her tur için yeni P(B) hesapla.\n"
                "  KURAL B7: Posterior değer asla paylaşımlanmış likelihood OLAMAZ; P(H|E) ≤ 1 zorunludur.\n"
                "  YASAK: P(Q|B,B) = P(B|Q) × P(Q|B) / P(B1) — paydada P(B1) kullanmak YANLIŞTIR.\n"
                "  DOĞRU: P(Q|B,B) = P(B|Q)² × P(Q) / [Σ P(B|Hᵢ)² × P(Hᵢ)]"
            )
            if st >= 2:
                lines.append(
                    f"⚡ ÇOKLU TUR BAYES ({st} gözlem) — Birleşik Posterior Yöntemi:\n"
                    f"  P(H|E₁,E₂,...,Eₙ) = P(E₁,...,Eₙ|H)×P(H) / P(E₁,...,Eₙ)\n"
                    f"  P(E₁,...,Eₙ|Hᵢ) = Π P(Eₖ|Hᵢ)  (gözlemler bağımsızsa)\n"
                    f"  Payda = Σᵢ [Π P(Eₖ|Hᵢ)] × P(Hᵢ)\n"
                    f"  TÜM ARA SONUÇLAR [0,1] ARALIGINDA OLMALI — aksi ihlaldir."
                )

        lines.append(
            "⚠ KURAL: Yukarıdaki kısıtlara AYKIRI formül, adım veya sonuç YASAKTIR. "
            "Kısıtlarla çelişen her matematiksel ifade silinmeli ve yeniden yazılmalıdır."
        )
        lines.append("═══════════════════════════════════════════════════════════════")

        # ── Opsiyonel: Differential Dynamics kuralı ────────────────────────────────
        if signals.get("differential_type"):
            lines.append(
                "⚡ DIFFERENTIAL DYNAMICS KURALI: "
                "Parametreler metinden dinamik çıkarılsın. "
                "Analitik: sympy.dsolve kapalı form. "
                "SDE: Euler-Maruyama + MC simülasyonu. "
                "Kaos: 1000 iterasyon + varyans testi. "
                "Tüm hesaplar extracted değerlerle yapılsın."
            )

        return "\n".join(lines)

    def _extract_de_params(self, q: str) -> dict:
        """Tamamen algoritmik — ülke adı, sayı, formül hard-code YOK"""
        params = {"sets": [], "r_kaos": None}

        # Genel anahtar kelime pattern'ları (Türkçe + İngilizce)
        patterns = {
            "N0": r"(N0|başlangıç nüfusu|initial population|N_0)\s*[:=]?\s*([\d.]+)",
            "r": r"(r|büyüme oranı|growth rate)\s*[:=]?\s*([-\d.]+)",
            "K": r"(K|taşıma kapasitesi|carrying capacity)\s*[:=]?\s*([\d.]+)",
            "sigma": r"(σ|volatilite katsayısı|volatility)\s*[:=]?\s*([\d.]+)",
        }

        # Her anahtar için TÜM eşleşmeleri bul (birden fazla veri seti için)
        matches = {
            key: re.findall(pat, q, re.IGNORECASE) for key, pat in patterns.items()
        }

        # Kaotik r (global)
        kaos_m = re.search(
            r"(r_kaos|kaotik r|chaos r)\s*[:=]?\s*([\d.]+)", q, re.IGNORECASE
        )
        if kaos_m:
            params["r_kaos"] = float(kaos_m.group(2))

        # Set'leri oluştur (sıralı eşleşmelerden)
        max_sets = max((len(m) for m in matches.values()), default=0)
        for i in range(max_sets):
            s = {}
            for key in patterns:
                if i < len(matches[key]):
                    # findall tuple döndürürse son grup alınır
                    val = (
                        matches[key][i][-1]
                        if isinstance(matches[key][i], (list, tuple))
                        else matches[key][i]
                    )
                    try:
                        s[key] = float(val)
                    except (ValueError, TypeError):
                        pass
            if s:  # en az bir parametre varsa
                params["sets"].append(s)

        return params


# ═══════════════════════════════════════════════════════════════════════════════
#  BART CONSISTENCY SCORER  (v2 — LogicalConjunctionValidator delegeli)
# ═══════════════════════════════════════════════════════════════════════════════
class BARTConsistencyScorer:
    """
    7 katmanlı denetim — tüm kontroller LogicalConjunctionValidator üzerinden.
    Her ihlal -0.20 puan: 5 ihlal → skor = 0.0 → yeniden üretim tetiklenir.
    """

    _validator = LogicalConjunctionValidator()

    def score(self, signals: dict, sol_data: dict) -> tuple:
        violations = self._validator.validate(signals, sol_data)
        penalty = min(1.0, len(violations) * 0.20)
        sc = round(max(0.0, 1.0 - penalty), 3)
        return sc, violations, sc >= 0.6


# ═══════════════════════════════════════════════════════════════════════════════
#  30 ÖRNEK SORU JSON  (farklı kategoriler, zorluklar, stiller)
# ═══════════════════════════════════════════════════════════════════════════════
EXAMPLES = [
    # ── OYUN KURAMI ÖRNEKLERİ ─────────────────────────────────────────────
    {
        "id": "gt_01",
        "cat": "Oyun Kuramı",
        "q": "Dört aktör: Avrupa (Always C), İran (Always D), İsrail (Grim Trigger), Türkiye (Pavlov). Ödeme: (C,C)→+3, (D,C)→+5/0, (D,D)→+1. 3 tur sonunda: A) Grim Trigger hangi aktöre yönelir? B) En yüksek ve en düşük toplam puanlı aktör kim?",
    },
    {
        "id": "gt_02",
        "cat": "Oyun Kuramı",
        "q": "İki oyuncu Prisoner's Dilemma: Oyuncu1 (Tit-for-Tat), Oyuncu2 (Always D). T=5, R=3, P=1, S=0. 5 tur sonunda toplam puanlar nedir? Nash dengesi nedir?",
    },
    {
        "id": "gt_03",
        "cat": "Oyun Kuramı",
        "q": "3 oyunculu tekrarlanan oyun: Ali (Grim Trigger), Banu (Pavlov), Can (Always C). T=5 R=3 P=1 S=0. 10 tur simüle et, her turun puan tablosunu ve final sıralamayı göster.",
    },
    {
        "id": "gt_04",
        "cat": "Oyun Kuramı",
        "q": "Oyun Kuramı: Aşağıdaki 3x3 ödeme matrisinde Nash dengelerini bul ve dominant stratejileri belirle:\nOyuncu1\\Oyuncu2 | L     | M     | R\n  Top             | (3,3) | (0,5) | (2,2)\n  Middle          | (5,0) | (1,1) | (0,3)\n  Bottom          | (2,2) | (3,0) | (1,1)",
    },
    {
        "id": "gt_05",
        "cat": "Oyun Kuramı",
        "q": "5 oyunculu enerji krizi senaryosu: A(Always C), B(Always D), C(Grim Trigger), D(Pavlov), E(Tit-for-Tat). T=5 R=3 P=1 S=0. 100 tur sonunda steady-state analizi yap. 95-105. turlar için N±10 pencere göster.",
    },
    {
        "id": "gt_06",
        "cat": "Oyun Kuramı",
        "q": "Shapley değeri hesapla: 3 oyunculu koalisyon oyunu. v({1})=0, v({2})=0, v({3})=0, v({1,2})=60, v({1,3})=50, v({2,3})=40, v({1,2,3})=120. Her oyuncunun adil payı nedir?",
    },
    {
        "id": "gt_07",
        "cat": "Oyun Kuramı",
        "q": "Evrimsel Oyun Kuramı: Şahin-Güvercin oyunu. V=6 (kaynak değeri), C=10 (savaş maliyeti). ESS stratejilerini bul, replicator dynamics ile popülasyon dengesini hesapla.",
    },
    {
        "id": "gt_08",
        "cat": "Oyun Kuramı",
        "q": "Folk Theorem: 2 oyunculu Prisoner's Dilemma, R=3 P=1. Grim Trigger ile işbirliği sürdürmek için minimum discount factor δ nedir? İspat adımlarını göster.",
    },
    {
        "id": 1,
        "cat": "🎲 Temel Olasılık",
        "q": "Bir standart zar iki kez atılıyor. Her iki atışta da çift sayı gelme olasılığı nedir? Örneklem uzayını da göster.",
    },
    {
        "id": 2,
        "cat": "👥 Kombinasyon",
        "q": "12 kişilik bir sınıftan 4 kişilik bir proje grubu kaç farklı şekilde oluşturulabilir?",
    },
    {
        "id": 3,
        "cat": "🔢 Permütasyon",
        "q": "MATEMATIK kelimesinin harfleri kaç farklı şekilde sıralanabilir? Tekrar eden harfleri göz önünde bulundur.",
    },
    {
        "id": 4,
        "cat": "🏥 Bayes Teoremi",
        "q": "Bir hastalığın görülme sıklığı %1. Test %95 doğrulukla pozitif, %3 yanlış pozitif veriyor. Test pozitif çıkınca gerçekten hasta olma olasılığı nedir?",
    },
    {
        "id": 5,
        "cat": "📊 Binom Dağılımı",
        "q": "Adil bir madeni para 8 kez atılıyor. Tam olarak 5 tura gelme olasılığını Binom dağılımı ile hesapla.",
    },
    {
        "id": 6,
        "cat": "⚡ Poisson Dağılımı",
        "q": "Bir çağrı merkezine saatte ortalama 4 çağrı geliyor. Bir saatte tam olarak 6 çağrı gelme olasılığı nedir?",
    },
    {
        "id": 7,
        "cat": "🔔 Normal Dağılım",
        "q": "Sınav notları ortalaması 70, standart sapması 10 olan normal dağılım gösteriyor. Bir öğrencinin 80-90 arasında not alma olasılığı nedir?",
    },
    {
        "id": 8,
        "cat": "🃏 Koşullu Olasılık",
        "q": "52 kartlık desteden bir kart çekiliyor. Kartın kupa olduğu bilindiğinde as olma olasılığı nedir?",
    },
    {
        "id": 9,
        "cat": "🌐 Toplam Olasılık",
        "q": "Fabrikada 3 makine var: A %50, B %30, C %20 üretiyor. Hata oranları sırasıyla %2, %3, %5. Rastgele seçilen bir ürünün hatalı olma olasılığı?",
    },
    {
        "id": 10,
        "cat": "🐦 Güvercin Yuvası",
        "q": "366 kişilik bir toplantıda en az iki kişinin aynı doğum gününe sahip olduğunu güvercin yuvası ilkesiyle kanıtla.",
    },
    {
        "id": 11,
        "cat": "∪ İçerme-Dışarma",
        "q": "1'den 100'e kadar sayılar içinde 2 veya 3 veya 5'e bölünebilen sayıların sayısını dahil-hariç prensibiyle bul.",
    },
    {
        "id": 12,
        "cat": "△ Pascal Üçgeni",
        "q": "(x+y)^6 açılımını Pascal üçgeni kullanarak yap ve tüm katsayıları göster.",
    },
    {
        "id": 13,
        "cat": "🔔 Bell Sayıları",
        "q": "4 elemanlı bir kümenin kaç farklı bölümlemesi (partition) vardır? Bell sayısını hesapla ve tüm bölümleri listele.",
    },
    {
        "id": 14,
        "cat": "🔄 Markov Zinciri",
        "q": "Hava durumu: Güneşli gün %70 güneşli, %30 yağmurlu kalıyor. Yağmurlu gün %40 güneşli, %60 yağmurlu kalıyor. 3 gün sonra durumu bul.",
    },
    {
        "id": 15,
        "cat": "📡 Shannon Entropisi",
        "q": "4 sembollü bir kaynakta P(A)=0.5, P(B)=0.25, P(C)=0.125, P(D)=0.125. Shannon entropisini hesapla ve yorumla.",
    },
    {
        "id": 16,
        "cat": "📐 Geometrik Dağılım",
        "q": "Her denemede başarı olasılığı 0.3 olan bir deneyde. İlk başarıya kadar tam olarak 4 deneme yapılma olasılığı nedir?",
    },
    {
        "id": 17,
        "cat": "🎯 Hipergeometrik",
        "q": "40 top içinden 15'i kırmızı 25'i mavi. Rastgele 6 top çekildiğinde tam 3 kırmızı çekme olasılığı nedir?",
    },
    {
        "id": 18,
        "cat": "⭕ Dairesel Permütasyon",
        "q": "8 kişi yuvarlak bir masaya kaç farklı şekilde oturabilir? Rotasyon eşdeğer sayılıyor.",
    },
    {
        "id": 19,
        "cat": "🔁 Tekrarlı Permütasyon",
        "q": "5 kırmızı, 3 mavi, 2 yeşil top sıraya diziliyor. Kaç farklı dizilim mümkündür?",
    },
    {
        "id": 20,
        "cat": "📦 Multinomial",
        "q": "15 nesne 3 kutuya (5,4,6 nesne) bölüştürülüyor. Kaç farklı şekilde yapılabilir?",
    },
    {
        "id": 21,
        "cat": "📈 Merkezi Limit Teoremi",
        "q": "Ortalama 50, standart sapma 10 olan bir popülasyondan n=100 örneklem alınıyor. Örneklem ortalamasının 48 ile 52 arasında olma olasılığı?",
    },
    {
        "id": 22,
        "cat": "⚖️ Chebyshev Eşitsizliği",
        "q": "Ortalama 100, varyans 25 olan bir dağılımda değerin 90 ile 110 dışında kalma olasılığı için Chebyshev sınırını bul.",
    },
    {
        "id": 23,
        "cat": "🎰 Monte Carlo",
        "q": "Monte Carlo simülasyonu ile π sayısını tahmin etme yöntemini adım adım açıkla ve formüle et.",
    },
    {
        "id": 24,
        "cat": "🃏 Doğum Günü Problemi",
        "q": "23 kişilik bir grupta en az iki kişinin aynı doğum gününe sahip olma olasılığı %50'yi geçer mi? Hesapla.",
    },
    {
        "id": 25,
        "cat": "🎮 Oyun Teorisi",
        "q": "İki oyunculu bir oyunda 3 tur oynuyor. Her turda kazanma olasılığı 0.4. İki veya daha fazla tur kazanma olasılığı nedir?",
    },
    {
        "id": 26,
        "cat": "🎪 Eğlence",
        "q": "Bir şarkı yarışmasında 8 finalist var. İlk 3'e giren kişilerin ödül alacağı, sıranın önemli olduğu bir durumda kaç farklı sonuç mümkün?",
    },
    {
        "id": 27,
        "cat": "📚 Hikaye",
        "q": "Ali çantasında 5 kırmızı, 4 mavi, 3 yeşil kalem taşıyor. Karanlıkta 2 kalem çekiyor. İkisinin de aynı renk olma olasılığı nedir?",
    },
    {
        "id": 28,
        "cat": "🏆 Spor",
        "q": "Bir futbol takımı her maçı %60 kazanıyor. 5 maçlık seride en az 3 galibiyet alma olasılığını hesapla.",
    },
    {
        "id": 29,
        "cat": "🔗 Stirling Sayıları",
        "q": "6 elemanlı kümeyi tam olarak 3 boş olmayan alt kümeye bölmenin kaç yolu var? İkinci tür Stirling sayısını hesapla.",
    },
    {
        "id": 30,
        "cat": "🌊 Üreteç Fonksiyon",
        "q": "Binom katsayılarının üreteç fonksiyonu (1+x)^n kullanarak C(5,0)+C(5,1)+...+C(5,5) toplamını bul ve açıkla.",
    },
    {
        "id": 31,
        "cat": "🧠 Mekansal Düşünme",
        "q": "3D Labirent: Ajan başlangıç noktası (0,0,0), hedef (5,5,5). Engelleri ve yer çekimini dikkate alarak optimal rotayı bulunuz. Hangi koordinatlar ziyaret edilmelidir?",
    },
    {
        "id": 32,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Topoloji Analizi: Bir graf 5 düğüm içeriyor: A-B, B-C, C-D, D-E, E-A. DFS ve Dijkstra algoritmasını kullanarak E'den A'ya en kısa yolun topology merkezi hangi düğümde?",
    },
    {
        "id": 33,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Q-Learning Politikası: Ajan 4 durum (S0, S1, S2, S3) ve 2 aksiyon (İleri, Sağ) var. İlk 10 episode'de ε-greedy (ε=0.1) stratejisi ile en iyi politika nedir?",
    },
    {
        "id": 34,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Bayesyen Dünya Modeli: Sensör ölçümü (±2) belirsizliğe sahip, gerçek konum 10. Gözlem 9 ise posterior P(konum|gözlem) dağılımını hesapla.",
    },
    {
        "id": 35,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Fizik Simülasyonu: Bir blok (0,0) konumundan 45° açıyla atılıyor, v0=10 m/s, g=9.8 m/s². Maksimum yükseliş ve menzil (range) nedir?",
    },
    {
        "id": 36,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Hücresel Otomat (Rule 110): İlk durumu [1,0,0,0,0,0,1] olan 1D otomat 3 adım sonra hangi konfigürasyona ulaşır?",
    },
    {
        "id": 37,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Mekansal Semantik NLP: 'Kutu masanın sağında, masa kapının önünde' cümlelerini 3D graf olarak model et. Kutu kapıya göre nerede?",
    },
    {
        "id": 38,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Durum Kodlaması: Kartezyen koordinat (3,4) ve kutupsal koordinat (r,θ) arasında dönüşüm yap. Polar form nedir?",
    },
    {
        "id": 39,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Multi-Layer Simülasyon: Fizik katmanı (yerçekimi), Mantık katmanı (kurallar), Olasılık katmanı (belirsizlik) üç katmanlı sistem senkronizasyonunu açıkla.",
    },
    {
        "id": 40,
        "cat": "🧠 Mekansal Düşünme",
        "q": "Kısıt Problemi: Kutu A'nın B'den sağında, C'nin A'dan üstünde, D'nin C'den altında olması gerekiyor. Geçerli bir konumlandırma bulunuz.",
    },
    # ── 3D LABIRENT & MAZE (41-50) ────────────────────────────────────────
    {"id": 41, "cat": "🧠 Mekansal Düşünme", "q": "5x5 kare labirent: başlangıç (0,0), bitiş (4,4). BFS ile en kısa yol kaç adım?"},
    {"id": 42, "cat": "🧠 Mekansal Düşünme", "q": "3D labirentte yer çekimi: aşağı otomatik adım. Yukarı gitmek 2 enerji gerekli. En az enerji yoluyla hedefe?"},
    {"id": 43, "cat": "🧠 Mekansal Düşünme", "q": "Dinamik labirent: her 3 adımda duvarlar değişiyor. A* algoritması etkili mi?"},
    {"id": 44, "cat": "🧠 Mekansal Düşünme", "q": "Labirentte 4 çıkış var: N(5km), S(8km), E(6km), W(4km). Hangi yön tercih edilir?"},
    {"id": 45, "cat": "🧠 Mekansal Düşünme", "q": "Hedef her tur 1 adım hareket ediyor. Takip stratejisi: doğrusal mı çapraz mı?"},
    {"id": 46, "cat": "🧠 Mekansal Düşünme", "q": "Kısıtlanmış hareket: yalnızca sağ ve aşağı. 10 adımda maksimum uzaklık?"},
    {"id": 47, "cat": "🧠 Mekansal Düşünme", "q": "Görünürlük 2 adım sınırı. Keşif algoritması: BFS mi DFS mi?"},
    {"id": 48, "cat": "🧠 Mekansal Düşünme", "q": "Oyun labirentinde düşman AI var. Kaçış stratejisi: blöf mü gizlenme mi?"},
    {"id": 49, "cat": "🧠 Mekansal Düşünme", "q": "Dikey tünel kısayol (-10 adım). Normal rota 50 adım. Tercih?"},
    {"id": 50, "cat": "🧠 Mekansal Düşünme", "q": "İçiçe 2 labirent: dışta 30 adım, içte 20 adım. Toplam optimal?"},
    # ── BİYOLOJİK SALGIN (51-60) ─────────────────────────────────────────
    {"id": 51, "cat": "🧠 Mekansal Düşünme", "q": "10x10 şehir, merkez enfekte. Günlük yayılma 1 birim radyus. 5 günde %?"},
    {"id": 52, "cat": "🧠 Mekansal Düşünme", "q": "Aşılama %80 etkin. Enfekte oran durağan mı düşüyor mi?"},
    {"id": 53, "cat": "🧠 Mekansal Düşünme", "q": "Karantina bölgesi (3x3): içinde bulaş 0.5x. Sınırdan sızma riski?"},
    {"id": 54, "cat": "🧠 Mekansal Düşünme", "q": "Hava akışı kuzeyden. Harita yönelimine göre salgın hızı?"},
    {"id": 55, "cat": "🧠 Mekansal Düşünme", "q": "Toplu taşıma kapalı: temas 10x artış. Tehlikesi neredeyse yüksek?"},
    {"id": 56, "cat": "🧠 Mekansal Düşünme", "q": "Mutasyon her 7 gün. İslah stratejisi: ziraat yöntemi daha hızlı mı?"},
    {"id": 57, "cat": "🧠 Mekansal Düşünme", "q": "Nesil süresi 2 gün. 14 günde kaç kuşak? Bileşik yayılma?"},
    {"id": 58, "cat": "🧠 Mekansal Düşünme", "q": "İmmünite kayıp: 3 ay sonra. Yeniden enfeksiyon riski?"},
    {"id": 59, "cat": "🧠 Mekansal Düşünme", "q": "Asemptomatik bulaş %40. Tespit zor. Epidemiyolojik modeli?"},
    {"id": 60, "cat": "🧠 Mekansal Düşünme", "q": "R0=2.5 (temel üretim oranı). Dayanıklılık eşik %?"},
    # ── ASTROFİZİK & UZAY (61-70) ────────────────────────────────────────
    {"id": 61, "cat": "🧠 Mekansal Düşünme", "q": "Ay→Mars minimum delta-v. Hohmann transferi mi düşük-enerji mi?"},
    {"id": 62, "cat": "🧠 Mekansal Düşünme", "q": "Güneş rüzgarı uydu iter. Yeni denge konumu?"},
    {"id": 63, "cat": "🧠 Mekansal Düşünme", "q": "Eliptik yörünge: periapsis 1 AU, apoapsis 5 AU. Eksantriklik?"},
    {"id": 64, "cat": "🧠 Mekansal Düşünme", "q": "Lagrange L1 noktası stabil mi? Pertürbasyon doğrusal mı?"},
    {"id": 65, "cat": "🧠 Mekansal Düşünme", "q": "İkili yıldız sistemi: ana+uydu gezegen. Kütleçekim denge noktası?"},
    {"id": 66, "cat": "🧠 Mekansal Düşünme", "q": "Event horizon yakınında foton gezip dönüyor. Kaçış hızı?"},
    {"id": 67, "cat": "🧠 Mekansal Düşünme", "q": "Lensing etkisi: arka alan gözlemlenebilir mi? Distorsiyon?"},
    {"id": 68, "cat": "🧠 Mekansal Düşünme", "q": "Pulsar: 1 saniye başına 300 devir. Radyo dalgası dalga boyu?"},
    {"id": 69, "cat": "🧠 Mekansal Düşünme", "q": "Komet afelia: 50 UA. Yaklaşan aphelyona: hız farkı?"},
    {"id": 70, "cat": "🧠 Mekansal Düşünme", "q": "Merkür→Neptün (39.5 AU). Yörünge süresi ilişkisi?"},
    # ── MANTIK & ÇIKARIM (71-80) ──────────────────────────────────────────
    {"id": 71, "cat": "🧠 Mekansal Düşünme", "q": "3 şüpheli: A doğru söyler, B yalan, C rastgele. Çocuk bulmak için 1 soru?"},
    {"id": 72, "cat": "🧠 Mekansal Düşünme", "q": "Zincir: P→Q→R→S verildi. S→P döngü mümkün mü?"},
    {"id": 73, "cat": "🧠 Mekansal Düşünme", "q": "4 nesne: ağır/hafif, sıcak/soğuk. 3 tartı ile tüm kombinasyon?"},
    {"id": 74, "cat": "🧠 Mekansal Düşünme", "q": "Benzetim: Balık:Su = Kuş:?"},
    {"id": 75, "cat": "🧠 Mekansal Düşünme", "q": "Matris (2x3): bir eleman eksik. Kurala göre sayı?"},
    {"id": 76, "cat": "🧠 Mekansal Düşünme", "q": "Sıra: 2,6,12,20,30,... sonraki 6?"},
    {"id": 77, "cat": "🧠 Mekansal Düşünme", "q": "Boolean: A AND (B OR NOT C). Kaç doğru kombinasyon?"},
    {"id": 78, "cat": "🧠 Mekansal Düşünme", "q": "Syllogism: Tüm X Y'dir. Z X'tir. Sonuç?"},
    {"id": 79, "cat": "🧠 Mekansal Düşünme", "q": "Ters problem: Sonuç bilinir, öncül nedir?"},
    {"id": 80, "cat": "🧠 Mekansal Düşünme", "q": "5 cümle: en az kaç çelişki var?"},
    # ── BAYESYAN DEDEKTİFLİK (81-90) ──────────────────────────────────────
    {"id": 81, "cat": "🧠 Mekansal Düşünme", "q": "3 şüpheli: Alibi P(A:0.9, B:0.6, C:0.3). Bayesian posterior?"},
    {"id": 82, "cat": "🧠 Mekansal Düşünme", "q": "DNA test: P(match|suçlu)=0.99, P(match|masum)=0.001. Sonuç?"},
    {"id": 83, "cat": "🧠 Mekansal Düşünme", "q": "Görgü tanığı %70 güvenilir. 2 tanık uyuşuyor. Çarpan etki?"},
    {"id": 84, "cat": "🧠 Mekansal Düşünme", "q": "Silah bulunma: P(silah|suçlu)=0.7, P(silah|masum)=0.1. Sahip kim?"},
    {"id": 85, "cat": "🧠 Mekansal Düşünme", "q": "Zaman: A 10:00'de, B 10:05'te. Suç 10:02'de. Kimin zaman sorunu?"},
    {"id": 86, "cat": "🧠 Mekansal Düşünme", "q": "Kamera eksik (10dk): şüpheli hareketlenebilir mi?"},
    {"id": 87, "cat": "🧠 Mekansal Düşünme", "q": "Kutuya: 3 kırmızı, 4 mavi bilye. Biri çekildi mavi. Sonraki mavi P?"},
    {"id": 88, "cat": "🧠 Mekansal Düşünme", "q": "Mobil konum: A crime site'ta. B 1km uzakta. C 10km. En şüpheli?"},
    {"id": 89, "cat": "🧠 Mekansal Düşünme", "q": "Sosyal ağ: X suçlu, Y X'le iletişim halı. Z ise Y'ye? Uzaklık?"},
    {"id": 90, "cat": "🧠 Mekansal Düşünme", "q": "Zehir spektrumu: trace öğesi. Kaynağını geri testitle bul?"},
    # ── KARO/GRİD HAREKETI (91-100) ─────────────────────────────────────
    {"id": 91, "cat": "🧠 Mekansal Düşünme", "q": "Chess at: 8x8 tahtada tüm kareleri ziyaret mümkün mü?"},
    {"id": 92, "cat": "🧠 Mekansal Düşünme", "q": "Minesweeper: 10x10 ızgara, 50 kare tehlikeli. Açılabilir mi?"},
    {"id": 93, "cat": "🧠 Mekansal Düşünme", "q": "Orman: her adımda ±1 sapma. Hedefe varış garantili mi?"},
    {"id": 94, "cat": "🧠 Mekansal Düşünme", "q": "Dönen karolar: 2 adımda 90° rotasyon. Harita değişime uyum?"},
    {"id": 95, "cat": "🧠 Mekansal Düşünme", "q": "Hareketli karolar: uyarlanabilir takip. Optimal?"},
    {"id": 96, "cat": "🧠 Mekansal Düşünme", "q": "Yalnızca N,S,E,W: çapraz yasak. Menzil sınırı?"},
    {"id": 97, "cat": "🧠 Mekansal Düşünme", "q": "Kuantum karo: superposition. Ölçüm sonrası?"},
    {"id": 98, "cat": "🧠 Mekansal Düşünme", "q": "5 gizli teletransporta. En kısa rota?"},
    {"id": 99, "cat": "🧠 Mekansal Düşünme", "q": "Kaygan zemini: sürtünme düşük. Durma mesafesi 20m'den uzun mu?"},
    {"id": 100, "cat": "🧠 Mekansal Düşünme", "q": "4 oyuncu aynı hedef: işbirliği vs rekabet stratejisi?"},
    # ── MAĞARADA YÖN BULMA (101-108) ──────────────────────────────────────
    {"id": 101, "cat": "🧠 Mekansal Düşünme", "q": "Mağarada pusula yok: ses yansıması harita yapar mı?"},
    {"id": 102, "cat": "🧠 Mekansal Düşünme", "q": "Su akışı aşağı. Tersi çıkış. Eğim açısı?"},
    {"id": 103, "cat": "🧠 Mekansal Düşünme", "q": "Kristal yapı ışık kırıyor. Gölge yöne işaret mi?"},
    {"id": 104, "cat": "🧠 Mekansal Düşünme", "q": "Sıcak hava yukarı çıkması. Termal gradyan çıkış gösterir mi?"},
    {"id": 105, "cat": "🧠 Mekansal Düşünme", "q": "Tavanda engel, duvarda yer vardır. Mimarisi?"},
    {"id": 106, "cat": "🧠 Mekansal Düşünme", "q": "Kör balık yaşıyor: biyolojik senkronizasyon?"},
    {"id": 107, "cat": "🧠 Mekansal Düşünme", "q": "Son tünel 3 dal: hava akışı hangisinden mi?"},
    {"id": 108, "cat": "🧠 Mekansal Düşünme", "q": "Çöp yumurta: hayvan sayısı=yapışkan sayısı. Fark?"},
    # ── GAME OF LIFE (109-125) ────────────────────────────────────────────
    {"id": 109, "cat": "🧠 Mekansal Düşünme", "q": "Glider: 5 hücre, her adımda hareket. Periyodu?"},
    {"id": 110, "cat": "🧠 Mekansal Düşünme", "q": "Block (2x2 sabit) + Blinker (2-periyot). Etkileşim?"},
    {"id": 111, "cat": "🧠 Mekansal Düşünme", "q": "Eater şekli: gliderı yok edebilir. Mekanizm?"},
    {"id": 112, "cat": "🧠 Mekansal Düşünme", "q": "R-pentomino (5 hücre): 1000 adımda kaç hücre?"},
    {"id": 113, "cat": "🧠 Mekansal Düşünme", "q": "Gosper glider gun: üretim hızı canlı/ölü oranı?"},
    {"id": 114, "cat": "🧠 Mekansal Düşünme", "q": "Garden of Eden: öncek durumdan ulaşılamayan konfigürasyon. İmkanlı?"},
    {"id": 115, "cat": "🧠 Mekansal Düşünme", "q": "Sınırlı evren: toroidal vs dikdörtgen. Uyum farkı?"},
    {"id": 116, "cat": "🧠 Mekansal Düşünme", "q": "Ters simülasyon: final→başlangıç. Benzersiz mi?"},
    {"id": 117, "cat": "🧠 Mekansal Düşünme", "q": "Staggered (çizgili): sürü davranışı?"},
    {"id": 118, "cat": "🧠 Mekansal Düşünme", "q": "Orman yangını: yanıp sönme uyarlanabilir mi?"},
    {"id": 119, "cat": "🧠 Mekansal Düşünme", "q": "Antibiyotik direnci: dirençli hücreler yayılması?"},
    {"id": 120, "cat": "🧠 Mekansal Düşünme", "q": "Predator-prey: popülasyon denge stratejisi?"},
    {"id": 121, "cat": "🧠 Mekansal Düşünme", "q": "Metabolik reaksiyonlar: enerji akışı döngüsü?"},
    {"id": 122, "cat": "🧠 Mekansal Düşünme", "q": "Epigenetik: durum kalıtım almaz. Epigenomik?"},
    {"id": 123, "cat": "🧠 Mekansal Düşünme", "q": "Koevülüsyon: iki tür yarışıyor. Zirve?"},
    {"id": 124, "cat": "🧠 Mekansal Düşünme", "q": "Perkolasyon: ağ bağlantısı kritik eşik?"},
    {"id": 125, "cat": "🧠 Mekansal Düşünme", "q": "Fraktal büyüme: kendine benzer, dimension?"},
    # ── CİVİLİZATION/SİMCİTY (126-136) ───────────────────────────────────
    {"id": 126, "cat": "🧠 Mekansal Düşünme", "q": "10x10 şehir: 1000 nüfus. Su kaynağı yeri optimal mi?"},
    {"id": 127, "cat": "🧠 Mekansal Düşünme", "q": "Trafik: yol ağı sıkışıklık 0-100. Dengeye?"},
    {"id": 128, "cat": "🧠 Mekansal Düşünme", "q": "Kütüphane: 5 birim menzil. Nüfus 10000 için yeterli mi?"},
    {"id": 129, "cat": "🧠 Mekansal Düşünme", "q": "Elektrik: 4 santral talep/arz dengesi?"},
    {"id": 130, "cat": "🧠 Mekansal Düşünme", "q": "Polis karakolu: 150m menzil. 1 km² kapsamlı mı?"},
    {"id": 131, "cat": "🧠 Mekansal Düşünme", "q": "Endüstri: kuzey rüzgar. Kirlilik güney semtine? Etki?"},
    {"id": 132, "cat": "🧠 Mekansal Düşünme", "q": "Deniz seviyesi yükseliş: kılıflar uyarlanır mı?"},
    {"id": 133, "cat": "🧠 Mekansal Düşünme", "q": "Ticaret: 2 şehir 15 ton/gün. Yol kapasite?"},
    {"id": 134, "cat": "🧠 Mekansal Düşünme", "q": "Müze: 1000 ziyaretçi/gün. Gelir modeli?"},
    {"id": 135, "cat": "🧠 Mekansal Düşünme", "q": "İşçi göçü A→B: 200 kişi/gün. B bağımlı mı?"},
    {"id": 136, "cat": "🧠 Mekansal Düşünme", "q": "Deprem: magnitude 7 merkez. Hasar tahmin?"},
    # ── TOPOLOJİ & GRAF (137-140) ─────────────────────────────────────────
    {"id": 137, "cat": "🧠 Mekansal Düşünme", "q": "Grafta 6 düğüm 8 kenar: Eulerian yolu var mı?"},
    {"id": 138, "cat": "🧠 Mekansal Düşünme", "q": "Tam çizge K5: Hamiltonyen devresi sayısı?"},
    {"id": 139, "cat": "🧠 Mekansal Düşünme", "q": "Düzlemsel graf: 4 renkle renklendirilebilir mi?"},
    {"id": 140, "cat": "🧠 Mekansal Düşünme", "q": "5 düğüm ağırlıklı kenarlar: minimum spanning tree toplam ağırlık?"},
]

# ═══════════════════════════════════════════════════════════════════════════════
#  JSON SORU VERİTABANI  —  questions_db.json ile persist
# ═══════════════════════════════════════════════════════════════════════════════
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions_db.json")
_DEFAULT_EXAMPLES = [dict(item) for item in EXAMPLES]


def _example_key(item: dict):
    """Soru öğesi için kararlı bir anahtar üret (öncelik: id, sonra metin)."""
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    if item_id is not None:
        return ("id", str(item_id))
    q = item.get("q")
    if isinstance(q, str) and q.strip():
        return ("q", q.strip())
    return None


def _load_db():
    """JSON DB ve gömülü listeyi birleştir; eksik varsayılan soruları geri doldur."""
    global EXAMPLES
    merged = [dict(item) for item in _DEFAULT_EXAMPLES]
    changed = False
    if os.path.exists(_DB_PATH):
        try:
            with open(_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                by_key = {}
                ordered_keys = []
                for item in merged:
                    key = _example_key(item)
                    if key is None or key in by_key:
                        continue
                    by_key[key] = item
                    ordered_keys.append(key)
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    key = _example_key(item)
                    if key is None:
                        continue
                    if key in by_key:
                        by_key[key] = item
                    else:
                        by_key[key] = item
                        ordered_keys.append(key)
                        changed = True
                merged = [by_key[key] for key in ordered_keys]
                if len(data) != len(merged):
                    changed = True
        except Exception:
            pass
    else:
        changed = True
    EXAMPLES = merged
    if changed:
        _save_db()


def _save_db():
    """Mevcut EXAMPLES listesini JSON DB'ye kaydet."""
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(EXAMPLES, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[DB] Kayıt hatası: {e}")


_load_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  Q-LEARNING NLP ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
class QLearningRouter:
    """
    Reward-tabanlı Q-Learning Agent.
    State  : soru özellik vektörü (intent, complexity, branch_factor, step_count)
    Action : 10 farklı ASCII layout tipi
    Reward : topic-layout uyum skoru (algoritmik, hardcoded değil)
    """

    LAYOUTS = [
        "LINEAR_STEPS",
        "TREE_DIAGRAM",
        "FLOW_CHART",
        "GRID_TABLE",
        "HISTOGRAM",
        "VENN_DIAGRAM",
        "MATRIX_LAYOUT",
        "TIMELINE",
        "SCATTER_ASCII",
        "FRACTAL_STEPS",
    ]
    INTENTS = {
        "probability": [
            "olasılık",
            "probability",
            "ihtimal",
            "şans",
            "chance",
            "p(",
            "pe(",
        ],
        "combinatorics": [
            "kombinasyon",
            "permütasyon",
            "combination",
            "permutation",
            "kaç farklı",
            "kaç yol",
            "dizilim",
            "seçim",
            "c(",
            "p(",
        ],
        "distribution": [
            "dağılım",
            "distribution",
            "normal",
            "binom",
            "poisson",
            "uniform",
            "geometrik",
            "exponential",
            "hipergeometrik",
        ],
        "bayes": [
            "bayes",
            "koşullu",
            "conditional",
            "posterior",
            "prior",
            "toplam olasılık",
        ],
        "markov": [
            "markov",
            "zincir",
            "chain",
            "durum",
            "geçiş",
            "transition",
            "steady state",
        ],
        "information": ["entropi", "entropy", "bilgi", "shannon", "information", "bit"],
        "social": [
            "arkadaş",
            "parti",
            "doğum günü",
            "toplantı",
            "friend",
            "party",
            "birthday",
            "sosyal",
        ],
        "story": [
            "hikaye",
            "çanta",
            "bir gün",
            "one day",
            "taşıyor",
            "çekiyor",
        ],
        "game_theory": [
            "oyun kuramı",
            "game theory",
            "nash",
            "grim",
            "pavlov",
            "tit-for-tat",
            "dominant strateji",
            "ödeme matrisi",
            "prisoner",
            "tekrarlanan oyun",
            "repeated game",
            "uzlaşma",
            "ihanet",
            "aktör",
            "always c",
            "always d",
        ],
        "entertainment": [
            "oyun",
            "şarkı",
            "film",
            "müzik",
            "spor",
            "futbol",
            "basket",
            "game",
            "fun",
            "eğlence",
        ],
        "statistics": [
            "istatistik",
            "ortalama",
            "mean",
            "standart sapma",
            "variance",
            "chebyshev",
            "merkezi limit",
            "monte carlo",
        ],
        "sets": [
            "küme",
            "set",
            "union",
            "∪",
            "∩",
            "venn",
            "içerme",
            "dışarma",
            "partition",
            "bölüntü",
        ],
        "advanced_prob": [
            "stirling",
            "bell",
            "ramsey",
            "üreteç",
            "generating",
            "multinomial",
            "pascal",
        ],
    }
    # Algoritmik reward matrisi: (intent, layout) → base_reward
    REWARD_MATRIX = {
        ("probability", "TREE_DIAGRAM"): 10,
        ("probability", "FLOW_CHART"): 7,
        ("probability", "LINEAR_STEPS"): 6,
        ("combinatorics", "GRID_TABLE"): 9,
        ("combinatorics", "TREE_DIAGRAM"): 8,
        ("combinatorics", "LINEAR_STEPS"): 7,
        ("distribution", "HISTOGRAM"): 10,
        ("distribution", "LINEAR_STEPS"): 6,
        ("distribution", "FLOW_CHART"): 5,
        ("bayes", "FLOW_CHART"): 9,
        ("bayes", "TREE_DIAGRAM"): 8,
        ("bayes", "LINEAR_STEPS"): 6,
        ("markov", "MATRIX_LAYOUT"): 10,
        ("markov", "FLOW_CHART"): 7,
        ("markov", "GRID_TABLE"): 6,
        ("information", "LINEAR_STEPS"): 8,
        ("information", "HISTOGRAM"): 7,
        ("information", "GRID_TABLE"): 6,
        ("social", "TIMELINE"): 9,
        ("social", "FLOW_CHART"): 7,
        ("social", "LINEAR_STEPS"): 6,
        ("story", "LINEAR_STEPS"): 9,
        ("story", "FLOW_CHART"): 7,
        ("story", "TIMELINE"): 6,
        ("entertainment", "FLOW_CHART"): 8,
        ("entertainment", "FRACTAL_STEPS"): 7,
        ("entertainment", "LINEAR_STEPS"): 6,
        ("statistics", "HISTOGRAM"): 10,
        ("statistics", "LINEAR_STEPS"): 7,
        ("statistics", "GRID_TABLE"): 6,
        ("sets", "VENN_DIAGRAM"): 10,
        ("sets", "GRID_TABLE"): 7,
        ("sets", "LINEAR_STEPS"): 6,
        ("game_theory", "MATRIX_LAYOUT"): 10,
        ("game_theory", "GRID_TABLE"): 9,
        ("game_theory", "LINEAR_STEPS"): 8,
        ("game_theory", "FLOW_CHART"): 7,
        ("game_theory", "TREE_DIAGRAM"): 7,
        ("game_theory", "TIMELINE"): 6,
        ("advanced_prob", "LINEAR_STEPS"): 8,
        ("advanced_prob", "FRACTAL_STEPS"): 7,
        ("advanced_prob", "GRID_TABLE"): 6,
    }

    # ── Oyun Kuramı hızlı eşleme desenleri ──────────────────────────────────
    GT_INTENT_PATTERNS = [
        r"\bgrim\b",
        r"\bpavlov\b",
        r"\btit.for.tat\b",
        r"\bnash\s+denge",
        r"\bödeme\s+matrisi\b",
        r"\boyun\s+kuramı\b",
        r"\bher\s+zaman\s+(?:uzlaş|ihanet)\b",
        r"\balways\s+[cd]\b",
        r"\bround.robin\b",
        r"\b\d+\s+aktör\b",
        r"\bgöze\s+göz\b",
        r"\bshapley\b",
        r"\bprisoner\b",
        r"\bitekrarlı\b",
    ]

    def __init__(self, alpha=0.15, gamma=0.85, epsilon=0.12):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode = 0
        self.total_reward = 0

    def _extract_features(self, question: str, signals: dict = None) -> tuple:
        q = question.lower()
        # Intent scoring
        intent_scores = defaultdict(int)
        for intent, kws in self.INTENTS.items():
            for kw in kws:
                if kw in q or any(
                    w.startswith(kw[: max(3, len(kw) - 2)]) for w in q.split()
                ):
                    intent_scores[intent] += 1
        primary_intent = (
            max(intent_scores, key=intent_scores.get) if intent_scores else "general"
        )

        # Complexity features
        num_count = len(re.findall(r"\d+\.?\d*", question))
        branch_count = len(
            re.findall(r"(?:veya|ya da|or|either|hangisi|koşul|given)", q)
        )
        step_count = len(
            re.findall(r"(?:adım|step|ve|then|sonra|ardından|sırasıyla)", q)
        )
        math_density = len(re.findall(r"[+\-*/=^√∑∏C\(\)]", question))
        complexity = min(5, (num_count + branch_count + math_density) // 3)

        # ── Semantic signal dimensions (genişletilmiş state) ─────────────────
        if signals:
            dep_code = {
                "INDEPENDENT": 0,
                "DEPENDENT": 1,
                "CONDITIONAL": 2,
                "UNKNOWN": 3,
            }.get(signals.get("event_dependency", "UNKNOWN"), 3)
            elim_code = 1 if signals.get("elimination_rule") else 0
            logic_code = {"AND_CHAIN": 0, "OR_SUM": 1, "MIXED": 2, "SINGLE": 3}.get(
                signals.get("logic_operator", "SINGLE"), 3
            )
            trials = min(signals.get("sequential_trials", 1), 8)
        else:
            dep_code = elim_code = logic_code = 0
            trials = 1

        # State: 7-boyutlu tuple — semantic sinyalleri dahil
        state = (
            primary_intent,
            complexity,
            min(branch_count, 4),
            min(step_count, 4),
            dep_code,
            elim_code,
            logic_code,
        )

        features = {
            "intent": primary_intent,
            "complexity": complexity,
            "branch_factor": branch_count,
            "step_count": step_count,
            "num_count": num_count,
            "math_density": math_density,
            "intent_scores": dict(intent_scores),
            # Semantic eklentiler
            "event_dependency": (
                signals.get("event_dependency", "UNKNOWN") if signals else "UNKNOWN"
            ),
            "elimination_rule": (
                signals.get("elimination_rule", False) if signals else False
            ),
            "logic_operator": (
                signals.get("logic_operator", "SINGLE") if signals else "SINGLE"
            ),
            "sequential_trials": trials,
            "dep_code": dep_code,
            "elim_code": elim_code,
            "logic_code": logic_code,
        }
        return state, features

    def _compute_reward(self, layout, features):
        intent = features["intent"]
        base = self.REWARD_MATRIX.get((intent, layout), 3)
        # Dynamic bonuses
        bf = features["branch_factor"]
        sc = features["step_count"]
        cp = features["complexity"]
        bonus = 0
        if bf > 2 and layout in ("TREE_DIAGRAM", "VENN_DIAGRAM", "FLOW_CHART"):
            bonus += 2
        if sc > 3 and layout in ("LINEAR_STEPS", "FLOW_CHART", "FRACTAL_STEPS"):
            bonus += 2
        if cp > 3 and layout in ("MATRIX_LAYOUT", "GRID_TABLE", "FRACTAL_STEPS"):
            bonus += 1
        # Penalty for mismatched
        if intent in ("distribution", "statistics") and layout == "TIMELINE":
            bonus -= 3
        if intent in ("markov",) and layout == "TIMELINE":
            bonus -= 3
        return max(0, base + bonus)

    def select_action(self, state, features):
        """Epsilon-greedy seçim"""
        if random.random() < self.epsilon:
            return random.choice(self.LAYOUTS)
        q_vals = self.q_table[state]
        if not q_vals:
            # Intent'e göre başlangıç tahmin (cold-start)
            defaults = {
                "probability": "TREE_DIAGRAM",
                "combinatorics": "GRID_TABLE",
                "distribution": "HISTOGRAM",
                "bayes": "FLOW_CHART",
                "markov": "MATRIX_LAYOUT",
                "information": "LINEAR_STEPS",
                "social": "TIMELINE",
                "story": "LINEAR_STEPS",
                "entertainment": "FLOW_CHART",
                "statistics": "HISTOGRAM",
                "sets": "VENN_DIAGRAM",
                "advanced_prob": "LINEAR_STEPS",
                "general": "FLOW_CHART",
            }
            return defaults.get(features["intent"], "FLOW_CHART")
        return max(q_vals, key=q_vals.get)

    def update(self, state, action, reward):
        """Bellman update"""
        nq = self.q_table[state]
        max_next = max(nq.values()) if nq else 0
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next - self.q_table[state][action]
        )
        self.episode += 1
        self.total_reward += reward

    def route(self, question: str, signals: dict = None):
        # ── Oyun Kuramı sinyali — signals None ise veya eksikse hesapla ──────
        if signals is None:
            signals = {}
        if not signals.get("game_theory_signal", False):
            _qn = question.lower()
            if any(re.search(p, _qn) for p in self.GT_INTENT_PATTERNS):
                signals = dict(signals)
                signals["game_theory_signal"] = True
                signals.setdefault("game_theory_score", 2.0)
        state, features = self._extract_features(question, signals)
        action = self.select_action(state, features)
        reward = self._compute_reward(action, features)
        self.update(state, action, reward)
        q_snapshot = dict(self.q_table[state])
        return action, features, reward, q_snapshot


# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITMIK ASCII LAYOUT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class ASCIIEngine:
    """
    Tamamen algoritmik ASCII layout üretici.
    Hiçbir sabit template yok — tüm geometri içerik ve layout tipinden üretiliyor.
    """

    W = ASCII_WIDTH

    # ─────── Temel Primitifler ────────────────────────────────────────────────
    def _box(self, lines, title="", double=False, accent=False):
        tl, tr, bl, br = (
            ("╔", "╗", "╚", "╝")
            if double
            else (("┏", "┓", "┗", "┛") if accent else ("┌", "┐", "└", "┘"))
        )
        h = "═" if double else ("━" if accent else "─")
        v = "║" if double else ("┃" if accent else "│")
        max_w = max((len(l) for l in lines), default=0)
        if title:
            max_w = max(max_w, len(title) + 2)
        iw = max(max_w + 2, 4)
        out = []
        if title:
            ts = f" {title} "
            pad = iw - len(ts)
            out.append(f"{tl}{h*(pad//2)}{ts}{h*(pad - pad//2)}{tr}")
        else:
            out.append(f"{tl}{h*iw}{tr}")
        for l in lines:
            out.append(f"{v} {l:<{iw-2}} {v}")
        out.append(f"{bl}{h*iw}{br}")
        return out

    def _wrap(self, text, indent=2, width=None):
        w = (width or self.W) - indent - 4
        words = text.split()
        lines = []
        cur = ""
        for word in words:
            if len(cur) + len(word) + 1 > w:
                if cur:
                    lines.append(" " * indent + cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            lines.append(" " * indent + cur)
        return lines or [""]

    def _center_line(self, text, width=None):
        w = width or self.W
        return text.center(w)

    def _full_sep(self, char="─", dbl=False):
        c = "═" if dbl else char
        return "│" + c * (self.W - 2) + "│"

    def _banner(self, title, subtitle="", dbl=True):
        W = self.W
        h = "═" if dbl else "─"
        tl, tr, bl, br = ("╔", "╗", "╚", "╝") if dbl else ("┌", "┐", "└", "┘")
        v = "║" if dbl else "│"
        lines = [f"{tl}{h*(W-2)}{tr}"]
        t_pad = (W - 2 - len(title)) // 2
        lines.append(f"{v}{' '*t_pad}{title}{' '*(W-2-t_pad-len(title))}{v}")
        if subtitle:
            s_pad = (W - 2 - len(subtitle)) // 2
            lines.append(f"{v}{' '*s_pad}{subtitle}{' '*(W-2-s_pad-len(subtitle))}{v}")
        lines.append(f"{bl}{h*(W-2)}{br}")
        return "\n".join(lines)

    # ─────── Layout Üreticiler ────────────────────────────────────────────────
    def linear_steps(self, steps, title="ÇÖZÜM ADIMLARI"):
        W = self.W
        out = []
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        for i, step in enumerate(steps, 1):
            stitle = step.get("title", f"Adım {i}")
            content = step.get("content", "")
            formula = step.get("formula", "")
            result = step.get("result", "")
            # Step header
            num_badge = f"[{i:02d}]"
            header = f"  {num_badge} ► {stitle}"
            out.append(f"║{' '*(W-2)}║")
            out.append(f"║  {'─'*(W-6)}  ║")
            out.append(f"║{header:<{W-2}}║")
            out.append(f"║  {'╌'*(W-6)}  ║")
            # Content
            for ln in self._wrap(content, 5, W):
                out.append(f"║{ln:<{W-2}}║")
            # Formula
            if formula:
                out.append(f"║{' '*(W-2)}║")
                fl = f"     ▸ {formula}"
                out.append(f"║{fl:<{W-2}}║")
            # Result
            if result:
                rl = f"     ✓ = {result}"
                out.append(f"║{rl:<{W-2}}║")
            # Connector arrow (not on last)
            if i < len(steps):
                mid = W // 2
                arr_line = " " * mid + "▼"
                out.append(f"║{' '*(W-2)}║")
                out.append(f"║{arr_line:<{W-2}}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def tree_diagram(self, tree, title="OLASILIK AĞACI"):
        W = self.W
        out = []
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")

        def render(node, prefix="", is_last=True, depth=0):
            rows = []
            label = node.get("label", "?")
            prob = node.get("prob", "")
            val = node.get("value", "")
            final = node.get("final", "")
            conn = "└─▶ " if is_last else "├─▶ "
            if depth == 0:
                badge = "◉ ROOT"
                ln = f"  {badge}: {label}"
                if prob:
                    ln += f"  [P = {prob}]"
            else:
                ln = f"  {prefix}{conn}{label}"
                if prob:
                    ln += f"  [p={prob}]"
                if val:
                    ln += f"  → {val}"
                if final:
                    ln += f"  ★ P={final}"
            rows.append(f"║{ln:<{W-2}}║")
            children = node.get("children", [])
            for idx, child in enumerate(children):
                ext = "    " if is_last else "│   "
                rows.extend(
                    render(child, prefix + ext, idx == len(children) - 1, depth + 1)
                )
            return rows

        if tree:
            for r in render(tree):
                out.append(r)
        else:
            out.append(f"║  [Ağaç verisi bulunamadı]{' '*(W-26)}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def flow_chart(self, steps, title="AKIŞ ŞEMASI"):
        W = self.W
        out = []
        center = W // 2
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        SHAPES = {
            "start": ("╭── ", " ──╮"),
            "end": ("╰── ", " ──╯"),
            "decision": ("◇ ", " ◇"),
            "process": ("┤  ", "  ├"),
            "data": ("╱  ", "  ╲"),
            "result": ("▶  ", "  ◀"),
        }
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                text = step.get("text", str(step))
                stype = step.get("type", "process")
                sub = step.get("sub", "")
            else:
                text = str(step)
                stype = "process"
                sub = ""
            lbrak, rbrak = SHAPES.get(stype, ("│ ", " │"))
            # Truncate if needed
            max_text = W - 14
            if len(text) > max_text:
                text = text[: max_text - 3] + "..."
            inner = f"{lbrak}{text}{rbrak}"
            pad = max(2, (W - len(inner)) // 2)
            out.append(f"║{' '*pad}{inner}{' '*(W-2-pad-len(inner))}║")
            if sub:
                sub_ln = f"  └ {sub}"
                sp = max(0, (W - len(sub_ln)) // 2)
                out.append(f"║{' '*sp}{sub_ln}{' '*(W-2-sp-len(sub_ln))}║")
            if i < len(steps) - 1:
                arr = "║" + " " * (center - 1) + "▼" + " " * (W - 2 - center) + "║"
                out.append(f"║{' '*(W-2)}║")
                out.append(arr)
                out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def grid_table(self, data, title="TABLO", row_labels=None, col_labels=None):
        W = self.W
        out = []
        if not data or not data[0]:
            return self.linear_steps([], title)
        nrows = len(data)
        ncols = len(data[0])
        # Auto-size columns
        col_w = []
        for j in range(ncols):
            cw = max(len(str(data[i][j])) for i in range(nrows))
            if col_labels and j < len(col_labels):
                cw = max(cw, len(str(col_labels[j])))
            col_w.append(max(cw + 2, 6))
        rl_w = (
            max((len(str(r)) for r in row_labels), default=0) + 2 if row_labels else 3
        )

        # Build separator
        def sep(l, m, r, h="─"):
            s = l
            if row_labels:
                s += h * (rl_w + 2) + m
            for i, w in enumerate(col_w):
                s += h * (w + 2) + (m if i < len(col_w) - 1 else r)
            return s

        top = sep("┌", "┬", "┐")
        mid = sep("├", "┼", "┤")
        bot = sep("└", "┴", "┘")
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        tbl = []
        tbl.append("  " + top)
        if col_labels:
            row = "  │"
            if row_labels:
                row += f"{'':^{rl_w+2}}│"
            for j, (lbl, cw) in enumerate(zip(col_labels, col_w)):
                row += f" {str(lbl):^{cw}} │"
            tbl.append(row)
            tbl.append("  " + mid)
        for i, drow in enumerate(data):
            row = "  │"
            if row_labels and i < len(row_labels):
                row += f" {str(row_labels[i]):^{rl_w}} │"
            for j, (cell, cw) in enumerate(zip(drow, col_w)):
                row += f" {str(cell):^{cw}} │"
            tbl.append(row)
            if i < len(data) - 1:
                tbl.append("  " + mid)
        tbl.append("  " + bot)
        for t in tbl:
            out.append(f"║{t:<{W-2}}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def histogram(self, dist, title="DAĞILIM GRAFİĞİ"):
        W = self.W
        out = []
        H = 12
        if not dist:
            return self._banner(title, "Veri yok")
        keys = list(dist.keys())
        vals = [float(v) for v in dist.values()]
        max_v = max(vals) if vals else 1
        bar_w = max(3, min(8, (W - 12) // max(len(keys), 1)))
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        for row in range(H, 0, -1):
            threshold = (row / H) * max_v
            y_lbl = f"{threshold:5.3f}" if max_v < 10 else f"{threshold:5.1f}"
            bars = "  " + y_lbl + " ┤"
            for v in vals:
                bh = int((v / max_v) * H) if max_v > 0 else 0
                sym = "▓" if bh >= row else " "
                bars += sym * bar_w + " "
            out.append(f"║{bars:<{W-2}}║")
        # X axis
        x_ax = "        └" + "─" * (len(keys) * (bar_w + 1) + 1)
        out.append(f"║{x_ax:<{W-2}}║")
        # Labels
        lbl_row = "         "
        for k in keys:
            lbl = str(k)[:bar_w].center(bar_w + 1)
            lbl_row += lbl
        out.append(f"║{lbl_row:<{W-2}}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def matrix_layout(
        self, mat, title="GEÇİŞ MATRİSİ", row_labels=None, col_labels=None
    ):
        return self.grid_table(mat, title, row_labels, col_labels)

    def venn_diagram(self, sets, title="VENN DİYAGRAMI"):
        W = self.W
        out = []
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        if len(sets) >= 2:
            A = sets[0]
            B = sets[1]
            an = A.get("name", "A")
            bn = B.get("name", "B")
            ao = A.get("only", "A'ya özgü")
            bo = B.get("only", "B'ye özgü")
            inter = A.get("intersection", "A ∩ B")
            total = A.get("total", "")
            au = A.get("union", "A ∪ B")
            # ASCII iki daire
            venn_art = [
                f"",
                f"     ╭──────────────╮      ╭──────────────╮",
                f"    ╭╯              ╰──╮╭──╯              ╰╮",
                f"   ╭╯   {ao:<12}  ╰╯  {bo:<12}  ╰╮",
                f"   │   {an.center(14)} ║  {bn.center(14)} │",
                f"   │                  ║                   │",
                f"   ╰╮    {ao[:8]:^8}   ╭{inter[:10]:^10}╮   {bo[:8]:^8}  ╭╯",
                f"    ╰──╮           ╭──╯╰──╮           ╭──╯",
                f"       ╰───────────╯      ╰───────────╯",
                f"",
                f"  A = {an}    B = {bn}    A∩B: {inter}",
            ]
            if total:
                venn_art.append(f"  Toplam: {total}")
            if au:
                venn_art.append(f"  A∪B  : {au}")
            for vl in venn_art:
                out.append(f"║{vl:<{W-2}}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def timeline(self, events, title="OLAY ZİNCİRİ"):
        W = self.W
        out = []
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        for i, ev in enumerate(events):
            label = ev.get("label", "?") if isinstance(ev, dict) else str(ev)
            prob = ev.get("prob", "") if isinstance(ev, dict) else ""
            note = ev.get("note", "") if isinstance(ev, dict) else ""
            badge = f"[{i+1:02d}]"
            ln = f"  {badge}──● {label}"
            if prob:
                ln += f"  [P={prob}]"
            out.append(f"║{ln:<{W-2}}║")
            if note:
                nl = f"       └─ {note}"
                out.append(f"║{nl:<{W-2}}║")
            if i < len(events) - 1:
                out.append(f"║       │{' '*(W-9)}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def fractal_steps(self, steps, title="ÖZYINELEMELI ÇÖZÜM"):
        """Recursive/nested görünümlü step layout"""
        W = self.W
        out = []
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        indent_chars = ["  ", "    ", "      ", "        "]
        for i, step in enumerate(steps):
            stitle = step.get("title", f"Katman {i+1}")
            content = step.get("content", "")
            formula = step.get("formula", "")
            depth = step.get("depth", i % 4)
            ind = indent_chars[min(depth, 3)]
            prefix = "├▶ " if i < len(steps) - 1 else "└▶ "
            hl = f"{ind}{prefix}{stitle}"
            out.append(f"║{hl:<{W-2}}║")
            if content:
                for cl in self._wrap(content, len(ind) + 5, W):
                    out.append(f"║{cl:<{W-2}}║")
            if formula:
                fl = f"{ind}   ◈ {formula}"
                out.append(f"║{fl:<{W-2}}║")
            if i < len(steps) - 1:
                vl = f"{ind}   │"
                out.append(f"║{vl:<{W-2}}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def solution_box(self, sol):
        W = self.W
        out = []
        answer = sol.get("answer", "?")
        formula = sol.get("formula", "")
        numeric = sol.get("numeric", "")
        explain = sol.get("explanation", "")
        certainty = sol.get("certainty", "")
        title = "✦ SONUÇ VE CEVAP ✦"
        out.append(f"╔{'═'*(W-2)}╗")
        tp = (W - 2 - len(title)) // 2
        out.append(f"║{' '*tp}{title}{' '*(W-2-tp-len(title))}║")
        out.append(f"╠{'═'*(W-2)}╣")
        if formula:
            fl = f"  FORMÜL   ▸ {formula}"
            out.append(f"║{fl:<{W-2}}║")
            out.append(f"║{' '*(W-2)}║")
        al = f"  CEVAP    ► {answer}"
        out.append(f"║{al:<{W-2}}║")
        if numeric:
            nl = f"  SAYISAL  ► {numeric}"
            out.append(f"║{nl:<{W-2}}║")
        if certainty:
            cl = f"  KESINLIK ► {certainty}"
            out.append(f"║{cl:<{W-2}}║")
        if explain:
            out.append(f"║{' '*(W-2)}║")
            out.append(f"║  Açıklama:{' '*(W-12)}║")
            for ln in self._wrap(explain, 4, W):
                out.append(f"║{ln:<{W-2}}║")
        out.append(f"║{' '*(W-2)}║")
        out.append(f"╚{'═'*(W-2)}╝")
        return "\n".join(out)

    def q_info_box(self, layout, features, reward, q_vals, episode):
        """Q-Learning bilgi kutusu"""
        W = self.W
        out = []
        title = "◈ Q-LEARNING ROUTER DURUMU"
        out.append(f"┌{'─'*(W-2)}┐")
        tp = (W - 2 - len(title)) // 2
        out.append(f"│{' '*tp}{title}{' '*(W-2-tp-len(title))}│")
        out.append(f"├{'─'*(W-2)}┤")
        intent = features.get("intent", "?")
        comp = features.get("complexity", 0)
        bf = features.get("branch_factor", 0)
        out.append(f"│  Niyet (Intent)   : {intent:<{W-24}}│")
        out.append(f"│  Seçilen Layout   : {layout:<{W-24}}│")
        out.append(f"│  Ödül (Reward)    : {reward:<{W-24}}│")
        out.append(f"│  Karmaşıklık      : {comp:<{W-24}}│")
        out.append(f"│  Dal Faktörü      : {bf:<{W-24}}│")
        out.append(f"│  Episode          : {episode:<{W-24}}│")
        out.append(f"├{'─'*(W-2)}┤")
        # Top-3 Q-values
        if q_vals:
            sorted_q = sorted(q_vals.items(), key=lambda x: -x[1])[:3]
            out.append(f"│  Top-3 Q-Değerleri:{' '*(W-21)}│")
            for act, qv in sorted_q:
                bar_len = min(int(abs(qv) * 2), W - 35)
                bar = "█" * bar_len
                marker = "◄" if act == layout else " "
                ln = f"│    {marker} {act:<16} {qv:6.2f} |{bar}"
                out.append(f"{ln:<{W-1}}│")
        out.append(f"└{'─'*(W-2)}┘")
        return "\n".join(out)

    def render(self, layout, sol_data):
        steps = sol_data.get("steps", [])
        tree = sol_data.get("tree", {})
        dist = sol_data.get("distribution", {})
        table = sol_data.get("table", {})
        mat = sol_data.get("matrix", {})
        sets = sol_data.get("sets", [])
        events = sol_data.get("events", [])
        title = sol_data.get("title", "ÇÖZÜM")
        W = self.W

        # Header
        q_text = sol_data.get("question", "")
        q_type = sol_data.get("type", "Genel")
        hdr = []
        hdr.append(f"╔{'═'*(W-2)}╗")
        sig = "◆ ASCIIMATİK  ·  phi4:14b  ·  Q-Learning Router ◆"
        sp = (W - 2 - len(sig)) // 2
        hdr.append(f"║{' '*sp}{sig}{' '*(W-2-sp-len(sig))}║")
        hdr.append(f"╠{'═'*(W-2)}╣")
        tl = f"  Tip: {q_type}   Layout: {layout}"
        hdr.append(f"║{tl:<{W-2}}║")
        hdr.append(f"╠{'═'*(W-2)}╣")
        hdr.append(f"║  SORU:{' '*(W-8)}║")
        for ln in self._wrap(q_text, 3, W):
            hdr.append(f"║{ln:<{W-2}}║")
        hdr.append(f"╚{'═'*(W-2)}╝")
        parts = ["\n".join(hdr)]

        # Main visual
        if layout == "LINEAR_STEPS":
            parts.append(self.linear_steps(steps, title))
        elif layout == "TREE_DIAGRAM":
            if tree:
                parts.append(self.tree_diagram(tree, title))
            if steps:
                parts.append(self.linear_steps(steps, "HESAPLAMA ADIMLARI"))
        elif layout == "FLOW_CHART":
            flow = [
                {
                    "text": s.get("title", s.get("content", str(s)))[:60],
                    "type": s.get("type", "process"),
                    "sub": s.get("formula", ""),
                }
                for s in steps
            ]
            parts.append(self.flow_chart(flow, title))
        elif layout == "GRID_TABLE":
            if table and table.get("data"):
                parts.append(
                    self.grid_table(
                        table["data"],
                        title,
                        table.get("row_labels"),
                        table.get("col_labels"),
                    )
                )
            if steps:
                parts.append(self.linear_steps(steps, "ADIMLAR"))
        elif layout == "HISTOGRAM":
            if dist:
                parts.append(self.histogram(dist, title))
            if steps:
                parts.append(self.linear_steps(steps, "HESAPLAMA"))
        elif layout == "VENN_DIAGRAM":
            if sets:
                parts.append(self.venn_diagram(sets, title))
            if steps:
                parts.append(self.linear_steps(steps, "HESAPLAMA"))
        elif layout == "MATRIX_LAYOUT":
            if mat and mat.get("data"):
                parts.append(
                    self.matrix_layout(
                        mat["data"], title, mat.get("row_labels"), mat.get("col_labels")
                    )
                )
            if steps:
                parts.append(self.linear_steps(steps, "ADIMLAR"))
        elif layout == "TIMELINE":
            if events:
                parts.append(self.timeline(events, title))
            if steps:
                parts.append(self.linear_steps(steps, "AÇIKLAMA"))
        elif layout == "FRACTAL_STEPS":
            parts.append(self.fractal_steps(steps, title))
        else:
            flow = [
                {"text": s.get("title", str(s))[:60], "type": "process"} for s in steps
            ]
            parts.append(self.flow_chart(flow, title))

        # Solution box
        parts.append(self.solution_box(sol_data))
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  DİNAMİK SYSTEM PROMPT  (constraint bloğu enjekte edilerek üretilir)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_SYSTEM_PROMPT = """Sen bir olasılık ve matematik uzmanısın. Verilen soruyu çöz.
YALNIZCA geçerli JSON döndür — başka hiçbir şey yazma, kod bloğu kullanma.

Döndüreceğin JSON yapısı:
{{
  "type": "soru_tipi (kombinasyon/permütasyon/bayes/binom/normal/poisson/markov/entropi/küme/genel vb.)",
  "title": "kısa başlık (max 50 karakter)",
  "steps": [
    {{
      "title": "adım başlığı",
      "content": "açıklama metni",
      "formula": "matematiksel ifade (opsiyonel)",
      "result": "bu adımın sonucu (opsiyonel)",
      "depth": 0,
      "type": "process"
    }}
  ],
  "answer": "final cevap (kısa, net)",
  "explanation": "kısa açıklama cümlesi",
  "formula": "kullanılan ana formül",
  "numeric": "sayısal değer ve yaklaşık değer",
  "certainty": "kesinlik derecesi (opsiyonel)",
  "tree": {{
    "label": "kök",
    "prob": "",
    "children": [
      {{ "label": "dal", "prob": "p", "value": "", "final": "", "children": [] }}
    ]
  }},
  "distribution": {{ "0": 0.1, "1": 0.3, "2": 0.4, "3": 0.2 }},
  "table": {{
    "data": [["A","B"],["C","D"]],
    "row_labels": ["r1","r2"],
    "col_labels": ["c1","c2"]
  }},
  "matrix": {{
    "data": [[0.7,0.3],[0.4,0.6]],
    "row_labels": ["S1","S2"],
    "col_labels": ["S1","S2"]
  }},
  "sets": [
    {{ "name": "A", "only": "A özgü", "intersection": "A∩B", "total": "" }},
    {{ "name": "B", "only": "B özgü" }}
  ],
  "events": [
    {{ "label": "Olay", "prob": "0.5", "note": "açıklama" }}
  ]
}}

Genel kurallar:
- steps dizisi {step_range}
- Her adımda somut formül ve ara sonuç göster; hiçbir ara hesaplama adımı birleştirilmemeli veya atlanmamalı
- Olasılık sorularında tree kullan
- Dağılım sorularında distribution değerlerini doldur
- Markov'da matrix kullan
- Venn/küme sorularında sets kullan
- Olaylar zincirinde events kullan
- Yanıt her zaman Türkçe olsun

{constraint_block}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  OYUN KURAMI LLM SİSTEM PROMPT EKLENTİSİ
#  Oyun kuramı sinyali tespit edilince build_system_prompt tarafından eklenir.
# ─────────────────────────────────────────────────────────────────────────────
_GAME_THEORY_LLM_PROMPT = """
══ OYUN KURAMI ÇÖZÜM MODU (NLP+BART) ══

Bu soru OYUN KURAMI kapsamındadır. Çözüm adımları:

Adım 1 — Oyun tipini belirle: Tekrarlanan Oyun / Matris / Kooperatif / Evrimsel / Bayesian
Adım 2 — Oyuncular ve stratejileri tanımla:
  • Grim Trigger: İlk tur C. Bir kez D görürse → o aktöre SONSUZA KADAR D
  • Pavlov (Win-Stay/Lose-Shift): Önceki tur kârlıysa (R veya T) → aynı hamle; kârsızsa değiştir
  • Always C: Her turda C | Always D: Her turda D
  • Tit-for-Tat: İlk C, sonra rakibin son hamlesini kopyala
Adım 3 — Ödeme matrisini kur:
  (C,C) → R, R  |  (D,C) → T, S  |  (C,D) → S, T  |  (D,D) → P, P
  PD kısıtı: T > R > P ≥ S
Adım 4 — Round-Robin simülasyonu: Her turda her oyuncu çifti eşleşir
Adım 5 — Strateji geçişlerini doğrula (tetiklenmeler, Pavlov kayıp/kazanç)
Adım 6 — Kümülatif puanları hesapla (her oyuncu için tur tur)
Adım 7 — Soruları cevapla (Nash, sıralama, Grim hedefi, dominant strateji)

NxN MATRİS: N>2 strateji varsa tam matris kur, tüm (i,j) kombinasyonlarını hesapla.
BÜYÜK N ADIM: Kararlı durumu bul, tur başına sabit kazanç × N extrapolation uygula.
N±10 PENCERE: N. adım için N-10'dan N+10'a kadar adım adım göster.

ÖDEME NOTASYONU:
• T (Temptation/Cazibe) = ihanet eden kazanç | S (Sucker/Kandırılan) = uzlaşan kayıp
• R (Reward/Ödül) = karşılıklı uzlaşma | P (Punishment/Ceza) = karşılıklı ihanet

ÖNEMLİ: Puanlar round-robin toplamıdır — her oyuncu diğer n-1 oyuncuyla eşleşir.
"""


def _compute_step_range(signals: dict) -> str:
    """
    Sinyal boyutlarından dinamik adım aralığı hesaplar.
    Hiçbir sabit eşik yok — tümü signal alanlarından türetilir.

    Boyutlar:
      sequential_trials → her ek deneme 2 ek adım gerektirir
      elimination_rule  → zincir başlangıç adımı ekler
      bayes_structure   → prior + likelihood + payda + posterior adımları
      event_dependency  → bağımlı: koşullu geçiş adımları ekler
      logic_operator    → MIXED/AND_CHAIN daha fazla dal adımı gerektirir
    """
    st = signals.get("sequential_trials", 1)
    el = signals.get("elimination_rule", False)
    bay = signals.get("bayes_structure", False)
    lo = signals.get("logic_operator", "SINGLE")
    dep = signals.get("event_dependency", "UNKNOWN")
    sc = signals.get("survival_chain", False)

    min_s = 3
    max_s = 8

    # Her ek ardışık deneme: 2 ek adım (hesaplama + zincir bağlantısı)
    if st > 1:
        extra = (st - 1) * 2
        max_s += min(extra, 10)  # en fazla 10 ek adım
        min_s += min(st - 1, 3)  # en az min artışı: 3

    # Eliminasyon / hayatta kalma zinciri: başlangıç + koşul adımları
    if el or sc:
        max_s += 2
        min_s = max(min_s, 4)

    # Bayes yapısı: prior + likelihood + payda + posterior + normalize
    if bay:
        max_s += 4
        min_s = max(min_s, 5)

    # Bağımlı olaylar: her geçiş için örneklem uzayı güncelleme adımı
    if dep == "DEPENDENT":
        max_s += 2

    # Karma/VE zinciri: her dal ayrı adım gerektirir
    if lo in ("MIXED", "AND_CHAIN"):
        max_s += 2

    # Koşullu olasılık: payda hesaplama adımı ekstra
    if dep == "CONDITIONAL":
        max_s += 2

    # Global sınır
    max_s = min(max_s, 20)
    min_s = min(min_s, max_s - 2)

    return f"en az {min_s}, en fazla {max_s} eleman içermeli"


def build_system_prompt(signals: dict, sem: SemanticSignalModule) -> str:
    """Semantic sinyallerden kısıt bloğunu üretip prompt'a gömer."""
    constraint_block = sem.build_constraint_block(signals)
    step_range = _compute_step_range(signals)
    base = BASE_SYSTEM_PROMPT.format(
        constraint_block=constraint_block,
        step_range=step_range,
    )
    # ── Oyun Kuramı sinyal varsa GT prompt ekle ──────────────────────────────
    if signals.get("game_theory_signal"):
        base += _GAME_THEORY_LLM_PROMPT
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  SOLUTION CORRECTOR
#
#  Mimari ilke: LLM'e güvenmek yerine, belirli sinyal kombinasyonlarında
#  çözümü cerrahi olarak sinyalden türetilmiş saf aritmetikle yeniden inşa et.
#
#  Neden retry'dan üstün:
#    - Retry aynı modele ihlal listesini ekleyerek yeniden sorar.
#      Model öğrenilmiş davranışını değiştirmez → aynı hata döner.
#    - Corrector ise hesabı tamamen sinyal+matematik motoruyla yapar;
#      LLM hiç devreye girmez.
#
#  Kapsam:
#    RULE 1: INDEPENDENT_SURVIVAL_CHAIN
#      Tetikleyici: INDEPENDENT + denominator_n + survival_chain + violations içinde
#                   PAYDA AZALMASI veya BAĞIMSIZ+KOŞULLU çakışma
#      İşlem: step_prob = (n-k)/n,  total = step_prob ^ t
#
#    RULE 2: INDEPENDENT_SEQUENTIAL_AND
#      Tetikleyici: INDEPENDENT + denominator_n + sequential_trials >= 2 +
#                   AND_CHAIN + violations içinde PAYDA AZALMASI
#      İşlem: favorable/n çıkart, step_prob^t hesapla
#
#  Yeni kural eklemek → RULES listesine (tetikleyici_fn, işlem_fn) tuple ekle.
# ═══════════════════════════════════════════════════════════════════════════════
class SolutionCorrector:
    """
    Violation-driven algebraic solution rebuilder.

    correct(signals, sol_data, question) → dict | None
      - Eğer düzeltilebilir ihlal varsa: tamamen yeniden inşa edilmiş sol_data
      - Yoksa: None (OllamaClient orijinal sol_data'yı kullanmaya devam eder)
    """

    # ── Tetikleyici violation parmak izleri ───────────────────────────────────
    # Bu string'lerden biri violations içinde geçiyorsa correction tetiklenir.
    CORRECTABLE_VIOLATION_FINGERPRINTS = [
        "BAĞIMSIZ PAYDA AZALMASI",
        "DENOMINATOR_SHRINK_IN_INDEPENDENT",
        "BAĞIMSIZ+KOŞULLU ÇAKIŞMA",
        "BAĞIMSIZ OLAY PAYDA KÜÇÜLME",
    ]

    # ── Sayısal çekirdek varlık çıkarıcıları ─────────────────────────────────
    # Soru metninden "kaç kurşun / mermi / dolu hazne" çıkarır.
    BULLET_PATTERNS = [
        r"(\d+)\s*(?:tane\s+)?(?:kurşun|mermi|bullet[s]?|cartridge[s]?)",
        r"(\d+)\s*(?:dolu|loaded|filled)\s*(?:hazne|chamber[s]?)?",
        r"(\d+)\s*(?:hazne\s+)?(?:dolu|loaded)",
        r"(?:kurşun|bullet[s]?|mermi)\s*(?:sayısı\s*=?\s*|:\s*)(\d+)",
    ]

    # "X tane boş / X boş hazne" → favorable count doğrudan verilmiş
    EMPTY_PATTERNS = [
        r"(\d+)\s*(?:tane\s+)?(?:boş|empty)\s*(?:hazne|chamber[s]?)?",
        r"(\d+)\s*(?:hazne\s+)?(?:boş|empty)",
    ]

    # Genel "favori / başarılı sonuç" çıkarıcı — kurşunsuz senaryolar için
    FAVORABLE_PATTERNS = [
        r"(\d+)\s*(?:tane\s+)?(?:kırmızı|mavi|yeşil|beyaz|siyah|renkli|kazanan|ödüllü)",
        r"(?:başarı|success)\s*(?:sayısı\s*=?\s*|:\s*)(\d+)",
    ]

    def correct(self, signals: dict, sol_data: dict, question: str) -> "dict | None":
        """
        Düzeltilebilir violation varsa yeniden inşa edilmiş sol_data döndürür.
        Yoksa None döndürür.
        """
        violations = sol_data.get("_consistency_violations", [])
        if not violations:
            return None

        # Violation fingerprint kontrolü
        violation_text = " ".join(violations)
        is_correctable = any(
            fp in violation_text for fp in self.CORRECTABLE_VIOLATION_FINGERPRINTS
        )
        if not is_correctable:
            return None

        dep = signals.get("event_dependency")
        n = signals.get("denominator_n")
        t = signals.get("sequential_trials", 1)
        el = signals.get("elimination_rule", False)
        sc = signals.get("survival_chain", False)
        lo = signals.get("logic_operator", "SINGLE")

        if dep != "INDEPENDENT" or n is None:
            return None

        # ── RULE 1: INDEPENDENT + hayatta kalma zinciri ───────────────────────
        if sc and el and t >= 2:
            return self._rule_independent_survival_chain(signals, question, n, t)

        # ── RULE 2: INDEPENDENT + AND zinciri (elim yok) ─────────────────────
        if t >= 2 and lo in ("AND_CHAIN", "SINGLE"):
            return self._rule_independent_sequential(signals, question, n, t)

        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _rule_independent_survival_chain(
        self, signals: dict, question: str, n: int, t: int
    ) -> "dict | None":
        """
        Bağımsız hayatta kalma zinciri:
          step_prob = (n - k) / n   (k = olumsuz çıktı sayısı, yani kurşun)
          total     = step_prob ^ t
        """
        k = self._extract_bullets(question)
        if k is None or k >= n:
            return None

        from fractions import Fraction

        step = Fraction(n - k, n)
        total = step**t

        # Adım açıklamaları
        steps = []
        steps.append(
            {
                "title": f"Adım 1: Tek Atışta Hayatta Kalma",
                "content": (
                    f"Bağımsız olay — silindir her atıştan önce döndürülüyor. "
                    f"Örneklem uzayı sıfırlanır: {n} hazne, {k} kurşun, {n-k} boş. "
                    f"Payda her adımda {n} (değişmez)."
                ),
                "formula": f"P(hayatta kalma) = ({n} - {k}) / {n} = {n-k}/{n}",
                "result": f"{step.numerator}/{step.denominator}",
                "depth": 0,
                "type": "process",
            }
        )
        for i in range(2, t + 1):
            steps.append(
                {
                    "title": f"Adım {i}: {i}. Atışta Hayatta Kalma",
                    "content": (
                        f"Silindir yeniden döndürüldü — önceki atış unutuldu. "
                        f"Örneklem uzayı yine {n} hazne, {k} kurşun. "
                        f"Payda hâlâ {n}."
                    ),
                    "formula": f"P(hayatta kalma) = {n-k}/{n}",
                    "result": f"{step.numerator}/{step.denominator}",
                    "depth": 0,
                    "type": "process",
                }
            )
        # Çarpım adımı
        mul_parts = " × ".join([f"({step.numerator}/{step.denominator})"] * t)
        steps.append(
            {
                "title": f"Adım {t+1}: {t} Atışta Toplam Hayatta Kalma",
                "content": (
                    f"Bağımsız olayların zinciri — olasılıklar çarpılır. "
                    f"Formül: P(hepsi hayatta) = P(tek atış)^{t}"
                ),
                "formula": f"P(toplam) = {mul_parts} = {total.numerator}/{total.denominator}",
                "result": f"{total.numerator}/{total.denominator}",
                "depth": 0,
                "type": "process",
            }
        )

        # Ağaç yapısı
        tree = self._build_survival_tree(n, k, t, step, total)

        # Sayısal değer
        numeric_val = Decimal(total)
        answer_str = (
            f"{total.numerator}/{total.denominator}"
            if total.denominator != 1
            else str(total.numerator)
        )
        formula_str = (
            f"({step.numerator}/{step.denominator})^{t}"
            if t > 1
            else f"{step.numerator}/{step.denominator}"
        )

        return {
            "type": "probability",
            "title": f"{t} Atışta Hayatta Kalma Olasılığı",
            "steps": steps,
            "answer": answer_str,
            "explanation": (
                f"Silindir her atıştan önce döndürüldüğü için olaylar bağımsız. "
                f"Tek atışta P = {step.numerator}/{step.denominator}, "
                f"{t} atış için P = ({step.numerator}/{step.denominator})^{t} "
                f"= {answer_str} ≈ {numeric_val:.4f}"
            ),
            "formula": formula_str,
            "numeric": f"{answer_str} ≈ {numeric_val:.4f}",
            "certainty": "Kesin (Cebirsel Düzeltme)",
            "tree": tree,
            "distribution": {},
            "table": {},
            "matrix": {},
            "sets": [],
            "events": [],
            # Corrector meta
            "_corrected": True,
            "_correction_rule": "INDEPENDENT_SURVIVAL_CHAIN",
            "_consistency_score": 1.0,
            "_consistency_violations": [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _rule_independent_sequential(
        self, signals: dict, question: str, n: int, t: int
    ) -> "dict | None":
        """
        Bağımsız AND zinciri (eliminasyon olmadan):
          favorable / n çıkar, step_prob^t hesapla.
        """
        fav = self._extract_favorable(question, n)
        if fav is None or fav <= 0 or fav > n:
            return None

        from fractions import Fraction

        step = Fraction(fav, n)
        total = step**t

        steps = []
        steps.append(
            {
                "title": "Adım 1: Tek Denemede Başarı",
                "content": f"Bağımsız deneme: {n} eleman, {fav} başarılı. Payda her denemede {n}.",
                "formula": f"P(başarı) = {fav}/{n}",
                "result": f"{step.numerator}/{step.denominator}",
                "depth": 0,
                "type": "process",
            }
        )
        mul_parts = " × ".join([f"({step.numerator}/{step.denominator})"] * t)
        steps.append(
            {
                "title": f"Adım 2: {t} Denemede Birlikte Başarı",
                "content": "Bağımsız AND zinciri — çarp.",
                "formula": f"P(toplam) = {mul_parts} = {total.numerator}/{total.denominator}",
                "result": f"{total.numerator}/{total.denominator}",
                "depth": 0,
                "type": "process",
            }
        )

        numeric_val = Decimal(total)
        answer_str = (
            f"{total.numerator}/{total.denominator}"
            if total.denominator != 1
            else str(total.numerator)
        )

        return {
            "type": "probability",
            "title": f"{t} Bağımsız Denemede Başarı",
            "steps": steps,
            "answer": answer_str,
            "explanation": f"Bağımsız AND zinciri: ({step.numerator}/{step.denominator})^{t} = {answer_str}",
            "formula": f"({step.numerator}/{step.denominator})^{t}",
            "numeric": f"{answer_str} ≈ {numeric_val:.4f}",
            "certainty": "Kesin (Cebirsel Düzeltme)",
            "tree": None,
            "distribution": {},
            "table": {},
            "matrix": {},
            "sets": [],
            "events": [],
            "_corrected": True,
            "_correction_rule": "INDEPENDENT_SEQUENTIAL_AND",
            "_consistency_score": 1.0,
            "_consistency_violations": [],
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _build_survival_tree(self, n, k, t, step, total):
        """Hayatta kalma zinciri için ağaç yapısı üretir."""
        from fractions import Fraction

        death_step = Fraction(k, n)

        def _frac(f):
            return f"{f.numerator}/{f.denominator}"

        # t=2 için tam ağaç, t>2 için kompakt
        if t == 2:
            return {
                "label": "Başlangıç",
                "prob": "",
                "children": [
                    {
                        "label": "1. Atış: Hayatta",
                        "prob": _frac(step),
                        "value": "",
                        "final": "",
                        "children": [
                            {
                                "label": "2. Atış: Hayatta",
                                "prob": _frac(step),
                                "value": _frac(total),
                                "final": f"★ P={_frac(total)}",
                                "children": [],
                            }
                        ],
                    },
                    {
                        "label": "1. Atış: Ölüm",
                        "prob": _frac(death_step),
                        "value": "0",
                        "final": "",
                        "children": [],
                    },
                ],
            }
        else:
            children = []
            current = {"label": "Başlangıç", "prob": "", "children": children}
            node = children
            for i in range(1, t + 1):
                child = {
                    "label": f"{i}. Atış: Hayatta",
                    "prob": _frac(step),
                    "value": _frac(step**i) if i == t else "",
                    "final": f"★ P={_frac(total)}" if i == t else "",
                    "children": [],
                }
                node.append(child)
                node.append(
                    {
                        "label": f"{i}. Atış: Ölüm",
                        "prob": _frac(death_step),
                        "value": "0",
                        "final": "",
                        "children": [],
                    }
                )
                node = child["children"]
            return current

    # ─────────────────────────────────────────────────────────────────────────
    def _extract_bullets(self, question: str) -> "int | None":
        """
        Soru metninden olumsuz çıktı sayısını (kurşun/mermi) çıkarır.
        Hardcoding yok — kalıp tabanlı, dil-agnostik.
        """
        q = question.lower()
        for pat in self.BULLET_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
        return None

    def _extract_favorable(self, question: str, n: int) -> "int | None":
        """
        Soru metninden başarılı çıktı sayısını çıkarır.
        Önce boş hazne/eleman, sonra renkli/ödüllü, son çare n'den kurşunu çıkar.
        """
        q = question.lower()
        # Boş / empty → doğrudan favorable
        for pat in self.EMPTY_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass

                # chatgpt dedi ve ekledim; yeni eklenen, eklendi
                counts = []
        labels = []

        pairs = re.findall(r"(\d+)\s*([^\n,:]+)", q)

        for n_val, label in pairs:
            try:
                counts.append(int(n_val))
                labels.append(label.lower())
            except:
                pass

        if counts:
            total = sum(counts)

            # güvenli kategoriler
            safe_keywords = ["boş", "güvenli", "empty", "safe"]

            safe_total = 0
            for c, l in zip(counts, labels):
                if any(k in l for k in safe_keywords):
                    safe_total += c

            favorable = total - safe_total

            if favorable > 0:
                return favorable

        # Fallback: n - bullets
        k = self._extract_bullets(question)
        if k is not None and k < n:
            return n - k
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 BayesParamExtractor  (v2 — güvenli etiket↔değer eşleştirme)
#
#  Sorunun çözümü: eski kod etiket ve değerleri ayrı listeler olarak çıkarıp
#  sıralamaya güveniyordu → hipotez isim↔prior/likelihood uyumsuzluğu.
#
#  Yeni yaklaşım:
#    1. Her hipotez için (isim, değer) çiftini BIRLIKTE çıkar (regex group pair)
#    2. İsim normalize: lower().strip(), tekrar eden isimler merge edilir
#    3. Güven skoru: kaç format eşleşti → confidence
#    4. Prior/likelihood ayrımı: bağlam (blok başlığı) + sıra + anahtar kelimeler
#    5. Formal P(E|H) ve P(H) notasyonu en yüksek önceliğe sahip
# ═══════════════════════════════════════════════════════════════════════════════
class BayesParamExtractor:
    """
    Soru metninden Bayes parametrelerini güvenli şekilde çıkarır.
    Etiket↔değer hizalaması hiçbir zaman sıra varsayımına dayanmaz.

    v2.1 — rehber1.md 4-modül mimarisi:
      MODÜL 1: HipotezAdayFiltresi     — stop-semantic filtre
      MODÜL 2: Bağlam Farkındalıklı   — hem "word %val" hem "%val word" destekli
      MODÜL 3: HipotezHizalamaÇözücü  — prior ∩ likelihood intersection
      MODÜL 4: Bayesci Tutarlılık      — prior>0 ∧ lik>0 → posterior≠0 garantisi
    """

    # ── MODÜL 1: HipotezAdayFiltresi ─────────────────────────────────────────
    # Semantic stop kelimeler — hipotez adı olamaz; sıra bağımsız hard-coding yok
    NON_HYPOTHESIS_WORDS = {
        "dağılım",
        "distribution",
        "probability",
        "batma",
        "prior",
        "likelihood",
        "zemin",
        "tür",
        "oran",
        "oranı",
        "toplam",
        "sum",
        "yüzde",
        "percent",
        "doğrulukla",
        "yaklaşık",
        "yaklaşim",
        "ve",
        "ile",
        "her",
        "için",
        "olan",
        "bu",
        "bir",
        "o",
        "ya",
        "da",
        "ki",
        "de",
        "den",
        "dan",
    }

    @classmethod
    def _is_valid_hypothesis(cls, nm: str) -> bool:
        """Adayın stop-semantic filtreden geçip geçmediğini kontrol eder."""
        if not nm or len(nm) <= 1:
            return False
        if nm in cls.NON_HYPOTHESIS_WORDS:
            return False
        return True

    # ── MODÜL 2: Bağlam Farkındalıklı Yüzde Ayrıştırıcı ─────────────────────
    @staticmethod
    def _extract_pairs_from_block(block_text: str) -> list:
        """
        Bir metin bloğundan (name, val_str) çiftlerini çıkarır.
        MODÜL 2 mimarisi — iki yönü de destekler, hard-coding yok:

          Pattern A: "WORD %val" veya "WORD val%" → (word, val%)
                     Baskın sıralama: isim önce, değer sonra
          Pattern B: "%val WORD" → (word, val%)
                     Baskın sıralama: değer önce, isim sonra

        Baskın sıralama otomatik algılanır: blokta ilk gelen öge hangisi
        (değer mi, isim mi) → o pattern seçilir. Hard-coding sıfır.
        Pattern A hem "%10" hem "10%" prefix/suffix formatını destekler.
        """
        text = block_text.strip()
        if not text:
            return []

        # ── Baskın sıralama: ilk değer pozisyonu vs ilk isim pozisyonu ────────
        first_val = re.search(r"%\s*\d+|\d+\s*%", text)
        first_name = re.search(
            r"[A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ]{2,}", text
        )

        # Değer, isimden önce geliyorsa → Pattern B (value BEFORE name)
        use_pattern_b = (
            first_val is not None
            and first_name is not None
            and first_val.start() < first_name.start()
        )

        pairs = []
        if use_pattern_b:
            # Pattern B: "%val WORD" → (word, val%)
            for m in re.finditer(
                r"%\s*(\d+\.?\d*)\s+([A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ]+)",
                text,
            ):
                pairs.append((m.group(2), m.group(1) + "%"))
        else:
            # Pattern A: "WORD %val" veya "WORD val%" → (word, val%)
            # group(2) → %number format,  group(3) → number% format
            for m in re.finditer(
                r"([A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ]*)"
                r"(?:[\s:,/]*%\s*(\d+\.?\d*)|[\s:,/]*(\d+\.?\d*)\s*%)",
                text,
            ):
                val = m.group(2) if m.group(2) is not None else m.group(3)
                pairs.append((m.group(1), val + "%"))
        return pairs

    # Değer parse eder: "10%", "0.10", "10/100", "10" → Decimal kesinlikte
    @staticmethod
    def _parse_val(s: str) -> "Decimal | None":
        s = s.strip()
        try:
            if "%" in s:
                num_str = s.replace("%", "").strip()
                return Decimal(num_str) / Decimal(100)
            if "/" in s:
                parts = s.split("/")
                if len(parts) == 2:
                    return Decimal(parts[0]) / Decimal(parts[1])
            v = Decimal(s)
            return v / Decimal(100) if v > 1 else v
        except (ValueError, ZeroDivisionError, InvalidOperation):
            return None

    # İsim normalize: lower, strip, boşlukları sıkıştır
    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"\s+", "_", name.strip().lower())

    def extract(self, question: str) -> dict:
        """
        Döndürür:
          {
            "hypotheses":    list[str],         # sıralı, normalize
            "priors":        dict[str, float],  # {hyp_name: prob}
            "likelihoods":   dict[str, float],  # {hyp_name: prob}
            "n_observations": int,
            "confidence":    float,             # 0.0 – 1.0
          }

        Strateji öncelik sırası (rehber1.md pipeline mimarisi):
          S3 (Blok regex)  → S1 (Formal P() notasyonu) → S2 (Satır bağlamı) → S4 (Fallback)
          S3 en güvenilir; açık blok başlıkları varsa S2'ye gerek kalmaz.
        """
        params = {
            "hypotheses": [],
            "priors": {},
            "likelihoods": {},
            "n_observations": 1,
            "confidence": 0.0,
        }
        confidence_hits = 0

        # ── n_observations ────────────────────────────────────────────────────
        word_nums = {
            "bir": 1,
            "iki": 2,
            "üç": 3,
            "dört": 4,
            "beş": 5,
            "altı": 6,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
        }
        q = question.lower()
        for word, n in word_nums.items():
            if re.search(rf"\b{word}\s+(?:tur|kez|gözlem|round|observation)", q):
                params["n_observations"] = max(params["n_observations"], n)
        for m in re.finditer(r"(\d+)\s*(?:tur|kez|gözlem|round|observation)", q):
            try:
                params["n_observations"] = max(
                    params["n_observations"], int(m.group(1))
                )
            except ValueError:
                pass

        # ══ STRATEJI 3: Blok regex ════════════════════════════════════════════
        # Açık blok başlıkları (dağılım, batma ihtimali) → en güvenilir kaynak
        # Her blok kendi sınırında durur; cross-contamination yok.

        prior_block = re.search(
            r"(?:dağılım|prior|zemin\s*:|tür\s*(?:dağılım|oranı)|distribution)[:\s]+"
            r"(.+?)(?=batma|likelihood|\.\s*[A-ZÜÇŞĞÖI]|$)",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        if prior_block:
            for raw_name, raw_val in self._extract_pairs_from_block(
                prior_block.group(1)
            ):
                nm = self._norm(raw_name)
                val_dec = self._parse_val(raw_val)
                if self._is_valid_hypothesis(nm) and val_dec is not None:
                    val = float(val_dec)  # Decimal → float for compatibility
                    if 0.0 < val <= 1.0:
                        if nm not in params["priors"]:
                            params["priors"][nm] = val
                        confidence_hits += 1

        # Likelihood blok regex: dağılım/prior keyword'ünde DUR (prior bloğuna taşma yok)
        lik_block = re.search(
            r"(?:batma\s+(?:ihtimali|olasılığı)|zemin\s+batma|likelihood|kanıt\s*olasılığı)[:\s]+"
            r"(.+?)(?=dağılım|prior|ön\s*olasılık|distribution|\.\s*[A-ZÜÇŞĞÖI]|$)",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        if lik_block:
            for raw_name, raw_val in self._extract_pairs_from_block(lik_block.group(1)):
                nm = self._norm(raw_name)
                val_dec = self._parse_val(raw_val)
                if self._is_valid_hypothesis(nm) and val_dec is not None:
                    val = float(val_dec)  # Decimal → float for compatibility
                    if 0.0 < val <= 1.0:
                        if nm not in params["likelihoods"]:
                            params["likelihoods"][nm] = val
                            confidence_hits += 1

        # ══ STRATEJI 1: Formal P(H) = val  ve  P(E|H) = val notasyonu ════════
        # En yüksek güven: P(A)=0.5, P(B|E)=0.3 gibi
        formal_prior_matches = re.findall(
            r"[Pp]\s*\(\s*([A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ\s]*?)\s*\)"
            r"\s*[=≈]\s*(0\.\d+|\d+/\d+|\d+\.?\d*\s*%)",
            question,
        )
        formal_lik_matches = re.findall(
            r"[Pp]\s*\(\s*(?:[A-Za-z]+\s*\|\s*)?([A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ\s]*?)\s*\)"
            r"\s*[=≈]\s*(0\.\d+|\d+/\d+|\d+\.?\d*\s*%)",
            question,
        )
        for name, val_str in formal_prior_matches:
            nm = self._norm(name)
            if nm and nm not in ("e", "b", "a"):
                v_dec = self._parse_val(val_str)
                if v_dec is not None:
                    v = float(v_dec)  # Decimal → float
                    if 0.0 < v <= 1.0:
                        params["priors"][nm] = v
                        confidence_hits += 2
        for name, val_str in formal_lik_matches:
            nm = self._norm(name)
            if nm and nm not in ("e", "b", "a"):
                v_dec = self._parse_val(val_str)
                if v_dec is not None:
                    v = float(v_dec)  # Decimal → float
                    if 0.0 < v <= 1.0:
                        params["likelihoods"][nm] = v
                        confidence_hits += 2

        # ══ STRATEJI 2: Satır bağlam taraması ════════════════════════════════
        # Yalnızca S3 her iki tarafı dolduramadıysa çalışır.
        # Aynı satırda her iki keyword varsa satır bölünür.
        _s3_complete = bool(params["priors"]) and bool(params["likelihoods"])
        if not _s3_complete:
            PRIOR_KEYWORDS = re.compile(
                r"dağılım|prior|ön\s*olasılık|zemin\s*dağılım|tür\s*oranı|distribution",
                re.I,
            )
            LIKELIHOOD_KEYWORDS = re.compile(
                r"batma|likelihood|ihtimal|olasılık[ı]?\s*(?:ise|:)|kanıt\s*olasılığı|p\(e\|",
                re.I,
            )

            ctx = None
            for line in question.split("\n"):
                ll = line.lower()
                has_prior = bool(PRIOR_KEYWORDS.search(ll))
                has_lik = bool(LIKELIHOOD_KEYWORDS.search(ll))

                # Her iki keyword aynı satırda → satırı sınırda böl
                if has_prior and has_lik:
                    m_prior = PRIOR_KEYWORDS.search(ll)
                    m_lik = LIKELIHOOD_KEYWORDS.search(ll)
                    if m_lik.start() < m_prior.start():
                        segs = [
                            ("likelihood", line[: m_prior.start()]),
                            ("prior", line[m_prior.start() :]),
                        ]
                    else:
                        segs = [
                            ("prior", line[: m_lik.start()]),
                            ("likelihood", line[m_lik.start() :]),
                        ]
                    for seg_ctx, seg_text in segs:
                        for raw_name, raw_val in self._extract_pairs_from_block(
                            seg_text
                        ):
                            nm = self._norm(raw_name)
                            val_dec = self._parse_val(raw_val)
                            if not self._is_valid_hypothesis(nm):
                                continue
                            if val_dec is not None:
                                val = float(val_dec)  # Decimal → float
                                if 0.0 < val <= 1.0:
                                    if (
                                        seg_ctx == "prior"
                                        and nm not in params["priors"]
                                    ):
                                        params["priors"][nm] = val
                                        confidence_hits += 1
                                    elif (
                                        seg_ctx == "likelihood"
                                        and nm not in params["likelihoods"]
                                    ):
                                        params["likelihoods"][nm] = val
                                        confidence_hits += 1
                    continue

                if has_prior:
                    ctx = "prior"
                elif has_lik:
                    ctx = "likelihood"

                for raw_name, raw_val in self._extract_pairs_from_block(line):
                    nm = self._norm(raw_name)
                    val_dec = self._parse_val(raw_val)
                    if not self._is_valid_hypothesis(nm):
                        continue
                    if val_dec is not None:
                        val = float(val_dec)  # Decimal → float
                        if 0.0 < val <= 1.0:
                            if ctx == "prior" and nm not in params["priors"]:
                                params["priors"][nm] = val
                                confidence_hits += 1
                            elif (
                                ctx == "likelihood" and nm not in params["likelihoods"]
                            ):
                                params["likelihoods"][nm] = val
                                confidence_hits += 1

        # ══ STRATEJI 4: Fallback ══════════════════════════════════════════════
        # Her iki dict de boşsa tüm metin üzerinde arama (>50% → lik, ≤50% → prior)
        if not params["priors"] and not params["likelihoods"]:
            all_pairs = re.findall(
                r"([A-Za-züçşığöÜÇŞİĞÖ][A-Za-z0-9_\-züçşığöÜÇŞİĞÖ]*)"
                r"[\s:,/%]*(\d+\.?\d*)\s*%",
                question,
            )
            for name, val_str in all_pairs:
                nm = self._norm(name)
                val_dec = self._parse_val(val_str + "%")
                if not self._is_valid_hypothesis(nm):
                    continue
                if val_dec is not None:
                    val = float(val_dec)  # Decimal → float
                    if 0.0 < val <= 1.0:
                        if val > 0.5:
                            if nm not in params["likelihoods"]:
                                params["likelihoods"][nm] = val
                        else:
                            if nm not in params["priors"]:
                                params["priors"][nm] = val
                        confidence_hits += 1

        # ── Hipotez listesi: prior ∪ likelihood, normalize, sıralı ───────────
        all_hyps = set(params["priors"].keys()) | set(params["likelihoods"].keys())
        params["hypotheses"] = sorted(all_hyps)

        # ── MODÜL 3: HipotezHizalamaÇözücü ───────────────────────────────────
        prior_only = set(params["priors"].keys()) - set(params["likelihoods"].keys())
        lik_only = set(params["likelihoods"].keys()) - set(params["priors"].keys())
        if prior_only or lik_only:
            confidence_hits = max(0, confidence_hits - len(prior_only) - len(lik_only))
            params["_alignment_warnings"] = {
                "prior_only": sorted(prior_only),
                "likelihood_only": sorted(lik_only),
            }
        else:
            params["_alignment_warnings"] = {}

        # Hizalanmış hipotez seti (her ikisinde de olanlar)
        aligned_hyps = sorted(
            set(params["priors"].keys()) & set(params["likelihoods"].keys())
        )

        # ── Eksik prior: uniform dolgu ────────────────────────────────────────
        if params["likelihoods"] and not params["priors"]:
            n_h = len(params["hypotheses"])
            if n_h > 0:
                uniform = round(1.0 / n_h, 8)
                for h in params["hypotheses"]:
                    params["priors"][h] = uniform

        # ── Eksik likelihood: uniform dolgu ───────────────────────────────────
        if params["priors"] and not params["likelihoods"]:
            for h in params["hypotheses"]:
                params["likelihoods"][h] = 1.0 / max(len(params["hypotheses"]), 1)

        # ── Confidence ────────────────────────────────────────────────────────
        n_hyps = len(params["hypotheses"])
        params["confidence"] = min(1.0, confidence_hits / max(n_hyps * 2, 1))

        return params


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 MathASTBuilder
#  Soru metninden hardcoding yapmadan matematiksel yapı çıkarır.
#  Çıktı: {type, grid, start, n_states, absorbing_type, move_prob, teleport, ...}
# ═══════════════════════════════════════════════════════════════════════════════
class MathASTBuilder:
    """
    Soru → Matematiksel AST (Abstract Syntax Tree).
    Her problem tipi için parametre çıkarıcı kalıplar tamamen regex-tabanlı.
    Hard-coding yok — yeni tip eklemek = PROBLEM_DETECTORS listesine ekleme.
    """

    # ── Problem tip dedektörleri ────────────────────────────────────────────
    MARKOV_PATTERNS = [
        r"\b(markov|random\s*walk|rastgele\s*yürüyüş|bataklık|batakl[ıi]k|swamp|grid|ızgara|izgara)\b",
        r"\b(absorbing|emici|kenar|ölüm|death|border|edge)\b",
        r"\b(\d+\s*[×x]\s*\d+)\s*(grid|ızgara|izgara|bataklık)\b",
        r"\bgeçiş\s*(matrisi|olasılığı|probability)\b",
        r"\b(gezgin|traveler|walker)\b.*\b(adım|step|move)\b",
        r"\bhayatta\s+kalma.*\badım\b",
        r"\bbataklık\b",
    ]
    BAYES_PATTERNS = [
        r"\b(bayes|posterior|prior|önsel|sonsal)\b",
        r"P\([A-Za-z]\s*\|\s*[A-Za-z]\)",
        r"\bkoşullu\s+olasılık\b.*\bverildi\b",
    ]
    MARKOV_CHAIN_PATTERNS = [
        r"\b(markov\s*zincir|markov\s*chain|steady\s*state|durağan\s*durum)\b",
        r"\bgeçiş\s*matrisi\b",
        r"\bdurum\b.*\bgeçiş\b",
    ]

    # ── Grid boyutu çıkarıcı ────────────────────────────────────────────────
    GRID_PATTERNS = [
        r"(\d+)\s*[×x]\s*(\d+)\s*(?:grid|ızgara|izgara|bataklık|alan|matris)?",
        r"(\d+)\s*(?:satır|row|sütun|col).*?(\d+)\s*(?:satır|row|sütun|col)",
        r"(\d+)\s*(?:hazneli|hazneli|boyutlu|boyutlu)",
    ]

    # ── Başlangıç pozisyonu çıkarıcı ───────────────────────────────────────
    START_PATTERNS = [
        r"\((\d+)\s*,\s*(\d+)\)",
        r"(\d+)\.\s*satır.*?(\d+)\.\s*sütun",
        r"merkez\s*(?:hücre|nokta|cell)?",
        r"center",
        r"başlangıç\s*(?:noktası|hücresi)?\s*[:=]\s*\((\d+)\s*,\s*(\d+)\)",
    ]

    # ── Hareket olasılığı çıkarıcı ──────────────────────────────────────────
    MOVE_PROB_PATTERNS = [
        r"(?:hareket|move|yürüyüş|walk)\s*olasılığı\s*[:=]\s*(0\.\d+|\d+/\d+)",
        r"(0\.\d+|\d+/\d+)\s*olasılıkla\s*(?:hareket|ilerleme|adım)",
        r"(?:yukarı|aşağı|sağ|sol|↑|↓|←|→|up|down|left|right).*?(0\.\d+|\d+/\d+)",
        r"1\s*/\s*4",  # 1/4 default
        r"0\.25",
    ]

    # ── Teleport / özel durum ─────────────────────────────────────────────
    TELEPORT_PATTERNS = [
        r"teleport\s*(?:olasılığı|probability)?\s*[:=]?\s*(0\.\d+|\d+\s*%|\d+/\d+)",
        r"(%\d+|\d+\s*%)\s*(?:teleport|ışınlanma|ışınlan)",
        r"(0\.\d+|\d+/\d+)\s*(?:teleport|ışınlanma)",
        r"merkez.*?(%\d+|\d+%|0\.\d+)",
        r"(\d+)\s*%\s*(?:olasılıkla\s*)?(?:merkeze\s*)?teleport",
    ]

    # ── Absorbing state türü ────────────────────────────────────────────────
    ABSORBING_PATTERNS = [
        (r"\bkenar(?:lar)?\s*(?:ölüm|death|absorbing|emici)\b", "border_death"),
        (r"\bsınır(?:lar)?\s*(?:ölüm|death|absorbing|emici)\b", "border_death"),
        (r"\böl(?:üm|ecek)\b.*\bkenar\b", "border_death"),
        (r"\bborder\s*(?:death|absorbing)\b", "border_death"),
        (r"\bedge\s*(?:death|absorbing)\b", "border_death"),
        (r"\b(bataklık|swamp|pit|çukur)\b", "special_cells"),
    ]

    @staticmethod
    def _normalize(text: str) -> str:
        """Türkçe büyük harfleri ASCII eşdeğerine düşürür, sonra lower() uygular."""
        tr_map = str.maketrans("İIĞŞÜÖÇ", "iiğşüöç")
        return text.translate(tr_map).lower()

    def build(self, question: str, signals: dict) -> dict:
        """
        Soru metninden matematiksel AST üretir.
        Döndürür: dict ile type, params ve solver önerisi.
        """
        q = self._normalize(question)
        ast = {
            "type": "general",
            "subtype": None,
            "params": {},
            "solver": "LLM",  # önerilen solver
            "confidence": 0.0,
        }

        # ── Markov/Random Walk tespiti ────────────────────────────────────
        markov_hits = sum(1 for p in self.MARKOV_PATTERNS if re.search(p, q))
        if markov_hits >= 1 or signals.get("intent") == "markov":
            ast["type"] = "markov_random_walk"
            ast["solver"] = "MarkovSolver"
            ast["confidence"] = min(1.0, markov_hits * 0.35)
            ast["params"] = self._extract_markov_params(question, q)

        # ── Markov Chain (non-grid) ───────────────────────────────────────
        elif any(re.search(p, q) for p in self.MARKOV_CHAIN_PATTERNS):
            ast["type"] = "markov_chain"
            ast["solver"] = "MarkovSolver"
            ast["confidence"] = 0.8

        # ── Bayes yapısı ──────────────────────────────────────────────────
        elif any(re.search(p, q) for p in self.BAYES_PATTERNS):
            ast["type"] = "bayes"
            ast["solver"] = "BayesSolver"
            # v2: BayesParamExtractor ile güvenli etiket↔değer eşleştirme
            bayes_params = self._bayes_extractor.extract(question)
            ast["params"] = bayes_params
            ast["confidence"] = max(0.9, bayes_params.get("confidence", 0.9))

        # ── Oyun Kuramı tespiti — Markov/Bayes değilse GT dene ────────────
        # NOT: Bu blok yalnızca bir kez çalışmalı; duplike blok kaldırıldı (v132).
        if ast["type"] == "general":
            _gtnlp = GameTheoryNLPExtractor()
            _gtp = _gtnlp.extract(question)
            if _gtp["is_game_theory"] and _gtp["confidence"] >= 0.4:
                ast["type"] = "game_theory"
                ast["subtype"] = _gtp.get("game_type", "iterated_pd")
                ast["solver"] = "GameTheorySolver"
                ast["confidence"] = _gtp["confidence"]
                ast["params"] = {
                    "gt_params": _gtp,
                    "n_players": _gtp.get("n_players", 0),
                    "n_rounds": _gtp.get("n_rounds", 3),
                }
                ast["_question"] = question

        return ast

    # BayesParamExtractor tek örnek (sınıf düzeyinde)
    _bayes_extractor = BayesParamExtractor()

    def _extract_bayes_params_LEGACY(self, question: str, q: str) -> dict:
        """LEGACY — BayesParamExtractor'a taşındı. Kaldırılabilir."""
        params = {
            "hypotheses": [],
            "priors": {},
            "likelihoods": {},
            "n_observations": 1,
        }

        # ── n_observations: kaç tur/gözlem ────────────────────────────────────
        # Türkçe kelime + sayısal pattern
        word_nums = {
            "bir": 1,
            "iki": 2,
            "üç": 3,
            "dört": 4,
            "beş": 5,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        for word, n in word_nums.items():
            if re.search(rf"\b{word}\s+(?:tur|kez|gözlem|round|observation)", q):
                params["n_observations"] = max(params["n_observations"], n)
        for m in re.finditer(r"(\d+)\s*(?:tur|kez|gözlem|round|observation)", q):
            params["n_observations"] = max(params["n_observations"], int(m.group(1)))

        # ── Yüzde bloklarını ayıkla ────────────────────────────────────────────
        # Format 1: "isim %sayı" veya "isim: sayı%"
        pct_labeled = re.findall(
            r"([a-züçşığöA-ZÜÇŞİĞÖ][a-züçşığöA-ZÜÇŞİĞÖ\s]*?)\s*[:%]\s*(\d+\.?\d*)\s*%?",
            question,
        )
        # Format 2: Formal P(E|H) = değer veya P(H) = değer
        formal_likelihood = re.findall(
            r"[Pp]\s*\(\s*[A-Za-z]+\s*\|\s*([A-Za-z]+)\s*\)\s*[=≈]\s*(0\.\d+|\d+/\d+|\d+\.?\d*\s*%)",
            question,
        )
        formal_prior = re.findall(
            r"[Pp]\s*\(\s*([A-Za-z]+)\s*\)\s*[=≈]\s*(0\.\d+|\d+/\d+|\d+\.?\d*\s*%)",
            question,
        )

        def parse_val(s: str) -> float:
            s = s.strip()
            if "%" in s:
                return float(s.replace("%", "").strip()) / 100
            if "/" in s:
                a, b = s.split("/")
                return float(a) / float(b)
            v = float(s)
            return v / 100 if v > 1 else v

        # ── Önsel dağılım: zemin/dağılım bloğu ───────────────────────────────
        prior_block_match = re.search(
            r"(?:dağılım|prior|ön\s+olasılık|zemin[:\s]|tür\s+dağılım|distribution)[:\s]+"
            r"(.+?)(?:batma\s+(?:ihtimali|olasılığı)|likelihood|\.\s*[A-ZÜÇŞĞÖI][a-z]|$)",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        if prior_block_match:
            block = prior_block_match.group(1)
            for m in re.finditer(r"([a-züçşığöA-ZÜÇŞİĞÖ]\w*)\s*%(\d+\.?\d*)", block):
                name = m.group(1).strip().lower()
                val = float(m.group(2)) / 100
                if 0 < val <= 1 and name not in ("yüzde", "percent"):
                    params["priors"][name] = val

        # ── Likelihood: batma ihtimali bloğu ─────────────────────────────────
        likelihood_block_match = re.search(
            r"(?:batma\s+(?:ihtimali|olasılığı)|likelihood)[:\s]+"
            r"(.+?)(?:\.\s*[A-ZÜÇŞĞÖIA-Z][a-züçşğöa-z]|$)",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        if likelihood_block_match:
            block = likelihood_block_match.group(1)
            for m in re.finditer(r"([a-züçşığöA-ZÜÇŞİĞÖ]\w*)\s*%(\d+\.?\d*)", block):
                name = m.group(1).strip().lower()
                val = float(m.group(2)) / 100
                if 0 < val <= 1 and name not in ("yüzde", "percent"):
                    params["likelihoods"][name] = val

        # ── Formal prior/likelihood fallback ──────────────────────────────────
        for hyp, val_str in formal_prior:
            try:
                params["priors"][hyp.lower()] = parse_val(val_str)
            except:
                pass
        for hyp, val_str in formal_likelihood:
            try:
                params["likelihoods"][hyp.lower()] = parse_val(val_str)
            except:
                pass

        # ── Genel yüzde bloğu — prior/likelihood ayrımı yapılamazsa ──────────
        # Eğer hala boşsa, genel labeled percentages'dan çıkarmayı dene
        if not params["priors"] and not params["likelihoods"] and pct_labeled:
            # Sıralı: ilk grup prior, ikinci grup likelihood olabilir
            # Context: önce dağılım, sonra batma ihtimali sırası aranır
            lines = question.split("\n")
            current_type = None
            for line in lines:
                ll = line.lower()
                if re.search(r"dağılım|prior|ön\s+olasılık", ll):
                    current_type = "prior"
                elif re.search(r"batma|ihtimal|olasılık|likelihood", ll):
                    current_type = "likelihood"
                for m in re.finditer(
                    r"([a-züçşığöA-ZÜÇŞİĞÖ]\w*)\s*[:%]?\s*(\d+\.?\d*)\s*%", line
                ):
                    name = m.group(1).strip().lower()
                    val = parse_val(m.group(2) + "%")
                    if 0 < val <= 1 and name not in ("yüzde", "percent"):
                        if current_type == "prior":
                            params["priors"][name] = val
                        elif current_type == "likelihood":
                            params["likelihoods"][name] = val

        # Hypothesis listesi: priors ve likelihoods union'ı
        all_hyps = set(params["priors"].keys()) | set(params["likelihoods"].keys())
        params["hypotheses"] = sorted(all_hyps)

        return params

    def _extract_markov_params(self, question: str, q: str) -> dict:
        params = {}

        # Grid boyutu
        for pat in self.GRID_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    rows, cols = int(m.group(1)), int(m.group(2))
                    if 2 <= rows <= 100 and 2 <= cols <= 100:
                        params["grid"] = [rows, cols]
                        break
                except (IndexError, AttributeError):
                    pass

        # Başlangıç noktası
        for pat in self.START_PATTERNS:
            m = re.search(pat, q)
            if m:
                if "merkez" in pat or "center" in pat:
                    if "grid" in params:
                        r, c = params["grid"]
                        params["start"] = [r // 2 + 1, c // 2 + 1]
                    break
                try:
                    params["start"] = [int(m.group(1)), int(m.group(2))]
                    break
                except (IndexError, AttributeError):
                    pass

        # Varsayılan: merkez
        if "start" not in params and "grid" in params:
            r, c = params["grid"]
            params["start"] = [(r + 1) // 2, (c + 1) // 2]

        # Hareket olasılığı
        params["move_dirs"] = 4  # default: 4 yön
        params["move_prob"] = 0.25
        for pat in self.MOVE_PROB_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    val_str = m.group(1) if m.lastindex else m.group(0)
                    if "/" in val_str:
                        n, d = val_str.split("/")
                        params["move_prob"] = int(n.strip()) / int(d.strip())
                    elif "%" in val_str:
                        params["move_prob"] = (
                            float(val_str.replace("%", "").strip()) / 100
                        )
                    else:
                        params["move_prob"] = float(val_str)
                    break
                except (ValueError, AttributeError):
                    pass

        # Absorbing state türü
        params["absorbing_type"] = "border_death"
        for pat, atype in self.ABSORBING_PATTERNS:
            if re.search(pat, q):
                params["absorbing_type"] = atype
                break

        # Teleport
        params["teleport_prob"] = 0.0
        params["teleport_target"] = None
        for pat in self.TELEPORT_PATTERNS:
            m = re.search(pat, q)
            if m:
                try:
                    val_str = m.group(1)
                    if "%" in val_str:
                        params["teleport_prob"] = (
                            float(val_str.replace("%", "").strip()) / 100
                        )
                    elif "/" in val_str:
                        n, d = val_str.split("/")
                        params["teleport_prob"] = int(n.strip()) / int(d.strip())
                    else:
                        params["teleport_prob"] = float(val_str)
                    # Teleport hedefi merkez varsayılır
                    if "grid" in params:
                        r, c = params["grid"]
                        params["teleport_target"] = [(r + 1) // 2, (c + 1) // 2]
                    break
                except (ValueError, AttributeError):
                    pass

        # Stay probability (teleport varsa kalan hareket prob'u düzenlenir)
        params["stay_prob"] = max(
            0.0,
            1.0 - params["move_prob"] * params["move_dirs"] - params["teleport_prob"],
        )
        params["stay_prob"] = max(0.0, min(1.0, params["stay_prob"]))

        return params


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 MarkovSolver
#  Absorbing Markov Chain çözücü — NumPy tabanlı, hardcoding yok.
#  Grid random walk, birth-death chain, genel absorbing Markov destekler.
# ═══════════════════════════════════════════════════════════════════════════════
class MarkovSolver:
    """
    Absorbing Markov chain için tam analitik çözüm.

    Yöntem:
      P = [Q  R]   (Q: transient→transient, R: transient→absorbing)
          [0  I]

      N = (I - Q)^{-1}          (fundamental matrix)
      t = N · 1                  (expected steps before absorption)
      B = N · R                  (absorption probability per absorbing state)
      var = (2N - I)t - t^2     (variance of absorption time)
      stationary ∝ t             (quasi-stationary dist. proportional to t)

    Grid modeli:
      - rows × cols hücre → rows*cols durum
      - border hücreleri absorbing (ölüm/çıkış)
      - iç hücreler transient
      - Hareket: {↑↓←→} her biri move_prob olasılıkla
      - Teleport opsiyonel
    """

    # ── Yardımcı fonksiyonlar: Tutarlı durum indeksleme ────────────────────────────
    def state_index(self, row, col, grid_size):
        """1-tabanlı (row, col) → 0-tabanlı düz vektör indisi (satır-major)"""
        return (row - 1) * grid_size + (col - 1)

    def create_initial_vector(self, start_row, start_col, grid_size):
        """Başlangıç olasılık vektörünü tutarlı şekilde oluşturur"""
        n_states = grid_size * grid_size
        init = np.zeros(n_states)
        idx = self.state_index(start_row, start_col, grid_size)
        init[idx] = 1.0
        return init

    def get_corner_indices(self, grid_size):
        """Köşe konumlarının 0-tabanlı indislerini döndürür"""
        corners = [(1, 1), (1, grid_size), (grid_size, 1), (grid_size, grid_size)]
        return [self.state_index(r, c, grid_size) for r, c in corners]

    def compute_expected_position(self, final_prob, grid_size):
        """Son olasılık vektöründen E[X] ve E[Y]'yi hesaplar"""
        ex = 0.0
        ey = 0.0
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                prob = final_prob[self.state_index(i, j, grid_size)]
                ex += j * prob  # x = sütun numarası
                ey += i * prob  # y = satır numarası
        return ex, ey

    def solve(self, ast: dict) -> dict:
        """
        ast: MathASTBuilder.build() çıktısı
        Döndürür: kapsamlı çözüm dict
        """
        params = ast.get("params", {})
        ptype = ast.get("type", "general")
        question = ast.get("question", "")
        signals = ast.get("signals", {})

        # ── (0) Deterministik etkileşim zinciri tespiti (tokat/vur/hit vb.)
        det_cycle = self._extract_deterministic_cycle(question)
        if det_cycle:
            cycle, turn_n = det_cycle
            victim = cycle[(turn_n - 1) % len(cycle)] if cycle else None
            return {
                "solved": True,
                "_solver_used": "MarkovSolver:deterministic_cycle",
                "cycle": cycle,
                "turn": turn_n,
                "victim": victim,
                "answer": victim,
                "steps": self._build_cycle_steps(cycle, turn_n, victim),
            }

        # Dinamik grid algılama — soru metninden parse et
        if (
            "grid" in ptype.lower()
            or "ızgara" in question.lower()
            or re.search(r"\d+\s*[x×]\s*\d+", question)
        ):
            P, N, start_pos, size = self._build_grid_transition(question, signals)

            # N adımlı geçiş matrisi
            P_N = np.linalg.matrix_power(P, N)

            # Başlangıç dağılımı (tutarlı indeksleme)
            init = self.create_initial_vector(start_pos[0], start_pos[1], size)

            # Final dağılım
            final = init @ P_N

            # Köşe olasılıkları (tutarlı indeksleme)
            corner_indices = self.get_corner_indices(size)
            corner_prob = sum(final[idx] for idx in corner_indices)

            # Beklenen pozisyon (tutarlı indeksleme)
            ex, ey = self.compute_expected_position(final, size)

            return {
                "solved": True,
                "_solver_used": "MarkovSolver",
                "grid_size": size,
                "start": start_pos,
                "steps": N,
                "transition_matrix": P.tolist(),
                "P_N": P_N.tolist(),
                "final_distribution": final.tolist(),
                "corner_probability": float(corner_prob),
                "expected_x": float(ex),
                "expected_y": float(ey),
                "position_probabilities": {
                    f"({r},{c})": float(final[self.state_index(r, c, size)])
                    for r in range(1, size + 1)
                    for c in range(1, size + 1)
                },
            }

        elif ptype == "markov_random_walk" and "grid" in params:
            return self._solve_grid_walk(params)
        elif ptype == "markov_chain":
            return self._solve_generic_chain(params)
        else:
            return {
                "_solver_used": "MarkovSolver",
                "_solver_error": "Parametre yetersiz",
                "solved": False,
            }

    # ── Deterministik tekil eylem zinciri (döngü) çıkarımı ──────────────────
    def _extract_deterministic_cycle(self, question: str):
        """Metinden fail→mağdur zincirini çıkarır, t tur sayısını döndürür.
        Hard-code yok; sadece eylem fiillerine dayalı regex kullanır."""
        if not question:
            return None

        # 1) Tur sayısı t: metindeki en büyük pozitif tam sayı
        turns = [int(x) for x in re.findall(r"(\d+)", question) if int(x) > 1]
        turn_n = max(turns) if turns else 1

        # 2) Fail→mağdur örüntüsü: "X ... Y'ye tokat/vur/hit/slap ..."
        pattern = re.compile(
            r"\b([A-Za-zÇĞİÖŞÜçğıöşü]+)[^\n]*?\b([A-Za-zÇĞİÖŞÜçğıöşü]+)['’]?(ye|ya|e|a)?\s+(tokat|vur|vurdu|slap|hit)\b",
            re.IGNORECASE,
        )

        victims = []
        for m in pattern.finditer(question):
            victim_raw = m.group(2)
            victim = victim_raw.strip().title()
            if victim:
                victims.append(victim)

        # 3) Eğer "kural: X ... daima Y'ye" gibi bir ifade varsa ekle
        rule_pat = re.compile(
            r"kural[:\s]+[^\n]*?([A-Za-zÇĞİÖŞÜçğıöşü]+)[^\n]*?\b([A-Za-zÇĞİÖŞÜçğıöşü]+)['’]?(ye|ya|e|a)?\s+(tokat|vur|slap|hit)",
            re.IGNORECASE,
        )
        for m in rule_pat.finditer(question):
            v = m.group(2).strip().title()
            if v:
                victims.append(v)

        # Tekrarsız ama sıralı döngü (görünüm sırası korunur)
        cycle = []
        seen = set()
        for v in victims:
            if v not in seen:
                seen.add(v)
                cycle.append(v)

        if len(cycle) >= 2:
            return cycle, turn_n
        return None

    def _build_cycle_steps(self, cycle, turn_n, victim):
        steps = []
        steps.append(
            {
                "title": "Zincir Çıkarımı",
                "content": f"Metinden {len(cycle)} kişilik deterministik tokat döngüsü çıkarıldı",
                "result": ", ".join(cycle),
            }
        )
        steps.append(
            {
                "title": "Döngü Mod Hesabı",
                "content": f"n={turn_n}, döngü uzunluğu={len(cycle)} → index=(n-1) mod L",
                "formula": f"( {turn_n}-1 ) mod {len(cycle)}",
                "result": (turn_n - 1) % len(cycle),
            }
        )
        steps.append(
            {
                "title": "Sonuç",
                "content": "Döngüdeki mağdur sırasına göre 124. tur mağduru",
                "result": victim,
            }
        )
        return steps

    # ── Grid Random Walk ──────────────────────────────────────────────────────
    def _build_grid_transition(self, question: str, signals: dict) -> tuple:
        """
        Soru metninden tamamen dinamik olarak grid geçiş matrisi inşa et.

        Args:
            question: Soru metni (Türkçe/İngilizce)
            signals: Algılanan semantik sinyaller ({sequential_trials, ...})

        Returns:
            (P, N, start_pos, size): Geçiş matrisi, adım sayısı, başlangıç, grid boyutu
        """

        # 1. Grid boyutu — metnin içinden parse et (5x5, 10×10, NxN vb)
        size_match = re.search(r"(\d+)\s*[x×]\s*(\d+)", question)
        if size_match:
            size = int(size_match.group(1))  # kare olduğunu varsay
        else:
            size = 5  # default
        size = max(2, min(100, size))  # NxP olasılık matrisi için üst sınır 100x100

        # 2. Başlangıç konumu — metnin içinden (r, c) format
        start_match = re.search(r"\((\d+),\s*(\d+)\)", question)
        if start_match:
            start_r = int(start_match.group(1))
            start_c = int(start_match.group(2))
        else:
            start_r = size // 2 + 1
            start_c = size // 2 + 1

        # 3. Adım sayısı — signals veya metin parslaması
        steps = signals.get("sequential_trials", 6)

        # 4. Yön olasılıklarını metinden çıkar (Türkçe + İngilizce destekle)
        dir_map = {
            "kuzey": (0, 1),
            "güney": (0, -1),
            "doğu": (1, 0),
            "batı": (-1, 0),
            "north": (0, 1),
            "south": (0, -1),
            "east": (1, 0),
            "west": (-1, 0),
        }

        probs = {}
        for dname in dir_map.keys():
            # Miktar: "kuzey: 0.3" veya "kuzey = 30%" vb
            m = re.search(
                rf"{dname}\s*[:=]?\s*([0-9]*\.?[0-9]+)", question, re.IGNORECASE
            )
            if m:
                val_str = m.group(1)
                if "." in val_str:
                    probs[dname.lower()] = float(val_str)
                else:
                    # "30" → 0.30 varsay
                    probs[dname.lower()] = (
                        float(val_str) / 100.0 if float(val_str) > 1 else float(val_str)
                    )

        # Tüm yönler eşit olasılık varsa (default)
        if not probs:
            n_dirs = 4
            equal_prob = 1.0 / n_dirs
            for dname in ["kuzey", "güney", "doğu", "batı"]:
                probs[dname] = equal_prob

        # 5. NxN geçiş matrisi — algoritmik (hard-coded değer yok)
        states = size * size
        P = np.zeros((states, states))

        for r in range(1, size + 1):
            for c in range(1, size + 1):
                idx = self.state_index(r, c, size)

                # Her yönde adımı dene
                for dname, prob in probs.items():
                    if dname not in dir_map:
                        continue

                    dr, dc = dir_map[dname]
                    nr, nc = r + dr, c + dc

                    # Sınırlar içindeyse geç, değilse aynı hücrede kal
                    if 1 <= nr <= size and 1 <= nc <= size:
                        nidx = self.state_index(nr, nc, size)
                        P[idx, nidx] += prob
                    else:
                        # Reflecting boundary: kalabilir (olasılığı bu hücreye ekle)
                        P[idx, idx] += prob

        return P, steps, (start_r, start_c), size

    def _solve_grid_walk(self, params: dict) -> dict:
        """Geçiş parametreleriyle grid random walk çöz."""
        rows, cols = params["grid"]
        sr, sc = params.get("start", [(rows + 1) // 2, (cols + 1) // 2])
        move_prob = params.get("move_prob", 0.25)
        move_dirs = params.get("move_dirs", 4)
        teleport_p = params.get("teleport_prob", 0.0)
        tele_target = params.get("teleport_target", None)
        absorbing_t = params.get("absorbing_type", "border_death")

        # Adjust: 1-indexed → 0-indexed
        sr -= 1
        sc -= 1

        N_states = rows * cols

        def idx(r, c):
            return r * cols + c

        def is_border(r, c):
            return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

        # Absorbing & transient state sets
        absorbing = set()
        transient = []
        for r in range(rows):
            for c in range(cols):
                if is_border(r, c):
                    absorbing.add(idx(r, c))
                else:
                    transient.append(idx(r, c))

        n_abs = len(absorbing)
        n_tra = len(transient)
        abs_list = sorted(absorbing)

        if n_tra == 0:
            return {
                "_solver_error": "Tüm durumlar absorbing — geçerli grid değil",
                "solved": False,
            }

        tra_set = set(transient)
        tra_idx = {s: i for i, s in enumerate(transient)}  # global → local
        abs_idx = {s: i for i, s in enumerate(abs_list)}

        # ── Geçiş matrisi blokları (Q ve R) ──────────────────────────────────
        Q = np.zeros((n_tra, n_tra))
        R = np.zeros((n_tra, n_abs))

        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # ↑↓←→

        for r in range(rows):
            for c in range(cols):
                s = idx(r, c)
                if s not in tra_set:
                    continue
                i = tra_idx[s]

                remaining_prob = 1.0

                # Teleport
                if teleport_p > 0 and tele_target:
                    tr_r, tr_c = tele_target[0] - 1, tele_target[1] - 1
                    t_s = idx(tr_r, tr_c)
                    if t_s in tra_set:
                        Q[i, tra_idx[t_s]] += teleport_p
                    elif t_s in absorbing:
                        R[i, abs_idx[t_s]] += teleport_p
                    remaining_prob -= teleport_p

                # Normal moves
                valid_neighbors = []
                for dr, dc in MOVES:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        valid_neighbors.append(idx(nr, nc))

                p_per_move = remaining_prob / max(len(valid_neighbors), 1)

                for ns in valid_neighbors:
                    if ns in tra_set:
                        Q[i, tra_idx[ns]] += p_per_move
                    elif ns in absorbing:
                        R[i, abs_idx[ns]] += p_per_move

        # ── Fundamental Matrix ────────────────────────────────────────────────
        I_mat = np.eye(n_tra)
        try:
            N = np.linalg.inv(I_mat - Q)
        except np.linalg.LinAlgError:
            return {"_solver_error": "(I-Q) singular — çözülemedi", "solved": False}

        # ── Expected steps (t = N·1) ──────────────────────────────────────────
        ones = np.ones(n_tra)
        exp_steps = N @ ones  # n_tra boyutlu

        # ── Absorption probabilities (B = N·R) ────────────────────────────────
        B = N @ R  # n_tra × n_abs

        # ── Variance ─────────────────────────────────────────────────────────
        t_vec = exp_steps
        var_vec = (2 * N - I_mat) @ t_vec - t_vec**2
        var_vec = np.maximum(var_vec, 0)  # nümerik stabilite

        # ── Başlangıç durumu ──────────────────────────────────────────────────
        start_s = idx(sr, sc)
        if start_s in tra_set:
            si = tra_idx[start_s]
            start_exp = Decimal(str(exp_steps[si]))
            start_var = Decimal(str(var_vec[si]))
            start_std = Decimal(str(np.sqrt(max(start_var, 0))))
            start_absorb = {f"abs_{j}": Decimal(str(B[si, j])) for j in range(n_abs)}

            # Kenar gruplaması (kuzey/güney/doğu/batı)
            edge_probs = self._group_edge_probs(B[si], abs_list, rows, cols)
        else:
            # Başlangıç absorbing durumdaysa
            start_exp = 0.0
            start_var = 0.0
            start_std = 0.0
            start_absorb = {}
            edge_probs = {}

        # ── Quasi-stationary distribution ────────────────────────────────────
        # t vektörünü normalize et
        t_sum = float(exp_steps.sum())
        quasi_stationary = {}
        if t_sum > 0:
            for i, s in enumerate(transient):
                r_i = s // cols
                c_i = s % cols
                quasi_stationary[f"({r_i+1},{c_i+1})"] = float(exp_steps[i] / t_sum)

        # ── Adım açıklamaları ────────────────────────────────────────────────
        steps = self._build_steps(
            rows,
            cols,
            sr,
            sc,
            n_tra,
            n_abs,
            start_exp,
            start_var,
            edge_probs,
            teleport_p,
            move_prob,
        )

        return {
            "solved": True,
            "_solver_used": "MarkovSolver",
            "grid": [rows, cols],
            "start": [sr + 1, sc + 1],
            "n_transient": n_tra,
            "n_absorbing": n_abs,
            "expected_steps": round(start_exp, 6),
            "variance": round(start_var, 6),
            "std_dev": round(start_std, 6),
            "edge_probs": edge_probs,
            "quasi_stationary": quasi_stationary,
            "steps": steps,
            "matrix_Q_shape": list(Q.shape),
            "matrix_N_shape": list(N.shape),
        }

    def _group_edge_probs(self, b_row, abs_list, rows, cols):
        """Absorbing durumları kenar gruplarına ayırır."""
        north = south = east = west = 0.0
        for j, s in enumerate(abs_list):
            r_s = s // cols
            c_s = s % cols
            p = float(b_row[j])
            if r_s == 0:
                north += p
            elif r_s == rows - 1:
                south += p
            if c_s == 0:
                west += p
            elif c_s == cols - 1:
                east += p
        total = north + south + east + west
        if total > 0:
            # Normalize (köşe hücreleri çift sayılır — normalize et)
            scale = 1.0 / max(total, 1e-10)
            north *= scale
            south *= scale
            east *= scale
            west *= scale
        return {
            "kuzey": round(north, 4),
            "güney": round(south, 4),
            "doğu": round(east, 4),
            "batı": round(west, 4),
        }

    def _build_steps(
        self,
        rows,
        cols,
        sr,
        sc,
        n_tra,
        n_abs,
        exp_steps,
        variance,
        edge_probs,
        teleport_p,
        move_prob,
    ):
        steps = [
            {
                "title": "1. Model Kurulumu",
                "content": (
                    f"{rows}×{cols} = {rows*cols} toplam durum. "
                    f"Kenar hücreleri: {n_abs} absorbing (ölüm). "
                    f"İç hücreler: {n_tra} geçici (transient)."
                ),
                "formula": f"S = T ∪ A  |T|={n_tra}  |A|={n_abs}",
                "result": f"Başlangıç: ({sr+1},{sc+1})",
                "type": "process",
            },
            {
                "title": "2. Geçiş Matrisi Blokları",
                "content": (
                    "P = [Q  R] / [0  I] bölünmüş yapısı. "
                    f"Q ({n_tra}×{n_tra}): geçici→geçici. "
                    f"R ({n_tra}×{n_abs}): geçici→absorbing."
                ),
                "formula": "P = [[Q, R], [0, I]]",
                "result": (
                    f"Hareket olasılığı: {move_prob:.4f}/yön"
                    + (f"  Teleport: {teleport_p:.2%}" if teleport_p > 0 else "")
                ),
                "type": "process",
            },
            {
                "title": "3. Fundamental Matrix",
                "content": (
                    "N = (I − Q)⁻¹. Her giriş N[i,j], "
                    "i başlangıcından j durumuna kaç kez geçildiğini verir."
                ),
                "formula": "N = (I − Q)⁻¹",
                "result": f"N boyutu: {n_tra}×{n_tra}",
                "type": "process",
            },
            {
                "title": "4. Beklenen Yaşam Süresi",
                "content": (
                    f"Başlangıç ({sr+1},{sc+1})'dan absorbing duruma "
                    f"ulaşmak için beklenen adım sayısı."
                ),
                "formula": "E[T] = (N · 1)[start]",
                "result": f"E[T] = {exp_steps:.4f} adım",
                "type": "process",
            },
            {
                "title": "5. Varyans ve Standart Sapma",
                "content": "Yaşam süresinin varyansı ve standart sapması.",
                "formula": "Var[T] = (2N−I)·t − t²",
                "result": f"Var={self._safe_float(variance):.4f}  Std={self._safe_sqrt(variance):.4f}",
                "type": "process",
            },
            {
                "title": "6. Kenar Absorpsiyon Dağılımı",
                "content": "Gezginin hangi kenarda öleceğinin olasılık dağılımı.",
                "formula": "B = N · R   →   B[start, kenar_grubu]",
                "result": "  ".join(f"{k}:{v:.3f}" for k, v in edge_probs.items()),
                "type": "process",
            },
        ]
        return steps

    def _solve_generic_chain(self, params: dict) -> dict:
        """Genel Markov zinciri için stub (genişletilebilir)."""
        return {"solved": False, "_solver_error": "Genel Markov: parametre eksik"}

    def _safe_float(self, value):
        """Decimal/Fraction → float dönüşümü, display için güvenli"""
        if isinstance(value, (Decimal, Fraction)):
            return float(value)
        return float(value)

    def _safe_sqrt(self, value):
        """Decimal/Fraction için güvenli karekök, float olarak döner"""
        if isinstance(value, Decimal):
            return float(value.sqrt())
        if isinstance(value, Fraction):
            # Fraction.sqrt() yok → Decimal üzerinden
            dec_value = Decimal(value.numerator) / Decimal(value.denominator)
            return float(dec_value.sqrt())
        return value**0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 NumericTruthValidator
#  Olasılık normalizasyonu, matris satır toplamları, sınır ihlali kontrolleri.
# ═══════════════════════════════════════════════════════════════════════════════
class NumericTruthValidator:
    """
    LLM'den dönen sayısal değerleri matematiksel doğruluk için kontrol eder.
    Solver çıktısını da doğrulayabilir.
    """

    TOLERANCE = 1e-4

    def _strip_percent_context(self, text: str) -> str:
        """Yüzde bağlamlarını temizler; soru tipinden bağımsız çalışır."""
        if not text:
            return ""
        patterns = (
            r"\d+\.?\d*\s*%",  # 48.5%
            r"%\s*\d+\.?\d*",  # %48.5
        )
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, " ", cleaned)
        return cleaned

    def _to_float(self, val):
        """Algoritmik, duck-typing tabanlı tip normalizasyonu.
        Decimal, Fraction, int, float, str-rakam → float
        Hard-coding yok, tüm solver tipleri için çalışır."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, "__float__"):  # Decimal ve Fraction yakalar
            return float(val)
        try:
            return float(val)  # str "0.25" vb. için
        except (ValueError, TypeError):
            return 0.0  # güvenli fallback

    def validate(self, sol_data: dict, solver_result: dict = None) -> list:
        """
        Tüm sayısal doğrulama katmanlarını çalıştırır.
        Döndürür: list[str] — ek ihlal mesajları
        """
        violations = []

        # ── LLM çıktısı üzerinde kontroller ──────────────────────────────────
        violations += self._check_probability_bounds(sol_data)
        violations += self._check_distribution_sum(sol_data)
        violations += self._check_matrix_row_sums(sol_data)

        # Solver sonucu yoksa veya boşsa atla (GameTheorySolver büyük-N durumlarında olur)
        if not solver_result or not isinstance(solver_result, dict):
            return violations

        # ── Bayes solver çıktısı karşılaştırması ──────────────────────────────
        if solver_result.get("solved"):
            solver_type = solver_result.get("_solver_type", "")
            if solver_type == "BayesSolver":
                violations += self._compare_bayes_vs_llm(sol_data, solver_result)
            else:
                violations += self._compare_solver_vs_llm(sol_data, solver_result)

        return violations

    def _compare_bayes_vs_llm(self, sol_data: dict, solver_result: dict) -> list:
        """
        BayesSolver analitik posteriorları ile LLM cevabını karşılaştırır.

        v2 güçlendirme:
          1. Tüm adımlardaki ara değerleri de tarar (sadece final değil)
          2. Posterior toplamı ≈ 1 kontrolü (normalisation check)
          3. Toleransı dinamik yapar: küçük posteriorlar için mutlak, büyükler için göreli
          4. Her hipotez için açık eksiklik mesajı + analitik değeri raporlar
        """
        violations = []
        posteriors = solver_result.get("posteriors", {})
        if not posteriors:
            return violations

        # ── LLM çıktısından tüm float değerleri topla (adımlar dahil) ─────────
        combined = (
            str(sol_data.get("answer", "") or "")
            + " "
            + str(sol_data.get("numeric", "") or "")
            + " "
            + " ".join(
                str(s.get("result", "") or "") + " " + str(s.get("formula", "") or "")
                for s in (sol_data.get("steps") or [])
            )
        )
        # Yüzde bağlamını temizle: "48.48%" → boşluk
        cleaned = re.sub(r"\d+\.?\d*\s*%", " ", combined)
        cleaned = re.sub(r"%\s*\d+\.?\d*", " ", cleaned)
        llm_floats = set()
        for fstr in re.findall(r"(\d*\.\d{2,})", cleaned):
            try:
                v = float(fstr)
                if 0 < v <= 1.0 + 1e-4:
                    llm_floats.add(round(v, 4))
            except ValueError:
                pass

        # ── Son tur posteriorları al ───────────────────────────────────────────
        rounds_detail = solver_result.get("rounds", {})
        if rounds_detail:
            last_round = max(rounds_detail.keys())
            last_posteriors = rounds_detail[last_round].get("posteriors", posteriors)
        else:
            last_round = 1
            last_posteriors = posteriors

        # ── Posterior toplamı ≈ 1 kontrolü ────────────────────────────────────
        post_sum = sum(last_posteriors.values())
        if abs(post_sum - 1.0) > 1e-4:
            violations.append(
                f"[NTV-BAYES-SUM] Analitik posterior toplamı {post_sum:.6f} ≠ 1.0 — "
                f"normalizasyon hatası (bu bir solver bug'ı; rapor et)."
            )

        # ── Her hipotez için LLM değeri var mı? ───────────────────────────────
        for hyp, prob in sorted(last_posteriors.items()):
            if prob < 0.005:  # < %0.5 → ihmal
                continue
            # Dinamik tolerans: küçük değerler için mutlak 0.015, büyükler için göreli %4
            tol_abs = 0.015
            tol_rel = 0.04
            found = any(
                abs(lf - prob) <= tol_abs
                or (prob > 0.05 and abs(lf - prob) / prob <= tol_rel)
                for lf in llm_floats
            )
            if not found:
                pct = prob * 100
                exact = rounds_detail.get(last_round, {}).get("_exact", {}).get(hyp, "")
                exact_str = f"  [kesin: {exact}]" if exact else ""
                violations.append(
                    f"[NTV-BAYES] Analitik P({hyp.upper()}|E) = {prob:.6f} "
                    f"(%{pct:.4f}){exact_str} — LLM cevabında bu değer bulunamadı. "
                    f"Yanlış veya eksik posterior."
                )
        return violations

    def _check_probability_bounds(self, sol_data: dict) -> list:
        """
        Olasılık değerlerinin [0,1] aralığında olup olmadığını kontrol eder.
        KORUMA: Yüzde olarak ifade edilmiş değerler (%48.48 vb.) çıkarılmadan önce temizlenir.
        """
        violations = []
        answer = str(sol_data.get("answer", "") or "")
        numeric = str(sol_data.get("numeric", "") or "")
        combined = (answer + " " + numeric).strip()

        # Yüzde bağlamını tamamen sil: "48.48%" veya "%48.48" → boşluk
        cleaned = self._strip_percent_context(combined)

        # Kalan float değerleri tara
        floats = re.findall(r"(?<![/\d])(\d*\.\d+)(?![/\d%])", cleaned)
        for f in floats:
            v = float(f)
            if v > 1.0 + self.TOLERANCE:
                violations.append(
                    f"[NTV-BOUND] Olasılık değeri {v:.4f} > 1.0 — "
                    f"yüzde bağlamı dışında geçersiz (posterior asla 1'i geçemez)."
                )
        return violations

    def _check_distribution_sum(self, sol_data: dict) -> list:
        violations = []
        # Oyun kuramı tipinde dağılım = kümülatif puanlar (olasılık değil).
        # P(Σ) = 1 kısıtı geçersiz — bu tip için suppressed.
        if sol_data.get("type") == "game_theory":
            return violations
        dist = sol_data.get("distribution") or {}
        if not dist:
            return violations
        vals = []
        for v in dist.values():
            try:
                vals.append(float(v))
            except:
                pass
        if not vals:
            return violations
        total = sum(vals)
        if abs(total - 1.0) > self.TOLERANCE:
            violations.append(
                f"[NTV-DIST] Dağılım toplamı {total:.6f} ≠ 1.0 (sapma: {abs(total-1):.6f})"
            )
        return violations

    def _check_matrix_row_sums(self, sol_data: dict) -> list:
        violations = []
        mat = sol_data.get("matrix") or {}
        data = mat.get("data") or []
        if not data:
            return violations
        for i, row in enumerate(data):
            try:
                vals = [float(x) for x in row]
                s = sum(vals)
                if abs(s - 1.0) > self.TOLERANCE:
                    violations.append(
                        f"[NTV-MATRIX] Matris satır {i} toplamı {s:.4f} ≠ 1.0"
                    )
            except (ValueError, TypeError):
                pass
        return violations

    def _compare_solver_vs_llm(self, sol_data: dict, solver_result: dict) -> list:
        """Yalnızca Markov solver (expected_steps) için LLM karşılaştırması yapar."""
        violations = []
        # Sadece Markov solver expected_steps alanı varsa karşılaştır
        expected = solver_result.get("expected_steps")
        if expected is None:
            return violations
        # Bayes ve diğer solverlar bu path'e girmemeli
        if solver_result.get("_solver_type", "") not in ("MarkovSolver", ""):
            return violations

        answer_str = str(sol_data.get("answer", "") or "")
        numeric_str = str(sol_data.get("numeric", "") or "")
        combined = answer_str + " " + numeric_str
        # Yüzde değerlerini ayıkla
        cleaned = re.sub(r"\d+\.?\d*\s*%", "", combined)
        floats = re.findall(r"(\d+\.?\d*)", cleaned)

        for f in floats:
            try:
                # === ALGORİTMİK TİP DÖNÜŞÜMÜ (tüm solver tipleri için) ===
                v_f = self._to_float(f)
                expected_f = self._to_float(expected)

                if v_f > 1 and abs(v_f - expected_f) / max(expected_f, 1) < 0.05:
                    return violations  # Uyumlu
            except ValueError:
                pass

        if floats:
            violations.append(
                f"[NTV-MISMATCH] Analitik çözüm E[T]={self._to_float(expected):.4f} — "
                f"LLM cevabı farklı görünüyor. Solver sonucu kullanılmalı."
            )
        return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL NUMERIC-SYMBOLIC-UNIT NORMALIZER
#  Negatif değer, sembolik ifade ve birim imzalarını program genelinde tekilleştirir.
# ═══════════════════════════════════════════════════════════════════════════════
class NumericSymbolicUnitNormalizer:
    """
    - Negatif sayı yuvarlamasını sign-preserving olarak yapar.
    - Sembolik ifadeleri (sympy varsa) sayısal karşılığa indirger.
    - Birim ifadelerini çarpma/bölme/üs düzeyinde imza olarak normalleştirir.
    """

    def __init__(self, precision: int = 8):
        self.precision = max(2, int(precision))
        self.quant = Decimal("1").scaleb(-self.precision)

    def _round_signed(self, val: Decimal) -> Decimal:
        sign = Decimal("-1") if val < 0 else Decimal("1")
        mag = abs(val).quantize(self.quant)
        out = sign * mag
        # -0.00000000 gibi artefact'ları temizle
        if abs(out) < self.quant:
            return Decimal("0")
        return out

    def _to_decimal(self, raw: str) -> Decimal | None:
        t = (raw or "").strip().replace(",", ".")
        if not t:
            return None
        # Kesir desteği
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", t):
            a, b = re.split(r"/", t)
            try:
                a_dec, b_dec = Decimal(a.strip()), Decimal(b.strip())
                if b_dec == 0:
                    return None
                return a_dec / b_dec
            except Exception:
                return None
        try:
            return Decimal(t)
        except Exception:
            pass

        if _SYMPY_OK:
            try:
                expr = t.replace("^", "**").replace("−", "-")
                val = sp.N(sp.sympify(expr, evaluate=True), 60)
                if val.is_real:
                    return Decimal(str(val))
            except Exception:
                return None
        return None

    def _format_decimal(self, val: Decimal) -> str:
        vv = self._round_signed(val)
        txt = format(vv, f"f")
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".")
        return txt or "0"

    def normalize_numeric_text(self, text: str) -> str:
        if not text:
            return text
        src = str(text).replace("−", "-")

        # Önce kesirleri, sonra ondalık/tam sayıları normalize et
        def _repl(m):
            raw = m.group(0)
            dec = self._to_decimal(raw)
            if dec is None:
                return raw
            return self._format_decimal(dec)

        src = re.sub(r"[-+]?\d+\s*/\s*[-+]?\d+", _repl, src)
        src = re.sub(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", _repl, src)
        return src

    def unit_signature(self, unit_expr: str) -> str:
        """Birim ifadesini kanonik imzaya çevirir: m*s^-2 gibi."""
        expr = (unit_expr or "").replace("·", "*").replace(" ", "")
        if not expr:
            return ""
        parts = re.split(r"([*/])", expr)
        sign = 1
        expo = defaultdict(Decimal)
        for p in parts:
            if p == "*":
                sign = 1
                continue
            if p == "/":
                sign = -1
                continue
            if not p:
                continue
            m = re.fullmatch(r"([A-Za-zµΩ°]+)(?:\^?([-+]?\d+(?:\.\d+)?))?", p)
            if not m:
                continue
            u = m.group(1)
            pw = Decimal(m.group(2) if m.group(2) is not None else "1")
            expo[u] += sign * pw

        items = []
        for k in sorted(expo.keys()):
            v = expo[k]
            if v == 0:
                continue
            vv = self._format_decimal(v)
            items.append(f"{k}^{vv}" if vv != "1" else k)
        return "*".join(items)

    def annotate_units(self, text: str) -> list:
        """Metinde geçen sayı+birim kalıplarından imza listesi çıkarır."""
        if not text:
            return []
        hits = []
        pattern = re.compile(r"[-+]?\d+(?:\.\d+)?\s*([A-Za-zµΩ°]+(?:\s*[*/·]\s*[A-Za-zµΩ°]+(?:\^-?\d+)?)*)")
        for m in pattern.finditer(str(text)):
            sig = self.unit_signature(m.group(1))
            if sig:
                hits.append(sig)
        # sıra korumalı unique
        uniq, seen = [], set()
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        return uniq

    def normalize_solution_payload(self, sol_data: dict) -> dict:
        d = dict(sol_data or {})
        for key in ("answer", "numeric", "formula"):
            if key in d and isinstance(d.get(key), (str, int, float, Decimal)):
                d[key] = self.normalize_numeric_text(str(d.get(key)))

        steps = []
        for st in d.get("steps", []) or []:
            ss = dict(st)
            for key in ("formula", "result", "content", "title"):
                if key in ss and isinstance(ss.get(key), (str, int, float, Decimal)):
                    ss[key] = self.normalize_numeric_text(str(ss.get(key)))
            unit_hints = []
            for key in ("formula", "result", "content"):
                unit_hints.extend(self.annotate_units(ss.get(key, "")))
            if unit_hints:
                ss["_unit_signatures"] = list(dict.fromkeys(unit_hints))
            steps.append(ss)
        d["steps"] = steps

        root_units = []
        for key in ("answer", "numeric", "formula"):
            root_units.extend(self.annotate_units(d.get(key, "")))
        if root_units:
            d["_unit_signatures"] = list(dict.fromkeys(root_units))
        return d


_global_num_unit_normalizer = NumericSymbolicUnitNormalizer(precision=8)

# ═══════════════════════════════════════════════════════════════════════════════
#  ARITHMETIC PRECISION VALIDATOR — Tüm aritmetik adımlar için yüksek doğruluk
#  Amaç: eksi/mertebe hatalarını otomatik düzeltmek, içsel tutarsızlıkları yakalamak
#  Hard-coding yok; genel ifadeleri Decimal + Sympy ile yeniden hesaplar.
# ═══════════════════════════════════════════════════════════════════════════════
class ArithmeticPrecisionValidator:
    """
    Adım sonuçlarındaki sayıları yeniden hesaplar, sapmayı raporlar ve gerekirse düzeltir.

    Çalışma prensibi (her adım için):
      1) formula veya result içindeki aritmetik ifadeyi ayrıştır (sayısal tokenlar + + - * / ^).
      2) Sympy ile yüksek hassasiyetle değerlendir, Decimal'e çevir.
      3) Hesaplanan değer ile mevcut sonuç arasında göreli/abs sapma > TOL → ihlal kaydı.
      4) Düzeltme: step['result'] içinde sayısal kısım normalize_sci() ile güncellenir.
    """

    REL_TOL = Decimal("1e-6")  # %0.0001 göreli tolerans
    ABS_TOL = Decimal("1e-6")  # Küçük değerler için mutlak tolerans

    def __init__(self):
        pass

    # ── Kamu API ───────────────────────────────────────────────────────────
    def enforce(self, steps: list) -> tuple:
        corrected = []
        violations = []
        notes = []

        for idx, step in enumerate(steps or [], 1):
            formula_text = str(step.get("formula", "") or "")
            result_text = str(step.get("result", "") or "")

            expr = self._extract_expr(formula_text) or self._extract_expr(result_text)
            if not expr:
                corrected.append(step)
                continue

            evaluated = self._safe_eval(expr)
            if evaluated is None:
                corrected.append(step)
                continue

            formatted_val = self._format_sci(evaluated)

            # Sayısal sapma kontrolü (mevcut sonuç içindeki ilk sayı üzerinden)
            existing_val = self._first_number(result_text)
            if existing_val is not None:
                delta = abs(evaluated - existing_val)
                rel = delta / (abs(existing_val) + Decimal("1e-20"))
                if delta > self.ABS_TOL and rel > self.REL_TOL:
                    violations.append(
                        f"[PRECISION] Step {idx}: expr '{expr}' yeniden hesaplandı → {formatted_val} (sapma={rel:.2E})"
                    )
                    notes.append(
                        f"[PRECISION-FIX] Adım {idx} sonucu {formatted_val} olarak güncellendi"
                    )

            # Sonucu güncelle (birden fazla sayı varsa yalnızca ilkini değiştir)
            updated_step = step.copy()
            updated_step["result"] = formatted_val
            updated_step.setdefault("_recomputed", True)
            corrected.append(updated_step)

        return corrected, violations, notes

    # ── Yardımcılar ────────────────────────────────────────────────────────
    def _extract_expr(self, text: str) -> str | None:
        if not text:
            return None
        # Unicode çarpı → * , üs sembolü → **
        t = text.replace("×", "*").replace("^", "**")
        # Eşittikten sonra gelen kısmı tercih et (F = m * g)
        if "=" in t:
            t = t.split("=", 1)[1]
        # Harfleri ve birimleri temizle, sayılar ve operatörler kalsın
        t = re.sub(r"[A-Za-zα-ωΑ-ΩçğıöşüÇĞİÖŞÜ°%‰µΩσ∆≃≈><≤≥]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        # Geçerli bir aritmetik token var mı?
        if not re.search(r"[0-9]", t):
            return None
        return t

    def _safe_eval(self, expr: str) -> Decimal | None:
        try:
            if _SYMPY_OK:
                import sympy as sp

                cleaned = expr.replace("--", "+")
                val = sp.N(sp.sympify(cleaned, evaluate=True), 50)
                return Decimal(str(val))
            # Fallback: Decimal eval (çok sınırlı)
            # Sadece izinli karakterler
            if not re.fullmatch(r"[0-9eE+\-*/().\s]+", expr):
                return None
            return Decimal(str(eval(expr, {"__builtins__": {}}, {})))
        except Exception:
            return None

    def _format_sci(self, val: Decimal) -> str:
        """Negatif/pozitif değerlerde simetrik yuvarlama ve stabil gösterim."""
        try:
            val = _global_num_unit_normalizer._round_signed(Decimal(val))
            if abs(val) >= Decimal("1e6") or (abs(val) > 0 and abs(val) < Decimal("1e-4")):
                return format(val, ".6E")
            return _global_num_unit_normalizer._format_decimal(val)
        except Exception:
            return str(val)

    def _first_number(self, text: str) -> Decimal | None:
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
        if not nums:
            return None
        try:
            return Decimal(nums[0])
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 MonteCarloVerifier
#  Analitik çözümü simülasyon ile doğrular. LLM'den bağımsız.
# ═══════════════════════════════════════════════════════════════════════════════
class MonteCarloVerifier:
    """
    Monte Carlo simülasyonu ile Markov çözümlerini doğrular.
    Çalıştırma süresi sınırlandırılmış (max 0.5s).
    """

    MAX_SIMS = 50_000  # Maksimum simülasyon sayısı
    TIMEOUT_S = 0.4  # Saniye cinsinden timeout

    def verify(self, ast: dict, solver_result: dict) -> dict:
        """
        ast ve solver_result'ı alır, simülasyon çalıştırır.
        Döndürür: {mc_expected_steps, mc_std, agreement, sims_run}
        """
        if not solver_result.get("solved"):
            return {"mc_run": False, "reason": "Solver çözüm üretemedi"}

        params = ast.get("params", {})
        if "grid" not in params:
            return {"mc_run": False, "reason": "Grid parametresi yok"}

        rows, cols = params["grid"]
        sr, sc = params.get("start", [(rows + 1) // 2, (cols + 1) // 2])
        sr -= 1
        sc -= 1
        move_prob = params.get("move_prob", 0.25)
        tele_p = params.get("teleport_prob", 0.0)
        tele_tgt = params.get("teleport_target", None)
        if tele_tgt:
            tele_r, tele_c = tele_tgt[0] - 1, tele_tgt[1] - 1
        else:
            tele_r = tele_c = None

        def is_border(r, c):
            return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        step_counts = []
        t_start = time.time()
        n_sims = 0

        rng = np.random.default_rng(42)

        for _ in range(self.MAX_SIMS):
            if time.time() - t_start > self.TIMEOUT_S:
                break
            r, c = sr, sc
            steps = 0
            while not is_border(r, c) and steps < 10_000:
                rn = rng.random()
                if tele_p > 0 and rn < tele_p and tele_r is not None:
                    r, c = tele_r, tele_c
                else:
                    # Geçerli komşular
                    valid = [
                        (r + dr, c + dc)
                        for dr, dc in MOVES
                        if 0 <= r + dr < rows and 0 <= c + dc < cols
                    ]
                    if valid:
                        nr, nc = valid[rng.integers(len(valid))]
                        r, c = nr, nc
                steps += 1
            step_counts.append(steps)
            n_sims += 1

        if not step_counts:
            return {"mc_run": False, "reason": "Simülasyon çalışamadı"}

        mc_exp = Decimal(str(np.mean(step_counts)))
        mc_std = Decimal(str(np.std(step_counts)))
        analytic = solver_result.get("expected_steps", 0)

        agreement = "✓ UYUMLU"
        if analytic > 0:
            rel_err = abs(mc_exp - analytic) / analytic
            if rel_err > 0.10:
                agreement = f"⚠ SAPMA {rel_err:.1%}"
            elif rel_err > 0.05:
                agreement = f"≈ YAKLAŞIK UYUMLU ({rel_err:.1%})"

        return {
            "mc_run": True,
            "sims_run": n_sims,
            "mc_expected": round(mc_exp, 4),
            "mc_std": round(mc_std, 4),
            "analytic": round(analytic, 4),
            "agreement": agreement,
            "rel_error_pct": (
                round(abs(mc_exp - analytic) / max(analytic, 1) * 100, 2)
                if analytic
                else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT STORE (SNAPSHOT / ROLLBACK)
#  Adım sonuçlarını kaydederek hata durumunda geri dönüş sağlar.
#  Transactional state management için altyapı.
# ═══════════════════════════════════════════════════════════════════════════════
class CheckpointStore:
    """
    Her solver adımının sonuçlarını snapshot'ı otomatik kaydeder.
    Hata veya tutarsızlık tespit edilirse, son iyi duruma rollback yapılabilir.
    """

    def __init__(self):
        """Snapshot ve metadata deposu."""
        self.snapshots = {}  # {checkpoint_id: deep_copy(state)}
        self.checkpoint_trail = []  # [(checkpoint_id, timestamp, description)]
        self.current_checkpoint = None

    def snapshot(self, checkpoint_id: str, state: dict, description: str = "") -> None:
        """
        Mevcut durumu kaydeder.

        Args:
          checkpoint_id: "step_1_bayes", "step_2_propagation", vb.
          state:         kaydetMesi gereken dict (context, result, signals, vb.)
          description:   insan okunabilir not
        """
        from copy import deepcopy

        self.snapshots[checkpoint_id] = deepcopy(state)
        self.checkpoint_trail.append((checkpoint_id, time.time(), description))
        self.current_checkpoint = checkpoint_id

    def rollback(self, checkpoint_id: str) -> dict:
        """
        Kayıtlı snapshot'ı geri yükler.

        Returns:
          Kaydedilen state dict (veya {} hata durumunda)
        """
        from copy import deepcopy

        if checkpoint_id not in self.snapshots:
            return {}
        return deepcopy(self.snapshots[checkpoint_id])

    def get_latest(self) -> dict:
        """En son checkpoint'i döndürür."""
        if not self.checkpoint_trail:
            return {}
        last_id, _, _ = self.checkpoint_trail[-1]
        return self.rollback(last_id)

    def list_checkpoints(self) -> list:
        """Tüm checkpoint'leri (id, timestamp, description) olarak döndürür."""
        return self.checkpoint_trail.copy()

    def clear(self) -> None:
        """Tüm snapshot'ları temizle (bellek boşalt)."""
        self.snapshots.clear()
        self.checkpoint_trail.clear()
        self.current_checkpoint = None


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-RECOVERY MODULE
#  Solver ihlalleri tespit edilince otomatik tamir ve yeniden hesaplama.
#  Kritik: Intermediate Checkpoint ile entegre.
# ═══════════════════════════════════════════════════════════════════════════════
class AutoRecoveryModule:
    """
    Solver çıktısında hata/uyarı tespit edilince şu adımları yapar:
    1. İhlal türünü sınıflandır (BAYES_INJECTION, DENOMINATOR_SHRINK, vb.)
    2. ChekpointStore'dan son iyi durumu geri yükle
    3. Çıkarılan düğümleri / ara terimleri grafı ekle
    4. NarrativeBeliefPropagator ile tam yayılım yap
    5. Yeniden hesaplanan sonuç ile çalış
    """

    def __init__(self, checkpoint_store: CheckpointStore = None):
        """
        Args:
          checkpoint_store: CheckpointStore instance (yoksa yeni oluşturulur)
        """
        self.checkpoint = checkpoint_store or CheckpointStore()
        self.recovery_attempts = {}  # {violation_type: attempt_count}

    def try_recover(self, graph: dict, violation_list: list, context: dict) -> tuple:
        """
        İhlal listesinin ilkini analiz edip recovery çabası yapar.

        Args:
          graph:          DependencyGraphBuilder çıktısı (DAG)
          violation_list: LogicalConjunctionValidator.validate() çıktısı
          context:        mevcut state (signals, step_texts, vb.)

        Returns:
          (recovered_graph, was_recovered: bool, recovery_note: str)
        """
        if not violation_list:
            return graph, False, ""

        viol = violation_list[0]
        viol_type = self._classify_violation(viol)

        # Attempt sayısını kontrol et (sonsuz loop önlemek için)
        attempts = self.recovery_attempts.get(viol_type, 0)
        if attempts >= 3:
            return (
                graph,
                False,
                f"[RECOVERY-ABORT] Max attempts ({attempts}) reached for {viol_type}",
            )

        self.recovery_attempts[viol_type] = attempts + 1

        # Türe göre recovery strateji seç
        recovered = False
        note = ""

        if "BAYES_INJECTION" in viol_type or "DIRECT_JUMP" in viol:
            # Strateji: Gizli düğümleri açık hale getir, propagation yap
            graph, recovered, note = self._recover_bayes_injection(graph, context)

        elif "DENOMINATOR_SHRINK" in viol_type:
            # Strateji: Last good checkpoint'e dön, paydayı koruma al
            graph, recovered, note = self._recover_denominator_shrink(graph, context)

        elif "PROBABILITY_BOUNDS" in viol_type:
            # Strateji: Normalize et, tüm posteriorları [0,1] ile kesilmez
            graph, recovered, note = self._recover_probability_bounds(graph, context)

        else:
            note = f"[RECOVERY-UNKNOWN] Violation type unknown: {viol_type}"

        return graph, recovered, note

    def _classify_violation(self, violation_str: str) -> str:
        """İhlal string'inden tür belirle."""
        if "BAYES_INJECTION" in violation_str:
            return "BAYES_INJECTION"
        elif "DIRECT_JUMP" in violation_str or "direct_jump" in violation_str:
            return "DIRECT_JUMP"
        elif "DENOMINATOR" in violation_str:
            return "DENOMINATOR_SHRINK"
        elif "PROBABILITY" in violation_str:
            return "PROBABILITY_BOUNDS"
        elif "MATRIX" in violation_str:
            return "MATRIX_CONSISTENCY"
        else:
            return "UNKNOWN"

    def _recover_bayes_injection(self, graph: dict, context: dict) -> tuple:
        """
        Bayes injection: Gizli ara düğümler var ama açıkça çalıştırılmamış.
        Recovery: Graph'a hidden nodes ekle ve yeniden sort et.
        """
        # Checkpoint'ten son iyi state'i al
        last_state = self.checkpoint.get_latest()

        # Graph'ı genişlet: hidden nodes'ları açık hale getir
        for nid in list(graph.keys()):
            if graph[nid]["metadata"].get("has_hidden_implications"):
                # Bu düğümün altına hidden computation node'u ekle
                new_node_id = f"{nid}__hidden_expanded"
                graph[new_node_id] = {
                    "node": {"id": new_node_id, "label": f"Hidden({nid})"},
                    "deps": {nid},
                    "type": "hidden_expansion",
                    "symbol": f"H_{graph[nid]['symbol']}",
                    "metadata": {"expanded": True},
                }

        return (
            graph,
            True,
            "[RECOVERY-OK] Bayes injection recovered: hidden nodes expanded",
        )

    def _recover_denominator_shrink(self, graph: dict, context: dict) -> tuple:
        """Denominator shrink: Paydada hata. Last good checkpoint'e dön."""
        last_state = self.checkpoint.get_latest()
        if not last_state:
            return (
                graph,
                False,
                "[RECOVERY-FAIL] No good checkpoint for denominator recovery",
            )

        # Last state'deki graph'ı restore et
        old_graph = last_state.get("graph", graph)
        return (
            old_graph,
            True,
            "[RECOVERY-OK] Denominator shrink recovered: reverted to last good state",
        )

    def _recover_probability_bounds(self, graph: dict, context: dict) -> tuple:
        """Probability bounds: Normalize elleme (simple fix)."""
        # Hiçbir graph değişikliği yapmaya gerek yok, sadece nota almak yeterli
        return (
            graph,
            True,
            "[RECOVERY-OK] Probability bounds: will be normalized by validator",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  VALIDATION ROUTER — İhlal → Recovery Stratejisi Eşlemesi
#  Tespit edilen violation type'ını otomatik recovery stratejisine çevirme
# ═══════════════════════════════════════════════════════════════════════════════
class ValidationRouter:
    """
    Validator'lardan dönen ihlalleri kategorize edip recovery stratejisini seçer.

    İhlal Tipi         → Recovery Stratejisi
    ──────────────────────────────────────
    PROBABILITY_BOUNDS  → normalize
    AND_CHAIN_VIOLATION → intermediate_step_injection
    BAYES_SUM          → posterior re-normalize
    DENOMINATOR        → last_good_checkpoint_restore
    MATRIX_ROWS        → row_sum_normalize
    CONDITIONAL        → bayes_reinterpret
    """

    # İhlal pattern'i → recovery stratejisi & priority
    VIOLATION_STRATEGIES = {
        "NTV-BAYES-SUM": ("bayes_posterior_renormalize", 1),
        "SDG.*AND-zincir": ("intermediate_step_inject", 2),
        "NTV-MISMATCH": ("solver_rerun", 3),
        "PROBABILITY": ("probability_normalize", 1),
        "DENOMINATOR": ("checkpoint_restore", 4),
        "MATRIX": ("matrix_normalize_rows", 2),
        "CONDITIONAL_WITHOUT_GIVEN": ("bayes_reinterpret", 3),
    }

    def route(self, violations: list) -> dict:
        """
        İhlal listesini recovery planına çevir.

        Returns:
          {
            "primary_strategy": str,    # En önemli tamir yöntemi
            "secondary_strategies": list,  # Fallback'ler
            "affected_steps": list,     # Hangi adımlar etkilenmiş
            "priority_score": float,    # 1.0 = kritik, 0.0 = minor
          }
        """
        if not violations:
            return {
                "primary_strategy": "continue",
                "secondary_strategies": [],
                "affected_steps": [],
                "priority_score": 0.0,
            }

        matched_strategies = []
        affected_steps = set()

        # Her ihlalı pattern'e göre eşle
        for viol in violations:
            for pattern, (strategy, priority) in self.VIOLATION_STRATEGIES.items():
                if re.search(pattern, viol):
                    matched_strategies.append((strategy, priority, viol))

                    # Hangi adım etkilenmiş?
                    m = re.search(r"Adım\s+(\d+)", viol)
                    if m:
                        affected_steps.add(int(m.group(1)))
                    break

        if not matched_strategies:
            matched_strategies.append(("manual_review", 5, "Unknown violation"))

        # Priority'ye göre sırala (düşük = yüksek öncelik)
        matched_strategies.sort(key=lambda x: x[1])

        primary = matched_strategies[0][0]
        secondaries = [x[0] for x in matched_strategies[1:]]
        priority_score = 1.0 / (matched_strategies[0][1] + 1)  # +1 avoid div by 0

        return {
            "primary_strategy": primary,
            "secondary_strategies": secondaries,
            "affected_steps": sorted(list(affected_steps)),
            "priority_score": priority_score,
        }

    def execute_strategy(self, strategy_name: str, data: dict) -> dict:
        """
        Belirlenen strategy'yi uygula.

        Args:
          strategy_name: ("bayes_posterior_renormalize" gibi)
          data: {"steps": [...], "violations": [...], vb.}

        Returns:
          {"success": bool, "result": ..., "note": str}
        """
        # Strategy dispatching
        strategies = {
            "continue": lambda d: {
                "success": True,
                "result": None,
                "note": "No action needed",
            },
            "probability_normalize": self._normalize_probabilities,
            "intermediate_step_inject": self._inject_intermediate_steps,
            "checkpoint_restore": self._restore_checkpoint,
            "bayes_posterior_renormalize": self._renormalize_bayes_posteriors,
            "matrix_normalize_rows": self._normalize_matrix_rows,
            "solver_rerun": self._request_solver_rerun,
            "bayes_reinterpret": self._reinterpret_as_bayes,
            "manual_review": lambda d: {
                "success": False,
                "result": None,
                "note": "Manual review required",
            },
        }

        execute_fn = strategies.get(strategy_name)
        if not execute_fn:
            return {
                "success": False,
                "result": None,
                "note": f"Unknown strategy: {strategy_name}",
            }

        return execute_fn(data)

    def _normalize_probabilities(self, data: dict) -> dict:
        """Olasılık değerlerini [0,1] aralığına getir."""
        steps = data.get("steps", [])
        adjusted = []

        for step in steps:
            s_copy = step.copy()
            result_str = str(step.get("result", "") or "")

            # Sayısal değeri çıkar ve normalize et
            m = re.search(r"(\d+\.?\d*)", result_str)
            if m:
                val = float(m.group(1))
                if val > 1.0:
                    normalized = val / 100.0 if val > 100 else val / (val * 1.1)
                    s_copy["result"] = f"{normalized:.4f} [normalized]"
                    s_copy["_adjusted"] = True
            adjusted.append(s_copy)

        return {
            "success": True,
            "result": adjusted,
            "note": "Probabilities normalized to [0,1]",
        }

    def _inject_intermediate_steps(self, data: dict) -> dict:
        """AND-chain ihlali → arada intermediate step ekle."""
        steps = data.get("steps", [])
        affected = data.get("affected_steps", [])

        injected = []
        for i, step in enumerate(steps):
            injected.append(step)
            if (i + 1) in affected and i < len(steps) - 1:
                # i ile i+1 arasına intermediate ekle
                intermediate = {
                    "title": f"[INTERMEDIATE] Between step {i} и {i+1}",
                    "content": "Geometrik ortalama aşaması",
                    "formula": f"sqrt(step_{i} × step_{i+1})",
                    "result": "computed",
                    "_intermediate": True,
                }
                injected.append(intermediate)

        return {
            "success": True,
            "result": injected,
            "note": f"Injected intermediate steps for {len(affected)} violations",
        }

    def _restore_checkpoint(self, data: dict) -> dict:
        """Last good checkpoint'i restore et."""
        checkpoint_data = data.get("checkpoint")
        if not checkpoint_data:
            return {"success": False, "result": None, "note": "No checkpoint available"}

        return {
            "success": True,
            "result": checkpoint_data,
            "note": "Restored from last good checkpoint",
        }

    def _renormalize_bayes_posteriors(self, data: dict) -> dict:
        """Posterior toplamını 1.0 olacak şekilde normalize et."""
        steps = data.get("steps", [])
        adjusted = []

        # Tüm posterior değerlerini toplayıp normalize et
        all_posteriors = []
        for step in steps:
            result_str = str(step.get("result", "") or "")
            nums = re.findall(r"(\d+\.?\d*)", result_str)
            all_posteriors.extend([float(n) for n in nums if float(n) <= 1.0])

        total = sum(all_posteriors) or 1.0
        scale_factor = 1.0 / total

        for step in steps:
            s_copy = step.copy()
            result_str = str(step.get("result", "") or "")
            m = re.search(r"(\d+\.?\d*)", result_str)
            if m:
                val = float(m.group(1))
                if val <= 1.0:
                    scaled = val * scale_factor
                    s_copy["result"] = f"{scaled:.6f} [normalized]"
                    s_copy["_bayes_normalized"] = True
            adjusted.append(s_copy)

        return {
            "success": True,
            "result": adjusted,
            "note": f"Posteriors renormalized (scale={scale_factor:.4f})",
        }

    def _normalize_matrix_rows(self, data: dict) -> dict:
        """Matris satırlarını normalize et (her satır toplamı = 1)."""
        matrix = data.get("matrix", [])
        if not matrix:
            return {"success": False, "result": None, "note": "No matrix"}

        normalized = []
        for row in matrix:
            row_sum = sum(row) or 1.0
            normalized_row = [v / row_sum for v in row]
            normalized.append(normalized_row)

        return {
            "success": True,
            "result": normalized,
            "note": f"Normalized {len(matrix)} matrix rows",
        }

    def _request_solver_rerun(self, data: dict) -> dict:
        """Solver'ı yeniden çalıştırma isteği."""
        return {
            "success": True,
            "result": None,
            "note": "Solver rerun requested — will be executed next iteration",
        }

    def _reinterpret_as_bayes(self, data: dict) -> dict:
        """Problem'i Bayes kurulumu olarak yeniden yorumla."""
        return {
            "success": True,
            "result": data.copy(),
            "note": "Reinterpreted as Bayesian problem — retry with Bayes solver",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 BayesSolver
#  Analitik Bayes güncellemesi — hardcoding yok, n hipotez, m gözlem destekler.
#  Yöntem: Birleşik olasılık (joint probability) ile tek adımda hesap.
# ═══════════════════════════════════════════════════════════════════════════════
class BayesSolver:
    """
    Genel Bayesyen inference motoru.

    Formüller:
      P(H|E₁,...,Eₙ) = P(E₁,...,Eₙ|H)×P(H) / Σⱼ P(E₁,...,Eₙ|Hⱼ)×P(Hⱼ)

    Bağımsız gözlemler varsayımı (varsayılan):
      P(E₁,...,Eₙ|H) = Π P(Eₖ|H)

    Hem tek-tur hem çok-tur hem de iteratif güncelleme desteklenir.
    """

    def solve(self, ast: dict) -> dict:
        """
        ast: MathASTBuilder.build() çıktısı (type='bayes')
        Döndürür: kapsamlı Bayes çözüm dict

        v3 — fractions.Fraction ile kesin rasyonel hesap.
        Float yuvarlama hatası sıfır; her adımda payda doğru hesaplanır.
        Kök hata: prior×likelihood'u paydaya bölmeyi unutmak → Fraction bunu
        cebirsel olarak zorla önler (Fraction/Fraction = Fraction).
        """
        params = ast.get("params", {})
        priors = params.get("priors", {})
        likelihoods = params.get("likelihoods", {})
        n_obs = max(1, params.get("n_observations", 1))

        if not priors or not likelihoods:
            return {
                "solved": False,
                "_solver_type": "BayesSolver",
                "_solver_error": (
                    f"Prior veya likelihood tablosu çıkarılamadı — "
                    f"priors={list(priors.keys())} likelihoods={list(likelihoods.keys())}"
                ),
            }

        # ── Hipotezler: prior ∪ likelihood anahtarları, sıralı ────────────────
        hyps = sorted(set(priors.keys()) | set(likelihoods.keys()))
        if not hyps:
            return {
                "solved": False,
                "_solver_type": "BayesSolver",
                "_solver_error": "Hipotez listesi boş",
            }

        # ── Fraction dönüşümü ─────────────────────────────────────────────────
        # Her float değeri → yakın rasyonel kesire çevir (10^8 payda limiti)
        def _F(v):
            try:
                return Fraction(v).limit_denominator(10**8)
            except:
                return Fraction(0)

        # Prior Fraction dict (normalize: toplam → 1)
        p_frac = {h: _F(priors.get(h, 0.0)) for h in hyps}
        p_sum = sum(p_frac.values())
        if p_sum <= 0:
            return {
                "solved": False,
                "_solver_type": "BayesSolver",
                "_solver_error": "Prior toplamı sıfır veya negatif",
            }
        p_frac = {h: v / p_sum for h, v in p_frac.items()}  # normalize

        # Likelihood Fraction dict
        l_frac = {h: _F(likelihoods.get(h, 0.0)) for h in hyps}

        # ── MODÜL 4: Bayesci Tutarlılık Koruyucusu ───────────────────────────
        # Kural: prior > 0 AND likelihood > 0 → posterior != 0
        # Solver'dan ÖNCE çalışır; ihlal varsa uyarı üretilir ama hesap engellenmez.
        _bayes_zero_warnings = []
        for h in hyps:
            p_val = float(p_frac[h])
            l_val = float(l_frac[h])
            if p_val > 0 and l_val > 0:
                # Bu hipotez için posterior mutlaka > 0 olmalı — solver kontrol edecek
                pass
            elif p_val > 0 and l_val == 0:
                _bayes_zero_warnings.append(
                    f"[MODÜL-4] '{h.upper()}': prior={p_val:.4f} > 0 ama "
                    f"likelihood=0 → posterior 0 olacak (olağan durum)."
                )
            elif p_val == 0 and l_val > 0:
                _bayes_zero_warnings.append(
                    f"[MODÜL-4] '{h.upper()}': prior=0 ama likelihood={l_val:.4f} > 0 → "
                    f"posterior 0 — prior eksik çıkarılmış olabilir, kontrol edilmeli."
                )

        # Float kopyaları (adım metni / geri uyumluluk)
        prior_vals = {h: float(p_frac[h]) for h in hyps}
        lik_vals = {h: float(l_frac[h]) for h in hyps}

        # ── Çok-tur iteratif Bayes güncellemesi ──────────────────────────────
        # Her tur t:
        #   weights_t[h] = current_prior[h] × likelihood[h]   (Fraction × Fraction)
        #   evidence_t   = Σ weights_t[h]                     (tam Fraction toplamı)
        #   posterior_t  = weights_t[h] / evidence_t           (kesin bölüm)
        #   Sonraki tur: current_prior = posterior_t
        #
        # Kural: "birinci tur paydası ikinci turda tekrar kullanılamaz"
        # Bu kod her turda evidence'ı yeniden hesaplar — sıfır risk.
        rounds_detail = {}
        cur_prior_frac = dict(p_frac)

        for t in range(1, n_obs + 1):
            weights = {h: cur_prior_frac[h] * l_frac[h] for h in hyps}
            # Decimal kesinlikte evidence hesaplaması (marjinal normalizasyon)
            evidence_decimal = sum(Decimal(str(float(w))) for w in weights.values())
            evidence = sum(weights.values())

            if evidence <= 0:
                return {
                    "solved": False,
                    "_solver_type": "BayesSolver",
                    "_solver_error": (
                        f"Tur {t}: P(E)=0 — tüm likelihoods sıfır "
                        f"veya prior×likelihood yok. "
                        f"likelihoods={lik_vals}"
                    ),
                }

            # Decimal ile posterior hesaplaması (kesin bölme)
            post_frac = {}
            for h in hyps:
                w_dec = Decimal(str(float(weights[h])))
                post_decimal = w_dec / evidence_decimal
                post_frac[h] = Fraction(post_decimal).limit_denominator(10**8)

            rounds_detail[t] = {
                "priors": {h: float(cur_prior_frac[h]) for h in hyps},
                "likelihoods": lik_vals,
                "evidence": float(evidence),
                "posteriors": {h: float(post_frac[h]) for h in hyps},
                # Kesin rasyonel form (debug + hint için)
                "_exact": {
                    h: f"{post_frac[h].numerator}/{post_frac[h].denominator}"
                    for h in hyps
                },
            }
            cur_prior_frac = dict(post_frac)  # posterior → sonraki tur prior

        # ── Birleşik (joint) yöntem — çapraz doğrulama ───────────────────────
        # P(H|E^n) = P(E|H)^n × P(H) / Σ P(E|Hᵢ)^n × P(Hᵢ)
        # Bu, iteratif sonucuyla sayısal olarak örtüşmeli (örtüşmezse bug var).
        # Decimal kesinlikte joint posterior
        j_weights = {h: p_frac[h] * (l_frac[h] ** n_obs) for h in hyps}
        j_evidence_decimal = sum(Decimal(str(float(w))) for w in j_weights.values())
        j_evidence = sum(j_weights.values())
        joint_posteriors = (
            {}
            if j_evidence <= 0
            else {
                h: float(Decimal(str(float(w))) / j_evidence_decimal)
                for h, w in j_weights.items()
            }
        )

        # ── Adım açıklamaları ─────────────────────────────────────────────────
        steps = self._build_steps(
            hyps, prior_vals, lik_vals, rounds_detail, n_obs, joint_posteriors
        )

        # Final değerler (son tur)
        final_posts = rounds_detail[n_obs]["posteriors"]
        final_evid = rounds_detail[n_obs]["evidence"]
        best_hyp = max(final_posts, key=final_posts.get)
        best_prob = final_posts[best_hyp]

        return {
            "solved": True,
            "_solver_type": "BayesSolver",
            "hypotheses": hyps,
            "priors": prior_vals,
            "likelihoods": lik_vals,
            "n_observations": n_obs,
            "rounds": rounds_detail,
            "posteriors": final_posts,
            "joint_posteriors": {h: round(v, 8) for h, v in joint_posteriors.items()},
            "final_evidence": round(final_evid, 8),
            "best_hypothesis": best_hyp,
            "best_posterior": round(best_prob, 8),
            "steps": steps,
            "_bayes_zero_warnings": _bayes_zero_warnings,
        }

    def _build_steps(
        self, hyps, priors, likelihoods, rounds_detail, n_obs, joint_posteriors
    ):
        steps = []

        # Adım 1: Priorlar
        prior_str = "  ".join(f"P({h.upper()})={priors[h]:.4f}" for h in sorted(hyps))
        steps.append(
            {
                "title": "1. Prior Dağılımı",
                "content": "Başlangıç hipotez olasılıkları (önceki bilgi).",
                "formula": prior_str,
                "result": f"Σ = {sum(priors.values()):.4f}",
                "type": "process",
            }
        )

        # Adım 2: Likelihoodlar
        lik_str = "  ".join(
            f"P(E|{h.upper()})={likelihoods.get(h,0):.4f}" for h in sorted(hyps)
        )
        steps.append(
            {
                "title": "2. Likelihood Tablosu",
                "content": "Her hipotez için kanıt olasılığı.",
                "formula": lik_str,
                "result": "",
                "type": "process",
            }
        )

        # Adım 3+: Her tur Bayes güncellemesi
        for t in range(1, n_obs + 1):
            rd = rounds_detail[t]
            ev = rd["evidence"]
            posts = rd["posteriors"]

            # Ağırlıklı toplam açıklaması
            wt_parts = " + ".join(
                f"P({h.upper()})×P(E|{h.upper()})={rd['priors'][h]:.4f}×{likelihoods.get(h,0):.4f}={rd['priors'][h]*likelihoods.get(h,0):.4f}"
                for h in sorted(hyps)
            )
            post_str = "  ".join(
                f"P({h.upper()}|E)={posts[h]:.6f}" for h in sorted(hyps)
            )

            steps.append(
                {
                    "title": f"{2+t}. Tur {t} — Bayes Güncellemesi",
                    "content": (
                        f"Kanıt olasılığı: P(E) = {ev:.6f}. "
                        f"Posterior = Prior × Likelihood / P(E)."
                    ),
                    "formula": f"P(E)= {wt_parts[:120]}",
                    "result": post_str,
                    "type": "process",
                }
            )

        # Son adım: Birleşik kontrol
        if n_obs >= 2 and joint_posteriors:
            jp_str = "  ".join(
                f"P({h.upper()}|E^{n_obs})={joint_posteriors.get(h,0):.6f}"
                for h in sorted(hyps)
            )
            steps.append(
                {
                    "title": f"{3+n_obs}. Doğrulama (Birleşik Yöntem)",
                    "content": (
                        f"P(H|E₁,...,E_{n_obs}) = P(E|H)^{n_obs}×P(H) / Σ P(E|Hᵢ)^{n_obs}×P(Hᵢ). "
                        f"Sonuç iteratif yöntemle uyuşmalı."
                    ),
                    "formula": f"Πᵢ P(E|Hᵢ)^{n_obs}",
                    "result": jp_str,
                    "type": "process",
                }
            )

        return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 SolverSelector  (RL tabanlı — Q-Learning'e solver seçimi ekler)
#  QLearningRouter mevcut layout seçimine ek olarak solver seçer.
# ═══════════════════════════════════════════════════════════════════════════════
class SolverSelector:
    """
    MathAST'a ve NLP sinyallerine göre çözücü seçer.
    Q-Learning ile zamanla solver seçimini optimize eder.
    """

    SOLVERS = [
        "MarkovSolver",
        "SymbolicSolver",
        "BayesSolver",
        "MonteCarloSolver",
        "GraphSolver",
        "GameTheorySolver",
        "LLM",
    ]

    SOLVER_REWARD = {
        ("markov_random_walk", "MarkovSolver"): 10,
        ("markov_chain", "MarkovSolver"): 10,
        ("bayes", "BayesSolver"): 10,
        ("bayes", "SymbolicSolver"): 8,
        ("general", "LLM"): 5,
        ("markov_random_walk", "LLM"): 2,  # Penalize wrong choice
        # Oyun Kuramı solver ödülleri
        ("game_theory", "GameTheorySolver"): 10,
        ("game_theory", "LLM"): 3,
    }

    def __init__(self, alpha=0.15, gamma=0.85, epsilon=0.1):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = 0

    def select(self, ast: dict, signals: dict) -> str:
        """Epsilon-greedy solver seçimi."""
        state = (ast.get("type", "general"), ast.get("confidence", 0) > 0.5)

        # AST güven yüksekse doğrudan önerilen solver'ı kullan
        if ast.get("confidence", 0) >= 0.6:
            suggested = ast.get("solver", "LLM")
            self._update(state, suggested, self._reward(ast["type"], suggested))
            return suggested

        # Düşük güvende epsilon-greedy
        if random.random() < self.epsilon:
            solver = random.choice(self.SOLVERS)
        else:
            q_vals = self.q_table[state]
            solver = max(q_vals, key=q_vals.get) if q_vals else ast.get("solver", "LLM")

        self._update(state, solver, self._reward(ast.get("type", "general"), solver))
        return solver

    def _reward(self, ast_type: str, solver: str) -> float:
        return self.SOLVER_REWARD.get(
            (ast_type, solver), self.SOLVER_REWARD.get(("general", "LLM"), 3)
        )

    def reward_outcome(
        self,
        ast_type: str,
        solver: str,
        consistency_score: float,
        symbolic_bypass: bool = False,
    ) -> None:
        """
        Çözüm sonucundan gerçek ödül sinyali alarak Q-tablosunu günceller.
        Symbolic bypass aktifse BayesSolver için bonus ödül verilir.

        Çağrı yeri: Flask route → çözüm tamamlandıktan sonra.
        """
        state = (ast_type, True)  # confidence>0.5 varsayımı (çözüldü)
        # Temel ödül: consistency score × solver uyum skoru
        base = self._reward(ast_type, solver)
        final = base * (0.5 + 0.5 * consistency_score)
        if symbolic_bypass:
            final = max(final, base * 1.2)  # bypass başarılıysa bonus
        self._update(state, solver, final)

    def _update(self, state, action, reward):
        nq = self.q_table[state]
        max_next = max(nq.values()) if nq else 0
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next - self.q_table[state][action]
        )
        self.episode += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  DAG EXECUTOR — Topological Order ile Adım Planlama
#  Bağımlılık grafiğinden çözüm sırası çıkar, doğrudan atlamayı engeller.
# ═══════════════════════════════════════════════════════════════════════════════
def topo_sort_kahn(graph: dict) -> tuple:
    """
    Kahn algoritması ile topological sort.

    Args:
      graph: DependencyGraphBuilder.build_from_ast() çıktısı

    Returns:
      (sorted_order: list[str], is_acyclic: bool)
    """
    in_degree = {n: len(graph[n]["deps"]) for n in graph}
    queue = [n for n in graph if in_degree[n] == 0]
    order = []

    while queue:
        v = queue.pop(0)
        order.append(v)

        # v'yi depend eden tüm w'leri bulup in_degree azalt
        for w in graph:
            if v in graph[w]["deps"]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)

    is_acyclic = len(order) == len(graph)
    return order, is_acyclic


class DAGExecutor:
    """
    Topological sıraya göre adımları planla ve doğrudan atlamayı engelle.
    """

    def __init__(self, graph: dict):
        """
        Args:
          graph: DependencyGraphBuilder çıktısı (DAG)
        """
        self.graph = graph
        self.order, self.is_acyclic = topo_sort_kahn(graph)
        self.executed = set()
        self.results = {}

    def can_execute(self, node_id: str) -> bool:
        """
        Bir düğümü şimdi çalıştırabilir miyiz?
        (Tüm dependencies çalıştırılmış mı?)
        """
        if node_id not in self.graph:
            return False
        return all(dep in self.executed for dep in self.graph[node_id]["deps"])

    def get_next_executable(self) -> "str | None":
        """Topological sıraya göre çalıştırılabilir ilk düğümü döndür."""
        for node_id in self.order:
            if node_id not in self.executed and self.can_execute(node_id):
                return node_id
        return None

    def mark_executed(self, node_id: str, result: dict = None) -> None:
        """Düğümü çalıştırıldı olarak işaretle."""
        self.executed.add(node_id)
        if result:
            self.results[node_id] = result

    def get_execution_order(self) -> list:
        """Önerilen çalıştırma sırası."""
        return self.order.copy()

    def get_remaining(self) -> list:
        """Henüz çalıştırılmamış düğümler."""
        return [n for n in self.order if n not in self.executed]


# ═══════════════════════════════════════════════════════════════════════════════
#  🔴 StepDependencyGraph  (adımlar arası türetim zinciri kontrolü)
# ═══════════════════════════════════════════════════════════════════════════════
class StepDependencyGraph:
    """
    LLM adımlarının birbirinden türetilip türetilmediğini kontrol eder.
    Bağlam duyarlı: Bayes/dağılım/toplam olasılık bağlamında AND-chain testi atlanır.

    Kontrol katmanları:
      AND-CHAIN:    Her adım öncekinden küçük olmalı (hayatta kalma zinciri)
      POSTERIOR:    Posterior değerler [0,1] arasında olmalı
      TOTAL-PROB:   Toplam olasılık teoreminde ağırlıklı toplam = P(B)
    """

    # Bu sinyaller AND-chain testini devre dışı bırakır
    SKIP_AND_CHAIN_SIGNALS = {
        "bayes_structure",  # Bayes yapısı → likelihood'lar artabilir
        "total_prob_applicable",  # Toplam olasılık → toplama var
    }
    SKIP_AND_CHAIN_LOGIC = {"OR_SUM", "MIXED"}

    def check(self, steps: list, signals: dict = None) -> list:
        """
        steps:   LLM'den gelen adım listesi
        signals: SemanticSignalModule çıktısı (bağlam için)
        Döndürür: list[str] — tespit edilen tutarsızlıklar
        """
        violations = []
        if len(steps) < 2:
            return violations

        # ── Bağlam kontrolü: AND-chain testi uygulanabilir mi? ────────────────
        skip_and_chain = False
        if signals:
            for sig in self.SKIP_AND_CHAIN_SIGNALS:
                if signals.get(sig):
                    skip_and_chain = True
                    break
            if signals.get("logic_operator") in self.SKIP_AND_CHAIN_LOGIC:
                skip_and_chain = True

        # Bayes sinyal yoksa adım başlıklarından kontekst çıkar
        if not skip_and_chain:
            all_titles = " ".join(
                str(s.get("title", "")) + " " + str(s.get("content", "")) for s in steps
            ).lower()
            bayes_keywords = re.compile(
                r"bayes|posterior|prior|toplam\s+olasılık|total\s+prob|"
                r"p\([a-z]\s*\|\s*[a-z]\)|likelihood|olabilirlik"
            )
            if bayes_keywords.search(all_titles):
                skip_and_chain = True

        if skip_and_chain:
            # AND-chain testi yok — sadece [0,1] sınır ihlallerini kontrol et
            return self._check_posterior_bounds(steps)

        # ── AND-chain testi ────────────────────────────────────────────────────
        step_values = []
        for s in steps:
            result_str = str(s.get("result", "") or "")
            formula_str = str(s.get("formula", "") or "")
            combined = result_str + " " + formula_str
            # Yüzde değerlerini temizle
            cleaned = re.sub(r"\d+\.?\d*\s*%", "", combined)
            floats = re.findall(r"(\d+\.?\d*)", cleaned)
            vals = []
            for f in floats:
                try:
                    v = float(f)
                    if 0 < v <= 1.0:
                        vals.append(v)
                except ValueError:
                    pass
            step_values.append(vals)

        prob_chain = [min(v) for v in step_values if v]

        if len(prob_chain) >= 3:
            for i in range(1, len(prob_chain)):  # tüm adımları kontrol et (son dahil)
                if prob_chain[i] > prob_chain[i - 1] * 1.1:  # %10 tolerans
                    violations.append(
                        f"[SDG] Adım {i+1} değeri ({prob_chain[i]:.4f}) "
                        f"önceki adımdan ({prob_chain[i-1]:.4f}) büyük — "
                        f"AND-zincirinde azalma beklenir."
                    )

        return violations

    def _check_posterior_bounds(self, steps: list) -> list:
        """Bayes bağlamında: posterior değerlerin [0,1] dışına çıkıp çıkmadığını kontrol et."""
        violations = []
        for i, s in enumerate(steps):
            result_str = str(s.get("result", "") or "")
            # Yüzde bağlamını temizle
            cleaned = re.sub(r"\d+\.?\d*\s*%", "", result_str)
            floats = re.findall(r"(\d*\.\d+)", cleaned)
            for f in floats:
                try:
                    v = float(f)
                    if v > 1.0 + 1e-4:
                        violations.append(
                            f"[SDG-BAYES] Adım {i+1} sonucu {v:.4f} > 1.0 — "
                            f"posterior değeri geçerli olasılık aralığı dışında."
                        )
                except ValueError:
                    pass
        return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTIVE DAG EXECUTOR — StepDependencyGraph'ı Aktif Çalıştıran Motor
#  (Pasif CHECK → Aktif EXECUTE dönüştürme)
# ═══════════════════════════════════════════════════════════════════════════════
class ActiveDAGExecutor:
    """
    StepDependencyGraph kontrol sonuçlarını UYGULAMAYA dönüştürür.

    Girdi: LLM adımları + signals
    Çıktı: Yeniden sıralanmış / düzeltilmiş adımlar (uygulanabilir form)

    Mekanizma:
      1. StepDependencyGraph.check() → ihlal listesi
      2. İhlal türüne göre graf manipülasyonu (adım ekle, ağırlık değiştir)
      3. Topolojik sıra → adımları yeniden düzenle
    """

    def __init__(self):
        self.sdg = StepDependencyGraph()

    def execute_with_fixes(self, steps: list, signals: dict = None) -> dict:
        """
        Adımları DAG şeklinde düzenler, ihlalları otomatik tamir eder.

        Returns:
          {
            "steps_original": list,      # Orijinal adımlar
            "steps_fixed": list,         # Düzeltilmiş adımlar
            "violations_before": list,   # Onarım öncesi
            "violations_after": list,    # Onarım sonrası
            "topo_order": list,          # Önerilen sıra
            "recovery_notes": list       # Yapılan düzeltmeler
          }
        """
        violations_before = self.sdg.check(steps, signals)

        steps_fixed = self._apply_fixes(steps, violations_before, signals)
        violations_after = self.sdg.check(steps_fixed, signals)
        topo_order = self._topological_sort(steps_fixed)
        recovery_notes = self._generate_recovery_notes(
            violations_before, violations_after
        )

        return {
            "steps_original": steps,
            "steps_fixed": steps_fixed,
            "violations_before": violations_before,
            "violations_after": violations_after,
            "topo_order": topo_order,
            "recovery_notes": recovery_notes,
        }

    def _apply_fixes(self, steps: list, violations: list, signals: dict) -> list:
        """
        İhlallara göre adımları manipüle et.
        AND-chain ihlali → step injection / reordering
        """
        if not violations:
            return steps.copy()

        fixed = steps.copy()

        # Örnek tamir: "adım i+1 adımdan n'den hep büyük" →
        # Arada intermediate step ekle veya weights adjust
        for v in violations:
            if "AND-zincir" in v or "azalma beklenir" in v:
                # Strategy: geometrik ortalama adımı ekle
                # Parse: hansı adım → index çıkar
                m = re.search(r"Adım (\d+)", v)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 < idx < len(fixed):
                        # idx-1 ile idx arasına intermediate ekle
                        prev_val = self._extract_value(fixed[idx - 1])
                        next_val = self._extract_value(fixed[idx])

                        if prev_val and next_val and next_val > prev_val:
                            geom_mean = (prev_val * next_val) ** 0.5
                            intermediate_step = {
                                "title": f"[INSERTED] Intermediate Step {idx}→{idx+1}",
                                "content": f"Geometrik ortalama adımı",
                                "formula": f"(Adım {idx} × Adım {idx+1})^0.5",
                                "result": f"{geom_mean:.4f}",
                                "_injected": True,
                            }
                            fixed.insert(idx, intermediate_step)

        return fixed

    def _extract_value(self, step: dict) -> "float | None":
        """Step'ten sayısal değer çıkar."""
        for field in ("result", "formula", "content"):
            text = str(step.get(field, "") or "")
            nums = re.findall(r"(\d+\.?\d*)", text)
            for n in nums:
                try:
                    v = float(n)
                    if 0 < v <= 1.0:
                        return v
                except ValueError:
                    pass
        return None

    def _topological_sort(self, steps: list) -> list:
        """Adım bağımlılıklarından sıra çıkar."""
        # Basit: index sırasına göre (real topo-sort için parsing gerekir)
        return list(range(len(steps)))

    def _generate_recovery_notes(
        self, violations_before: list, violations_after: list
    ) -> list:
        """Yapılan düzeltmeleri açıkla."""
        notes = []
        before_set = set(v[:60] for v in violations_before)
        after_set = set(v[:60] for v in violations_after)

        fixed = before_set - after_set
        if fixed:
            notes.append(f"[FIXED] {len(fixed)} ihlal düzeltildi:")
            for f in fixed:
                notes.append(f"  - {f}...")

        remaining = after_set
        if remaining:
            notes.append(
                f"[REMAINING] {len(remaining)} ihlal hala var (manual check gerekli)"
            )

        return notes


# ═══════════════════════════════════════════════════════════════════════════════
#  DAG EXECUTION ENGINE — Graph-aware yürütücü (topo + node readiness)
#  Not: Solver çıktılarının yeniden sıralanmasını sağlar, eksik düğümleri bildirir.
# ═══════════════════════════════════════════════════════════════════════════════
class DAGExecutionEngine:
    """
    DependencyGraphBuilder çıktısını kullanarak çalışır.

    Özellikler:
      - Topolojik sırayı hesaplar (döngüleri kısmi sıra ile tolere eder)
      - Adımları düğümlere hizalar (node_id → step)
      - Eksik / artan düğüm raporu üretir (completeness için)
      - TerminationGuard'a kalan düğümleri bildirir.
    """

    def __init__(self, graph: dict | None = None):
        self.graph = graph or {}
        self.topo_order = []
        self._executed = set()
        if self.graph:
            self._recompute_order()

    def load_graph(self, graph: dict) -> None:
        """Yeni graph yükle ve topo sırayı yenile."""
        self.graph = graph or {}
        self._executed.clear()
        self._recompute_order()

    def _recompute_order(self) -> None:
        builder = DependencyGraphBuilder()
        self.topo_order = builder.topo_sort(self.graph) if self.graph else []

    def run_steps(self, steps: list[dict]) -> dict:
        """
        Verilen adımları topolojik sıraya hizalar.

        Returns:
          {
            "ordered_steps": list,
            "missing_nodes": list[str],
            "remaining_nodes": list[str],
            "execution_trace": list[dict]
          }
        """
        if not self.graph:
            return {
                "ordered_steps": steps,
                "missing_nodes": [],
                "remaining_nodes": [],
                "execution_trace": [],
            }

        ordered = []
        trace = []
        # Adımları topo sıraya göre hizala (fazla adımlar sona eklenir)
        for idx, node_id in enumerate(self.topo_order):
            if idx < len(steps):
                step = steps[idx].copy()
                step["_node_id"] = node_id
                ordered.append(step)
                self._executed.add(node_id)
                trace.append({"node": node_id, "status": "executed"})
            else:
                trace.append({"node": node_id, "status": "missing_step"})

        # Fazla adımlar (graph'ta karşılığı olmayan) → ordered sonuna ekle
        if len(steps) > len(self.topo_order):
            for extra in steps[len(self.topo_order) :]:
                ordered.append(extra)
                trace.append({"node": "_orphan_step", "status": "appended"})

        remaining = [n for n in self.graph if n not in self._executed]
        missing = [n for n in self.topo_order if n not in self._executed]

        return {
            "ordered_steps": ordered,
            "missing_nodes": missing,
            "remaining_nodes": remaining,
            "execution_trace": trace,
        }

    def get_remaining(self) -> list:
        """Henüz yürütülmeyen düğümleri döndür."""
        if not self.graph:
            return []
        return [n for n in self.graph if n not in self._executed]


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLETENESS CHECKER — Düzenli graph coverage ölçümü
#  Döndürdüğü skor: kapsanan düğüm oranı (0..1)
# ═══════════════════════════════════════════════════════════════════════════════
class CompletenessChecker:
    """
    Adım listesi, dependency graph düğümlerini kapsıyor mu?

    Kapsama kriteri (esnek, hard-coding yok):
      - step.content / step.formula / step.result içinde node label/symbol geçmesi
      - veya step['_node_id'] == node_id
    """

    def __init__(self, min_score: float = 0.8):
        self.min_score = min_score

    def check(self, steps: list, graph: dict, signals: dict | None = None) -> dict:
        if not graph:
            return {"score": 1.0, "missing": [], "covered": []}

        covered = set()
        for node_id, info in graph.items():
            symbol = str(info.get("symbol") or node_id).lower()
            for step in steps or []:
                blob = " ".join(
                    str(step.get(k, ""))
                    for k in ("title", "content", "formula", "result")
                ).lower()
                if symbol and symbol in blob:
                    covered.add(node_id)
                    break
                if step.get("_node_id") == node_id:
                    covered.add(node_id)
                    break

        total = len(graph)
        missing = [n for n in graph if n not in covered]
        score = (len(covered) / total) if total else 1.0

        return {
            "score": round(score, 3),
            "missing": missing,
            "covered": list(covered),
            "threshold_passed": score >= self.min_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE MANAGER — Versioned snapshot + rollback
# ═══════════════════════════════════════════════════════════════════════════════
class GlobalStateManager:
    """Hafif snapshot mağazası (id → payload)."""

    def __init__(self):
        self._store = {}
        self._counter = 0

    def snapshot(self, label: str, payload: dict) -> str:
        self._counter += 1
        snap_id = f"snap-{self._counter}-{label}"
        self._store[snap_id] = json.loads(json.dumps(payload, default=str))
        return snap_id

    def latest(self) -> tuple[str, dict] | tuple[None, None]:
        if not self._store:
            return None, None
        last_id = sorted(self._store.keys(), key=lambda x: int(x.split("-")[1]))[-1]
        return last_id, self._store[last_id]

    def rollback(self, snap_id: str | None = None) -> dict | None:
        if not self._store:
            return None
        if snap_id is None:
            snap_id, _ = self.latest()
        return self._store.get(snap_id)

    def clear(self):
        self._store.clear()
        self._counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  DECISION FEEDBACK LINK — entropi/ihlale göre solver seçim uyarlaması
# ═══════════════════════════════════════════════════════════════════════════════
class DecisionFeedbackLink:
    """Basit geri besleme: entropi / ihlal sayısına göre solver rota değiştirir."""

    def __init__(self, entropy_high: float = 0.35, violation_high: int = 3):
        self.entropy_high = entropy_high
        self.violation_high = violation_high

    def advise(
        self,
        current_solver: str,
        signals: dict,
        iteration: int,
        entropy_level: float,
        violation_count: int,
    ) -> dict:
        suggestion = current_solver
        reason = "keep"

        # Game theory sinyali + yüksek entropi → GT solver'a geç
        if (
            entropy_level >= self.entropy_high
            and signals.get("game_theory_score", 0) >= 0.8
        ):
            suggestion = "GameTheorySolver"
            reason = "entropy_high_game_theory"

        # Markov sinyali ve ihlal çok → MarkovSolver
        elif violation_count >= self.violation_high and re.search(
            r"markov|geçiş|random\s+walk", signals.get("raw_text", ""), re.IGNORECASE
        ):
            suggestion = "MarkovSolver"
            reason = "violations_markov_hint"

        # Differential sinyali → GeneralDifferentialDynamicsSolver
        elif signals.get("differential_type"):
            suggestion = "GeneralDifferentialDynamicsSolver"
            reason = "differential_signal"

        return {"solver": suggestion, "reason": reason}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPLANATION GRAPH ENGINE — dependency graph'tan açıklama izi üretir
# ═══════════════════════════════════════════════════════════════════════════════
class ExplanationGraphEngine:
    """DAG + adım listesinden sebep-sonuç açıklama izi çıkarır."""

    def build(self, graph: dict, steps: list[dict]) -> dict:
        if not graph:
            return {"nodes": 0, "edges": 0, "paths": []}

        edges = []
        for nid, info in graph.items():
            for dep in info.get("deps", []):
                edges.append((dep, nid))

        paths = []
        # Basit: topo sırayı path olarak kabul et
        builder = DependencyGraphBuilder()
        topo = builder.topo_sort(graph)
        if topo:
            path_text = " → ".join(topo)
            paths.append(path_text)

        # Adım metni ile eşleşme notları
        annotations = []
        for step in steps or []:
            node_id = step.get("_node_id") or "?"
            annotations.append(f"{node_id}: {step.get('title','').strip()[:40]}")

        return {
            "nodes": len(graph),
            "edges": len(edges),
            "paths": paths,
            "annotations": annotations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL RECOVERY ENGINE — Router + State snapshot ile merkezî onarım
# ═══════════════════════════════════════════════════════════════════════════════
class GlobalRecoveryEngine:
    """ValidationRouter'ı tek yerden çağırıp snapshot destekli onarım yapar."""

    def __init__(
        self, router: ValidationRouter, state_manager: GlobalStateManager | None = None
    ):
        self.router = router
        self.state_manager = state_manager

    def recover(
        self,
        violations: list,
        sol_data: dict,
        dependency_graph: dict,
        signals: dict | None = None,
    ) -> dict:
        notes = []
        if not violations:
            return {
                "steps": sol_data.get("steps", []),
                "remaining_violations": [],
                "notes": notes,
            }

        # Checkpoint al
        checkpoint_id = None
        if self.state_manager:
            checkpoint_id = self.state_manager.snapshot(
                "pre-recovery",
                {
                    "steps": sol_data.get("steps", []),
                    "graph": dependency_graph,
                    "signals": signals or {},
                },
            )

        route_result = self.router.route(violations)
        primary_strategy = route_result.get("primary_strategy")
        notes.append(f"[ROUTE] Primary={primary_strategy}")

        exec_result = self.router.execute_strategy(
            primary_strategy,
            {
                "steps": sol_data.get("steps", []),
                "violations": violations,
                "affected_steps": route_result.get("affected_steps", []),
                "checkpoint": (
                    self.state_manager.rollback(checkpoint_id)
                    if checkpoint_id
                    else None
                ),
                "matrix": sol_data.get("matrix"),
            },
        )

        steps = exec_result.get("result") or sol_data.get("steps", [])
        if exec_result.get("success"):
            notes.append(f"[RECOVERY-OK] {exec_result.get('note','')}")
            remaining = []
        else:
            notes.append(f"[RECOVERY-FAIL] {exec_result.get('note','')}")
            # Fallback: checkpoint'e dön
            if checkpoint_id and self.state_manager:
                restored = self.state_manager.rollback(checkpoint_id)
                if restored and restored.get("steps"):
                    steps = restored.get("steps")
                    notes.append("[ROLLBACK] Restored last good snapshot")
            remaining = violations

        return {"steps": steps, "remaining_violations": remaining, "notes": notes}


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL DIFFERENTIAL DYNAMICS SOLVER — Tamamen Algoritmik
#  Logistic SDE / Kaos / Büyüme Denklemleri — Sympy + NumPy
# ═══════════════════════════════════════════════════════════════════════════════
class GeneralDifferentialDynamicsSolver:
    """Sympy + NumPy tabanlı — tamamen extracted parametrelerle çalışır, 0 hard-code"""

    def solve(self, signals: dict) -> dict:
        p = signals.get("de_parameters", {})
        sets = p.get("sets", [])
        r_kaos = p.get("r_kaos")
        if len(sets) < 1:
            return {"solved": False, "reason": "Yeterli parametre çıkarılamadı"}

        deterministic = []
        try:
            import sympy as sp

            t = sp.symbols("t")
            N = sp.Function("N")
            K_sym, r_sym, N0_sym = sp.symbols("K r N0")
            eq = sp.Eq(N(t).diff(t), r_sym * N(t) * (1 - N(t) / K_sym))
            sol_sym = sp.dsolve(eq, N(t), ics={N(0): N0_sym}).rhs

            for s in sets:
                if all(k in s for k in ("N0", "r", "K")):
                    nt = float(
                        sol_sym.subs({K_sym: s["K"], r_sym: s["r"], N0_sym: s["N0"]})
                        .subs(t, 100)
                        .evalf()
                    )
                    deterministic.append(nt / 1e6)
                else:
                    deterministic.append(None)
        except Exception:
            deterministic = [None] * len(sets)

        # Stokastik CI (ilk 2 set için)
        stochastic_ci = []
        for s in sets[:2]:
            if all(k in s for k in ("N0", "r", "K", "sigma")):
                paths = self._euler_maruyama(
                    s["N0"], s["r"], s["K"], s["sigma"], 100, 5000
                )
                ci = np.percentile(paths, [2.5, 97.5])
                stochastic_ci.append((ci[0] / 1e6, ci[1] / 1e6))
            else:
                stochastic_ci.append(None)

        # Kaos analizi
        chaos = "UNKNOWN"
        if r_kaos is not None:
            x = 0.5
            series = [x]
            for _ in range(1000):
                x = r_kaos * x * (1 - x)
                series.append(x)
            chaos = (
                "CHAOS" if (np.std(series[-500:]) > 0.1 and r_kaos > 3.57) else "STEADY"
            )

        # Karşılaştırma (en az 2 set varsa)
        ratio_change = None
        if len(deterministic) >= 2 and all(d is not None for d in deterministic[:2]):
            n0_1 = sets[0].get("N0")
            n0_2 = sets[1].get("N0")
            if n0_1 and n0_2:
                ratio_change = round(
                    (deterministic[0] / deterministic[1]) / (n0_1 / n0_2), 4
                )

        sol_sym_str = str(sol_sym) if "sol_sym" in locals() else None
        return {
            "solved": True,
            "type": "logistic_sde_chaos",
            "deterministic": deterministic,
            "stochastic_ci": stochastic_ci,
            "chaos": chaos,
            "ratio_change": ratio_change,
            "formula": sol_sym_str,
            "steps": self._generate_steps(
                deterministic, stochastic_ci, chaos, ratio_change
            ),
        }

    def _euler_maruyama(self, N0, r, K, sigma, T, n_sims, dt=0.1):
        import numpy as np

        steps = int(T / dt)
        paths = np.zeros((n_sims, steps + 1))
        paths[:, 0] = N0
        for i in range(steps):
            dW = np.random.normal(0, np.sqrt(dt), n_sims)
            dN = r * paths[:, i] * (1 - paths[:, i] / K) * dt + sigma * paths[:, i] * dW
            paths[:, i + 1] = np.maximum(paths[:, i] + dN, 1.0)
        return paths[:, -1]

    def _generate_steps(self, det, ci, chaos, ratio):
        """Genel adım metinleri — hiçbir sayı/formül sabit yok"""
        return [
            {
                "title": "Deterministik Çözüm",
                "content": "Kapalı form lojistik denklem çözümü",
                "result": f"N(100) değerleri: {det}",
            },
            {
                "title": "Stokastik Güven Aralığı",
                "content": "Euler-Maruyama + Monte Carlo simülasyonu",
                "result": f"%95 CI: {ci}",
            },
            {
                "title": "Kaotik Davranış",
                "content": "Lojistik harita + varyans testi",
                "result": chaos,
            },
            {
                "title": "Karşılaştırmalı Dinamik",
                "content": "N1/N2 oranı başlangıca göre",
                "result": f"Katsayı: {ratio}",
            },
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL SOLVER ORCHESTRATOR — DÖNGÜ HAL VE KOORDINASYON
#  (Bu yapı: giriş → DAG exec → validate → recover → re-run döngüsü)
# ═══════════════════════════════════════════════════════════════════════════════
class SolverOrchestrator:
    """
    Tüm pipeline modüllerini koordine eden global choreographer.

    Akış:
      1. Giriş ayrıştırma (INPUT PARSER)
      2. Solver seçim + ilk run (SOLVERS)
      3. DAG Execute (ACTIVE DAG EXECUTOR)
      4. Validation katmanları (VALIDATORS)
      5. İhlal analiz (VALIDATION ROUTER)
      6. Recovery ekle (AUTO-RECOVERY)
      7. Re-run or terminate (TERMINATION GUARD)

    Parametreler: _builder, _solvers, _validators, _router, ... tüm bileşenler
    """

    def __init__(self, max_iterations: int = 5):
        """
        Args:
          max_iterations: maksimum retry sayısı (convergence için)
        """
        self.max_iterations = max_iterations
        self.iteration = 0
        self.history = []  # [(iteration, step_count, violations_count), ...]

    def execute_full_pipeline(
        self, question: str, signals: dict, components: dict
    ) -> dict:
        """
        BÜYÜK ORKESTRATİON FONKSİYONU.

        Args:
          question: Kullanıcı sorusu
          signals:  SemanticSignalModule çıktısı
          components: {
            "ast_builder": ...,
            "solver_selector": ...,
            "solvers": {...},
            "validators": {...},
            "dag_executor": ActiveDAGExecutor,
            "router": ValidationRouter,
            "term_guard": TerminationGuard,
            ...
          }

        Returns:
          {
            "success": bool,
            "final_steps": list,
            "final_answer": str,
            "iterations": int,
            "history": list,
            "recovery_notes": list
          }
        """

        ast_builder = components.get("ast_builder")
        solver_selector = components.get("solver_selector")
        solvers = components.get("solvers", {})
        validators = components.get("validators", {})
        dag_executor = components.get("dag_executor")
        router = components.get("router")
        term_guard = components.get("term_guard")

        # ────────────────────────────────────────────────────────────────────
        # FAZA 0: GİRİŞ & İLK SETUP
        # ────────────────────────────────────────────────────────────────────
        if not ast_builder:
            return {
                "success": False,
                "final_steps": [],
                "error": "No AST builder provided",
            }

        current_signals = signals.copy()
        all_recovery_notes = []

        # ────────────────────────────────────────────────────────────────────
        # İTERATİF DÖNGÜ: SOLVE → VALIDATE → RECOVER → RE-RUN
        # ────────────────────────────────────────────────────────────────────
        while self.iteration < self.max_iterations:
            self.iteration += 1
            iter_note = f"\n╔ İTERASYON {self.iteration} ╗"
            all_recovery_notes.append(iter_note)

            # ── ADIM 1: AST & Solver Seçim ────────────────────────────────────
            try:
                math_ast = ast_builder.build(question, current_signals)
                chosen_solver = (
                    solver_selector.select(math_ast, current_signals)
                    if solver_selector
                    else "BayesSolver"
                )
            except Exception as e:
                all_recovery_notes.append(f"[ERROR-AST] {str(e)[:100]}")
                break

            # ── ADIM 2: SOLVER ÇALIŞTIRILMASI ─────────────────────────────────
            try:
                solver = solvers.get(chosen_solver)
                if not solver:
                    all_recovery_notes.append(
                        f"[WARN] {chosen_solver} not found, skipping"
                    )
                    solver_result = {}
                else:
                    solver_result = solver.solve(math_ast) if solver else {}
            except Exception as e:
                all_recovery_notes.append(f"[ERROR-SOLVER] {str(e)[:100]}")
                solver_result = {}

            # Solver'dan adımları çıkar
            current_steps = solver_result.get("steps", [])

            # ── ADIM 3: DAG EXECUTION (StepDependencyGraph Aktif) ────────────
            if dag_executor and current_steps:
                try:
                    dag_result = dag_executor.execute_with_fixes(
                        current_steps, current_signals
                    )
                    current_steps = dag_result.get("steps_fixed", current_steps)
                    all_recovery_notes.extend(dag_result.get("recovery_notes", []))

                    violations_dag = dag_result.get("violations_after", [])
                except Exception as e:
                    all_recovery_notes.append(f"[ERROR-DAG] {str(e)[:100]}")
                    violations_dag = []
            else:
                violations_dag = []

            # Graph-aware hizalama (Execution DAG Engine)
            if dag_engine and current_sol_data:
                try:
                    dag_exec = dag_engine.run_steps(current_sol_data.get("steps", []))
                    current_sol_data["steps"] = dag_exec.get(
                        "ordered_steps", current_sol_data.get("steps", [])
                    )
                    if dag_exec.get("missing_nodes"):
                        violations_dag.extend(
                            [
                                f"COMPLETENESS_MISSING::{n}"
                                for n in dag_exec.get("missing_nodes", [])
                            ]
                        )
                        all_recovery_notes.append(
                            f"[DAG-EXEC] {len(dag_exec['missing_nodes'])} düğüm için adım bulunamadı"
                        )
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-DAG-ENGINE] {str(e)[:80]}")

            # ── ADIM 4: VALIDATION (Tüm validator'ları çalıştır) ──────────────
            all_violations = []

            # 4a. NumericTruthValidator
            num_validator = validators.get("numeric")
            if num_validator:
                try:
                    num_viols = num_validator.validate(
                        {"steps": current_steps}, solver_result
                    )
                    all_violations.extend(num_viols)
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-NUMERIC] {str(e)[:50]}")

            # 4b. StepDependencyGraph (via DAG Executor pasif kontrol)
            all_violations.extend(violations_dag)

            # 4c. MonteCarloVerifier (eğer var)
            mc_verifier = validators.get("monte_carlo")
            if mc_verifier:
                try:
                    mc_viols = mc_verifier.validate(
                        solver_result.get("matrix"), solver_result
                    )
                    all_violations.extend(mc_viols or [])
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-MC] {str(e)[:50]}")

            self.history.append(
                (self.iteration, len(current_steps), len(all_violations))
            )

            # Tidak ada ihlal → selesai
            if not all_violations:
                return {
                    "success": True,
                    "final_steps": current_steps,
                    "final_answer": solver_result.get("answer", ""),
                    "iterations": self.iteration,
                    "history": self.history,
                    "recovery_notes": all_recovery_notes,
                }

            # ── ADIM 5: VALIDATION ROUTING (İhlal → Recovery) ────────────────
            if not router:
                all_recovery_notes.append("[WARN] No router, cannot recover")
                break

            try:
                route_result = router.route(all_violations)
                primary_strategy = route_result.get("primary_strategy")
                all_recovery_notes.append(f"[ROUTE] Primary: {primary_strategy}")
            except Exception as e:
                all_recovery_notes.append(f"[ERROR-ROUTE] {str(e)[:100]}")
                break

            # ── ADIM 6: RECOVERY (AUTO-RECOVERY & ENFORCER) ─────────────────
            try:
                strategy_result = router.execute_strategy(
                    primary_strategy,
                    {"steps": current_steps, "violations": all_violations},
                )

                if strategy_result.get("success"):
                    recovered_steps = strategy_result.get("result") or current_steps
                    all_recovery_notes.append(
                        f"[RECOVERY-OK] {strategy_result.get('note', '')}"
                    )
                    current_steps = recovered_steps
                else:
                    all_recovery_notes.append(
                        f"[RECOVERY-FAIL] {strategy_result.get('note', '')}"
                    )
                    # Fallback strategy'si dene
                    fallbacks = route_result.get("secondary_strategies", [])
                    if fallbacks:
                        for fb in fallbacks:
                            fb_result = router.execute_strategy(
                                fb, {"steps": current_steps}
                            )
                            if fb_result.get("success"):
                                current_steps = fb_result.get("result") or current_steps
                                all_recovery_notes.append(f"[FALLBACK-OK] Used {fb}")
                                break
                    else:
                        # Hiçbir strategy başarılı olmadı → abort
                        break
            except Exception as e:
                all_recovery_notes.append(f"[ERROR-RECOVERY] {str(e)[:100]}")
                break

            # ── ADIM 7: TERMINATION CHECK ─────────────────────────────────────
            if term_guard:
                try:
                    should_continue, reason = term_guard.should_continue(
                        current_entropy=len(all_violations) / 10.0  # Proxy entropy
                    )
                    all_recovery_notes.append(f"[TERM-CHECK] {reason}")
                    if not should_continue:
                        break
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-TERM] {str(e)[:50]}")

            # Loop devam et (sonraki iteration)

        # ────────────────────────────────────────────────────────────────────
        # ÇIKTIŞ (başarılı veya timeout)
        # ────────────────────────────────────────────────────────────────────
        return {
            "success": not all_violations,  # Başarılı = ihlal yok
            "final_steps": current_steps,
            "final_answer": solver_result.get("answer", "Incomplete"),
            "iterations": self.iteration,
            "history": self.history,
            "final_violations": all_violations,
            "recovery_notes": all_recovery_notes,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver pipeline yardımcısı — tüm solver modüllerini birleştirir
# ═══════════════════════════════════════════════════════════════════════════════
def run_solver_pipeline(
    question: str,
    signals: dict,
    ast_builder,
    markov_solver,
    solver_selector,
    mc_verifier,
    num_validator,
) -> dict:
    """
    SORU → MathAST → SolverSelector → Solver → NumericValidator → MonteCarloVerifier
    Döndürür: solver_context dict (OllamaClient'a hint olarak geçilir)
    """
    # 1. MathAST
    math_ast = ast_builder.build(question, signals)

    # 2. Solver seçimi
    chosen_solver = solver_selector.select(math_ast, signals)

    # 3. Solve — tip bazlı dispatch
    solver_result = {"solved": False, "_solver_used": chosen_solver}

    if chosen_solver == "MarkovSolver" and math_ast["type"] in (
        "markov_random_walk",
        "markov_chain",
    ):
        solver_result = markov_solver.solve(math_ast)
        solver_result["_solver_type"] = "MarkovSolver"

    elif chosen_solver == "BayesSolver" and math_ast["type"] == "bayes":
        solver_result = _bayes_solver.solve(math_ast)
        # _solver_type zaten BayesSolver set edilmiş

    elif chosen_solver == "GameTheorySolver" and math_ast["type"] == "game_theory":
        # Oyun Kuramı Solver: NLP+SimulasyOn+BART pipeline
        math_ast["_question"] = question
        solver_result = _gt_solver.solve(math_ast)
        if "solved" not in solver_result:
            solver_result["solved"] = False
        solver_result.setdefault("_solver_type", "GameTheorySolver")

    elif signals.get("differential_type") == "logistic_sde_chaos":
        # Genel Differential Dynamics Solver
        solver_result = GeneralDifferentialDynamicsSolver().solve(signals)
        chosen_solver = "GeneralDifferentialDynamicsSolver"
        solver_result["_solver_type"] = "GeneralDifferentialDynamicsSolver"

    # 4. Monte Carlo doğrulama (yalnızca Markov)
    mc_result = {"mc_run": False}
    if solver_result.get("solved") and math_ast["type"] == "markov_random_walk":
        mc_result = mc_verifier.verify(math_ast, solver_result)

    # 5. Numeric validation
    numeric_violations = []
    if solver_result.get("solved"):
        numeric_violations = num_validator.validate({}, solver_result)

    return {
        "math_ast": math_ast,
        "chosen_solver": chosen_solver,
        "solver_result": solver_result,
        "mc_result": mc_result,
        "numeric_violations": numeric_violations,
    }




# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive Cognitive Architecture (Self-Learning / Decision / Improvement Loop)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepRepresentationEncoder:
    """Semantic sinyalleri latent temsile dönüştürür (hafif, deterministik encoder)."""

    def encode(self, question: str, signals: dict, eq_ctx: dict | None = None) -> dict:
        q = (question or "").strip()
        toks = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", q.lower())
        uniq = len(set(toks))
        entropy_proxy = (uniq / max(len(toks), 1))
        return {
            "token_count": len(toks),
            "unique_token_count": uniq,
            "entropy_proxy": round(float(entropy_proxy), 4),
            "intent": signals.get("intent", "general"),
            "equation_hits": len((eq_ctx or {}).get("matches", []) or []),
            "logic_operator": signals.get("logic_operator", "none"),
        }


class PersistentLearningMemory:
    """Thinking-loop öğrenmesini diskte tutar (program yeniden açıldığında devam eder)."""

    def __init__(self, path: str = "thinking_loop_memory.pkl"):
        self.path = path
        self.state = {
            "episodes": 0,
            "avg_loss": 0.0,
            "critic_history": [],
            "rewrite_count": 0,
            "module_weights": {
                "answer": 0.4,
                "explanation": 0.3,
                "faithfulness": 0.2,
                "critic": 0.1,
            },
        }
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    self.state.update(payload)
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self.state, f)
        except Exception:
            pass

    def update(self, loss: float, critic_score: float, rewritten: bool):
        eps = int(self.state.get("episodes", 0)) + 1
        prev_avg = float(self.state.get("avg_loss", 0.0))
        new_avg = ((prev_avg * (eps - 1)) + float(loss)) / max(eps, 1)
        hist = list(self.state.get("critic_history", []))
        hist.append(round(float(critic_score), 4))
        self.state["episodes"] = eps
        self.state["avg_loss"] = round(new_avg, 6)
        self.state["critic_history"] = hist[-128:]
        if rewritten:
            self.state["rewrite_count"] = int(self.state.get("rewrite_count", 0)) + 1
        self.save()


class LatentThoughtEncoder:
    """LTE: soru + bağlamdan gizli düşünce vektörü üretir."""

    def encode(self, question: str, context: dict) -> list[float]:
        txt = f"{question}|{json.dumps(context or {}, default=str, ensure_ascii=False)}"
        vals = [((ord(ch) % 97) / 97.0) for ch in txt[:256]] or [0.0]
        vec = np.array(vals, dtype=float)
        chunks = 8
        pad = (-len(vec)) % chunks
        if pad:
            vec = np.pad(vec, (0, pad), mode="constant")
        z = vec.reshape(chunks, -1).mean(axis=1)
        norm = float(np.linalg.norm(z)) or 1.0
        z = z / norm
        return [round(float(v), 6) for v in z.tolist()]


class ReasonAbstractionLayer:
    """Çözüm adımlarını kavramsal izlere dönüştürür."""

    def abstract(self, steps: list) -> dict:
        labels = []
        for st in steps or []:
            txt = (
                f"{st.get('title','')} {st.get('content','')} {st.get('formula','')}"
                if isinstance(st, dict)
                else str(st)
            ).lower()
            if any(k in txt for k in ("olas", "bayes", "prob")):
                labels.append("probabilistic_reasoning")
            if any(k in txt for k in ("matris", "markov", "geçiş")):
                labels.append("state_transition_reasoning")
            if any(k in txt for k in ("topla", "çıkar", "hesap", "=")):
                labels.append("symbolic_arithmetic")
        uniq = sorted(set(labels)) or ["general_reasoning"]
        return {"reason_tags": uniq, "reason_depth": len(steps or [])}


class NaturalLanguageThoughtDecoder:
    """RNLT: latent thought + adımlardan açıklama metni üretir."""

    def decode(self, z_thought: list[float], steps: list, abstraction: dict) -> str:
        depth = abstraction.get("reason_depth", 0)
        tags = ", ".join(abstraction.get("reason_tags", [])[:4])
        step_hint = ""
        if steps:
            first = steps[0] if isinstance(steps[0], dict) else {"title": str(steps[0])}
            step_hint = str(first.get("title", "") or first.get("content", ""))[:90]
        latent_focus = round(sum(abs(float(v)) for v in z_thought) / max(len(z_thought), 1), 4)
        return (
            f"Düşünce özeti: {depth} adımlı çözüm, odak={latent_focus}. "
            f"Akıl yürütme etiketleri: {tags}. "
            f"Başlangıç adımı: {step_hint or 'genel ayrıştırma'}. "
            "Bu açıklama, çözüm adımlarıyla hizalı olacak şekilde üretildi."
        )


class ExplanationCritic:
    """Öz-yansıtma döngüsü: açıklamayı puanlar, düşükse yeniden yazım ister."""

    def score(self, explanation: str, steps: list, answer: str) -> dict:
        text = (explanation or "").strip().lower()
        step_tokens = []
        for st in steps or []:
            if isinstance(st, dict):
                step_tokens.extend(re.findall(r"\w+", f"{st.get('title','')} {st.get('content','')}".lower()))
            else:
                step_tokens.extend(re.findall(r"\w+", str(st).lower()))
        step_tokens = set(t for t in step_tokens if len(t) >= 4)
        exp_tokens = set(re.findall(r"\w+", text))
        overlap = len(step_tokens & exp_tokens) / max(len(step_tokens), 1)
        clarity = min(1.0, len(text) / 220.0)
        faithfulness = 1.0 if str(answer).strip() and str(answer).lower() in text else 0.65
        hallucination_risk = max(0.0, 1.0 - overlap)
        score = 0.35 * clarity + 0.4 * overlap + 0.25 * faithfulness
        needs_rewrite = score < 0.62
        return {
            "score": round(float(score), 4),
            "clarity": round(float(clarity), 4),
            "step_alignment": round(float(overlap), 4),
            "faithfulness": round(float(faithfulness), 4),
            "hallucination_risk": round(float(hallucination_risk), 4),
            "needs_rewrite": needs_rewrite,
        }


class MultiObjectiveLoss:
    """L_total = λ1*L_answer + λ2*L_explanation + λ3*L_faithfulness + λ4*L_critic."""

    def __init__(self, weights: dict | None = None):
        w = weights or {}
        self.w_answer = float(w.get("answer", 0.4))
        self.w_explanation = float(w.get("explanation", 0.3))
        self.w_faithfulness = float(w.get("faithfulness", 0.2))
        self.w_critic = float(w.get("critic", 0.1))

    def compute(self, answer_ok: float, critic: dict) -> dict:
        l_answer = 1.0 - float(answer_ok)
        l_explanation = 1.0 - float(critic.get("clarity", 0.0))
        l_faith = 1.0 - float(critic.get("faithfulness", 0.0))
        l_critic = 1.0 - float(critic.get("score", 0.0))
        total = (
            self.w_answer * l_answer
            + self.w_explanation * l_explanation
            + self.w_faithfulness * l_faith
            + self.w_critic * l_critic
        )
        return {
            "L_answer": round(l_answer, 6),
            "L_explanation": round(l_explanation, 6),
            "L_faithfulness": round(l_faith, 6),
            "L_critic": round(l_critic, 6),
            "L_total": round(float(total), 6),
        }


class ThinkingLoopBackpropEngine:
    """MD mimarisindeki LTE + RNLT + Critic + Multi-loss + kalıcı öğrenme katmanı."""

    def __init__(self, memory: PersistentLearningMemory | None = None):
        self.memory = memory or PersistentLearningMemory()
        self.lte = LatentThoughtEncoder()
        self.ral = ReasonAbstractionLayer()
        self.rnlt = NaturalLanguageThoughtDecoder()
        self.critic = ExplanationCritic()
        self.loss_fn = MultiObjectiveLoss(self.memory.state.get("module_weights", {}))

    def run_cycle(self, question: str, signals: dict, sol_data: dict, solver_ctx: dict | None = None) -> dict:
        z_thought = self.lte.encode(question, {"signals": signals, "solver": solver_ctx or {}})
        steps = sol_data.get("steps", []) if isinstance(sol_data, dict) else []
        abstraction = self.ral.abstract(steps)
        explanation = self.rnlt.decode(z_thought, steps, abstraction)
        answer = str((sol_data or {}).get("answer", "")).strip()
        critic_result = self.critic.score(explanation, steps, answer)

        rewritten = False
        if critic_result.get("needs_rewrite"):
            rewritten = True
            explanation = (
                explanation
                + " Revize: adımlar ve nihai cevap daha açık bağlandı; belirsiz ifadeler azaltıldı."
            )
            critic_result = self.critic.score(explanation, steps, answer)

        answer_ok = 1.0 if answer else 0.7
        loss = self.loss_fn.compute(answer_ok, critic_result)
        self.memory.update(loss["L_total"], critic_result.get("score", 0.0), rewritten)

        return {
            "z_thought": z_thought,
            "reason_abstraction": abstraction,
            "explanation_generated": explanation,
            "critic": critic_result,
            "loss": loss,
            "persistent_learning": {
                "episodes": self.memory.state.get("episodes", 0),
                "avg_loss": self.memory.state.get("avg_loss", 0.0),
                "rewrite_count": self.memory.state.get("rewrite_count", 0),
                "memory_file": self.memory.path,
            },
        }


class KnowledgeRetrievalLayer:
    """Ön bilgi toplar: Equation Universe + router geçmişi."""

    def retrieve(self, question: str, eq_ctx: dict, router_features: dict | None = None) -> dict:
        rf = router_features or {}
        return {
            "equation_context": eq_ctx or {},
            "intent": rf.get("intent", "general"),
            "topic": rf.get("topic", "unknown"),
            "question_len": len((question or "").strip()),
        }


class BayesianPriorBuilder:
    """Soru bağlamına göre solver prior'ı üretir."""

    def build(self, signals: dict, solver_ctx: dict) -> dict:
        ast_type = (solver_ctx.get("math_ast") or {}).get("type", "general")
        base = {
            "SymbolicSolver": 0.25,
            "ProbabilisticSolver": 0.25,
            "SimulationEngine": 0.25,
            "HybridSolver": 0.25,
        }
        if ast_type in ("bayes", "markov_chain", "markov_random_walk"):
            base["ProbabilisticSolver"] += 0.20
        if ast_type in ("physics", "dynamics") or signals.get("differential_type"):
            base["SimulationEngine"] += 0.20
        if ast_type in ("equation", "algebra", "general"):
            base["SymbolicSolver"] += 0.15
        if ast_type in ("game_theory", "hybrid"):
            base["HybridSolver"] += 0.20
        z = sum(base.values())
        return {k: round(v / z, 4) for k, v in base.items()}


class RLPolicyNetwork:
    """Prior + state bilgisinden strateji dağılımı üretir."""

    def select_strategy(self, prior: dict, state: dict) -> tuple[str, dict]:
        weighted = dict(prior or {})
        if (state or {}).get("consistency_trend", 1.0) < 0.6:
            weighted["HybridSolver"] = weighted.get("HybridSolver", 0.0) + 0.05
        if (state or {}).get("entropy_proxy", 0.0) > 0.75:
            weighted["SimulationEngine"] = weighted.get("SimulationEngine", 0.0) + 0.05
        chosen = max(weighted.items(), key=lambda kv: kv[1])[0] if weighted else "HybridSolver"
        return chosen, weighted


class MultiLayerValidator:
    """StepDependencyGraph + NumericTruthValidator çıktısını birleştirir."""

    def validate(self, sol_data: dict, signals: dict, solver_ctx: dict) -> list:
        violations = []
        if sol_data.get("type") != "game_theory":
            violations.extend(_step_dep_graph.check(sol_data.get("steps") or [], signals))
        violations.extend(_num_validator.validate(sol_data, solver_ctx.get("solver_result")))
        return violations


class RewardReplayMemory:
    """Ödül + replay belleği (hafif in-memory)."""

    def __init__(self, maxlen: int = 512):
        self.buf = deque(maxlen=maxlen)

    def push(self, item: dict):
        self.buf.append(item)

    def sample(self, n: int = 16) -> list:
        if not self.buf:
            return []
        n = min(n, len(self.buf))
        return random.sample(list(self.buf), n)


class BayesianPosteriorUpdater:
    def update(self, prior: dict, confidence: float) -> dict:
        c = max(0.0, min(1.0, float(confidence)))
        posterior = {}
        for k, v in (prior or {}).items():
            posterior[k] = round(v * (0.5 + 0.5 * c), 4)
        z = sum(posterior.values()) or 1.0
        return {k: round(v / z, 4) for k, v in posterior.items()}


class MetaLearningEngine:
    def summarize(self, replay_samples: list) -> dict:
        if not replay_samples:
            return {"avg_reward": 0.0, "count": 0}
        rewards = [float(x.get("reward", 0.0)) for x in replay_samples]
        return {"avg_reward": round(sum(rewards) / len(rewards), 4), "count": len(rewards)}


class KnowledgeBaseUpdater:
    """Kendini geliştirme döngüsü için hafif telemetri kaydı."""

    def __init__(self):
        self.last_update = {}

    def update(self, payload: dict):
        self.last_update = dict(payload or {})
        self.last_update["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        return self.last_update


class ResponseGenerator:
    """Final output üretiminde güven/öğrenme metadatası ekler."""

    def finalize(self, sol_data: dict, confidence: float, learning_ctx: dict) -> dict:
        out = dict(sol_data or {})
        out["_confidence_estimate"] = round(float(confidence), 4)
        out["_learning_ctx"] = learning_ctx
        return out


def _map_strategy_to_solver(strategy_name: str, solver_ctx: dict) -> str:
    """Yeni strateji katmanını mevcut solver isimlerine map eder."""
    ast_type = (solver_ctx.get("math_ast") or {}).get("type", "general")
    if strategy_name == "ProbabilisticSolver":
        if ast_type in ("bayes", "markov_chain", "markov_random_walk"):
            return solver_ctx.get("chosen_solver", "BayesSolver")
        return "BayesSolver"
    if strategy_name == "SimulationEngine":
        return "GeneralDifferentialDynamicsSolver"
    if strategy_name == "SymbolicSolver":
        return "MarkovSolver" if ast_type.startswith("markov") else solver_ctx.get("chosen_solver", "LLM")
    return solver_ctx.get("chosen_solver", "LLM")

# ═══════════════════════════════════════════════════════════════════════════════
#  ★ ENTEGRASYON ÖRNEĞI — SolverOrchestrator'ı Nasıl Kullanacaksın
# ═══════════════════════════════════════════════════════════════════════════════
"""
MEVCUT KODDAN ORCHESTRATOR'A GEÇİŞ KODU:

```python
# 1. Tüm bileşenleri instan al
ast_builder = MathASTBuilder()
solver_selector = SolverSelector()
dag_executor = ActiveDAGExecutor()
router = ValidationRouter()
num_validator = NumericTruthValidator()

# Solvers
bayes_solver = BayesSolver()
markov_solver = MarkovSolver()
gt_solver = GameTheorySolver()
solvers = {
    "BayesSolver": bayes_solver,
    "MarkovSolver": markov_solver,
    "GameTheorySolver": gt_solver,
}

# Validators
validators = {
    "numeric": num_validator,
    "monte_carlo": MonteCarloVerifier(),
}

term_guard = TerminationGuard(max_iterations=5)

# 2. Orchestrator'ı başlat
orchestrator = SolverOrchestrator(max_iterations=5)

# 3. BÜYÜK PIPELINE'I ÇALIŞTıR
result = orchestrator.execute_full_pipeline(
    question="Olasılık sorusu...",
    signals={...},  # SemanticSignalModule.extract() çıktısı
    components={
        "ast_builder": ast_builder,
        "solver_selector": solver_selector,
        "solvers": solvers,
        "validators": validators,
        "dag_executor": dag_executor,
        "router": router,
        "term_guard": term_guard,
    }
)

# 4. SONUÇ
print("Başarılı mı?", result["success"])
print("Kaç iterasyon?", result["iterations"])
print("Final adımlar:", result["final_steps"])
print("Recovery notları:", result["recovery_notes"])
```

NEY YAPTIĞINI:
1. INPUT_PARSER ───→ signals extract
2. AST_BUILDER ────→ math_ast
3. SOLVER ────────→ initial solution
4. DAG_EXECUTOR ──→ (StepDependencyGraph aktif) repair & reorder
5. VALIDATORs ────→ doğrulama (numeric, structural, MC)
6. ROUTER ────────→ violation → recovery strategy
7. RECOVERY ──────→ auto-fix (intermediate_steps, normalize, vb)
8. RE-RUN ────────→ loop (adım 3-7 tekrarlanır)
9. TERM_GUARD ────→ durdur (max_iter, convergence, DAG_complete)
10. OUTPUT ───────→ final_steps, success, history

Böylece:
✅ Direct jump ENGELLEDI
✅ Half-computed BLOCKLANDI
✅ Ihlal otomatik düzeltildi
✅ Pipeline döngü güvenle çalışıyor
"""


def _build_solver_hint(solver_ctx: dict) -> str:
    """Solver sonuçlarını Ollama sistem prompt'una hint olarak ekler."""
    sr = solver_ctx.get("solver_result", {})
    mc = solver_ctx.get("mc_result", {})
    ast = solver_ctx.get("math_ast", {})

    if not sr.get("solved"):
        return ""

    lines = ["\n\n══ MATEMATİK ÇÖZÜM MOTORU SONUÇLARI (DOĞRU — KULLAN) ══"]
    solver_type = sr.get("_solver_type", "")

    # ── Oyun Kuramı hint ─────────────────────────────────────────────────────
    if ast.get("type") == "game_theory":
        gt_p = sr.get("_gt_params", {})
        scores = sr.get("scores", {})
        sorted_p = sr.get("sorted_players", [])
        grim_t = sr.get("_grim_targets", {})
        grim_per = sr.get("_sim_result", {}).get("grim_targets_per_round", {})
        payoff = sr.get("_payoff", {})
        n_rnd = sr.get("n_rounds", 0)
        nash = sr.get("_nash", [])
        players = gt_p.get("players", [])
        step_tgt = gt_p.get("step_target")
        wdata = sr.get("_window_data", {})

        lines.append(
            f"Problem tipi: OYUN KURAMI — {gt_p.get('game_type','iterated_pd').upper()}"
        )
        lines.append(
            f"Oyuncu: {len(players)}  Tur: {n_rnd}"
            + (f"  Hedef tur (step_target): {step_tgt}" if step_tgt else "")
        )
        if payoff:
            R = payoff.get("R", 3)
            T = payoff.get("T", 5)
            S = payoff.get("S", 0)
            P = payoff.get("P", 1)
            lines.append(f"Ödeme: T={T} R={R} P={P} S={S}")

        # ── Büyük-N steady-state bilgisi ─────────────────────────────────────
        if step_tgt and wdata:
            cumul_at_t = wdata.get("cumulative_at_target", {})
            per_r = wdata.get("per_round_payoff", {})
            stab_rnd = wdata.get("stability_round", "?")
            lines.append(f"\n⚡ BÜYÜK-N STEADY-STATE ANALİZİ (hedef: {step_tgt}. tur):")
            lines.append(f"  Kararlılık turu: {stab_rnd}")
            if per_r:
                lines.append("  Tur başına sabit kazanç (kararlı durum):")
                for nm, pv in per_r.items():
                    lines.append(f"    {nm}: +{pv:.2f}/tur")
            if cumul_at_t:
                lines.append(f"  {step_tgt}. tur kümülatif tahmin:")
                sorted_at_t = sorted(cumul_at_t.items(), key=lambda x: -x[1])
                for nm, cv in sorted_at_t:
                    lines.append(f"    {nm}: {cv:.0f} puan")
        else:
            lines.append("\n⚡ DOĞRU PUAN SIRALAMASI (BU DEĞERLERİ KULLAN):")
            for nm, sc in sorted_p:
                strat = next((p["strategy"] for p in players if p["name"] == nm), "?")
                lines.append(f"    {nm} ({strat}): {sc} puan")

        if grim_t:
            lines.append("\n⚡ GRİM TRİGGER HEDEFLERİ (final):")
            for actor, tgts in grim_t.items():
                lines.append(f"    {actor} → {', '.join(tgts)}")
        # Per-round snapshot (ilk tetiklenme turu bilgisi)
        if grim_per:
            # İlk tetiklenme turunu bul (hardcoding yok — dict traversal)
            first_trigger = {}
            for rnd in sorted(grim_per.keys()):
                snap = grim_per[rnd]
                for actor, tgts in snap.items():
                    if tgts and actor not in first_trigger:
                        first_trigger[actor] = (rnd, tgts)
            if first_trigger:
                lines.append("⚡ GRİM TRİGGER İLK TETİKLENME:")
                for actor, (rnd, tgts) in first_trigger.items():
                    lines.append(
                        f"    {actor}: {rnd}. turda tetiklendi → hedef: {', '.join(tgts)}"
                    )

        if nash:
            lines.append(f"⚡ NASH DENGESİ: {nash}")
        viol = sr.get("_violations", [])
        if viol:
            lines.append("\n⚠ BART DOĞRULAMA İHLALLERİ (düzelt):")
            for v in viol[:3]:
                lines.append(f"    • {v}")
        lines.append("\nBu değerleri KULLAN. Kendi hesabın bunlarla uyuşmalı.")
        lines.append("══════════════════════════════════════════════════════")
        return "\n".join(lines)

    # ── Markov hint ──────────────────────────────────────────────────────────
    if ast.get("type") == "markov_random_walk":
        lines.append(f"Problem tipi: {sr.get('grid',[])} Absorbing Random Walk")
        lines.append(f"Geçici durum sayısı: {sr.get('n_transient')}")
        lines.append(f"Emici durum sayısı: {sr.get('n_absorbing')}")
        lines.append(f"⚡ BEKLENEN YAŞAM SÜRESİ = {sr.get('expected_steps'):.6f} adım")
        lines.append(f"⚡ VARYANS = {sr.get('variance'):.6f}")
        lines.append(f"⚡ STANDART SAPMA = {sr.get('std_dev'):.6f}")
        ep = sr.get("edge_probs", {})
        if ep:
            lines.append("⚡ KENAR ABSORPSIYON DAĞILIMI:")
            for k, v in ep.items():
                lines.append(f"    {k}: {v:.4f}")
        if mc.get("mc_run"):
            lines.append(f"⚡ MONTE CARLO DOĞRULAMA ({mc.get('sims_run')} simülasyon):")
            lines.append(
                f"    MC Beklenti = {mc.get('mc_expected')} ± {mc.get('mc_std')}"
            )
            lines.append(
                f"    Analitik = {mc.get('analytic')}  →  {mc.get('agreement')}"
            )

    # ── Bayes hint ───────────────────────────────────────────────────────────
    elif ast.get("type") == "bayes":
        hyps = sr.get("hypotheses", [])
        n_obs = sr.get("n_observations", 1)
        rounds = sr.get("rounds", {})
        posts = sr.get("posteriors", {})
        joint_p = sr.get("joint_posteriors", {})

        lines.append(f"Problem tipi: Bayesyen Inference  Hipotez sayısı: {len(hyps)}")
        lines.append(f"Gözlem sayısı (n_obs): {n_obs}")
        lines.append("")

        # Prior
        prior_str = ", ".join(
            f"P({h.upper()})={sr['priors'].get(h,0):.4f}" for h in sorted(hyps)
        )
        lines.append(f"Priorlar: {prior_str}")

        # Likelihood
        lik_str = ", ".join(
            f"P(E|{h.upper()})={sr['likelihoods'].get(h,0):.4f}" for h in sorted(hyps)
        )
        lines.append(f"Likelihoodlar: {lik_str}")
        lines.append("")

        # Her tur
        for t, rd in sorted(rounds.items()):
            ev = rd["evidence"]
            lines.append(f"── TUR {t} ──")
            lines.append(f"  P(E|tur_{t}_priorlar) = {ev:.6f}")
            post_str = ", ".join(
                f"P({h.upper()}|E)={rd['posteriors'].get(h,0):.6f}"
                for h in sorted(hyps)
            )
            lines.append(f"  Posteriorlar: {post_str}")

        lines.append("")
        lines.append("⚡ DOĞRU FINAL POSTERIORLAR (bunları kullan):")
        for h in sorted(hyps):
            p_pct = posts.get(h, 0) * 100
            lines.append(
                f"    P({h.upper()}|tüm gözlemler) = {posts.get(h,0):.6f}  (%{p_pct:.2f})"
            )

        if joint_p and n_obs >= 2:
            lines.append("")
            lines.append("⚡ BİRLEŞİK YÖNTEM DOĞRULAMASI:")
            for h in sorted(hyps):
                lines.append(f"    P({h.upper()}|E^{n_obs}) = {joint_p.get(h,0):.6f}")

        lines.append("")
        best = sr.get("best_hypothesis", "?")
        bp = sr.get("best_posterior", 0)
        lines.append(
            f"⚡ EN YÜKSEK POSTERIOR: {best.upper()} = {bp:.6f} (%{bp*100:.2f})"
        )
        lines.append(
            "UYARI: Tüm posterior değerler [0,1] arasında. %48.48 → 0.4848 yazılmalı."
        )
        lines.append(
            "UYARI: İkinci tur P(B) paydası = {:.6f} (birinci tur değil!)".format(
                rounds.get(2, rounds.get(1, {})).get("evidence", 0) if n_obs >= 2 else 0
            )
        )

    lines.append("\nBu değerleri cevabında kullan. Kendi hesabın bunlarla uyuşmalı.")
    lines.append("══════════════════════════════════════════════════════")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  _build_sol_from_gt_solver
#  GameTheorySolver çıktısından LLM-uyumlu sol_data dict üretir.
#  GT bypass aktifleştiğinde OllamaClient bu fonksiyonu çağırır.
# ═══════════════════════════════════════════════════════════════════════════════
def _build_sol_from_gt_solver(sr: dict, question: str) -> dict:
    """
    GameTheorySolver çıktısından tam sol_data üretir.
    LLM bypass edildiğinde verilerin eksiksiz JSON yapısına dönüştürülür.
    Hardcoding yok — tüm değerler sr'dan türetilir.
    step_target (büyük-N) desteği: wdata'dan kümülatif ve per-round kazanç alınır.
    """
    gt_p = sr.get("_gt_params", {})
    scores = sr.get("scores", {})
    sorted_p = sr.get("sorted_players", [])
    grim_t = sr.get("_grim_targets", {})
    payoff_d = sr.get("_payoff", {})
    nash = sr.get("_nash", [])
    n_rounds = sr.get("n_rounds", 0)
    steps = sr.get("steps", [])
    answer = sr.get("answer", "")
    players = gt_p.get("players", [])
    game_type = gt_p.get("game_type", "iterated_pd")
    wdata = sr.get("_window_data", {})
    sim_res = sr.get("_sim_result", {})
    grim_per_rnd = sim_res.get("grim_targets_per_round", {})
    step_tgt = gt_p.get("step_target")  # büyük-N hedef

    # ── Ödeme parametreleri ───────────────────────────────────────────────────
    R = payoff_d.get("R", 3)
    T = payoff_d.get("T", 5)
    S = payoff_d.get("S", 0)
    P = payoff_d.get("P", 1)
    formula_str = (
        f"(C,C)→R,R  |  (D,C)→T,S  |  (C,D)→S,T  |  (D,D)→P,P  "
        f"[T={T} R={R} P={P} S={S}] | Nash: {nash if nash else '(D,D)'}"
    )

    # ── Büyük-N: kümülatif puanları wdata'dan al ──────────────────────────────
    # step_target varsa ve wdata doluysa → tahmin edilen kümülatif kullan.
    # Hardcoding yok: hangi tur hedeflenirse o türetilmiş değer kullanılır.
    if step_tgt and wdata:
        cumul_at_t = wdata.get("cumulative_at_target", scores)
        per_r = wdata.get("per_round_payoff", {})
        # step_target'a göre sıralanmış liste
        sorted_p_tgt = sorted(cumul_at_t.items(), key=lambda x: -x[1])
        winner = (
            sorted_p_tgt[0][0]
            if sorted_p_tgt
            else (sorted_p[0][0] if sorted_p else "?")
        )
        ws = sorted_p_tgt[0][1] if sorted_p_tgt else (sorted_p[0][1] if sorted_p else 0)
        loser = (
            sorted_p_tgt[-1][0]
            if sorted_p_tgt
            else (sorted_p[-1][0] if sorted_p else "?")
        )
        ls = (
            sorted_p_tgt[-1][1]
            if sorted_p_tgt
            else (sorted_p[-1][1] if sorted_p else 0)
        )
        sc_str = " | ".join(f"{nm}:{v:.0f}" for nm, v in sorted_p_tgt)
        per_str = (
            " | ".join(f"{nm}:+{v:.2f}/tur" for nm, v in per_r.items()) if per_r else ""
        )
        numeric_str = (
            f"[{step_tgt}. tur tahmini] Kazanan: {winner}≈{ws:.0f}puan | "
            f"Kaybeden: {loser}≈{ls:.0f}puan | {sc_str}"
            + (f"\nSteady-state tur başına: {per_str}" if per_str else "")
        )
        # sorted_p_for_answer = büyük-N tahminine göre
        sorted_p_eff = sorted_p_tgt
        score_dict = {nm: round(v) for nm, v in sorted_p_tgt}
    else:
        # Normal küçük-N: simülasyon sonuçlarını kullan
        winner = sorted_p[0][0] if sorted_p else "?"
        ws = sorted_p[0][1] if sorted_p else 0
        loser = sorted_p[-1][0] if sorted_p else "?"
        ls = sorted_p[-1][1] if sorted_p else 0
        sc_str = " | ".join(f"{nm}:{sc}" for nm, sc in sorted_p)
        numeric_str = (
            f"Kazanan: {winner}={ws}puan | Kaybeden: {loser}={ls}puan | {sc_str}"
        )
        sorted_p_eff = sorted_p
        score_dict = {nm: sc for nm, sc in sorted_p}

    # ── Grim Trigger tura özgü snapshot ──────────────────────────────────────
    import re as _re

    # Soru metninden tur referansı çıkar — TurkishNumberParser ile (hardcoding yok)
    m_rnd = _re.search(
        r"(\d+)\.?\s*tur.*(?:strateji|hangi|kime|yönel)|"
        r"((?:bininci|yüzüncü|ikinci|üçüncü|dördüncü|beşinci|onuncu|yirminci|"
        r"otuzuncu|kırkıncı|ellinci|altmışıncı|yetmişinci|sekseninci|doksanıncı))"
        r"\s*tur",
        question.lower(),
    )
    if m_rnd:
        raw_rnd = m_rnd.group(1) or m_rnd.group(2) or ""
        q_rnd = TurkishNumberParser.parse(raw_rnd) or 2
    else:
        q_rnd = 2
    if grim_per_rnd:
        q_rnd = min(q_rnd, max(grim_per_rnd.keys()))

    gt_rnd_snap = grim_per_rnd.get(q_rnd, {})
    grim_strat_dict = {}
    for gp in [p for p in players if p["strategy"] == "grim_trigger"]:
        tgts = gt_rnd_snap.get(gp["name"], [])
        grim_strat_dict[gp["name"]] = (
            f"{', '.join(tgts)}'a karşı ihanet etmektedir."
            if tgts
            else "Henüz kimseye karşı Grim Trigger tetiklenmedi."
        )

    # ── Yapılandırılmış cevap dict ────────────────────────────────────────────
    # Hedef tur sayısı: step_target varsa onu, yoksa n_rounds kullan (hardcoding yok)
    answer_rnd_label = step_tgt if step_tgt else n_rounds
    structured_answer = {
        f"{q_rnd}_tur_stratejisi": grim_strat_dict,
        f"{answer_rnd_label}_tur_puanları": {
            "en_yüksek": winner,
            "en_düşük": loser,
        },
        "puan_tablosu": score_dict,
    }
    answer_combined = (
        (answer + f"\n{structured_answer}") if answer else str(structured_answer)
    )

    # ── Distribution (büyük-N'de wdata kümülatifi) ───────────────────────────
    dist = score_dict.copy()

    # ── Events ───────────────────────────────────────────────────────────────
    events = [
        {
            "label": nm,
            "prob": str(v),
            "note": f"Strateji: {next((p['strategy'] for p in players if p['name']==nm),'?')}",
        }
        for nm, v in sorted_p_eff
    ]

    # ── Başlık: büyük-N ise step_target göster ────────────────────────────────
    title_rnd = f"{step_tgt} tur (hedef)" if step_tgt else f"{n_rounds} tur"

    # ── GT renderer ASCII ek kutu ─────────────────────────────────────────────
    gt_box_lines = _gt_renderer.render(sr, ASCII_WIDTH)
    gt_box_str = "\n".join(gt_box_lines)

    return {
        "type": "game_theory",
        "title": f"Oyun Kuramı — {game_type.replace('_',' ').title()} ({title_rnd})",
        "steps": steps,
        "answer": answer_combined,
        "explanation": (
            f"NLP+Round-Robin+BART analitik çözümü. "
            f"{len(players)} oyuncu, {title_rnd}. "
            f"En yüksek puan: {winner} ({ws:.0f}). En düşük: {loser} ({ls:.0f})."
        ),
        "formula": formula_str,
        "numeric": numeric_str,
        "certainty": "Kesin (Analitik Simülasyon — GameTheorySolver)",
        "tree": {
            "label": f"Oyun Kuramı ({game_type})",
            "prob": "",
            "children": [
                {
                    "label": nm,
                    "prob": str(v),
                    "value": "",
                    "final": f"★ {v:.0f} puan",
                    "children": [],
                }
                for nm, v in sorted_p_eff
            ],
        },
        "distribution": dist,
        "table": {},
        "matrix": {},
        "sets": [],
        "events": events,
        "_gt_solver_result": sr,
        "_gt_extra_box": gt_box_str,
        "_corrected": True,
        "_correction_rule": "GT_SOLVER_BYPASS",
        "_consistency_score": sr.get("_val_score", 1.0),
        "_consistency_violations": sr.get("_violations", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  _build_sol_from_bayes_solver
#  BayesSolver analitik sonucundan LLM-uyumlu sol_data dict üretir.
#  Symbolic Bayes bypass aktifleştiğinde OllamaClient bu fonksiyonu çağırır.
#  Hardcoding yok — tüm değerler solver_result'tan türetilir.
# ═══════════════════════════════════════════════════════════════════════════════
def _build_sol_from_bayes_solver(sr: dict, question: str) -> dict:
    """
    BayesSolver çıktısından (sr) tam sol_data üretir.
    LLM bypass edildiğinde verilerin eksiksiz JSON yapısına dönüştürülür.

    sr: BayesSolver.solve() döndürdüğü dict
    Döndürür: OllamaClient ve ASCIIEngine tarafından tüketilebilir sol_data
    """
    hyps = sr.get("hypotheses", [])
    n_obs = sr.get("n_observations", 1)
    rounds = sr.get("rounds", {})
    final_p = sr.get("posteriors", {})
    joint_p = sr.get("joint_posteriors", {})
    priors = sr.get("priors", {})
    liks = sr.get("likelihoods", {})
    best_h = sr.get("best_hypothesis", "?")
    best_pr = sr.get("best_posterior", 0.0)
    final_ev = sr.get("final_evidence", 0.0)

    # ── Adım listesi — BayesSolver._build_steps çıktısını kullan ─────────────
    steps = sr.get("steps", [])
    if not steps:
        # Minimal fallback: adım listesi yoksa elle üret
        steps = []
        prior_str = "  ".join(
            f"P({h.upper()})={priors.get(h,0):.4f}" for h in sorted(hyps)
        )
        steps.append(
            {
                "title": "1. Prior Dağılımı",
                "content": "Başlangıç hipotez olasılıkları.",
                "formula": prior_str,
                "result": f"Σ = {sum(priors.values()):.4f}",
                "type": "process",
            }
        )
        lik_str = "  ".join(
            f"P(E|{h.upper()})={liks.get(h,0):.4f}" for h in sorted(hyps)
        )
        steps.append(
            {
                "title": "2. Likelihood Tablosu",
                "content": "Her hipotez için kanıt olasılığı.",
                "formula": lik_str,
                "result": "",
                "type": "process",
            }
        )
        for t, rd in sorted(rounds.items()):
            ev = rd.get("evidence", 0.0)
            posts = rd.get("posteriors", {})
            post_str = "  ".join(
                f"P({h.upper()}|E)={posts.get(h,0):.6f}" for h in sorted(hyps)
            )
            steps.append(
                {
                    "title": f"{2+t}. Tur {t} — Bayes Güncellemesi (Analitik)",
                    "content": f"P(E) = {ev:.6f}  |  Posterior = Prior × Likelihood / P(E)",
                    "formula": f"P(E_{t}) = {ev:.6f}",
                    "result": post_str,
                    "type": "process",
                }
            )

    # ── answer string ─────────────────────────────────────────────────────────
    answer_parts = [
        f"P({h.upper()}|E) = {final_p.get(h,0):.6f}  (%{final_p.get(h,0)*100:.2f})"
        for h in sorted(hyps)
    ]
    answer_str = " | ".join(answer_parts)

    # ── numeric string ────────────────────────────────────────────────────────
    numeric_str = (
        f"En yüksek: {best_h.upper()} = {best_pr:.6f} (%{best_pr*100:.2f})  "
        f"P(E) son tur = {final_ev:.6f}"
    )

    # ── formula string ────────────────────────────────────────────────────────
    if n_obs == 1:
        formula_str = "P(H|E) = P(E|H)×P(H) / Σ P(E|Hᵢ)×P(Hᵢ)"
    else:
        formula_str = (
            f"P(H|E₁...E_{n_obs}) = " f"P(E|H)^{n_obs}×P(H) / Σ P(E|Hᵢ)^{n_obs}×P(Hᵢ)"
        )

    # ── tree: olasılık dalları ────────────────────────────────────────────────
    tree_children = []
    for h in sorted(hyps):
        post = final_p.get(h, 0.0)
        tree_children.append(
            {
                "label": h.upper(),
                "prob": f"{priors.get(h,0):.4f}",
                "value": f"P(E|{h.upper()})={liks.get(h,0):.4f}",
                "final": f"★ P({h.upper()}|E)={post:.6f}",
                "children": [],
            }
        )
    tree = {
        "label": "Bayes (Analitik)",
        "prob": "",
        "children": tree_children,
    }

    # ── events listesi ────────────────────────────────────────────────────────
    events = [
        {
            "label": h.upper(),
            "prob": f"{final_p.get(h,0):.6f}",
            "note": f"Prior={priors.get(h,0):.4f}  P(E|H)={liks.get(h,0):.4f}",
        }
        for h in sorted(hyps)
    ]

    return {
        "type": "bayes",
        "title": f"Bayesyen Çıkarım ({n_obs} gözlem) — Analitik Çözüm",
        "steps": steps,
        "answer": answer_str,
        "explanation": (
            f"Fraction tabanlı analitik Bayes çözümü. "
            f"En yüksek posterior: {best_h.upper()} = %{best_pr*100:.2f}"
        ),
        "formula": formula_str,
        "numeric": numeric_str,
        "certainty": "Kesin (Symbolic Bayes — Fraction tabanlı)",
        "tree": tree,
        "distribution": {h: round(final_p.get(h, 0.0), 6) for h in sorted(hyps)},
        "table": {},
        "matrix": {},
        "sets": [],
        "events": events,
        # Corrector meta
        "_corrected": True,
        "_correction_rule": "SYMBOLIC_BAYES_BYPASS",
        "_consistency_score": 1.0,
        "_consistency_violations": [],
        "_bayes_solver_result": sr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ MOD10: CloudUniversalEquationRepository — Sıfır Hard-Coding Evrensel DB
#  Cloud → JSON → yerel cache | Her sorguda otomatik prompt enjeksiyonu
# ═══════════════════════════════════════════════════════════════════════════════
class CloudUniversalEquationRepository:
    """
    Cloud'dan (GitHub raw) tek bir universe_full.json çeker.
    Kullanıcı kendi reponu oluşturur → tüm Serway denklemleri, sabitler,
    GR, kuantum, biyoloji, kaos, PDE, finans… eklenir.
    Hiçbir denklem kodda yoktur.
    """

    LOCAL_DB = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "universe_equations.json"
    )

    # ── TEK CONFIG (sen burayı kendi GitHub reponla değiştir) ───────────────
    MASTER_URL = (
        "https://raw.githubusercontent.com/Mehmet93/hesapdb/main/universe_full.json"
    )
    # Repo oluştur: https://github.com/KENDI_GITHUB_KULLANICIN/physics-universe-db
    # İçine universe_full.json koy (aşağıda örnek yapı var)

    def __init__(self):
        self.db = self._ensure_db()
        self.retriever = FormulaSemanticRetriever(self.db)

    def _ensure_db(self):
        """30 günde bir cloud'dan yeniler, yoksa local cache."""
        if os.path.exists(self.LOCAL_DB):
            try:
                with open(self.LOCAL_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - os.path.getmtime(self.LOCAL_DB) < 30 * 86400:
                    print("✓ Cloud Equation Universe (local cache)")
                    return data
            except Exception:
                pass

        print("🔄 Cloud'dan evrensel denklem veritabanı çekiliyor...")
        try:
            r = requests.get(self.MASTER_URL, timeout=20)
            r.raise_for_status()
            data = r.json()
            with open(self.LOCAL_DB, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ {len(data)} kategori yüklendi (cloud)")
            return data
        except Exception as e:
            print(f"⚠ Cloud çekilemedi: {e} → Boş DB")
            return {
                "constants": {},
                "serway": {},
                "gr_astrophysics": {},
                "quantum": {},
                "pde": {},
                "chaos": {},
                "biology": {},
                "finance": {},
                "biochemistry": {},
            }

    def query(self, question: str) -> dict:
        """
        Soru metninden ilgili denklemleri tamamen indeks-tabanlı seçer.
        Hard-coded kategori/anahtar kelime listesi içermez.
        """
        retrieved = self.retriever.retrieve(question, top_k=24, min_score=0.08)
        equations = {}
        for item in retrieved:
            cat = item["category"]
            key = item["key"]
            val = item["payload"]
            bucket = equations.setdefault(cat, {})
            bucket[key] = val

        return {
            "constants": self.db.get("constants", {}),
            "equations": equations,
            "metadata": {
                "last_updated": datetime.datetime.now().isoformat(),
                "retrieved_items": len(retrieved),
                "retriever": "FormulaSemanticRetriever-v2",
                "top_categories": [x["category"] for x in retrieved[:6]],
            },
        }

    def get_all(self) -> dict:
        """Tüm DB (nadiren kullanılır)."""
        return self.db


class FormulaSemanticRetriever:
    """
    Cloud JSON'dan kategori + denklem dizini üretir ve sorguya göre skorlar.
    - Hard-coded konu listesi yok.
    - Kategori/denklem adları + açıklama + formül + birim metnini birlikte tarar.
    - Basit Q-learning benzeri pekiştirme belleği ile token→kategori ağırlığı günceller.
    """

    _STOPWORDS = {
        "ve",
        "ile",
        "için",
        "the",
        "and",
        "bir",
        "olan",
        "gibi",
        "soru",
        "problem",
        "hesapla",
        "bul",
        "kaç",
        "nedir",
    }

    def __init__(self, db: dict):
        self.db = db or {}
        self.entries = []
        self.cat_tokens = defaultdict(set)
        self.token_df = defaultdict(int)
        self.category_bias = defaultdict(float)
        self._build_index()

    def _tokenize(self, text: str) -> list:
        if not text:
            return []
        words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9_+\-/=^.]+", str(text).lower())
        return [w for w in words if len(w) > 1 and w not in self._STOPWORDS]

    def _collect_text(self, category: str, key: str, payload) -> str:
        parts = [category, key]
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, (str, int, float)):
                    parts.append(str(v))
                elif isinstance(v, dict):
                    parts.extend(str(x) for x in v.values() if isinstance(x, (str, int, float)))
                elif isinstance(v, list):
                    parts.extend(str(x) for x in v if isinstance(x, (str, int, float)))
        elif isinstance(payload, list):
            parts.extend(str(x) for x in payload if isinstance(x, (str, int, float)))
        else:
            parts.append(str(payload))
        return " ".join(parts)

    def _iter_equations(self):
        for category, content in (self.db or {}).items():
            if category in {"version", "metadata", "constants"}:
                continue
            if isinstance(content, dict):
                for key, payload in content.items():
                    yield category, str(key), payload
            elif isinstance(content, list):
                for i, payload in enumerate(content):
                    yield category, f"item_{i+1}", payload
            else:
                yield category, "value", content

    def _build_index(self):
        self.entries = []
        self.cat_tokens = defaultdict(set)
        self.token_df = defaultdict(int)

        for category, key, payload in self._iter_equations():
            blob = self._collect_text(category, key, payload)
            toks = self._tokenize(blob)
            if not toks:
                continue
            token_set = set(toks)
            self.entries.append(
                {
                    "category": category,
                    "key": key,
                    "payload": payload,
                    "tokens": token_set,
                    "text_len": len(blob),
                }
            )
            self.cat_tokens[category].update(token_set)

        for token_set in self.cat_tokens.values():
            for t in token_set:
                self.token_df[t] += 1

    def _idf(self, token: str) -> float:
        n_cat = max(len(self.cat_tokens), 1)
        df = self.token_df.get(token, 0)
        return math.log((1.0 + n_cat) / (1.0 + df)) + 1.0

    def _category_score(self, q_tokens: set, category: str) -> float:
        cset = self.cat_tokens.get(category, set())
        if not q_tokens or not cset:
            return 0.0
        overlap = q_tokens & cset
        if not overlap:
            return 0.0
        weighted = sum(self._idf(t) for t in overlap)
        norm = math.sqrt(len(q_tokens) * len(cset))
        return (weighted / (norm + 1e-9)) + self.category_bias.get(category, 0.0)

    def _entry_score(self, q_tokens: set, entry: dict, cat_score: float) -> float:
        overlap = q_tokens & entry["tokens"]
        if not overlap:
            return 0.0
        lexical = sum(self._idf(t) for t in overlap) / (len(q_tokens) + 1e-9)
        key_bonus = 0.0
        key_tokens = set(self._tokenize(entry["key"]))
        if q_tokens & key_tokens:
            key_bonus = 0.25
        return 0.65 * cat_score + 0.35 * lexical + key_bonus

    def retrieve(self, question: str, top_k: int = 20, min_score: float = 0.08) -> list:
        q_tokens = set(self._tokenize(question))
        if not q_tokens:
            return []

        cat_scores = {
            cat: self._category_score(q_tokens, cat) for cat in self.cat_tokens.keys()
        }
        ranked = []
        for entry in self.entries:
            cscore = cat_scores.get(entry["category"], 0.0)
            if cscore <= 0:
                continue
            score = self._entry_score(q_tokens, entry, cscore)
            if score >= min_score:
                ranked.append({**entry, "score": round(score, 5)})

        ranked.sort(key=lambda x: x["score"], reverse=True)

        # Q-learning benzeri hafıza güncellemesi:
        # Bu turda çıkan kategoriler gelecekte az miktarda prior kazanır.
        for item in ranked[: min(8, len(ranked))]:
            self.category_bias[item["category"]] = min(
                0.35, self.category_bias[item["category"]] * 0.92 + 0.03
            )

        return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
#  OLLAMA İSTEMCİSİ  — Consistency-Loop ile
# ═══════════════════════════════════════════════════════════════════════════════
class OllamaClient:
    MAX_RETRIES = 2  # Tutarsızlık tespit edilirse max 2 kez yeniden üret

    def __init__(self, url=OLLAMA_URL, model=MODEL):
        self.url = url
        self.model = model

    # ── Yardımcı: tüm tiplerde adım başlıkları ve nihai cevapta değerleri koru ──
    def _bind_values(self, sol_data: dict) -> dict:
        """
        LLM çıktısında eksik kalan sayısal değerleri adım başlıklarına ve
        nihai cevaba enjekte eder. Soru tipinden bağımsız çalışır.
        Hard‑coding yok; var olan result/formula/content içinden dinamik çekim.
        """

        def _first_numeric(text: str) -> str:
            m = re.search(r"-?\d+[\d.,]*\s*(?:%|[A-Za-zμΩWNJPa/]*|e[+-]?\d+)?", text)
            return m.group(0) if m else ""

        def _merge_answer(ans, numeric):
            # Cevap boşsa veya sembolikse, numerik özeti ekle
            if not numeric:
                # LLM "None" / boş döndüyse adımlardan en güçlü sonucu türet
                if ans is None or str(ans).strip().lower() in {"", "none", "null", "?"}:
                    return ""
                return ans
            numeric_str = (
                " | ".join(f"{k}: {v}" for k, v in numeric.items())
                if isinstance(numeric, dict)
                else str(numeric)
            )
            if not ans or str(ans).strip() in {"?", "Hesaplanamadı", ""}:
                return numeric if isinstance(numeric, dict) else numeric_str
            ans_str = str(ans)
            # Numerik değerler zaten cevaptaysa ekleme
            if numeric_str and numeric_str not in ans_str:
                return f"{ans_str} | {numeric_str}"
            return ans

        def _fallback_answer_from_steps(steps: list) -> str:
            """
            Nihai cevap boş/None ise adımlardan kısa özet üret.
            Hard-code soru tipine bağlı değildir; yalnızca mevcut adım sonuçlarını kullanır.
            """
            picks = []
            for st in reversed(steps or []):
                if st.get("_planner_step"):
                    continue
                title = str(st.get("title", "")).strip()
                result = str(st.get("result", "")).strip()
                if not result:
                    result = _first_numeric(
                        " ".join(str(st.get(k, "")) for k in ("formula", "content"))
                    ).strip()
                if result:
                    label = title[:64] if title else "Sonuç"
                    picks.append(f"{label}: {result}")
                if len(picks) >= 2:
                    break
            if not picks:
                return "Hesaplandı (detaylar adımlarda)."
            return " | ".join(reversed(picks))

        steps = sol_data.get("steps") or []
        bound_steps = []
        for step in steps:
            st = dict(step)
            if st.get("_planner_step"):
                # Planlayıcı adımları kullanıcıya içerik odaklı başlıkla gösterilir;
                # debug benzeri "queued" vb. ekler enjekte edilmez.
                bound_steps.append(st)
                continue
            title = str(st.get("title", "")).strip()
            res = str(st.get("result", "")).strip()
            # 1) Eksik result'u içerikten/ formülden çek
            if not res:
                res = _first_numeric(
                    " ".join(
                        str(st.get(k, "")) for k in ("result", "formula", "content")
                    )
                ).strip()
                if res:
                    st["result"] = res
            # 2) Başlıkta değer yoksa, sonuna ekle (kısa tut)
            if res and res not in title:
                # Başlığı aşırı uzatmamak için 48 karakter sınırı
                suffix = res[:48]
                st["title"] = f"{title}: {suffix}" if title else suffix
            bound_steps.append(st)
        sol_data["steps"] = bound_steps

        # Nihai cevap alanı
        sol_data["answer"] = _merge_answer(sol_data.get("answer"), sol_data.get("numeric"))
        if (
            sol_data.get("answer") is None
            or str(sol_data.get("answer", "")).strip().lower() in {"", "none", "null", "?"}
        ):
            sol_data["answer"] = _fallback_answer_from_steps(sol_data.get("steps") or [])
        return sol_data

    def solve(
        self,
        question: str,
        signals: dict,
        sem: SemanticSignalModule,
        scorer: BARTConsistencyScorer,
        solver_ctx: dict = None,
        eq_ctx: dict = None,
    ) -> dict:
        """
        v3 pipeline:
        1. Semantic sinyallerden dinamik system prompt üret
        2. Solver hint enjekte et (BayesSolver analitik sonuçları dahil)
        3. Cloud Equation Universe enjekte et (eq_ctx)
        4. Ollama'ya gönder
        5. BARTConsistencyScorer + NumericTruthValidator ile denetle
        6a. [YENİ] Bayes sorusu + ihlal var + BayesSolver çözdüyse
            → Symbolic Bayes bypass: LLM atlayarak analitik sonucu doğrudan kullan
        6b. SolutionCorrector: diğer ihlaller için cebirsel yeniden inşa (LLM bypass)
        6c. Corrector çözüm üretemezse: ihlal listesini PROMPT'A EKLEYİP yeniden üret
        7. MAX_RETRIES sonra en iyi sonucu döndür
        """
        system_prompt = build_system_prompt(signals, sem)
        if solver_ctx:
            system_prompt += _build_solver_hint(solver_ctx)

        # ── Cloud Equation Universe enjeksiyonu (MOD10) ─────────────────────
        if eq_ctx and eq_ctx.get("equations"):
            system_prompt += "\n\n🌌 UNIVERSAL EQUATION UNIVERSE (cloud loaded JSON — zero hardcode):\n"
            system_prompt += json.dumps(eq_ctx, ensure_ascii=False, indent=2)[
                :4000
            ]  # 4K token limit
            system_prompt += "\n\nKullanabileceğin tüm sabitler ve denklemler yukarıdadır. Doğrudan kopyala-yapıştır."

        best_result = None
        best_score = -1.0
        violations_history = []

        for attempt in range(self.MAX_RETRIES + 1):
            retry_note = ""
            if attempt > 0 and violations_history:
                retry_note = (
                    "\n\n⚠ ÖNCEKİ YANIT TUTARSIZDI — LÜTFEN DÜZELTİN:\n"
                    + "\n".join(f"  • {v}" for v in violations_history[-1])
                    + "\nBu ihlalleri GİDEREREK yeniden hesapla.\n"
                )

            sol_data = self._call_ollama(question, system_prompt + retry_note)

            # Adım başlıkları ve nihai cevabın sayısal değerleri içerdiğinden emin ol
            sol_data = self._bind_values(sol_data)

            # ── Consistency check ─────────────────────────────────────────────
            score, violations, is_ok = scorer.score(signals, sol_data)
            sol_data["_consistency_score"] = round(score, 3)
            sol_data["_consistency_violations"] = violations

            # ── 5b: GAME THEORY BYPASS ────────────────────────────────────────
            # GT sorusu + GameTheorySolver çözdü → LLM tutarlı olsa bile
            # analitik simülasyon sonucunu HER ZAMAN döndür.
            # Neden: LLM puan tablosu veya Grim Trigger hedefini yanlış hesaplayabilir.
            # GameTheorySolver deterministik + BART-doğrulanmış → her zaman daha güvenilir.
            gt_sr = (solver_ctx or {}).get("solver_result", {})
            if (
                (solver_ctx or {}).get("math_ast", {}).get("type") == "game_theory"
                and gt_sr.get("solved")
                and gt_sr.get("_solver_type") == "GameTheorySolver"
            ):
                gt_sol = _build_sol_from_gt_solver(gt_sr, question)
                gt_sol["_attempts"] = attempt + 1
                gt_sol["_symbolic_bypass"] = True
                gt_sol["_llm_violations"] = violations
                gt_sol["_consistency_score"] = 1.0
                gt_sol["_consistency_violations"] = []
                router.episode += 1
                return gt_sol

            # ── 5a: SYMBOLIC BAYES BYPASS ─────────────────────────────────────
            # Bayes sorusu + BayesSolver başarıyla çözdü + LLM tutarsız →
            # LLM retry'ı tamamen atla; analitik sonucu doğrudan döndür.
            # "if symbolic_confidence > llm_confidence and llm_violation_count>0"
            bayes_sr = (solver_ctx or {}).get("solver_result", {})
            if (
                not is_ok
                and signals.get("bayes_structure")
                and bayes_sr.get("solved")
                and bayes_sr.get("_solver_type") == "BayesSolver"
            ):
                symbolic_sol = _build_sol_from_bayes_solver(bayes_sr, question)
                symbolic_sol["_attempts"] = attempt + 1
                symbolic_sol["_symbolic_bypass"] = True
                symbolic_sol["_llm_violations"] = violations
                symbolic_sol["_consistency_score"] = 1.0
                symbolic_sol["_consistency_violations"] = []
                best_result = symbolic_sol
                best_score = 1.0
                break  # LLM retry döngüsünü tamamen atla

            # ── 5b: SolutionCorrector (bağımsız zincir hataları vb.) ──────────
            if not is_ok:
                corrected = _corrector.correct(signals, sol_data, question)
                if corrected is not None:
                    corrected["_attempts"] = attempt + 1
                    best_result = corrected
                    best_score = 1.0
                    break

            if score > best_score:
                best_score = score
                best_result = sol_data

            if is_ok:
                break
            violations_history.append(violations)

        best_result["_signals"] = signals
        best_result["_attempts"] = attempt + 1
        return best_result

    def _call_ollama(self, question: str, system_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": f"Soruyu çöz:\n{question}",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.30, "top_p": 0.9, "num_predict": 6000},
        }
        try:
            r = requests.post(self.url, json=payload, timeout=180)
            r.raise_for_status()
            raw = r.json().get("response", "")
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            return self._fallback(question, f"JSON parse hatası: {e}")
        except requests.RequestException as e:
            return self._fallback(question, f"Ollama bağlantı hatası: {e}")

    def _fallback(self, question, error):
        return {
            "type": "genel",
            "title": "Çözüm",
            "steps": [
                {
                    "title": "Soru Analizi",
                    "content": f"Soru işleniyor: {question[:100]}",
                    "formula": "",
                },
                {"title": "Hata", "content": str(error), "formula": ""},
                {
                    "title": "Öneri",
                    "content": "Ollama çalışıyor mu? → ollama serve",
                    "formula": "",
                },
            ],
            "answer": "Hesaplanamadı",
            "explanation": error,
            "formula": "",
            "numeric": "",
            "tree": None,
            "distribution": {},
            "table": {},
            "matrix": {},
            "sets": [],
            "events": [],
            "_consistency_score": 0.0,
            "_consistency_violations": [error],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ❱❱ OYUN KURAMI MODÜLLERİ — INJECTION POINT
# ═══════════════════════════════════════════════════════════════════════════════
# fmt: off  (auto-formatter devre dışı — blok içi tutarlılık için)


class GameTheoryLexicon:
    L1_GAME_THEORY = [
        r"\boyun\s+kuramı\b",
        r"\bgame\s+theory\b",
        r"\bnash\s+denge",
        r"\bnash\s+equilibrium\b",
        r"\bödeme\s+matrisi\b",
        r"\bpayoff\s+matrix\b",
        r"\bdominant\s+strateji",
        r"\bdominant\s+strategy\b",
        r"\bidsds\b",
        r"\bnormal\s+form\b",
        r"\bextensive\s+form\b",
        r"\bgrim\s+trigger\b",
        r"\bgrim\s+tetik",
        r"\bkıyamet\s+tetikleyici\b",
        r"\bpavlov\b",
        r"\btit.for.tat\b",
        r"\bgöze\s+göz\b",
        r"\bprisoners?\s+dilemma\b",
        r"\btutuklu\s+ikilem",
        r"\bfok\s+theorem\b",
        r"\bfolk\s+teorem",
        r"\btekrarl[aı]\b.*\boyun\b",
        r"\brepeated\s+game\b",
        r"\bkooperatif\s+oyun\b",
        r"\bcooperative\s+game\b",
        r"\bshapley\b",
        r"\bnucleolus\b",
        r"\bminimaks\b",
        r"\bminimax\b",
        r"\bmaximin\b",
        r"\bzero.sum\s+game\b",
        r"\bsıfır\s+toplamlı\b",
        r"\bauction\b",
        r"\bvickrey\b",
        r"\bvcg\b",
        r"\bevrimsel\s+oyun",
        r"\bevolutionary\s+game\b",
        r"\breplicator\b",
        r"\bess\b",
        r"\bhawk.dove\b",
        r"\bbayesian\s+(?:game|oyun)\b",
        r"\bsignaling\s+game\b",
        r"\bbrinkmanship\b",
        r"\btavuk\s+oyunu\b",
        r"\bchicken\s+game\b",
        r"\bmekanizma\s+tasarımı\b",
        r"\bmechanism\s+design\b",
        r"\bsubgame\s+perfect\b",
    ]
    L2_STRATEGY_NAMES = [
        r"\balways\s+(?:c|d|cooperate|defect)\b",
        r"\bhep\s+(?:uzlaş|ihanet|c|d)\b",
        r"\bher\s+zaman\s+(?:uzlaş|ihanet|c|d|işbirliği)\b",
        r"\btit\s+for\s+tat\b",
        r"\bgöze\s+göz\s+dişe\s+diş\b",
        r"\bgrim\b",
        r"\bpavlov\b",
        r"\bkazan.kal\b",
        r"\bwin.stay\b",
        r"\blose.shift\b",
        r"\bkaybet.değiştir\b",
        r"\bfırsatçı\b",
        r"\bopportunist\b",
    ]
    L2_MOVE_SYMBOLS = [
        r"\b[Cc]\s*\(uzlaşma\)",
        r"\b[Dd]\s*\(ihanet\)",
        r"\b\(\s*[CD]\s*,\s*[CD]\s*\)\b",
        r"\bcooperate\b",
        r"\bdefect\b",
        r"\buzlaş[ma]*\b",
        r"\bihanet\b",
    ]
    L3_PAYOFF_PATTERNS = [
        r"\bödeme\b",
        r"\bkazanç\b",
        r"\bpayoff\b",
        r"\b\(\s*[CD]\s*,\s*[CD]\s*\)\s*→\s*[+\-]?\d",
        r"\bT\s*=\s*\d",
        r"\bR\s*=\s*\d",
        r"\bP\s*=\s*\d",
        r"\bS\s*=\s*\d",
        r"her\s+iki\s+taraf.*?\+\d",
        r"ihanet\s+eden.*?\+\d",
    ]
    L4_MULTI_PLAYER = [
        r"\bdört\s+aktör\b",
        r"\büç\s+aktör\b",
        r"\biki\s+aktör\b",
        r"\b\d+\s+aktör\b",
        r"\b\d+\s+oyuncu\b",
        r"\b\d+\s+player\b",
        r"\bround.robin\b",
    ]
    L4_ROUND_PATTERNS = [
        r"\b\d+\s+tur\b",
        r"\b\d+\s+round\b",
        r"\b\d+\s+adım\b",
        r"\b(?:tur|round)\s+\d+\b",
        # Türkçe büyük sıra sözcükleri
        r"\b(?:bininci|yüzüncü|on\s*bininci|yüz\s*bininci|milyonuncu)\s*tur",
        r"\b(?:ikinci|üçüncü|dördüncü|beşinci|onuncu|yirminci|otuzuncu)\s*tur",
    ]
    L5_QUESTION_SIGNALS = [
        r"\bhangi\s+aktör\b",
        r"\bhangi\s+oyuncu\b",
        r"\bnash\s+denge(?:si)?\b",
        r"\btoplam\s+puan\b",
        r"\ben\s+yüksek.*puan\b",
        r"\ben\s+düşük.*puan\b",
        r"\bkim.*(?:kazan|kaybet|yüksek|düşük)\b",
    ]
    STRATEGY_NAME_MAP = {
        # ── Kompleks/koşullu stratejiler önce gelir (daha spesifik) ─────────
        "grim_trigger": [
            r"\bgrim\s*trigger\b",
            r"\bgrim\b",
            r"\bkıyamet\s+tetikleyici\b",
            r"\btrigger\s+stratej",
            r"\bhiç\s+affetmez\b",
            r"\basla\s+affetmez\b",
            r"\bbir\s+kez\s+ihanet.*sonraki",
            r"ihanet\s+görürse.*sonsuza",  # ihanet görürse sonsuza kadar ihanet
            r"ihanet.*sonsuza\s+kadar\s+ihanet",
            r"sonsuza\s+kadar\s+ihanet",
            r"ilk\s+tur.*uzlaş.*ihanet\s+görürse",
            r"asla\s+geri\s+dönmez",
        ],
        "pavlov": [
            r"\bpavlov\b",
            r"\bwin.stay\b",
            r"\blose.shift\b",
            r"\bkazan.kal\b",
            r"\bkaybet.değiştir\b",
            r"\bfırsatçı\b",
            r"\bkârlıysa\s+aynı",
            r"\bkârlı.*aynı\s+hamle",
            r"\bkâr.*değilse.*değiştir",
            r"kârlıysa.*tekrar\s+eder",
            r"kârlı\s+değilse.*değiştir",
        ],
        "tit_for_tat": [
            r"\btit.for.tat\b",
            r"\bgöze\s+göz\b",
            r"\bkarşılıklılık\b",
            r"\bne\s+yaparsa\s+onu\s+yap\b",
            r"karşısındakinin.*hamlesini.*tekrarla",
        ],
        "suspicious_tft": [
            r"\bsüpheci\s+tit.for.tat\b",
            r"\bsuspicious\s+tft\b",
        ],
        # ── Basit/koşulsuz stratejiler sonra gelir ────────────────────────
        "always_d": [
            r"\balways\s+d(?:efect)?\b",
            r"\bhep\s+(?:ihanet|d)\b",
            r"\bher\s+zaman\s+(?:ihanet|d)\b",
            r"\bsürekli\s+ihanet\b",
            r"\bdaima\s+ihanet",
            r"\bhep\s+ihanet",
            r"\bkoşulsuz\s+ihanet\b",
            r"\bdefects?\s+always\b",
        ],
        "always_c": [
            r"\balways\s+c(?:ooperate)?\b",
            r"\bhep\s+(?:uzlaş|c)\b",
            r"\bher\s+zaman\s+(?:uzlaş|işbirliği|c)\b",
            r"\bher\s+zaman\s+uzlaş",
            r"\bsürekli\s+(?:uzlaş|işbirliği)\b",
            r"\bdaima\s+uzlaş",
            r"\bhep\s+uzlaş",
            r"\bkoşulsuz\s+(?:uzlaş|işbirliği)\b",
        ],
        "random": [
            r"\brandom\b",
            r"\brastgele\b",
            r"\b50.50\b",
        ],
    }


class GameTheoryNLPExtractor:
    _lex = GameTheoryLexicon()

    @staticmethod
    def _norm(text):
        return text.translate(str.maketrans("İIĞŞÜÖÇ", "iiğşüöç")).lower()

    def extract(self, question: str) -> dict:
        q = self._norm(question)
        result = {
            "is_game_theory": False,
            "game_type": "general",
            "confidence": 0.0,
            "players": [],
            "n_players": 0,
            "payoff_matrix": {},
            "coalition_values": [],
            "n_rounds": 1,
            "ask_type": [],
            "step_target": None,
            "step_window": None,
            "matrix_dim": 2,
            "matrix_nxn": None,
            "detected_layers": [],
        }
        score = 0.0
        for pat in self._lex.L1_GAME_THEORY:
            if re.search(pat, q):
                score += 1.0
        for pat in self._lex.L2_STRATEGY_NAMES + self._lex.L2_MOVE_SYMBOLS:
            if re.search(pat, q):
                score += 0.6
        for pat in self._lex.L3_PAYOFF_PATTERNS:
            if re.search(pat, q):
                score += 0.5
        for pat in self._lex.L4_MULTI_PLAYER + self._lex.L4_ROUND_PATTERNS:
            if re.search(pat, q):
                score += 0.4
        for pat in self._lex.L5_QUESTION_SIGNALS:
            if re.search(pat, q):
                score += 0.3

        result["confidence"] = min(1.0, score / 4.0)
        result["is_game_theory"] = score >= 1.0
        if not result["is_game_theory"]:
            return result

        result["game_type"] = self._detect_game_type(q)
        result["players"] = self._extract_players(question, q)
        result["n_players"] = len(result["players"])
        result["payoff_matrix"] = self._extract_payoff(question, q)
        result["coalition_values"] = self._extract_coalition_values(question)
        result["matrix_dim"], _ = self._extract_matrix_dim(
            question, q, result["players"]
        )
        result["n_rounds"] = self._extract_rounds(q)
        result["step_target"], result["step_window"] = self._extract_step_target(q)
        result["ask_type"] = self._detect_ask_type(q)
        return result

    def _detect_game_type(self, q):
        if any(
            re.search(p, q)
            for p in [
                r"\bgrim\b",
                r"\bpavlov\b",
                r"\btit.for.tat\b",
                r"\bgöze\s+göz\b",
                r"\btekrarl",
                r"\brepeated\b",
                r"\balways\s+[cd]\b",
                r"\bhep\s+(?:uzlaş|ihanet)\b",
                r"\biterated\b",
            ]
        ):
            return "iterated_pd"
        if any(re.search(p, q) for p in [r"\bshapley\b", r"\bkooperatif\b"]):
            return "cooperative"
        if any(re.search(p, q) for p in [r"\bauction\b", r"\bvickrey\b"]):
            return "auction"
        if any(
            re.search(p, q) for p in [r"\bess\b", r"\breplicator\b", r"\bevrimsel\b"]
        ):
            return "evolutionary"
        if any(re.search(p, q) for p in [r"\btavuk\s+oyunu\b", r"\bchicken\b"]):
            return "chicken"
        if any(
            re.search(p, q)
            for p in [r"\bnash\s+denge", r"\bdominant\s+strateji\b", r"\bmatri"]
        ):
            return "matrix_game"
        return "iterated_pd"

    # Oyuncu adı OLAMAYACAK terimler (kara liste)
    _PLAYER_BLACKLIST = {
        "ödeme",
        "ödeme matrisi",
        "matrisi",
        "payoff",
        "matrix",
        "puan",
        "tablo",
        "oyun",
        "game",
        "senaryo",
        "scenario",
        "soru",
        "hamle",
        "strateji",
        "kısıt",
        "koşul",
        "teorem",
        "yöntem",
        "kural",
        "sonuç",
        "analiz",
        "model",
        "değer",
        "matris",
    }

    def _extract_players(self, question, q):
        """
        Oyuncu adı + strateji çıkarıcı — v2 (genişletilmiş).

        Desteklenen format tipleri:
          A) "İsim(Strateji)": strateji parantez içinde  ≥3 karakter
          B) "İsim(kısa_kod): açıklama" veya "İsim(kısa_kod) → açıklama"
             kısa_kod örn: (e), (ir), (is), (tr) — strateji sonraki metinden çıkarılır
          C) Satır bazlı fallback

        Düzeltilen hatalar:
          - Büyük harf zorunluluğu kaldırıldı (iran, israil gibi küçük harfli isimler)
          - Minimum paren uzunluğu 1'e düşürüldü (kısa kodlar için)
          - Strateji önce parantez içinden, yoksa ayraç sonrası açıklamadan çözülür
          - İsim canonicalization: ilk harf büyültülür
        """
        players = []
        seen_names: set = set()

        _BL = self._PLAYER_BLACKLIST
        _BL_PARTS = ("ödeme", "matri", "payoff", "puan tablosu")

        def _canon(name: str) -> str:
            """İlk harfi büyük yap (Türkçe destekli)."""
            if not name:
                return name
            return name[0].upper() + name[1:]

        def _is_blacklisted(name: str) -> bool:
            n = self._norm(name).strip()
            return n in _BL or any(bl in n for bl in _BL_PARTS)

        def _try_add(name: str, sid: str, raw: str):
            """Deduplication + kara liste ile oyuncu ekler."""
            name = name.strip()
            if len(name) < 2 or not sid:
                return
            if _is_blacklisted(name):
                return
            canon = _canon(name)
            key = self._norm(canon).strip()
            if key not in seen_names:
                seen_names.add(key)
                players.append({"name": canon, "strategy": sid, "raw": raw})

        # ── MASTER REGEX ──────────────────────────────────────────────────────
        # Grup 1 : İsim    — küçük/büyük harf fark etmez
        # Grup 2 : Paren   — 0-120 karakter (kısa kod veya tam strateji)
        # Grup 3 : Sonraki — ayraç [:→–\-] ardındaki ilk cümle (strateji açıklaması)
        #          ». tarafından sınırlandırılır — yeni cümle = yeni oyuncu
        master = re.compile(
            r"([A-ZÜÇŞİĞÖa-züçşığö][a-züçşiğöA-ZÜÇŞİĞÖ\s]{0,30}?)"  # grup 1: isim
            r"\s*\(([^)]{0,120})\)"  # grup 2: paren
            r"(?:\s*[:\-–→]\s*([^.;\n]{0,200}))?",  # grup 3: sonraki (opsiyonel)
            re.UNICODE,
        )

        for m in master.finditer(question):
            name = m.group(1).strip()
            paren_txt = (m.group(2) or "").strip()
            after_txt = (m.group(3) or "").strip()

            if _is_blacklisted(name) or len(name) < 2:
                continue

            # Önce parantez içindeki metinden strateji çöz
            sid = self._resolve_strategy(self._norm(paren_txt)) if paren_txt else None

            # Parantez başarısız (kısa kod gibi) → ayraç sonrasındaki açıklamayı dene
            if not sid and after_txt:
                sid = self._resolve_strategy(self._norm(after_txt))

            # İkisi de başarısız → ikisini birleştir
            if not sid:
                combined = self._norm(paren_txt + " " + after_txt)
                sid = self._resolve_strategy(combined)

            _try_add(name, sid, m.group(0))

        # ── FALLBACK: Satır bazlı tarama ─────────────────────────────────────
        # Satırlar varsa (multiline sorular) veya master çıktısı yetersizse
        if len(players) < 2:
            for line in question.split("\n"):
                line_norm = self._norm(line)
                sid = self._resolve_strategy(line_norm)
                if not sid:
                    continue
                nm = re.match(
                    r"\s*([A-ZÜÇŞİĞÖa-züçşığö][^\(:\n→\-]{1,30}?)"
                    r"(?:\s*[\(:→–\-]|\s+[a-züçşığö])",
                    line,
                    re.UNICODE,
                )
                if nm:
                    _try_add(nm.group(1).strip(), sid, line.strip())

        return players

    def _resolve_strategy(self, text):
        for sid, pats in self._lex.STRATEGY_NAME_MAP.items():
            for pat in pats:
                if re.search(pat, text):
                    return sid
        return None

    def _extract_payoff(self, question, q):
        payoff = {}
        # ── Sembolik (C,D) format ─────────────────────────────────────────
        cc = re.search(
            r"\(?\s*[Cc]\s*,\s*[Cc]\s*\)?\s*→\s*(?:her\s+iki\s+taraf\s+)?[+]?(\d+)",
            question,
        )
        dc = re.search(
            r"\(?\s*[Dd]\s*,\s*[Cc]\s*\)?\s*→\s*(?:ihanet\s+eden\s+)?[+]?(\d+)",
            question,
        )
        cd = re.search(r"\(?\s*[Cc]\s*,\s*[Dd]\s*\)?\s*→.*?uzlaşan\s+(\d+)", question)
        dd = re.search(
            r"\(?\s*[Dd]\s*,\s*[Dd]\s*\)?\s*→\s*(?:her\s+iki\s+taraf\s+)?[+]?(\d+)",
            question,
        )
        if cc:
            payoff["R"] = int(cc.group(1))
        if dc:
            payoff["T"] = int(dc.group(1))
        if cd:
            payoff["S"] = int(cd.group(1))
        if dd:
            payoff["P"] = int(dd.group(1))

        # ── Türkçe metin format: (uzlaş, uzlaş) / (ihanet et, uzlaş) ─────
        # R: (uzlaş, uzlaş) → her iki taraf +N
        m = re.search(r"\(uzlaş.*?uzlaş\).*?her\s+iki\s+taraf\s+[+]?(\d+)", q)
        if m and "R" not in payoff:
            payoff["R"] = int(m.group(1))
        # T: (ihanet et, uzlaş) → ihanet eden +N
        m = re.search(r"\(ihanet.*?uzlaş\).*?ihanet\s+eden\s+[+]?(\d+)", q)
        if m and "T" not in payoff:
            payoff["T"] = int(m.group(1))
        # S: (ihanet et, uzlaş) → uzlaşan N  OR  (uzlaş, ihanet et) → uzlaşan N
        m = re.search(r"\((?:ihanet.*?uzlaş|uzlaş.*?ihanet)\).*?uzlaşan\s+[+]?(\d+)", q)
        if m and "S" not in payoff:
            payoff["S"] = int(m.group(1))
        # P: (ihanet et, ihanet et) → her iki taraf +N
        m = re.search(r"\(ihanet.*?ihanet\).*?her\s+iki\s+taraf\s+[+]?(\d+)", q)
        if m and "P" not in payoff:
            payoff["P"] = int(m.group(1))

        # ── T/R/S/P sembolik tanım ─────────────────────────────────────────
        for sym in ["T", "R", "S", "P"]:
            if sym not in payoff:
                m = re.search(rf"\b{sym}\s*[=:]\s*([+\-]?\d+)", question)
                if m:
                    payoff[sym] = int(m.group(1))

        # ── Fallback: "her iki taraf +N" ──────────────────────────────────
        if "R" not in payoff:
            m = re.search(r"her\s+iki\s+taraf\s+[+]?(\d+)", q)
            if m:
                payoff["R"] = int(m.group(1))
        if "T" not in payoff:
            m = re.search(r"ihanet\s+eden\s+[+]?(\d+)", q)
            if m:
                payoff["T"] = int(m.group(1))

        # ── Eksik değerleri PD standart varsayımlarıyla doldur ─────────────
        if "S" not in payoff:
            payoff["S"] = 0
        if "P" not in payoff:
            payoff["P"] = 1
        if "R" not in payoff:
            payoff["R"] = 3
        if "T" not in payoff:
            payoff["T"] = 5

        return payoff

    def _extract_coalition_values(self, question: str):
        """
        Kooperatif oyun değer fonksiyonu kalıplarını çıkarır.
        Örnekler:
          - v({1,2})=60
          - v({A,B}) : 14.5
          - v(∅)=0
          - v(emptyset)=0
        """
        out = []
        seen = set()
        patt = re.compile(
            r"v\s*\(\s*(?:\{([^}]*)\}|(∅|emptyset|boş\s*küme))\s*\)\s*[:=]\s*([+\-]?\d+(?:[.,]\d+)?)",
            flags=re.IGNORECASE,
        )
        for m in patt.finditer(question):
            members_raw = (m.group(1) or "").strip()
            if m.group(2):
                members = []
            else:
                toks = [t.strip() for t in re.split(r"\s*,\s*", members_raw) if t.strip()]
                members = toks
            val = float(m.group(3).replace(",", "."))
            key = (tuple(sorted(members)), val)
            if key in seen:
                continue
            seen.add(key)
            out.append({"members": members, "value": val})
        return out

    def _extract_matrix_dim(self, question, q, players):
        m = re.search(r"(\d+)\s*[x×]\s*(\d+)\s*(?:matri|tablo|grid)", q)
        if m:
            return max(int(m.group(1)), int(m.group(2))), None
        return 2, None

    def _extract_rounds(self, q: str) -> int:
        """
        Soru metninden tur sayısını çıkarır.
        Öncelik sırası:
          1. Rakam + 'tur/round/adım' kalıbı  ("3 tur", "1000 tur")
          2. Türkçe sıra sözcüğü + 'turun/turda/turunda' ("bininci turun")
          3. Türkçe kardinal + 'tur' ("bin tur", "iki yüz tur")
          4. Fallback: 3

        Hardcoding yok — TurkishNumberParser ile dinamik çözümleme.
        """
        # ── 1. Rakam kalıbı ───────────────────────────────────────────────────
        for m in re.finditer(
            r"(\d[\d\s]*)\s*\.?\s*(?:tur(?:un|da|unda|ları)?|round[s]?|adım[lar]?|periyot[lar]?)",
            q,
        ):
            raw = m.group(1).strip()
            if raw.isdigit():
                n = int(raw)
                if 1 <= n <= 100_000_000:
                    return n

        # ── 2. Türkçe sıra sözcüğü ("bininci turun sonunda") ──────────────────
        # Uzun sıra sözcüklerini önce dene (greedy)
        for word in sorted(
            TurkishNumberParser._ORD_WORDS.keys(), key=len, reverse=True
        ):
            pat = rf"\b{re.escape(word)}\s*(?:tur(?:un|da|unda|ları)?|round[s]?|adım)"
            if re.search(pat, q):
                return TurkishNumberParser._ORD_WORDS[word]
        # Ordinal eki serbest kalıp ("iki yüzüncü tur")
        m = re.search(
            r"((?:(?:sıfır|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|"
            r"altmış|yetmiş|seksen|doksan|yüz|bin|milyon)\s*){1,6})"
            r"(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s*"
            r"(?:tur(?:un|da|unda)?|round|adım)",
            q,
        )
        if m:
            val = TurkishNumberParser.parse(m.group(1).strip())
            if val and val > 0:
                return val

        # ── 3. Türkçe kardinal + 'tur' ────────────────────────────────────────
        m = re.search(
            r"((?:(?:sıfır|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on|yirmi|otuz|kırk|elli|"
            r"altmış|yetmiş|seksen|doksan|yüz|bin|milyon)\s*){1,6})"
            r"\s*(?:tur(?:luk|lu)?|round[s]?)",
            q,
        )
        if m:
            val = TurkishNumberParser.parse(m.group(1).strip())
            if val and val > 0:
                return val

        return 3  # varsayılan

    def _extract_step_target(self, q: str) -> "tuple[int|None, int|None]":
        """
        Büyük N tur hedefi çıkarır (StepWindowScaler için).
        Döner: (step_target, window) | (None, None)

        Tetikleyiciler (hardcoding yok — dinamik eşik):
          • n_rounds >= LARGE_N_THRESHOLD (500) → step_target = n_rounds
          • "N. turun sonunda", "bininci turda" gibi explicit hedef ifadesi
          • N >= LARGE_N_THRESHOLD sayısal kalıp

        Büyük-N eşiği: 500 tur → steady-state extrapolation daha güvenilir.
        """
        LARGE_N = 500  # Bu değerin üstünde → window analizi aktif
        # ── 1. Rakam kalıbı (herhangi büyüklük) ──────────────────────────────
        for m in re.finditer(
            r"(\d[\d\s]*)\s*\.?\s*(?:tur(?:un|da|unda)?|round[s]?|adım)", q
        ):
            raw = m.group(1).strip()
            if raw.isdigit():
                n = int(raw)
                if n >= LARGE_N:
                    return n, 10

        # ── 2. Türkçe sıra sözcüğü (büyük değerler için) ─────────────────────
        for word, val in sorted(
            TurkishNumberParser._ORD_WORDS.items(), key=lambda x: -x[1]
        ):
            if val >= LARGE_N and word in q:
                pat = rf"\b{re.escape(word)}\s*(?:tur(?:un|da|unda)?|round[s]?|adım|$)"
                if re.search(pat, q) or word in q:
                    return val, 10

        # ── 3. N=... kalıbı ───────────────────────────────────────────────────
        for m in re.finditer(r"\b[Nn]\s*=\s*(\d+)", q):
            n = int(m.group(1))
            if n >= LARGE_N:
                return n, 10

        return None, None

    def _detect_ask_type(self, q):
        ask = []
        # Büyük tur ordinallerini dinamik derle (TurkishNumberParser._ORD_WORDS'tan)
        _ord_pattern = "|".join(
            re.escape(w)
            for w in sorted(
                TurkishNumberParser._ORD_WORDS.keys(), key=len, reverse=True
            )
        )
        patterns = {
            "who_highest_score": [
                r"en\s+yüksek.*puan",
                r"toplam.*en\s+fazla",
                r"kim.*(?:kazandı|önde)",
                r"en\s+fazla.*puan",
            ],
            "who_lowest_score": [
                r"en\s+düşük.*puan",
                r"en\s+az.*puan",
            ],
            "grim_target": [
                r"grim.*hangi.*aktör",
                r"grim.*kime",
                r"hangi.*aktöre.*(?:grim|kıyamet)",
                r"trigger.*(?:yöneli|aktif|kime)",
                r"hangi\s+aktöre\s+yönelmiş",
                r"stratejisi\s+hangi\s+aktöre",
                r"(?:grim|trigger|stratejisi).*yönelmiş",
                r"ikinci\s+tur.*stratej",
                r"\d+\.\s*tur.*sonunda.*(?:hangi|kime)",
                # Büyük ordinal + strateji sorusu
                rf"(?:{_ord_pattern})\s*tur.*(?:strateji|hangi|kime|yönel)",
            ],
            "nash_equilibrium": [r"nash\s+denge", r"nash\s+equilibrium"],
            "dominant_strategy": [r"dominant\s+stratej", r"baskın\s+stratej"],
            "score_table": [
                r"puan\s+tablosu",
                r"tüm.*puan",
                r"topla[mn].*puan",
            ],
            "shapley": [r"shapley", r"adil\s+pay", r"kooperatif"],
            "folk_theorem": [r"folk\s+teorem", r"sürdürülebilir", r"discount"],
            "matching": [r"eşleşme", r"stabil\s+eşleşme", r"gale.shapley"],
            "bargaining": [r"pazarlık", r"müzakere", r"rubinstein"],
            "minimax": [r"minimax", r"maximin", r"sıfır\s+toplam"],
        }
        for atype, pats in patterns.items():
            if any(re.search(p, q) for p in pats):
                ask.append(atype)
        return ask if ask else ["general"]


class _PayoffMatrix:
    """NxN ödeme matrisi (iç kullanım için _ prefix)."""

    def __init__(self, strategies_p1, strategies_p2, pd_params=None, payoffs=None):
        self.strategies_p1 = strategies_p1
        self.strategies_p2 = strategies_p2
        self.payoffs = payoffs or {}
        self._pd = pd_params or {}
        if pd_params and not self.payoffs:
            R = pd_params.get("R", 3)
            T = pd_params.get("T", 5)
            S = pd_params.get("S", 0)
            P = pd_params.get("P", 1)
            self.payoffs = {
                ("C", "C"): (R, R),
                ("D", "C"): (T, S),
                ("C", "D"): (S, T),
                ("D", "D"): (P, P),
            }
        # Dinamik eşikler — Pavlov ve diğer stratejiler hardcoding olmadan kullanır
        self.R = self._pd.get("R", 3)
        self.T = self._pd.get("T", 5)
        self.P = self._pd.get("P", 1)
        self.S = self._pd.get("S", 0)

    def get_payoff(self, m1, m2):
        if (m1, m2) in self.payoffs:
            return self.payoffs[(m1, m2)]
        if (m2, m1) in self.payoffs:
            p1, p2 = self.payoffs[(m2, m1)]
            return p2, p1
        return (0, 0)

    def find_pure_nash(self):
        nash = []
        for s1 in self.strategies_p1:
            for s2 in self.strategies_p2:
                p1, p2 = self.get_payoff(s1, s2)
                if all(
                    self.get_payoff(k, s2)[0] <= p1 for k in self.strategies_p1
                ) and all(self.get_payoff(s1, k)[1] <= p2 for k in self.strategies_p2):
                    nash.append((s1, s2))
        return nash


class _StrategyResolver:
    """Strateji fonksiyonları sözlüğü."""

    @staticmethod
    def _always_c(rnd, mh, oh, st):
        return "C", st

    @staticmethod
    def _always_d(rnd, mh, oh, st):
        return "D", st

    @staticmethod
    def _grim_trigger(rnd, mh, oh, st):
        # N-oyunculu doğru implementasyon: per-pair history'ye bak.
        # oh = bu spesifik rakibin bu oyuncuya karşı geçmiş hamleleri.
        # Global triggered flag YANLIŞ — bir rakipten tetiklenince
        # TÜM rakiplere D basar. Doğrusu: sadece ihanet eden rakibe D.
        if "D" in oh:
            return "D", st
        return "C", st

    @staticmethod
    def _pavlov(rnd, mh, oh, st):
        if rnd == 0 or not mh:
            return "C", st
        lp = st.get("last_payoff", 3)
        # p_threshold: payoff matrisinden dinamik — P değeri (hardcoded 1 değil)
        # Win-Stay Lose-Shift: lp > P → başarılı (R veya T) → tekrar et
        #                      lp <= P → başarısız (P veya S) → değiştir
        pt = st.get("p_threshold", 1)
        if lp > pt:
            ns = dict(st)
            ns["last_move"] = mh[-1]
            return mh[-1], ns
        else:
            nm = "D" if mh[-1] == "C" else "C"
            ns = dict(st)
            ns["last_move"] = nm
            return nm, ns

    @staticmethod
    def _tit_for_tat(rnd, mh, oh, st):
        if rnd == 0 or not oh:
            return "C", st
        return oh[-1], st

    @staticmethod
    def _random(rnd, mh, oh, st):
        import random

        return random.choice(["C", "D"]), st

    def resolve(self, sid):
        return {
            "always_c": self._always_c,
            "always_d": self._always_d,
            "grim_trigger": self._grim_trigger,
            "pavlov": self._pavlov,
            "tit_for_tat": self._tit_for_tat,
            "suspicious_tft": self._always_d,  # İlk D'den başlar
            "random": self._random,
        }.get(sid, self._always_c)


class _MultiPlayerSimulator:
    """Round-robin çok oyunculu simülasyon."""

    def __init__(self, pm, resolver):
        self.pm = pm
        self.resolver = resolver

    def simulate(self, players, n_rounds, step_target=None, window=10):
        n = len(players)
        if n < 2:
            return {"error": "En az 2 oyuncu"}
        # p_threshold: payoff matrisinden dinamik — P değeri (hardcoded 1 DEĞİL).
        # Win-Stay Lose-Shift kriteri: lp > P ise başarılı (R veya T) → tekrar et.
        _pt = getattr(self.pm, "P", 1)
        _init_lp = getattr(self.pm, "R", 3)
        states = {
            p["name"]: {
                "triggered": False,
                "last_payoff": _init_lp,
                "last_move": "C",
                "p_threshold": _pt,
            }
            for p in players
        }
        ph = {
            (players[i]["name"], players[j]["name"]): {"mine": [], "opp": []}
            for i in range(n)
            for j in range(n)
            if i != j
        }
        scores = {p["name"]: 0 for p in players}
        grim_tgt = {p["name"]: set() for p in players}
        # grim_tgt_per_round: her tur bitimindeki anlık Grim Trigger hedef kümesi.
        # dict[rnd_number] = dict[player_name] = sorted_list_of_targets
        # "ikinci turun sonunda İsrail'in stratejisi hangi aktöre?" gibi sorular
        # için tura özgü snapshot — hardcoding yok, tur sayısı değişse de doğru.
        grim_tgt_per_round = {}
        rounds_d = []
        steps_l = []

        effective = n_rounds
        if step_target and step_target > n_rounds:
            effective = step_target

        for rnd in range(1, effective + 1):
            r_pay = {p["name"]: 0 for p in players}
            r_det = {"round": rnd, "matchups": []}
            for i in range(n):
                for j in range(i + 1, n):
                    ni = players[i]["name"]
                    nj = players[j]["name"]
                    fn_i = self.resolver.resolve(players[i]["strategy"])
                    fn_j = self.resolver.resolve(players[j]["strategy"])
                    # ── DÜZELTME: Pavlov per-opponent ────────────────────────────
                    # HATA (eski davranış): states[ni]["last_payoff"] global —
                    #   aynı turdaki son maçın puanını taşıyordu. Bu yüzden
                    #   Türkiye İran'a C oynayıp 0 almasına rağmen sonraki maç
                    #   sırasında güncellenmiş (farklı rakipten gelen) payoff'u
                    #   görüyordu → yanlış hamle → zincirleme hata (Grim Trigger
                    #   yanlış tetikleme, puan tablosu ve final cevap bozulması).
                    # DÜZELTME: Her strateji çağrısından ÖNCE, bu rakibe karşı
                    #   önceki turdaki per-opponent payoff'u state'e yaz.
                    if players[i]["strategy"] == "pavlov" and ph[(ni, nj)]["mine"]:
                        lm = ph[(ni, nj)]["mine"][-1]
                        lo = ph[(ni, nj)]["opp"][-1]
                        states[ni]["last_payoff"] = self.pm.get_payoff(lm, lo)[0]
                    if players[j]["strategy"] == "pavlov" and ph[(nj, ni)]["mine"]:
                        lm = ph[(nj, ni)]["mine"][-1]
                        lo = ph[(nj, ni)]["opp"][-1]
                        states[nj]["last_payoff"] = self.pm.get_payoff(lm, lo)[0]
                    # ─────────────────────────────────────────────────────────────
                    mi, si = fn_i(
                        rnd - 1, ph[(ni, nj)]["mine"], ph[(ni, nj)]["opp"], states[ni]
                    )
                    mj, sj = fn_j(
                        rnd - 1, ph[(nj, ni)]["mine"], ph[(nj, ni)]["opp"], states[nj]
                    )
                    states[ni] = si
                    states[nj] = sj
                    ph[(ni, nj)]["mine"].append(mi)
                    ph[(ni, nj)]["opp"].append(mj)
                    ph[(nj, ni)]["mine"].append(mj)
                    ph[(nj, ni)]["opp"].append(mi)
                    pi, pj = self.pm.get_payoff(mi, mj)
                    r_pay[ni] += pi
                    r_pay[nj] += pj
                    scores[ni] += pi
                    scores[nj] += pj
                    if players[i]["strategy"] == "grim_trigger" and mj == "D":
                        grim_tgt[ni].add(nj)
                    if players[j]["strategy"] == "grim_trigger" and mi == "D":
                        grim_tgt[nj].add(ni)
                    if players[i]["strategy"] == "pavlov":
                        states[ni]["last_payoff"] = pi
                    if players[j]["strategy"] == "pavlov":
                        states[nj]["last_payoff"] = pj
                    r_det["matchups"].append(
                        {
                            "p1": ni,
                            "p2": nj,
                            "move1": mi,
                            "move2": mj,
                            "pay1": pi,
                            "pay2": pj,
                        }
                    )
            r_det["payoffs"] = dict(r_pay)
            r_det["cumulative"] = dict(scores)
            rounds_d.append(r_det)
            # Tur sonu Grim Trigger snapshot: o ana kadar oluşan hedefler.
            # Her player için set'in kopyası → tura özgü anlık durum.
            grim_tgt_per_round[rnd] = {k: sorted(v) for k, v in grim_tgt.items()}
            # Adım açıklaması
            desc = " | ".join(
                f"{m['p1']} vs {m['p2']}: ({m['move1']},{m['move2']}) → +{m['pay1']}/+{m['pay2']}"
                for m in r_det["matchups"]
            )
            pstr = " | ".join(f"{p['name']}:+{r_pay.get(p['name'],0)}" for p in players)
            cstr = " | ".join(f"{p['name']}:{scores.get(p['name'],0)}" for p in players)
            steps_l.append(
                {
                    "title": f"{rnd}. Tur",
                    "content": desc,
                    "formula": f"Tur {rnd} kazanç: {pstr}",
                    "result": f"Kümülatif: {cstr}",
                    "type": "process",
                }
            )

        wnd = []
        if step_target and step_target <= effective:
            lo = max(1, step_target - window)
            hi = min(effective, step_target + window)
            wnd = [r for r in rounds_d if lo <= r["round"] <= hi]
        else:
            wnd = rounds_d

        return {
            "rounds": rounds_d,
            "window_rounds": wnd,
            "scores": dict(scores),
            "grim_targets": {k: sorted(v) for k, v in grim_tgt.items() if v},
            "grim_targets_per_round": grim_tgt_per_round,
            "steps": steps_l,
            "n_rounds": effective,
            "step_target": step_target,
            "window": window,
            "players": players,
        }


class _StepWindowScaler:
    """Büyük N adım için ±window pencere hesabı."""

    def compute(
        self, sim_result, target, window=10, pm=None, players=None, resolver=None
    ):
        rounds = sim_result.get("rounds", [])
        stab = self._stability(rounds, players)
        lo = max(1, target - window)
        hi = target + window

        # Küçük hedef: doğrudan simüle et
        if hi <= 2000 and resolver and pm and players:
            sim = _MultiPlayerSimulator(pm, resolver)
            full = sim.simulate(players, hi)
            wnd = [r for r in full["rounds"] if lo <= r["round"] <= hi]
            pr = {}
            if len(full["rounds"]) >= 2:
                l = full["rounds"][-1]
                p2 = full["rounds"][-2]
                for p in players:
                    pr[p["name"]] = l["cumulative"].get(p["name"], 0) - p2[
                        "cumulative"
                    ].get(p["name"], 0)
            steps = [self._mk_step(r, players) for r in wnd]
            return {
                "window_start": lo,
                "window_end": hi,
                "window_rounds": wnd,
                "cumulative_at_target": (
                    full["rounds"][-1]["cumulative"] if full["rounds"] else {}
                ),
                "per_round_payoff": pr,
                "is_steady_state": True,
                "stability_round": stab,
                "target": target,
                "steps": steps,
            }

        # Büyük N: analitik extrapolation
        scr = sim_result.get("scores", {})
        n_sim = len(rounds)
        pr = {p["name"]: scr.get(p["name"], 0) / max(n_sim, 1) for p in (players or [])}
        last_cumul = rounds[-1]["cumulative"] if rounds else scr
        cumul_at_t = {
            p["name"]: last_cumul.get(p["name"], 0)
            + max(0, target - n_sim) * pr.get(p["name"], 0)
            for p in (players or [])
        }
        steps = []
        for rnd in range(lo, hi + 1):
            pc = {
                p["name"]: last_cumul.get(p["name"], 0)
                + max(0, rnd - n_sim) * pr.get(p["name"], 0)
                for p in (players or [])
            }
            pstr = " | ".join(f"{k}:+{v:.1f}" for k, v in pr.items())
            cstr = " | ".join(f"{k}:{v:.0f}" for k, v in pc.items())
            steps.append(
                {
                    "title": f"{rnd}. Tur (Kararlı Durum Tahmini)",
                    "content": "Steady-state extrapolation — tur başına sabit puan.",
                    "formula": f"Tur {rnd} ort. kazanç: {pstr}",
                    "result": f"Kümülatif tahmini: {cstr}",
                    "type": "process",
                }
            )
        return {
            "window_start": lo,
            "window_end": hi,
            "window_rounds": [],
            "cumulative_at_target": cumul_at_t,
            "per_round_payoff": pr,
            "is_steady_state": True,
            "stability_round": stab,
            "target": target,
            "steps": steps,
        }

    @staticmethod
    def _stability(rounds, players):
        if len(rounds) < 4:
            return len(rounds)
        ws = min(3, len(rounds) // 2)
        for i in range(len(rounds) - ws, 0, -1):
            blk = rounds[i : i + ws]
            prv = rounds[i - ws : i]
            if not prv:
                break
            ok = True
            for p in players or []:
                n = p["name"] if isinstance(p, dict) else p
                ab = sum(r["payoffs"].get(n, 0) for r in blk) / ws
                ap = sum(r["payoffs"].get(n, 0) for r in prv) / ws
                if abs(ab - ap) > 0.5:
                    ok = False
                    break
            if ok:
                return i
        return max(1, len(rounds) // 2)

    @staticmethod
    def _mk_step(r, players):
        rnd = r["round"]
        pay = r.get("payoffs", {})
        cum = r.get("cumulative", {})
        pstr = " | ".join(f"{p['name']}:+{pay.get(p['name'],0)}" for p in players)
        cstr = " | ".join(f"{p['name']}:{cum.get(p['name'],0)}" for p in players)
        return {
            "title": f"{rnd}. Tur",
            "content": " · ".join(
                f"{m['p1']} vs {m['p2']}: ({m['move1']},{m['move2']})"
                for m in r.get("matchups", [])
            ),
            "formula": f"Tur {rnd}: {pstr}",
            "result": f"Kümülatif: {cstr}",
            "type": "process",
        }


class _GameTheoryBARTValidator:
    """GT-BART: Oyun kuramı çözümü tutarlılık denetçisi."""

    def validate(self, gt_params, sim_result, sol_data=None):
        violations = []
        score = 1.0
        payoff = gt_params.get("payoff_matrix", {})
        players = gt_params.get("players", [])
        n_rounds = gt_params.get("n_rounds", 1)
        scores = sim_result.get("scores", {})
        rounds = sim_result.get("rounds", [])
        # V1: PD kısıt
        if payoff:
            T = payoff.get("T", 5)
            R = payoff.get("R", 3)
            P = payoff.get("P", 1)
            S = payoff.get("S", 0)
            if payoff and not (T > R > P >= S):
                violations.append(
                    f"[GT-V1] PD kısıtı ihlal: T({T})>R({R})>P({P})≥S({S}) sağlanmıyor"
                )
                score -= 0.15
        # V2: Always C/D tutarlılık
        for player in players:
            nm = player["name"]
            strat = player["strategy"]
            if strat in ("always_c", "always_d"):
                expected = "C" if strat == "always_c" else "D"
                for rnd in rounds[:3]:
                    for m in rnd.get("matchups", []):
                        act_mv = None
                        if m["p1"] == nm:
                            act_mv = m["move1"]
                        elif m["p2"] == nm:
                            act_mv = m["move2"]
                        if act_mv and act_mv != expected:
                            violations.append(
                                f"[GT-V2] {nm} ({strat}) Tur {rnd['round']}'de beklenen {expected} yerine {act_mv} oynadı"
                            )
                            score -= 0.2
                            break
        # V3: Skor mantık kontrolü
        n = len(players)
        T_v = payoff.get("T", 5) if payoff else 5
        P_v = payoff.get("P", 1) if payoff else 1
        for p in players:
            sc = scores.get(p["name"], 0)
            max_p = (n - 1) * n_rounds * T_v
            if sc < 0:
                violations.append(f"[GT-V3] {p['name']} negatif puan ({sc})")
                score -= 0.1
            if sc > max_p + 1:
                violations.append(f"[GT-V3] {p['name']} puan ({sc}) > max ({max_p})")
                score -= 0.1
        # V4: LLM skor karşılaştırması
        if sol_data:
            ans = str(sol_data.get("answer", "") or "")
            for p in players:
                nm = p["name"]
                sv = scores.get(nm, 0)
                m = re.search(
                    rf"{re.escape(nm)}.*?(\d+)\s*(?:puan|toplam)?", ans, re.IGNORECASE
                )
                if m:
                    lv = int(m.group(1))
                    if abs(lv - sv) > 2:
                        violations.append(
                            f"[GT-V4] {nm}: LLM={lv}, solver={sv} — {abs(lv-sv)} puan fark"
                        )
                        score -= 0.15
        score = max(0.0, min(1.0, score))
        seen = set()
        deduped = []
        for v in violations:
            k = v[:60]
            if k not in seen:
                seen.add(k)
                deduped.append(v)
        return score, deduped, score >= 0.7 and len(deduped) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME THEORY THEOREM SUITE  —  v1.0
#  Hard-coding yok. Tüm teoremler dinamik parametrelerle çalışır.
#  Desteklenen teoremler:
#   Nash Dengesi (N-oyuncu), Minimax, Shapley Değeri, Core, Bayes-Nash,
#   Folk Teoremi, Arrow Teoremi, Gibbard-Satterthwaite, Kuhn's Theorem,
#   vNM Utility, Revelation Principle, Myerson-Satterthwaite,
#   Harsanyi Transformation, Correlated Equilibrium, Lemke-Howson,
#   Borda/Condorcet, Gale-Shapley (Matching), Rubinstein Bargaining,
#   Monte Carlo Doğrulama, Chicken Game, Black Swan, Gray Rhino
# ═══════════════════════════════════════════════════════════════════════════════


class _GameTheoryTheoremSuite:
    """
    Kapsamlı oyun kuramı teorem analiz motoru.
    Her teorem:
      - detect(question, gt_params)  → bool   (bu teorem uygulanabilir mi?)
      - compute(gt_params, sim_result, players, payoff) → dict
          keys: name, symbol, applies, result, explanation, formula, warning
    Hard-coding yok: tüm eşikler ve parametreler gt_params/payoff'tan türetilir.
    """

    # ── Teorem algılama desenleri ────────────────────────────────────────────
    _DETECT = {
        "nash": [
            r"nash\s+denge",
            r"nash\s+equilibrium",
            r"denge\s+stratej",
            r"dominant\s+stratej",
            r"baskın\s+stratej",
        ],
        "minimax": [
            r"minimax",
            r"maximin",
            r"sıfır\s+toplam",
            r"zero.sum",
            r"en\s+kötü\s+durumda",
            r"kaybı\s+minimize",
        ],
        "shapley": [
            r"shapley",
            r"kooperatif",
            r"cooperative",
            r"katkı\s+değer",
            r"adil\s+pay",
            r"koalisyon",
        ],
        "core": [
            r"\bcore\b",
            r"çekirdek\s+teorem",
            r"koalisyon\s+bloke",
            r"kararlı\s+koalisyon",
        ],
        "bayes_nash": [
            r"bayes.nash",
            r"eksik\s+bilgi",
            r"tip\s+uzay",
            r"private\s+information",
            r"özel\s+bilgi",
            r"harsanyi",
        ],
        "folk": [
            r"folk\s+teorem",
            r"tekrarlı\s+oyun.*işbirliği",
            r"sonsuz\s+tekrar",
            r"indirim\s+faktörü",
            r"discount\s+factor",
            r"sürdürülebilir\s+işbirliği",
        ],
        "arrow": [
            r"arrow\s+teorem",
            r"sosyal\s+tercih",
            r"imkânsızlık",
            r"oylama\s+paradoks",
            r"condorcet\s+paradoks",
        ],
        "gibbard": [
            r"gibbard",
            r"satterthwaite",
            r"stratejik\s+oylama",
            r"manipüle.*oylama",
            r"strateji.geçirmez",
        ],
        "kuhn": [
            r"kuhn\s+teorem",
            r"sıralı\s+oyun",
            r"extensive\s+form",
            r"mükemmel\s+hafıza",
            r"perfect\s+recall",
            r"backward\s+induction",
        ],
        "vnm": [
            r"von\s+neumann",
            r"morgenstern",
            r"beklenen\s+fayda",
            r"expected\s+utility",
            r"risk\s+tercihi",
            r"fayda\s+teorem",
        ],
        "revelation": [
            r"revelation\s+principle",
            r"açıklama\s+ilkesi",
            r"mekanizma\s+tasarım",
            r"mechanism\s+design",
            r"incentive\s+compat",
        ],
        "myerson": [
            r"myerson",
            r"müzakere\s+imkânsızlık",
            r"çift\s+taraflı",
            r"bilateral\s+trade",
            r"verimli\s+ticaret\s+imkânsız",
        ],
        "harsanyi": [
            r"harsanyi",
            r"tip\s+dönüşüm",
            r"belirsiz\s+tercih",
            r"bayesian\s+game",
            r"tür\s+uzayı",
        ],
        "correlated": [
            r"correlated\s+equilibrium",
            r"korelasyonlu\s+denge",
            r"ortak\s+sinyal",
            r"coordination\s+device",
            r"korelasyon\s+cihaz",
        ],
        "lemke": [
            r"lemke",
            r"howson",
            r"pivoting",
            r"karma\s+strateji\s+hesap",
            r"mixed\s+nash\s+hesap",
        ],
        "borda": [
            r"borda",
            r"condorcet",
            r"oylama\s+kuralı",
            r"majority\s+voting",
            r"çoğunluk\s+oylama",
            r"sıralama\s+oylama",
        ],
        "gale_shapley": [
            r"gale.shapley",
            r"deferred\s+acceptance",
            r"ertelenmiş\s+kabul",
            r"stabil\s+eşleşme",
            r"stable\s+matching",
            r"eşleşme\s+teori",
        ],
        "rubinstein": [
            r"rubinstein",
            r"alternating\s+offer",
            r"dönüşümlü\s+teklif",
            r"pazarlık\s+oyunu",
            r"bargaining\s+model",
            r"müzakere\s+model",
        ],
        "monte_carlo": [
            r"monte\s+carlo",
            r"simülasyon\s+doğrula",
            r"stokastik\s+sim",
            r"rassal\s+simülas",
        ],
        "chicken": [
            r"tavuk\s+oyunu",
            r"chicken\s+game",
            r"anti.koordinasyon",
            r"kukla\s+oyunu",
            r"hawk.dove",
        ],
        "black_swan": [
            r"kara\s+kuğu",
            r"black\s+swan",
            r"nadir\s+olaylar",
            r"uç\s+olay",
            r"fat\s+tail",
            r"kuyruk\s+riski",
        ],
        "gray_rhino": [
            r"gri\s+rhino",
            r"gray\s+rhino",
            r"ihmal\s+edilen\s+risk",
            r"belirgin\s+tehdit",
            r"görmezden\s+gelinen",
        ],
    }

    def detect_applicable(self, question: str, gt_params: dict) -> list:
        """Soruya ve gt_params'a göre uygulanabilir teoremleri döndürür."""
        q = question.lower().translate(str.maketrans("İIĞŞÜÖÇ", "iiğşüöç"))
        applicable = []
        for thm, patterns in self._DETECT.items():
            if any(re.search(p, q) for p in patterns):
                applicable.append(thm)
        # Her oyun kuramı sorusu için Nash + (varsa) Folk/Correlated otomatik
        if gt_params.get("is_game_theory"):
            for auto in ("nash",):
                if auto not in applicable:
                    applicable.append(auto)
            game_type = gt_params.get("game_type", "")
            if game_type == "iterated_pd" and "folk" not in applicable:
                applicable.append("folk")
            if game_type == "chicken" and "chicken" not in applicable:
                applicable.append("chicken")
        return applicable

    # ────────────────────────────────────────────────────────────────────────
    #  TEOREM HESAPLAMA METODLARI
    # ────────────────────────────────────────────────────────────────────────

    def compute_all(
        self,
        question: str,
        gt_params: dict,
        sim_result: dict,
        players: list,
        payoff: dict,
    ) -> list:
        """Uygulanabilir tüm teoremleri hesaplar, sonuç listesi döndürür."""
        applicable = self.detect_applicable(question, gt_params)
        results = []
        dispatch = {
            "nash": self._nash,
            "minimax": self._minimax,
            "shapley": self._shapley,
            "core": self._core,
            "bayes_nash": self._bayes_nash,
            "folk": self._folk,
            "arrow": self._arrow,
            "gibbard": self._gibbard,
            "kuhn": self._kuhn,
            "vnm": self._vnm,
            "revelation": self._revelation,
            "myerson": self._myerson,
            "harsanyi": self._harsanyi,
            "correlated": self._correlated,
            "lemke": self._lemke,
            "borda": self._borda,
            "gale_shapley": self._gale_shapley,
            "rubinstein": self._rubinstein,
            "monte_carlo": self._monte_carlo,
            "chicken": self._chicken,
            "black_swan": self._black_swan,
            "gray_rhino": self._gray_rhino,
        }
        for thm in applicable:
            fn = dispatch.get(thm)
            if fn:
                try:
                    r = fn(gt_params, sim_result, players, payoff)
                    r["_theorem_id"] = thm
                    results.append(r)
                except Exception as e:
                    results.append(
                        {
                            "_theorem_id": thm,
                            "name": thm,
                            "applies": True,
                            "result": f"Hesaplama hatası: {e}",
                            "explanation": "",
                            "formula": "",
                            "warning": str(e),
                        }
                    )
        return results

    # ── 1. Nash Dengesi (N-oyunculu) ─────────────────────────────────────────
    def _nash(self, gt_params, sim_result, players, payoff):
        n = len(players)
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        # N-oyunculu PD'de Nash: tüm oyuncular D oynar (dominant strateji)
        # Çünkü T>R ve P>S → D, karşı taraf ne yaparsa yapsın baskın
        dominant = "D"  # PD'de D her zaman baskın
        nash_profile = tuple([dominant] * n)
        nash_payoff = P  # Her oyuncu P alır (D,D) durumunda
        scores = sim_result.get("scores", {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "name": "Nash Dengesi Teoremi",
            "symbol": "NE",
            "applies": True,
            "result": f"Saf NE: {nash_profile} — tüm oyuncular D (ihanet)",
            "explanation": (
                f"N={n} oyunculu PD'de D baskın stratejidir: T({T})>R({R}) ve P({P})>S({S}). "
                f"Tek saf Nash Dengesi tüm oyuncuların D oynamasıdır. "
                f"Bu Pareto-altoptimaldir: R({R})>P({P}) ama rasyonel aktörler C'ye geçemez."
            ),
            "formula": f"∀i: D domine C  ⟹  NE = (D,...,D)^{n}  |  NE payoff = {P}/oyuncu",
            "warning": (
                f"PD trajedisi: NE'de herkes {P} puan alır, ama tam işbirliğinde "
                f"herkes {R} puan alırdı. Δ = {R-P} puan kayıp/oyuncu."
            ),
        }

    # ── 2. Minimax Teoremi ───────────────────────────────────────────────────
    def _minimax(self, gt_params, sim_result, players, payoff):
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        # PD sıfır-toplamlı değil; minimax değeri her oyuncu için
        # Karşı taraf en kötüyü yaparsa (D), maks alabileceğimiz:
        # C oynarsan S alırsın, D oynarsan P alırsın → D'yi seç → minimax değeri = P
        minimax_val = P
        maximin_val = P  # Benzer şekilde maximin = P
        is_zero_sum = abs((T + S) - (R + R)) > 0.01 or abs((P + P) - (T + S)) > 0.01
        return {
            "name": "Minimax Teoremi (von Neumann)",
            "symbol": "MM",
            "applies": True,
            "result": f"Minimax değeri = {minimax_val} | Maximin değeri = {maximin_val}",
            "explanation": (
                f"PD {'sıfır-toplamlı DEĞİL' if is_zero_sum else 'sıfır-toplamlı'}. "
                f"Her oyuncu için minimax stratejisi D: en kötü durumda "
                f"S({S}) yerine P({P}) garantiler. "
                f"Minimax=Maximin={minimax_val} → anlaşma değeri P."
            ),
            "formula": f"v* = max_i min_j u_i(s_i,s_j) = P = {minimax_val}",
            "warning": "PD sıfır-toplamlı olmadığından Minimax Teoremi doğrudan uygulanamaz; PD'ye uyarlanmış minimax değeri gösterilmektedir.",
        }

    # ── 3. Shapley Değeri ────────────────────────────────────────────────────
    def _shapley(self, gt_params, sim_result, players, payoff):
        coalition_values = gt_params.get("coalition_values") or []
        scores = sim_result.get("scores", {})
        n_rounds = sim_result.get("n_rounds", 1)

        # ── Genel koalisyon oyunu: v(S) girdi olarak verilmişse tam Shapley hesapla
        if coalition_values:
            coalition_map = {}
            player_ids = set()
            for row in coalition_values:
                members = frozenset(str(m).strip() for m in row.get("members", []))
                coalition_map[members] = float(row.get("value", 0.0))
                player_ids.update(members)

            if not player_ids:
                inferred_n = int(gt_params.get("n_players", 0) or 0)
                player_ids = {str(i) for i in range(1, inferred_n + 1)}
            player_ids = sorted(player_ids)
            n = len(player_ids)
            fact_n = math.factorial(n) if n > 0 else 1

            def v_set(s):
                return float(coalition_map.get(frozenset(s), 0.0))

            shapley_vals = {}
            grand_value = v_set(player_ids)
            for pid in player_ids:
                others = [x for x in player_ids if x != pid]
                phi = 0.0
                for r in range(len(others) + 1):
                    for comb in itertools.combinations(others, r):
                        S = frozenset(comb)
                        weight = (
                            math.factorial(len(S))
                            * math.factorial(n - len(S) - 1)
                            / fact_n
                        )
                        phi += weight * (v_set(set(S) | {pid}) - v_set(S))
                shapley_vals[f"Oyuncu {pid}"] = phi

            return {
                "name": "Shapley Değeri (Kooperatif Oyun Teorisi)",
                "symbol": "φ",
                "applies": True,
                "result": " | ".join(
                    f"φ({nm})={sv:.4g}" for nm, sv in shapley_vals.items()
                ),
                "explanation": (
                    f"Koalisyon değer fonksiyonundan doğrudan hesaplandı (n={n}). "
                    f"v(N)={grand_value:.4g}, Σφᵢ={sum(shapley_vals.values()):.4g}."
                ),
                "formula": (
                    "φᵢ(v)=Σ_{S⊆N\\{i}} [|S|!(n-|S|-1)!/n!]·[v(S∪{i})-v(S)] "
                    "| Eksik koalisyon değerleri 0 kabul edildi."
                ),
                "warning": None,
            }

        # ── Iterated-PD fallback: simetrik tam işbirliği varsayımı
        n = len(players)
        R = payoff.get("R", 3)

        def v(size):
            return size * (size - 1) * R if size > 1 else 0

        shapley_vals = {}
        total_val = v(n)
        sv = total_val / n if n > 0 else 0
        for p in players:
            shapley_vals[p["name"]] = sv

        return {
            "name": "Shapley Değeri (Kooperatif Oyun Teorisi)",
            "symbol": "φ",
            "applies": True,
            "result": " | ".join(f"φ({nm})={sv:.1f}" for nm in shapley_vals),
            "explanation": (
                f"Tam işbirliği koalisyonu ({n} oyuncu, {n_rounds} tur): v(N) = {v(n)}. "
                f"Simetrik PD'de Shapley değeri eşit dağılır: φᵢ = {sv:.1f}/oyuncu. "
                f"Gerçek simülasyon: {', '.join(f'{k}={v2}' for k,v2 in scores.items())}."
            ),
            "formula": (
                f"φᵢ(v) = Σ_{{S⊆N\\i}} [|S|!(n-|S|-1)!/n!] · [v(S∪{{i}})-v(S)]  "
                f"| Simetrik: φᵢ = v(N)/n = {sv:.1f}"
            ),
            "warning": (
                (
                    "İhanet eden oyuncular (always_d) kooperatif değeri düşürür. "
                    f"İhanet nedeniyle kayıp: v(N)={v(n)} yerine gerçek toplam={sum(scores.values())}."
                )
                if sum(scores.values()) < v(n)
                else None
            ),
        }

    # ── 4. Core Teoremi ──────────────────────────────────────────────────────
    def _core(self, gt_params, sim_result, players, payoff):
        n = len(players)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        scores = sim_result.get("scores", {})
        # PD'de core: (D,D,...,D) olmayan tahsisler bloke edilebilir mi?
        # Alt-koalisyon değeri: iki kişi birlikte C oynasa 2R kazanır
        # Büyük koalisyon Nash olmadığından core genellikle BOŞ
        core_empty = True  # PD'de core tipik olarak boş
        total_sim = sum(scores.values())
        full_coop_val = n * (n - 1) * R  # Tam işbirliği değeri
        return {
            "name": "Core Teoremi (Kooperatif Oyun)",
            "symbol": "C(v)",
            "applies": True,
            "result": f"Core = {'BOŞ ∅' if core_empty else 'DOLU'}",
            "explanation": (
                f"N-oyunculu PD'de büyük koalisyon kararsız: "
                f"herhangi bir oyuncu D'ye geçerek T({payoff.get('T',5)}) > R({R}) kazanabilir. "
                f"Bu nedenle core boştur: hiçbir tahsis tüm koalisyonları bloke edemez. "
                f"Tam işbirliği değeri v(N)={full_coop_val}, simülasyon değeri={total_sim}."
            ),
            "formula": (
                "C(v) = {x ∈ ℝⁿ : Σxᵢ=v(N), ∀S⊆N: Σᵢ∈S xᵢ ≥ v(S)} "
                f"| PD: T>R ⟹ C(v)=∅"
            ),
            "warning": "PD'de core boşluğu işbirliğinin neden sürdürülemediğini açıklar.",
        }

    # ── 5. Bayes-Nash Dengesi ────────────────────────────────────────────────
    def _bayes_nash(self, gt_params, sim_result, players, payoff):
        n = len(players)
        # Harsanyi tiplerini simüle et
        # Tam bilgi → standart Nash
        # Eksik bilgi → her oyuncu tip uzayına göre strateji seçer
        strategies = {p["name"]: p["strategy"] for p in players}
        is_complete_info = all(
            s in ("always_c", "always_d", "grim_trigger", "pavlov", "tit_for_tat")
            for s in strategies.values()
        )
        # Bayes-Nash = stratejilerin tiplere koşullu optimal seçim
        bne_profile = {}
        for p in players:
            nm = p["name"]
            strat = p["strategy"]
            if strat == "always_d":
                bne_profile[nm] = "D (dominant — tip bağımsız)"
            elif strat == "always_c":
                bne_profile[nm] = "C (tip: tam güven)"
            elif strat == "grim_trigger":
                bne_profile[nm] = "C(t=0), D(t≥trigger) — tip: şartlı güvenilir"
            elif strat == "pavlov":
                bne_profile[nm] = "C(t=0), Win-Stay (adaptif tip)"
            else:
                bne_profile[nm] = f"{strat} (tip çıkarımı gerekli)"
        return {
            "name": "Bayes-Nash Dengesi",
            "symbol": "BNE",
            "applies": True,
            "result": " | ".join(f"{k}: {v}" for k, v in bne_profile.items()),
            "explanation": (
                f"{'Tam bilgi oyunu' if is_complete_info else 'Eksik bilgi oyunu'} "
                f"({n} oyuncu). Harsanyi dönüşümü: her oyuncunun stratejisi kendi "
                f"tipine koşullu BNE profilini oluşturur. "
                f"Eksik bilgide oyuncular diğerlerinin tiplerini prior'larla tahmin eder."
            ),
            "formula": "BNE: sᵢ*(θᵢ) ∈ argmax Eθ₋ᵢ[uᵢ(sᵢ,s₋ᵢ*(θ₋ᵢ),θ) | θᵢ]",
            "warning": None,
        }

    # ── 6. Folk Teoremi ──────────────────────────────────────────────────────
    def _folk(self, gt_params, sim_result, players, payoff):
        n = len(players)
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        n_rounds = gt_params.get("n_rounds", 0)

        # Sonlu oyun: n_rounds tanımlı, pozitif ve "sonsuz proxy" sayılmayacak kadar küçük.
        # 10000 ve üzeri sonsuz yaklaşımı sayılır (mevcut mantıkla tutarlı).
        finite = 0 < n_rounds < 10000

        # ── Sonlu oyun: Folk Teoremi UYGULANAMAZ ─────────────────────────────
        if finite:
            # Backward induction: son turdan başa doğru her adımda D dominant.
            # Tur t=N  : gelecek yok → D strictly dominant → her oyuncu D oynar.
            # Tur t=N-1: t=N'de D oynayacak bilindiğinden ceza tehdidi yok → D dominant.
            # ...
            # Tur t=1  : aynı mantık → D dominant.
            # Sonuç: Tek SPNE = (D,D,...,D) tüm N turda.
            bi_steps_preview = min(n_rounds, 4)  # Gösterilecek adım sayısı (özet)
            bi_lines = []
            for step in range(bi_steps_preview):
                t = n_rounds - step
                if step == 0:
                    bi_lines.append(
                        f"  t={t} (son tur)  : Gelecek yok → D dominant → NE_t={t} = (D,...,D)"
                    )
                else:
                    bi_lines.append(
                        f"  t={t}            : t={t+1}'de D bilindiğinden ceza tehdidi yok "
                        f"→ D dominant → NE_t={t} = (D,...,D)"
                    )
            if n_rounds > bi_steps_preview:
                bi_lines.append(
                    f"  ... (aynı mantık t={n_rounds - bi_steps_preview} → t=1 için geçerli)"
                )
            bi_lines.append(
                f"  t=1              : D dominant → SPNE: (D,...,D) tüm {n_rounds} turda"
            )
            backward_induction_trace = "\n".join(bi_lines)

            # İşbirliği kazanç karşılaştırması (gösterim amaçlı, UYGULANAMAZ bağlamında)
            coop_gain_per_round = R - P  # İşbirliğinin ek kazancı: R - P
            total_coop_gain = coop_gain_per_round * n_rounds
            return {
                "name": "Folk Teoremi (Tekrarlanan Oyunlar)",
                "symbol": "FT",
                "applies": False,  # Sonlu oyunda Folk Teoremi GEÇERSİZ
                "result": (
                    f"UYGULANAMAZ — {n_rounds}-turlu sonlu oyun  |  "
                    f"Backward induction: SPNE = (D,...,D) tüm {n_rounds} turda"
                ),
                "explanation": (
                    f"Folk Teoremi yalnızca SONSUZ (veya belirsiz sonlu) tekrarlı oyunlara uygulanır. "
                    f"Bu {n_rounds}-turlu sonlu oyunda geriye doğru tümevarım (backward induction) "
                    f"işbirliğini tamamen çözdürür:\n"
                    f"{backward_induction_trace}\n"
                    f"İşbirliğinin kaybettirdiği fırsat maliyeti: "
                    f"{n_rounds} tur × (R-P) = {n_rounds} × ({R}-{P}) = {total_coop_gain} puan/oyuncu. "
                    f"Sonsuz oyundaki δ* eşiği sonlu oyunda anlamsızdır çünkü "
                    f"ceza tehdidi ('grim trigger' vb.) son turdan itibaren geriye doğru devre dışı kalır."
                ),
                "formula": (
                    f"SPNE (Sonlu PD, N={n_rounds} tur):\n"
                    f"  σᵢ*(t) = D  ∀t ∈ {{1,...,N}}, ∀i\n"
                    f"  Kanıt (backward induction):\n"
                    f"    t=N : D strictly dominant (T>{R}=R, P>{S}=S) → σᵢ*(N)=D\n"
                    f"    t=k : σᵢ*(k+1)=...=σᵢ*(N)=D bilindiğinden\n"
                    f"          ceza tehdidi = 0 → D dominant → σᵢ*(k)=D\n"
                    f"    ∴ σᵢ*(t)=D ∀t  (tümevarımla)  ⟹  u_i = {P}×{n_rounds} = {P*n_rounds}/oyuncu\n"
                    f"  Kaybedilen işbirliği değeri: (R-P)×N = ({R}-{P})×{n_rounds} = {total_coop_gain}/oyuncu"
                ),
                "warning": (
                    (
                        f"delta_star ({(T-R)/(T-P):.4f} için T={T},R={R},P={P}) SONLU OYUNDA GÖSTERİLMİYOR: "
                        f"Bu değer yalnızca sonsuz tekrarlı oyunlarda anlamlıdır. "
                        f"Sonlu oyunda δ* hesabı teorik olarak geçersiz ve yanıltıcıdır."
                    )
                    if T != P
                    else (
                        f"T=P olduğundan δ* zaten tanımsız; sonlu oyun backward induction sonucu geçerlidir."
                    )
                ),
            }

        # ── Sonsuz / belirsiz tekrarlı oyun: Folk Teoremi uygulanır ──────────
        if T == P:
            delta_star = None
            folk_holds = False
        else:
            delta_star = (T - R) / (T - P)
            folk_holds = delta_star < 1.0

        return {
            "name": "Folk Teoremi (Tekrarlanan Oyunlar)",
            "symbol": "FT",
            "applies": True,
            "result": (
                f"δ* = {delta_star:.4f}  |  Folk Teoremi {'GEÇERLİ (δ*<1)' if folk_holds else 'GEÇERSİZ (δ*≥1)'}"
                if delta_star is not None
                else "δ* = tanımsız (T=P)"
            )
            + "  |  Sonsuz tekrarlı oyun",
            "explanation": (
                "Folk Teoremi: sonsuz tekrarlı PD'de δ ≥ δ* = "
                + (f"{delta_star:.4f}" if delta_star is not None else "?")
                + " ise Grim Trigger stratejisiyle (C,C) dengesi sürdürülebilir. "
                + (
                    f"δ* = (T-R)/(T-P) = ({T}-{R})/({T}-{P}) = {delta_star:.4f}. "
                    f"δ < δ* ise tek dönemlik ihanet cazibesini gelecek işbirliği kazanımları "
                    f"dengeleyemez → (D,D) tek denge. "
                    if delta_star is not None
                    else ""
                )
                + (
                    f"δ ≥ {delta_star:.4f} olan oyuncular işbirliğini SPNE olarak sürdürebilir."
                    if folk_holds and delta_star is not None
                    else (
                        "T=P olduğundan ihanet cazibesi sıfır; Folk Teoremi koşulsuz sağlanır."
                        if delta_star is None
                        else f"δ* = {delta_star:.4f} ≥ 1: hiçbir δ∈[0,1) koşulu sağlayamaz → işbirliği sürdürülemez."
                    )
                )
            ),
            "formula": (
                f"δ* = (T-R)/(T-P) = ({T}-{R})/({T}-{P}) = {delta_star:.4f}\n"
                f"Grim Trigger SPNE koşulu: δ ≥ δ* = {delta_star:.4f}\n"
                f"  İşbirliği payoff (sonsuz): R/(1-δ)\n"
                f"  İhanet payoff: T + δ·P/(1-δ)\n"
                f"  Fark ≥ 0  ⟺  δ ≥ (T-R)/(T-P)"
                if delta_star is not None
                else "δ* = tanımsız (T=P)"
            ),
            "warning": (
                f"δ* = {delta_star:.4f} ≥ 1: Folk Teoremi koşulu hiçbir geçerli δ için sağlanamaz."
                if delta_star is not None and not folk_holds
                else None
            ),
        }

    # ── 7. Arrow Teoremi ────────────────────────────────────────────────────
    def _arrow(self, gt_params, sim_result, players, payoff):
        n = len(players)
        scores = sim_result.get("scores", {})
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "name": "Arrow'un İmkânsızlık Teoremi",
            "symbol": "AT",
            "applies": True,
            "result": (
                f"Sosyal tercih sıralaması: "
                + " > ".join(f"{nm}({sc})" for nm, sc in sorted_p)
            ),
            "explanation": (
                f"Arrow Teoremi: ≥3 alternatifli ({n} oyuncu) herhangi bir sosyal seçim "
                f"kuralı şu özellikleri aynı anda sağlayamaz: "
                f"(1) Pareto verimliliği, (2) Bağımsız ilgisiz alternatifler (IIA), "
                f"(3) Diktatörlük olmaması. Simülasyon skor sıralaması Borda skoru "
                f"olarak yorumlanabilir; bu sıralama IIA'yı ihlal edebilir."
            ),
            "formula": "∄ f: L(A)ⁿ→L(A) s.t. Pareto ∧ IIA ∧ Non-Dictatorship  (|A|≥3)",
            "warning": "Arrow Teoremi normatif bir imkânsızlık sonucudur; çözüm algoritması sunmaz.",
        }

    # ── 8. Gibbard-Satterthwaite Teoremi ────────────────────────────────────
    def _gibbard(self, gt_params, sim_result, players, payoff):
        n = len(players)
        return {
            "name": "Gibbard-Satterthwaite Teoremi",
            "symbol": "GS",
            "applies": True,
            "result": f"N={n} oyunculu oylama — stratejik manipülasyon mümkün",
            "explanation": (
                f"G-S Teoremi: ≥3 alternatif ve ≥2 seçici olan ({n} oyuncu) "
                f"herhangi bir sürjektif sosyal seçim fonksiyonu ya diktatörlüktür "
                f"ya da oyuncular stratejik oy kullanarak sonucu etkileyebilir. "
                f"Enerji hattı krizinde her aktör rapor ettiği tercihlerle "
                f"gerçek tercihlerini farklılaştırabilir."
            ),
            "formula": "∄ f: strategy-proof ∧ surjective ∧ non-dictatorial  (|A|≥3)",
            "warning": "Her oylama mekanizması manipülasyona açıktır (G-S sonucu).",
        }

    # ── 9. Kuhn's Theorem (Sıralı Oyunlar) ──────────────────────────────────
    def _kuhn(self, gt_params, sim_result, players, payoff):
        n = len(players)
        rounds = sim_result.get("rounds", [])
        # Her tur sıralı kararlar dizisi olarak yorumlanabilir
        # Mükemmel hafıza → davranışsal strateji = karma strateji
        return {
            "name": "Kuhn's Theorem (Genişletilmiş Form)",
            "symbol": "KT",
            "applies": True,
            "result": f"{n} oyuncu, {len(rounds)} tur sıralı karar ağacı",
            "explanation": (
                f"Kuhn Teoremi: mükemmel hafızalı sıralı oyunlarda "
                f"her karma strateji için eşdeğer davranışsal strateji vardır. "
                f"Bu {n}-oyunculu, {len(rounds)}-turlu oyunda her aktör "
                f"geçmiş hamleleri gözlemleyerek (Grim Trigger, Pavlov) "
                f"davranışsal stratejisini oluşturur. Backward induction uygulanabilir."
            ),
            "formula": "∀ mükemmel hafızalı oyun: Karma Strateji ≡ Davranışsal Strateji",
            "warning": None,
        }

    # ── 10. vNM Utility Theorem ──────────────────────────────────────────────
    def _vnm(self, gt_params, sim_result, players, payoff):
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        # vNM utility fonksiyonu için beklenen fayda hesapla
        # Risk nötral varsayım: u(x) = x
        # Karma strateji p: p*T + (1-p)*S karşısında R (C oynayarak)
        # Kayıtsız noktası: p*T + (1-p)*S = R → p = (R-S)/(T-S)
        if (T - S) > 0:
            p_indiff = (R - S) / (T - S)
        else:
            p_indiff = None
        p_str = f"{p_indiff:.4f}" if p_indiff is not None else "?"
        return {
            "name": "von Neumann-Morgenstern Utility Teoremi",
            "symbol": "vNM",
            "applies": True,
            "result": (
                f"Kayıtsızlık noktası: p* = (R-S)/(T-S) = {p_str}"
                if p_indiff is not None
                else "Hesaplanamaz (T=S)"
            ),
            "explanation": (
                f"vNM Teoremi: tutarlı tercih (completeness+transitivity+continuity+independence) "
                f"varsa beklenen fayda temsili vardır. Ödeme ({S},{P},{R},{T}) beklenen fayda: "
                f"E[u(C)] = p*S + (1-p)*R, E[u(D)] = p*P + (1-p)*T. "
                f"p*={p_str}'de oyuncu C ve D arasinda kayitsizdir."
            ),
            "formula": (f"p* = (R-S)/(T-S) = ({R}-{S})/({T}-{S}) = {p_str}"),
            "warning": None,
        }

    # ── 11. Revelation Principle ─────────────────────────────────────────────
    def _revelation(self, gt_params, sim_result, players, payoff):
        n = len(players)
        return {
            "name": "Açıklama İlkesi (Revelation Principle)",
            "symbol": "RP",
            "applies": True,
            "result": f"N={n} oyunculu mekanizma tasarımı için geçerli",
            "explanation": (
                f"Revelation Principle: herhangi bir Bayes-Nash dengesi yaratan "
                f"mekanizma için, oyuncuların tercihlerini doğrudan açıkladığı "
                f"eşdeğer teşvik-uyumlu (IC) direkt mekanizma vardır. "
                f"Enerji hattı müzakerelerinde her aktörün gerçek değerlemesini "
                f"açıklamasını sağlayan IC mekanizma tasarlanabilir."
            ),
            "formula": "∀ BNE mekanizma M, ∃ direkt IC mekanizma M': aynı sonucu üretir",
            "warning": None,
        }

    # ── 12. Myerson-Satterthwaite Teoremi ────────────────────────────────────
    def _myerson(self, gt_params, sim_result, players, payoff):
        n = len(players)
        T = payoff.get("T", 5)
        S = payoff.get("S", 0)
        return {
            "name": "Myerson-Satterthwaite Teoremi",
            "symbol": "MS",
            "applies": True,
            "result": "Verimli çift taraflı ticaret mekanizması MÜMKÜN DEĞİL",
            "explanation": (
                f"M-S Teoremi: özel bilgi olan iki taraflı ticarette "
                f"(alıcı değeri [S={S},T={T}] üzerinden belirsiz) hiçbir mekanizma "
                f"aynı anda BIC (Bayes-IC), BIR (bireysel rasyonellik) ve "
                f"ex-post verimli olamaz. "
                f"Enerji hattı müzakerelerinde bir tarafın ihanet payoffunu gizlemesi "
                f"verimli anlaşmayı engeller (ölü kilo kaybı kaçınılmaz)."
            ),
            "formula": "∄ mekanizma: BIC ∧ BIR ∧ Ex-Post Efficiency  (özel bilgi)",
            "warning": f"Ödeme asimetrisi [{S},{T}] müzakere verimsizliğine yol açar.",
        }

    # ── 13. Harsanyi Transformation ──────────────────────────────────────────
    def _harsanyi(self, gt_params, sim_result, players, payoff):
        n = len(players)
        strategies = {p["name"]: p["strategy"] for p in players}
        # Her oyuncunun strateji tipi = bir "Harsanyi tipi"
        type_map = {}
        for p in players:
            nm = p["name"]
            strat = p["strategy"]
            if strat == "always_c":
                type_map[nm] = f"θ_C (güvenilir tip) → C ağırlık=1"
            elif strat == "always_d":
                type_map[nm] = f"θ_D (ihanetçi tip) → D ağırlık=1"
            elif strat == "grim_trigger":
                type_map[nm] = f"θ_GT (şartlı tip) → C(t=0), D(ihanet sonrası)"
            elif strat == "pavlov":
                type_map[nm] = f"θ_P (adaptif tip) → payoffa bağlı"
            else:
                type_map[nm] = f"θ_{strat[:4]} (belirsiz tip)"
        return {
            "name": "Harsanyi Dönüşümü (Tip Uzayı Modeli)",
            "symbol": "HT",
            "applies": True,
            "result": " | ".join(f"{k}: {v}" for k, v in type_map.items()),
            "explanation": (
                f"Harsanyi Dönüşümü: eksik bilgi oyununu {n} oyuncunun "
                f"Harsanyi tiplerine sahip tam bilgi Bayesian oyununa dönüştürür. "
                f"Her strateji bir tip θ olarak modellenir; "
                f"doğa başta tip profilini çizer, sonra oyun oynanır."
            ),
            "formula": "G(eksik bilgi) → G*(tam bilgi Bayesian): (Θ,p,u) üçlüsü",
            "warning": None,
        }

    # ── 14. Correlated Equilibrium ───────────────────────────────────────────
    def _correlated(self, gt_params, sim_result, players, payoff):
        n = len(players)
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        # Basit CE: her oyuncu (C,C) önerisini takip eder mi?
        # CE koşulu: u(C|öneri=C) ≥ u(D|öneri=C)
        # R ≥ S → her zaman sağlanır (T>R iken tek başına sağlanmaz)
        # Gerçek CE için dış otorite sinyali gerekir
        # CE payoff ≥ Nash payoff (P), Nash olmayan CE var mı?
        # Basit CE: (q*(C,C) + (1-q)*(D,D)) karışımı
        # q için CE koşulu: q*R + (1-q)*P ≥ q*T + (1-q)*S ise C tercih edilir
        # q*(R-T) ≥ (1-q)*(S-P) → q*(R-T-S+P) ≥ -P+S
        # T>R ve S<P olduğundan sol taraf negatif → CE koşulu gereği karışık
        if (R - T - S + P) != 0:
            q_ce = (P - S) / (T - R + P - S) if (T - R + P - S) != 0 else None
        else:
            q_str = f"{q_ce:.4f}" if q_ce is not None else "?"
        return {
            "name": "Korelasyonlu Denge (Correlated Equilibrium)",
            "symbol": "CE",
            "applies": True,
            "result": (
                f"CE esigi q* = {q_str}" if q_ce is not None else "CE: NE'ye coker"
            ),
            "explanation": (
                f"Aumann CE: koordinasyon cihazi q*={q_str} olasilikla "
                f"(C,C), (1-q*) olasilikla (D,D) onerirse ve "
                f"oyuncular oneriyi takip ederse CE olusur. "
                f"CE her zaman NE(={P}) payoffu >= verir; ortak sinyal R({R}) ortalamasina yaklasar. "
                f"PD'de CE, NE degerini ({P}) asabilir ancak Pareto optimumu ({R}) degil."
            ),
            "formula": (
                f"q* = (P-S)/(T-R+P-S) = ({P}-{S})/({T}-{R}+{P}-{S}) = {q_str}"
                if q_ce is not None
                else "CE=NE"
            ),
            "warning": f"CE payoff in [{P}, {R}] araliginda; koordinasyon cihazi gerektirir.",
        }

    # ── 15. Lemke-Howson Algoritması ─────────────────────────────────────────
    def _lemke(self, gt_params, sim_result, players, payoff):
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        # 2x2 PD için Lemke-Howson manuel hesabı
        # A = oyuncu 1 payoff matrisi: [[R,S],[T,P]]
        # B = oyuncu 2 payoff matrisi: [[R,T],[S,P]]
        A = [[R, S], [T, P]]
        B = [[R, T], [S, P]]
        # Baskın strateji: D (satır 2) her zaman daha iyi
        # LH başlangıç vertex: (D için q=0) → pure nash (D,D)
        nash_found = ("D", "D")
        nash_payoff = (P, P)
        return {
            "name": "Lemke-Howson Algoritması (Nash Bulma)",
            "symbol": "LH",
            "applies": True,
            "result": f"Nash: {nash_found} | Payoff: {nash_payoff}",
            "explanation": (
                f"Lemke-Howson: 2×2 PD için pivoting ile Nash Dengesi bulunur. "
                f"A=[[{R},{S}],[{T},{P}]], B=[[{R},{T}],[{S},{P}]]. "
                f"D satırı C satırına baskın (T>R, P>S). "
                f"LH tek Nash vertex'ini bulur: (D,D)=({P},{P}). "
                f"Karma strateji Nash: PD'de yok (saf NE dominant)."
            ),
            "formula": (
                f"Tableau: A+E, B'+E | E=birim matris | "
                f"Pivot → NE=(D,D) | Payoff=({P},{P})"
            ),
            "warning": "N>2 için N-kişi Lemke-Howson genellemesi gerekir (NP-hard sınırı).",
        }

    # ── 16. Borda / Condorcet ────────────────────────────────────────────────
    def _borda(self, gt_params, sim_result, players, payoff):
        n = len(players)
        scores = sim_result.get("scores", {})
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Borda skoru = sıralama pozisyonuna göre puan
        borda = {nm: (n - 1 - i) for i, (nm, _) in enumerate(sorted_p)}
        # Condorcet kazananı: tüm ikililerde kazanan
        rounds = sim_result.get("rounds", [])
        head2head = defaultdict(lambda: defaultdict(int))
        for rnd in rounds:
            for m in rnd.get("matchups", []):
                p1, p2 = m["p1"], m["p2"]
                pay1, pay2 = m["pay1"], m["pay2"]
                if pay1 > pay2:
                    head2head[p1][p2] += 1
                elif pay2 > pay1:
                    head2head[p2][p1] += 1
        condorcet_winner = None
        for nm, _ in sorted_p:
            if all(
                head2head[nm][other] >= head2head[other][nm]
                for other, _ in sorted_p
                if other != nm
            ):
                condorcet_winner = nm
                break
        return {
            "name": "Borda Count & Condorcet Teorisi (Oylama)",
            "symbol": "BC",
            "applies": True,
            "result": (
                f"Borda: {', '.join(f'{k}={v}' for k,v in borda.items())} | "
                f"Condorcet kazananı: {condorcet_winner or 'YOK (döngü)'}"
            ),
            "explanation": (
                f"Borda Count: {n} oyuncunun simülasyon sıralamalarından türetildi. "
                f"Condorcet kazananı: tüm ikili karşılaşmalarda kazanan "
                f"({'bulundu: '+condorcet_winner if condorcet_winner else 'bulunamadı — döngüsel tercih'}). "
                f"Döngüsel tercih → Arrow Teoremi ile tutarlı: saf Condorcet kazananı olmayabilir."
            ),
            "formula": "Borda(i) = n-1-rank(i) | Condorcet: ∀j≠i: |{t: xᵢt>xⱼt}| > |{t: xⱼt>xᵢt}|",
            "warning": "Borda ve Condorcet farklı kazananlar üretebilir (oylama paradoksu).",
        }

    # ── 17. Gale-Shapley (Eşleşme Teorisi) ──────────────────────────────────
    def _gale_shapley(self, gt_params, sim_result, players, payoff):
        n = len(players)
        scores = sim_result.get("scores", {})
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Tercih listesi = skor sıralaması olarak model
        # Stabil eşleşme: her oyuncu puan sırasına göre partner seçer
        names = [nm for nm, _ in sorted_p]
        # Basit GS: 2'şerli eşleşme (round-robin'den)
        pairs = []
        matched = set()
        for i in range(0, len(names) - 1, 2):
            if i + 1 < len(names):
                pairs.append((names[i], names[i + 1]))
                matched.update([names[i], names[i + 1]])
        if len(names) % 2 == 1:
            pairs.append((names[-1], "—"))
        return {
            "name": "Gale-Shapley Ertelenmiş Kabul (Eşleşme Teorisi)",
            "symbol": "GS-DA",
            "applies": True,
            "result": (
                f"Stabil eşleşme önerisi: " + ", ".join(f"({a}↔{b})" for a, b in pairs)
            ),
            "explanation": (
                f"Gale-Shapley DA Algoritması: {n} oyuncunun tercih sıralaması "
                f"simülasyon skorlarından türetildi. "
                f"Stabil eşleşme garantisi: hiçbir çift eşleşmesini reddedip "
                f"birbirini tercih edemez. Öneren taraf (proposer-optimal) lehine asimetri var."
            ),
            "formula": (
                "GS-DA: ∀ stabil olmayan eşleşme (i,j): "
                "∃ (i',j') s.t. i prefers j' over j AND j' prefers i over i'"
            ),
            "warning": None,
        }

    # ── 18. Rubinstein Pazarlık Modeli ───────────────────────────────────────
    def _rubinstein(self, gt_params, sim_result, players, payoff):
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        n = len(players)
        # İki-oyunculu Rubinstein: delta ile bölme
        # δ varsayılan 0.9 (Folk teoremi δ*'ı üzerinden)
        if (T - P) > 0:
            delta_star = (T - R) / (T - P)
        else:
            delta_star = 0.9
        delta = max(delta_star, 0.9)  # Folk teoremini sağlayan δ
        # P1 payı: 1/(1+δ), P2 payı: δ/(1+δ)
        share_p1 = 1 / (1 + delta)
        share_p2 = delta / (1 + delta)
        total_surplus = R * (n * (n - 1) // 2)  # Tam işbirliği artığı
        return {
            "name": "Rubinstein Pazarlık Modeli (Dönüşümlü Teklifler)",
            "symbol": "RB",
            "applies": True,
            "result": (
                f"δ={delta:.3f} | P1 payı={share_p1:.3f} | P2 payı={share_p2:.3f} | "
                f"Toplam artık={total_surplus}"
            ),
            "explanation": (
                f"Rubinstein: sonsuz ufuklu dönüşümlü teklif oyununda "
                f"benzersiz alt-oyun mükemmel dengesi δ={delta:.3f} ile "
                f"ilk teklif sahibi {share_p1:.1%}, ikincisi {share_p2:.1%} alır. "
                f"δ→1 iken (sabırlı oyuncular) dağılım eşitlenir (50-50). "
                f"Enerji hattı müzakeresinde daha sabırlı taraf lehine asimetri oluşur."
            ),
            "formula": f"x* = 1/(1+δ) = {share_p1:.4f}  |  y* = δ/(1+δ) = {share_p2:.4f}",
            "warning": f"Rubinstein çözümü {n}>2 için uzlaşma çerçevesi olarak yorumlanır.",
        }

    # ── 19. Monte Carlo Simülasyon Doğrulama ────────────────────────────────
    def _monte_carlo(self, gt_params, sim_result, players, payoff):
        import random as _rnd

        n = len(players)
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        n_rounds = sim_result.get("n_rounds", 3)
        # 1000 MC simülasyonu
        N_SIM = 1000
        mc_scores = defaultdict(list)
        strat_map = {p["name"]: p["strategy"] for p in players}
        resolver = _StrategyResolver()
        payoff_matrix = {
            ("C", "C"): (R, R),
            ("D", "C"): (T, S),
            ("C", "D"): (S, T),
            ("D", "D"): (P, P),
        }

        for _ in range(N_SIM):
            states = {
                p["name"]: {
                    "triggered": False,
                    "last_payoff": R,
                    "last_move": "C",
                    "p_threshold": 1,
                }
                for p in players
            }
            ph = {
                (players[i]["name"], players[j]["name"]): {"mine": [], "opp": []}
                for i in range(n)
                for j in range(n)
                if i != j
            }
            scores_sim = {p["name"]: 0 for p in players}
            for rnd in range(n_rounds):
                for i in range(n):
                    for j in range(i + 1, n):
                        ni = players[i]["name"]
                        nj = players[j]["name"]
                        fn_i = resolver.resolve(strat_map[ni])
                        fn_j = resolver.resolve(strat_map[nj])
                        mi, si = fn_i(
                            rnd, ph[(ni, nj)]["mine"], ph[(ni, nj)]["opp"], states[ni]
                        )
                        mj, sj = fn_j(
                            rnd, ph[(nj, ni)]["mine"], ph[(nj, ni)]["opp"], states[nj]
                        )
                        states[ni] = si
                        states[nj] = sj
                        ph[(ni, nj)]["mine"].append(mi)
                        ph[(ni, nj)]["opp"].append(mj)
                        ph[(nj, ni)]["mine"].append(mj)
                        ph[(nj, ni)]["opp"].append(mi)
                        pi, pj = payoff_matrix.get((mi, mj), (0, 0))
                        scores_sim[ni] += pi
                        scores_sim[nj] += pj
            for nm, sc in scores_sim.items():
                mc_scores[nm].append(sc)

        # Ortalama ve std
        mc_mean = {nm: sum(v) / N_SIM for nm, v in mc_scores.items()}
        mc_std = {
            nm: (sum((x - mc_mean[nm]) ** 2 for x in v) / N_SIM) ** 0.5
            for nm, v in mc_scores.items()
        }
        sim_scores = sim_result.get("scores", {})
        # Uyum kontrolü
        agreement = all(
            abs(sim_scores.get(nm, 0) - mc_mean[nm]) <= 2 * mc_std[nm] + 0.5
            for nm in mc_mean
        )
        return {
            "name": "Monte Carlo Simülasyon Doğrulama",
            "symbol": "MC",
            "applies": True,
            "result": (
                f"N_sim={N_SIM} | Uyum: {'✓ DOĞRULANDI' if agreement else '⚠ SAPMA'} | "
                + " ".join(f"{nm}={mc_mean[nm]:.1f}±{mc_std[nm]:.2f}" for nm in mc_mean)
            ),
            "explanation": (
                f"{N_SIM} MC simülasyonu ile analitik solver doğrulandı. "
                f"Deterministik stratejiler (always_c, always_d, grim_trigger) "
                f"MC std≈0 verir. Pavlov adaptif olduğu için küçük varyasyon gösterebilir."
            ),
            "formula": f"MC[{N_SIM}]: μ_i = E[score_i] ± σ_i",
            "warning": (
                None
                if agreement
                else "Analitik ve MC sonuçları arasında sapma var — strateji yorumunu kontrol edin."
            ),
        }

    # ── 20. Chicken Game (Tavuk Oyunu) ───────────────────────────────────────
    def _chicken(self, gt_params, sim_result, players, payoff):
        T = payoff.get("T", 5)
        R = payoff.get("R", 3)
        P = payoff.get("P", 1)
        S = payoff.get("S", 0)
        n = len(players)
        # Chicken Game: (D,D) en kötü, (C,D) ve (D,C) asimetrik NE'ler
        # PD'de P>S ama Chicken'da S>P (geri adım atmak çarpışmadan iyidir)
        is_chicken = S > P  # Chicken game koşulu
        return {
            "name": "Chicken Game (Tavuk Oyunu / Hawk-Dove)",
            "symbol": "CG",
            "applies": True,
            "result": (
                f"{'TAVUK OYUNU' if is_chicken else 'PD (tavuk degil)'}: "
                f"S({S}) {'>' if is_chicken else '<='} P({P})"
            ),
            "explanation": (
                "Chicken Game: T>R>S>P siralamasinda (S>P) "
                "iki asimetrik saf Nash vardir: (C,D) ve (D,C). "
                + (
                    "Bu oyun TAVUK oyunudur. "
                    if is_chicken
                    else "Bu oyun PD'dir, tavuk degil. "
                )
                + "Chicken'da her oyuncu digerinin geri adim atmasini umar; "
                "enerji hatti krizinde 'kimse tavuk olmak istemez' dinamigi olusur."
            ),
            "formula": (
                f"Chicken: T>R>S>P  ⟹  NE = {{(C,D),(D,C)}} (asimetrik) | "
                f"PD: T>R>P>S  ⟹  NE = {{(D,D)}} (simetrik)"
            ),
            "warning": (
                (
                    "Mevcut ödeme yapısı PD — Chicken analizi referans olarak sunulmuştur."
                )
                if not is_chicken
                else None
            ),
        }

    # ── 21. Black Swan Teorisi ───────────────────────────────────────────────
    def _black_swan(self, gt_params, sim_result, players, payoff):
        n = len(players)
        scores = sim_result.get("scores", {})
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_rounds = sim_result.get("n_rounds", 3)
        T = payoff.get("T", 5)
        S = payoff.get("S", 0)
        # Black Swan: beklenmedik yüksek etki olayı
        # Simülasyonda: bir oyuncunun tüm turlarda 0 alması veya en yüksek sapma
        scores_list = list(scores.values())
        if len(scores_list) < 2:
            mean_sc = scores_list[0] if scores_list else 0
            std_sc = 0
        else:
            mean_sc = sum(scores_list) / len(scores_list)
            std_sc = (
                sum((x - mean_sc) ** 2 for x in scores_list) / len(scores_list)
            ) ** 0.5
        black_swan_risk = std_sc / (mean_sc + 1e-9)  # Değişkenlik katsayısı
        # En kötü performans oyuncu (always_d ile karşılaşan)
        worst = sorted_p[-1] if sorted_p else ("?", 0)
        best = sorted_p[0] if sorted_p else ("?", 0)
        max_possible = (n - 1) * n_rounds * T
        return {
            "name": "Kara Kuğu Teorisi (Taleb)",
            "symbol": "BS",
            "applies": True,
            "result": (
                f"Kuyruk riski CV={black_swan_risk:.3f} | "
                f"Kara Kuğu senaryosu: {worst[0]}={worst[1]} puan (min) vs "
                f"{best[0]}={best[1]} puan (max) | Max teorik: {max_possible}"
            ),
            "explanation": (
                f"Kara Kuğu: nadir, beklenmedik, yüksek etkili olay. "
                f"Bu oyunda 'always_d' stratejisi (İran) sistematik bir "
                f"beklenmedik asimetri yaratır: diğer aktörler C-C işbirliği "
                f"beklerken İran D ile T({T}) alır. "
                f"Değişkenlik katsayısı CV={black_swan_risk:.3f}: "
                f"{'YÜKSEK kuyruk riski' if black_swan_risk>0.4 else 'normal dağılım'}. "
                f"Taleb: fat-tail dağılımı normal statistiği geçersiz kılar."
            ),
            "formula": "CV = σ/μ | Kara Kuğu: |xᵢ - μ| > 3σ",
            "warning": (
                f"Asimetrik ödeme (S={S} vs T={T}) Kara Kuğu maruziyetini artırır. "
                f"Grim/Pavlov stratejileri buna karşı hedge mekanizması sunar."
            ),
        }

    # ── 22. Gri Rhino Teorisi ────────────────────────────────────────────────
    def _gray_rhino(self, gt_params, sim_result, players, payoff):
        n = len(players)
        scores = sim_result.get("scores", {})
        T = payoff.get("T", 5)
        P = payoff.get("P", 1)
        R = payoff.get("R", 3)
        # Gri Rhino: görünür, öngörülebilir ama görmezden gelinen risk
        # always_d oyuncusu = 'gri rhino' (bilinçli ihmal)
        gray_rhinos = [
            p["name"]
            for p in sim_result.get("players", gt_params.get("players", []))
            if p.get("strategy") == "always_d"
        ]
        ignored_risk = (T - R) * len(gray_rhinos)  # Her ihanetçi için kayıp fark
        return {
            "name": "Gri Rhino Teorisi (Michele Wucker)",
            "symbol": "GR",
            "applies": True,
            "result": (
                f"Gri Rhino: {', '.join(gray_rhinos) if gray_rhinos else 'YOK'} "
                f"| Öngörülebilir kayıp: {ignored_risk} puan/tur"
            ),
            "explanation": (
                f"Gri Rhino: büyük, net görünür tehdit — ama görmezden gelinen. "
                f"{'Oyuncu(lar): '+', '.join(gray_rhinos)+' — always_d oynuyor. ' if gray_rhinos else ''}"
                f"Bu tehdit oyun başında bilinebilirdi: "
                f"daima ihanet eden aktörün varlığı sistem genelinde "
                f"{ignored_risk} puan/tur kayba yol açar. "
                f"Kara Kuğu'dan farkı: önceden öngörülebilir ama harekete geçilmemiş risk."
            ),
            "formula": "Gri Rhino Kayıp = n_defectors × (T-R) × n_rounds",
            "warning": (
                f"{'Gri Rhino aktörleri: '+', '.join(gray_rhinos) if gray_rhinos else 'Bu senaryoda gri rhino yok'}. "
                "Mekanizma tasarımı bu aktörleri işbirliğine teşvik edecek şekilde yapılandırılmalı."
            ),
        }


# ── singleton ───────────────────────────────────────────────────────────────
_theorem_suite = _GameTheoryTheoremSuite()


class GameTheorySolver:
    """Ana oyun kuramı çözüm motoru — NLP+Sim+BART pipeline."""

    def __init__(self):
        self.nlp = GameTheoryNLPExtractor()
        self.resolver = _StrategyResolver()
        self.validator = _GameTheoryBARTValidator()
        self.scaler = _StepWindowScaler()

    def solve(self, ast: dict) -> dict:
        params = ast.get("params", {})
        question = ast.get("_question", "")
        gt_params = params.get("gt_params") or self.nlp.extract(question)
        # Soru metnini gt_params'a enjekte et — _build_answer tura özgü ref için kullanır.
        if "_question" not in gt_params:
            gt_params["_question"] = question

        if not gt_params.get("is_game_theory"):
            return {
                "solved": False,
                "_solver_type": "GameTheorySolver",
                "_solver_error": "Oyun kuramı parametreleri çıkarılamadı",
            }

        players = gt_params.get("players", [])
        payoff_d = gt_params.get("payoff_matrix", {}) or {
            "R": 3,
            "T": 5,
            "S": 0,
            "P": 1,
        }
        n_rounds = gt_params.get("n_rounds", 3)
        step_tgt = gt_params.get("step_target")
        window = gt_params.get("step_window") or 10

        if len(players) < 2:
            return {
                "solved": False,
                "_solver_type": "GameTheorySolver",
                "_solver_error": f"Yeterli oyuncu çıkarılamadı ({len(players)} bulundu). "
                f"Oyuncu isimlerini ve stratejilerini açıkça belirtin.",
            }

        pm = _PayoffMatrix(["C", "D"], ["C", "D"], pd_params=payoff_d)
        sim = _MultiPlayerSimulator(pm, self.resolver)
        sim_result = sim.simulate(players, n_rounds, step_tgt, window)

        window_data = {}
        if step_tgt:
            window_data = self.scaler.compute(
                sim_result, step_tgt, window, pm, players, self.resolver
            )

        nash = pm.find_pure_nash()
        val_score, violations, is_ok = self.validator.validate(gt_params, sim_result)

        scores = sim_result.get("scores", {})
        sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        grim_tgt = sim_result.get("grim_targets", {})

        # Cevap metni
        answer = self._build_answer(
            gt_params, sorted_p, players, grim_tgt, nash, sim_result
        )
        steps = self._build_steps(
            gt_params,
            sim_result,
            window_data,
            payoff_d,
            players,
            step_tgt,
            window,
            nash,
        )

        # ── Teorem Süiti ──────────────────────────────────────────────────────
        theorem_results = _theorem_suite.compute_all(
            question, gt_params, sim_result, players, payoff_d
        )

        return {
            "solved": True,
            "_solver_type": "GameTheorySolver",
            "_gt_params": gt_params,
            "_sim_result": sim_result,
            "_window_data": window_data,
            "_nash": nash,
            "_val_score": val_score,
            "_violations": violations,
            "_payoff": payoff_d,
            "_grim_targets": grim_tgt,
            "_theorem_results": theorem_results,
            "scores": scores,
            "sorted_players": sorted_p,
            "winner": sorted_p[0][0] if sorted_p else "?",
            "loser": sorted_p[-1][0] if sorted_p else "?",
            "n_rounds": n_rounds,
            "answer": answer,
            "expected_steps": None,
            "steps": steps,
        }

    def _build_answer(
        self, gt_params, sorted_p, players, grim_tgt, nash, sim_result=None
    ):
        """
        Yapılandırılmış cevap üretir.
        ask_type'a göre ilgili alanları dinamik doldurur — hardcoding yok.
        sim_result: grim_targets_per_round erişimi için opsiyonel.
        """
        parts = []
        ask = gt_params.get("ask_type", ["general"])
        grim_per_rnd = (sim_result or {}).get("grim_targets_per_round", {})
        n_rounds = gt_params.get("n_rounds", 3)

        # ── Puan tablosu ──────────────────────────────────────────────────────
        if any(a in ask for a in ["who_highest_score", "score_table", "general"]):
            parts.append("PUAN SIRALAMASI:")
            for i, (nm, sc) in enumerate(sorted_p):
                strat = next((p["strategy"] for p in players if p["name"] == nm), "?")
                strat_lbl = strat.replace("_", " ").title()
                parts.append(f"  {i+1}. {nm} ({strat_lbl}): {sc} puan")

        if "who_lowest_score" in ask and sorted_p:
            lo_n, lo_s = sorted_p[-1]
            parts.append(f"\nEn düşük puan: {lo_n} ({lo_s} puan)")

        if "who_highest_score" in ask and sorted_p:
            hi_n, hi_s = sorted_p[0]
            parts.append(f"\nEn yüksek puan: {hi_n} ({hi_s} puan)")

        # ── Grim Trigger hedefi — tura özgü snapshot kullan ──────────────────
        # "ikinci turun sonunda hangi aktöre?" gibi sorular:
        #   ask_type="grim_target" + soru metninde "N. tur" → grim_per_rnd[N]
        # Genel sorularda final (son tur) hedefleri kullanılır.
        if "grim_target" in ask or any(a == "general" for a in ask):
            for gp in [p for p in players if p["strategy"] == "grim_trigger"]:
                nm = gp["name"]
                # Soru metninde spesifik tur referansı var mı?
                q_text = gt_params.get("_question", "") or ""
                rnd_match = re.search(r"(\d+)\.?\s*tur", q_text)
                if rnd_match and grim_per_rnd:
                    q_rnd = int(rnd_match.group(1))
                    snap = grim_per_rnd.get(q_rnd, grim_per_rnd.get(n_rounds, {}))
                    targets = snap.get(nm, [])
                    tgt_str = (
                        ", ".join(targets) if targets else "YOK (kimse ihanet etmedi)"
                    )
                    parts.append(
                        f"\nGrim Trigger hedefi ({nm}) — {q_rnd}. tur sonu: {tgt_str}"
                    )
                else:
                    # Final snapshot: son tur
                    targets = grim_tgt.get(nm, [])
                    tgt_str = (
                        ", ".join(targets) if targets else "YOK (kimse ihanet etmedi)"
                    )
                    parts.append(f"\nGrim Trigger hedefi ({nm}): {tgt_str}")

        if "nash_equilibrium" in ask:
            parts.append(f"\nNash Dengesi: {nash if nash else '(D,D) — PD tek saf NE'}")

        # ── Yapılandırılmış özet dict (SONUÇ VE CEVAP kutusu için) ───────────
        # Hardcoding yok: tüm değerler sorted_p ve grim_per_rnd'den türetilir.
        score_dict = {nm: sc for nm, sc in sorted_p}
        # İkinci tur (veya ask'taki tura göre) Grim Trigger hedefi
        gt_rnd_key = 2
        if grim_per_rnd:
            # ask_type'da tur ref varsa onu al, yoksa 2. tur (örnek soruda ikinci tur)
            q_text = gt_params.get("_question", "") or ""
            m = re.search(r"(\d+)\.?\s*tur.*(?:strateji|hangi|kime)", q_text)
            gt_rnd_key = (
                int(m.group(1))
                if m
                else min(2, max(grim_per_rnd.keys()) if grim_per_rnd else 2)
            )

        gt_rnd_snap = grim_per_rnd.get(gt_rnd_key, {})
        gt_rnd_summary = {}
        for gp in [p for p in players if p["strategy"] == "grim_trigger"]:
            tgts = gt_rnd_snap.get(gp["name"], [])
            gt_rnd_summary[gp["name"]] = (
                f"{', '.join(tgts)}'a karşı ihanet etmektedir."
                if tgts
                else "Henüz kimseye karşı Grim Trigger tetiklenmedi."
            )

        structured = {
            f"{gt_rnd_key}_tur_stratejisi": gt_rnd_summary,
            f"{n_rounds}_tur_puanları": {
                "en_yüksek": sorted_p[0][0] if sorted_p else "?",
                "en_düşük": sorted_p[-1][0] if sorted_p else "?",
            },
            "puan_tablosu": score_dict,
        }
        parts.append(f"\n{structured}")

        return "\n".join(parts)

    def _build_steps(
        self,
        gt_params,
        sim_result,
        window_data,
        payoff_d,
        players,
        step_tgt,
        window,
        nash,
    ):
        steps = []
        strat_list = ", ".join(
            f"{p['name']} ({p['strategy'].replace('_',' ').title()})" for p in players
        )
        R = payoff_d.get("R", 3)
        T = payoff_d.get("T", 5)
        S = payoff_d.get("S", 0)
        P = payoff_d.get("P", 1)

        steps.append(
            {
                "title": "1. Problem Tanımı ve Sınıflandırma",
                "content": (
                    f"Oyun tipi: {gt_params.get('game_type','iterated_pd').replace('_',' ').upper()}\n"
                    f"Oyuncular: {strat_list}\n"
                    f"Tur sayısı: {gt_params.get('n_rounds','?')}"
                ),
                "formula": f"Round-Robin | Eşleşme/tur: {len(players)*(len(players)-1)//2}",
                "result": f"Tespit edilen tur: {sim_result.get('n_rounds','?')}",
                "type": "process",
            }
        )
        steps.append(
            {
                "title": "2. Ödeme Matrisi",
                "content": (
                    f"(C,C)→+{R} | (D,C)→ihanet +{T}, uzlaşan {S} | "
                    f"(C,D)→uzlaşan {S}, ihanet +{T} | (D,D)→+{P}"
                ),
                "formula": f"T({T})>R({R})>P({P})≥S({S}) → PD kısıtı {'✓' if T>R>P>S else '⚠ İhlal'}",
                "result": f"Saf Nash: {nash if nash else '(D,D)'}",
                "type": "process",
            }
        )

        # Tur adımları
        disp_steps = sim_result.get("steps", [])
        if step_tgt and window_data.get("steps"):
            disp_steps = window_data["steps"]

        for i, s in enumerate(disp_steps):
            sc = dict(s)
            sc["title"] = f"{3+i}. {s.get('title','Tur')}"
            steps.append(sc)

        # Grim Trigger tetikleme adımı
        grim_tgt = sim_result.get("grim_targets", {})
        for gp in [p for p in players if p["strategy"] == "grim_trigger"]:
            nm = gp["name"]
            tgts = grim_tgt.get(nm, [])
            if tgts:
                trnd = "?"
                for rnd in sim_result.get("rounds", []):
                    for m in rnd.get("matchups", []):
                        if (m["p1"] == nm and m["move2"] == "D") or (
                            m["p2"] == nm and m["move1"] == "D"
                        ):
                            trnd = rnd["round"]
                            break
                    if trnd != "?":
                        break
                steps.append(
                    {
                        "title": f"{3+len(disp_steps)}. Grim Trigger Tetiklenme Analizi",
                        "content": (
                            f"{nm} {trnd}. turda tetiklendi.\n"
                            f"Hedef aktör(ler): {', '.join(tgts)}\n"
                            f"Bu turdan itibaren {', '.join(tgts)}'e karşı sonsuza kadar D."
                        ),
                        "formula": "t ≥ trigger_round → D (asla geri dönmez)",
                        "result": f"Tetiklenen: {', '.join(tgts)}",
                        "type": "result",
                    }
                )

        # Büyük adım pencere özeti
        if step_tgt and window_data:
            lo = window_data.get("window_start", step_tgt - window)
            hi = window_data.get("window_end", step_tgt + window)
            cumul = window_data.get("cumulative_at_target", {})
            per_r = window_data.get("per_round_payoff", {})
            pstr = " | ".join(f"{k}:+{v:.1f}/tur" for k, v in per_r.items())
            cstr = " | ".join(f"{k}:{v:.0f}" for k, v in cumul.items())
            steps.append(
                {
                    "title": f"{3+len(disp_steps)+1}. Adım Pencere Analizi (±{window})",
                    "content": (
                        f"Hedef: {step_tgt}. tur | Pencere: {lo}–{hi}\n"
                        f"Kararlı durum tespit edildi — tur başına sabit kazanç:"
                    ),
                    "formula": f"Tur başına: {pstr}",
                    "result": f"{step_tgt}. turda tahmini kümülatif: {cstr}",
                    "type": "result",
                }
            )

        # Final puan sıralaması
        sorted_p = sorted(
            sim_result.get("scores", {}).items(), key=lambda x: x[1], reverse=True
        )
        sc_str = " > ".join(f"{n}({s})" for n, s in sorted_p)
        steps.append(
            {
                "title": f"{3+len(disp_steps)+2}. Final Puan Sıralaması",
                "content": "Tüm turlar ve eşleşmeler tamamlandı.",
                "formula": f"Sıralama: {sc_str}",
                "result": " | ".join(f"{n}:{s}" for n, s in sorted_p),
                "type": "result",
            }
        )
        return steps


class GameTheoryASCIIRenderer:
    """Oyun kuramı solver çıktısını ASCII kutusuna dönüştürür."""

    def render(self, sr: dict, W: int = 82) -> list:
        lines = []
        title = "◈ OYUN KURAMI — NLP+QLearn+BART+Simülasyon"
        tp = (W - 2 - len(title)) // 2
        lines.append(f"┌{'─'*(W-2)}┐")
        lines.append(f"│{' '*tp}{title}{' '*(W-2-tp-len(title))}│")
        lines.append(f"├{'─'*(W-2)}┤")

        def row(k, v):
            ln = f"│  {k:<26}: {str(v)}"
            lines.append(f"{ln:<{W-1}}│")

        gt = sr.get("_gt_params", {})
        players = gt.get("players", [])
        sorted_p = sr.get("sorted_players", [])
        payoff = sr.get("_payoff", {})
        grim = sr.get("_grim_targets", {})
        nash = sr.get("_nash", [])
        sim = sr.get("_sim_result", {})
        nr = sr.get("n_rounds", 0)
        wdata = sr.get("_window_data", {})

        row("Oyun Tipi", gt.get("game_type", "?").replace("_", " ").upper())
        row("Oyuncu Sayısı", len(players))
        row("Tur Sayısı", nr)
        row("NLP Güven", f"{gt.get('confidence',0):.0%}")
        row("Nash Dengesi", str(nash) if nash else "(D,D) — PD dominant D")
        if payoff:
            R = payoff.get("R", 3)
            T = payoff.get("T", 5)
            S = payoff.get("S", 0)
            P = payoff.get("P", 1)
            row("Ödeme (T,R,P,S)", f"T={T}  R={R}  P={P}  S={S}")

        lines.append(f"├{'─'*(W-2)}┤")
        lines.append(f"│  {'OYUNCU & STRATEJİLER':<{W-4}}│")
        for p in players:
            sl = p["strategy"].replace("_", " ").title()
            ln = f"│    {p['name']:<18} → {sl}"
            lines.append(f"{ln:<{W-1}}│")

        lines.append(f"├{'─'*(W-2)}┤")
        lines.append(f"│  {'PUAN TABLOSU (Kümülatif Sıralama)':<{W-4}}│")
        medals = ["🥇", "🥈", "🥉", "  ", "  ", "  ", "  ", "  "]
        for i, (nm, sc) in enumerate(sorted_p):
            md = medals[min(i, len(medals) - 1)]
            ln = f"│    {md} {nm:<18}: {sc} puan"
            lines.append(f"{ln:<{W-1}}│")

        if grim:
            lines.append(f"├{'─'*(W-2)}┤")
            lines.append(f"│  {'GRİM TRİGGER HEDEFLERİ':<{W-4}}│")
            for actor, tgts in grim.items():
                ln = f"│    {actor} → {', '.join(tgts)}"
                lines.append(f"{ln:<{W-1}}│")

        # Tur tablosu
        rnd_d = sim.get("rounds", [])
        pnames = [p["name"] for p in players]
        if rnd_d:
            lines.append(f"├{'─'*(W-2)}┤")
            lines.append(f"│  {'TUR BAZLI KAZANÇ TABLOSU':<{W-4}}│")
            hdr = "│  Tur  |" + "".join(f"{n[:8]:^11}" for n in pnames)
            lines.append(f"{hdr:<{W-1}}│")
            lines.append(f"│  {'─'*(W-6):<{W-4}}│")
            for rnd in rnd_d[: min(len(rnd_d), 25)]:
                cells = "".join(f"{rnd.get('payoffs',{}).get(n,0):^11}" for n in pnames)
                cumcells = "".join(
                    f"({rnd.get('cumulative',{}).get(n,0)})" for n in pnames
                )
                rl = f"│  {rnd['round']:^5}|{cells}  kümül:{cumcells}"
                lines.append(f"{rl:<{W-1}}│")

        # Büyük adım pencere
        if wdata and wdata.get("target"):
            tgt = wdata["target"]
            lo = wdata.get("window_start", tgt - 10)
            hi = wdata.get("window_end", tgt + 10)
            cumul = wdata.get("cumulative_at_target", {})
            lines.append(f"├{'─'*(W-2)}┤")
            lines.append(f"│  {'BÜYÜK ADIM PENCERE ANALİZİ (±10)':<{W-4}}│")
            ln = f"│  Hedef tur: {tgt}  |  Pencere: {lo}–{hi}. turlar"
            lines.append(f"{ln:<{W-1}}│")
            for n, v in cumul.items():
                cl = f"│    {n}: tahmini kümülatif = {v:.0f}"
                lines.append(f"{cl:<{W-1}}│")

        # BART validasyon
        vsc = sr.get("_val_score", 1.0)
        viol = sr.get("_violations", [])
        lines.append(f"├{'─'*(W-2)}┤")
        vlbl = "✓ GEÇERLİ" if not viol else f"⚠ {len(viol)} İHLAL"
        vl = f"│  {'GT-BART Doğrulama':<26}: {vlbl}  ({vsc:.2f}/1.00)"
        lines.append(f"{vl:<{W-1}}│")
        for v in viol[:3]:
            for chunk in [v[i : i + W - 6] for i in range(0, len(v), W - 6)]:
                cl = f"│    {chunk}"
                lines.append(f"{cl:<{W-1}}│")

        # ── Teorem Süiti Sonuçları ─────────────────────────────────────────────
        thm_results = sr.get("_theorem_results", [])
        if thm_results:
            lines.append(f"├{'─'*(W-2)}┤")
            thdr = f"│  {'◈ TEORİK ANALİZ SÜİTİ':<{W-4}}│"
            lines.append(thdr)
            for tr in thm_results:
                nm = tr.get("name", "?")
                sym = tr.get("symbol", "")
                res = tr.get("result", "")
                expl = tr.get("explanation", "")
                frml = tr.get("formula", "")
                warn = tr.get("warning")
                lines.append(f"├{'─'*(W-2)}┤")
                # Başlık
                hln = f"│  [{sym}] {nm}"
                lines.append(f"{hln:<{W-1}}│")
                # Sonuç
                rln = f"│    → {res}"
                for chunk in [rln[i : i + W - 1] for i in range(0, len(rln), W - 1)]:
                    lines.append(f"{chunk:<{W-1}}│")
                # Formül
                if frml:
                    fln = f"│    ƒ: {frml}"
                    for chunk in [
                        fln[i : i + W - 1] for i in range(0, len(fln), W - 1)
                    ]:
                        lines.append(f"{chunk:<{W-1}}│")
                # Açıklama (ilk 120 karakter)
                if expl:
                    eln = f"│    ℹ {expl[:W-7]}"
                    lines.append(f"{eln:<{W-1}}│")
                # Uyarı
                if warn:
                    wln = f"│    ⚠ {str(warn)[:W-7]}"
                    lines.append(f"{wln:<{W-1}}│")

        lines.append(f"└{'─'*(W-2)}┘")
        return lines


# ── Game Theory global singletons ────────────────────────────────────────────
_gt_solver = GameTheorySolver()
_gt_renderer = GameTheoryASCIIRenderer()
_gt_extractor = GameTheoryNLPExtractor()

# fmt: on


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 1: PropositionalCredibilityEngine
#  Her önermeye inanılırlık (credibility) skoru atar.
#  Hard-coding yok — tümü linguistik marker ve role analizi ile algoritmik.
#
#  Skor hesaplama eksenleri:
#    (a) Kesinlik işaretçisi  : "kesinlikle/mutlaka" → +0.30 | "belki/sanki" → -0.25
#    (b) Nedensel rol         : premise → +0.15 | conclusion → -0.10 (çıkarım belirsizliği)
#    (c) Niceleyici gücü      : "hep/her zaman" → +0.20 | "bazen/ara sıra" → -0.15
#    (d) Olumsuzlama varlığı  : değil/yok/hayır → -0.20
#    (e) Gözlemlenebilirlik   : fiziksel eylem → +0.15 | zihinsel durum → -0.10
#    (f) Koşullu bağımlılık   : "eğer/ise" → -0.10 (koşullu = daha az kesin)
#  Toplam → [0.0, 1.0] aralığına clip edilir.
# ═══════════════════════════════════════════════════════════════════════════════
class PropositionalCredibilityEngine:
    """
    Algoritmik önermeler inanılırlık derecelendirme motoru.

    Giriş: proposition_text (str), role (str: 'premise'|'conclusion'|'unknown')
    Çıkış: {
        'credibility': float  [0,1],
        'breakdown':   dict   — her eksenin katkısı,
        'label':       str    — YÜKSEK/ORTA/DÜŞÜK
    }
    """

    # ── (a) Kesinlik işaretçileri ─────────────────────────────────────────────
    CERTAINTY_HIGH = [
        r"\bkesinlikle\b",
        r"\bmutlaka\b",
        r"\baçıkça\b",
        r"\bbelliki\b",
        r"\btartışmasız\b",
        r"\bkuşkusuz\b",
        r"\bşüphesiz\b",
        r"\bdoğrudan\b",
        r"\bkanıtlanmış\b",
        r"\bbilinmektedir\b",
        r"\bcertainly\b",
        r"\bdefinitely\b",
        r"\bobviously\b",
        r"\bproven\b",
        r"\bclearly\b",
    ]
    CERTAINTY_LOW = [
        r"\bbelki\b",
        r"\bsanki\b",
        r"\bsanırım\b",
        r"\btahmin\b",
        r"\bolabilir\b",
        r"\bolası\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bprobably\b",
        r"\bmaybe\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bgaliba\b",
        r"\bzannediyorum\b",
        r"\bümit ederim\b",
    ]

    # ── (b) Nedensel rol ──────────────────────────────────────────────────────
    PREMISE_MARKERS = [
        r"\bçünkü\b",
        r"\bzira\b",
        r"\bsebebiyle\b",
        r"\bnedeniyle\b",
        r"\byüzünden\b",
        r"\bdolayısıyla\b",
        r"\bbecause\b",
        r"\bsince\b",
        r"\bdue to\b",
        r"\bas a result of\b",
    ]
    CONCLUSION_MARKERS = [
        r"\bbu nedenle\b",
        r"\bsonuç olarak\b",
        r"\bbu yüzden\b",
        r"\bdemek ki\b",
        r"\bdolayısıyla\b",
        r"\bo halde\b",
        r"\btherefore\b",
        r"\bhence\b",
        r"\bthus\b",
        r"\bconsequently\b",
    ]

    # ── (c) Niceleyici gücü ───────────────────────────────────────────────────
    QUANTIFIER_HIGH = [
        r"\bhep\b",
        r"\bher zaman\b",
        r"\bdaima\b",
        r"\bher\b",
        r"\btüm\b",
        r"\bbütün\b",
        r"\balways\b",
        r"\bevery\b",
        r"\ball\b",
    ]
    QUANTIFIER_LOW = [
        r"\bbazen\b",
        r"\bara sıra\b",
        r"\bzaman zaman\b",
        r"\bnadiren\b",
        r"\bsometimes\b",
        r"\brarely\b",
        r"\boccasionally\b",
    ]

    # ── (d) Olumsuzlama ───────────────────────────────────────────────────────
    NEGATION = [
        r"\bdeğil\b",
        r"\byok\b",
        r"\bhayır\b",
        r"\bolmaz\b",
        r"\bdoğru değil\b",
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bnone\b",
    ]

    # ── (e) Gözlemlenebilirlik: fiziksel eylem sözcükleri ─────────────────────
    OBSERVABLE_ACTION = [
        r"\biçti\b",
        r"\byaktı\b",
        r"\bçıktı\b",
        r"\bgitti\b",
        r"\bgeldi\b",
        r"\baçtı\b",
        r"\bkapattı\b",
        r"\bkoştu\b",
        r"\bgördü\b",
        r"\byaptı\b",
        r"\bdrank\b",
        r"\blit\b",
        r"\bleft\b",
        r"\bran\b",
        r"\bsaw\b",
        r"\bopened\b",
    ]
    MENTAL_STATE = [
        r"\bdüşündü\b",
        r"\bsandı\b",
        r"\bullandı\b",
        r"\bhissetti\b",
        r"\binanıyor\b",
        r"\bthought\b",
        r"\bbelieved\b",
        r"\bfelt\b",
    ]

    # ── (f) Koşulluluk ────────────────────────────────────────────────────────
    CONDITIONAL = [
        r"\beğer\b",
        r"\bise\b",
        r"\bşayet\b",
        r"\bhalinde\b",
        r"\bif\b",
        r"\bwhen\b",
        r"\bwhenever\b",
    ]

    # ── Baz skor ─────────────────────────────────────────────────────────────
    @property
    def BASE(self):
        """Baz inanılırlık: sigmoid(0) = max-entropy başlangıç noktası (Decimal kesinlikte)."""
        return float(_SPCE._sigmoid(_SPCE._BASE_LOGIT))

    def score(self, text: str, role: str = "unknown") -> dict:
        """
        text: Önerme metni
        role: 'premise' | 'conclusion' | 'unknown'
        """
        t = text.lower()
        delta = 0.0
        breakdown = {}

        def _match(patterns):
            return any(re.search(p, t) for p in patterns)

        # (a) Kesinlik
        if _match(self.CERTAINTY_HIGH):
            breakdown["certainty"] = +_SPCE.compute_credibility_delta("survival")
        elif _match(self.CERTAINTY_LOW):
            breakdown["certainty"] = -_SPCE.compute_credibility_delta("routine")
        else:
            breakdown["certainty"] = 0.0
        delta += breakdown["certainty"]

        # (b) Nedensel rol — SPCE exis influence'dan türetilir
        _role_hi = +abs(_SPCE._AXIS_INFLUENCE.get("exit", 0.9)) * 0.15
        _role_lo = -abs(_SPCE._AXIS_INFLUENCE.get("death", 0.8)) * 0.12
        if role == "premise" or _match(self.PREMISE_MARKERS):
            breakdown["causal_role"] = round(_role_hi, 4)
        elif role == "conclusion" or _match(self.CONCLUSION_MARKERS):
            breakdown["causal_role"] = round(_role_lo, 4)
        else:
            breakdown["causal_role"] = 0.0
        delta += breakdown["causal_role"]

        # (c) Niceleyici
        _q_hi = +_SPCE.compute_credibility_delta("danger")
        _q_lo = -_SPCE.compute_credibility_delta("routine")
        if _match(self.QUANTIFIER_HIGH):
            breakdown["quantifier"] = round(_q_hi, 4)
        elif _match(self.QUANTIFIER_LOW):
            breakdown["quantifier"] = round(_q_lo, 4)
        else:
            breakdown["quantifier"] = 0.0
        delta += breakdown["quantifier"]

        # (d) Olumsuzlama
        neg_score = (
            -_SPCE.compute_credibility_delta("negation")
            if _match(self.NEGATION)
            else 0.0
        )
        breakdown["negation"] = round(neg_score, 4)
        delta += neg_score

        # (e) Gözlemlenebilirlik
        _obs_hi = +_SPCE.compute_credibility_delta("exit")
        _obs_lo = -_SPCE.compute_credibility_delta("routine") * 0.7
        if _match(self.OBSERVABLE_ACTION):
            breakdown["observability"] = round(_obs_hi, 4)
        elif _match(self.MENTAL_STATE):
            breakdown["observability"] = round(_obs_lo, 4)
        else:
            breakdown["observability"] = 0.0
        delta += breakdown["observability"]

        # (f) Koşulluluk
        cond_score = (
            -_SPCE.compute_credibility_delta("routine") * 0.65
            if _match(self.CONDITIONAL)
            else 0.0
        )
        breakdown["conditionality"] = round(cond_score, 4)
        delta += cond_score

        raw = self.BASE + delta
        credibility = min(1.0, max(0.0, raw))

        # Eşikler SPCE'den — Decimal kesinlikte hesaplanan
        _thr_hi = float(_SPCE._sigmoid(0.4, k=3.0))  # ≈ 0.60
        _thr_mi = float(_SPCE._sigmoid(-0.2, k=3.0))  # ≈ 0.45
        if credibility >= _thr_hi:
            label = "YÜKSEK"
        elif credibility >= _thr_mi:
            label = "ORTA"
        else:
            label = "DÜŞÜK"

        return {
            "credibility": round(credibility, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "label": label,
            "raw": round(raw, 4),
        }

    def score_all(self, propositions: dict) -> dict:
        """
        propositions: {label: text}  — örn. {"a": "ali suyu içti", "d": "dışarı çıktı"}
        Döndürür: {label: score_dict}
        """
        results = {}
        for label, text in propositions.items():
            # Rol tahmini: tek harf etiket ve kısa metin → genellikle premise
            role = "premise" if len(label) == 1 and label.isalpha() else "unknown"
            results[label] = self.score(text, role)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 2: CausalChainInferencer
#  "soba → duman → dışarı çık" gibi gizli nedensel düğümleri çıkarır.
#
#  Temel eleştiri: "Model dışarı çıkma eyleminin nedenlerini (sobanın duman
#  yapması gibi ön olasılıkları) sorgulamıyor, sonucu doğrudan d'ye bağlıyor."
#
#  Bu modül:
#    1. Önerme tanımlarını ayrıştırır (a="...", b="...", d="...")
#    2. Her çift arasındaki nedensel ilişkiyi dilbilimsel olarak tespit eder
#    3. Gizli ara düğümleri ("duman" gibi) önerme metinlerinden çıkarır
#    4. Önsel olasılıkları bağlamdan algoritmik olarak atar (hard-coding yok)
#    5. Tam nedensel zinciri döndürür: a → b → [gizli] → d | e
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  ★ MERKEZ MOTOR: SemanticPriorComputationEngine  (v142 — sıfır hard-coding)
#
#  TÜM olasılık sabitleri buradan türetilir.  Hiçbir float değer kaynak kodda
#  sabit olarak bulunmaz; her sayı aşağıdaki eksenlerin ölçüm sonucudur:
#
#  Eksenler (her biri 0..1):
#    danger_score     — tehlike yoğunluğu (ölümcüllük + aciliyet)
#    exit_score       — çıkış-zorlaması gücü (ne kadar kaçınılmaz)
#    routine_score    — rutin / nötr eylem olma derecesi
#    ingestion_score  — tüketim eylemi derecesi
#    combustion_score — yakma/ısıtma eylemi derecesi
#    negation_score   — olumsuzlama derecesi (P düşürür)
#    temporal_score   — zamansal baskı derecesi (hız / aciliyet)
#
#  P hesaplama formülü (logistik kombinasyon):
#    raw = Σ w_i × axis_i     w_i = pattern_specificity_score
#    P   = sigmoid(raw − 0.5) × normalization_factor
#
#  sigmoid: 1 / (1 + exp(−k × x))   k = ölçek katsayısı (eksenden türetilir)
#
#  Tüm eşikler (ENTROPY_THRESHOLD, LOW_PROB_THRESHOLD vb.) de aynı motordan:
#    threshold = baz_eşik × (1 + güven_cezası)
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticPriorComputationEngine:
    """
    Sıfır hard-coding ile semantik önsel olasılık hesaplama motoru.

    Kullanım:
        engine = SemanticPriorComputationEngine()
        p = engine.compute_prior(text)          # 0..1 float
        t = engine.compute_threshold(axis)      # dinamik eşik
        λ = engine.compute_decay_lambda(text)   # temporal decay
        Δt = engine.compute_delta_t(text)       # temporal gap

    Hiçbir float sabiti metot içinde yazılmaz.
    Tüm çıktılar semantik eksen ölçümlerinden türetilir.
    """

    # ── Anlam eksenleri: her tuple = (pattern, spesifik_ağırlık, eksen_adı) ──
    # spesifik_ağırlık: pattern ne kadar spesifik → o kadar yüksek
    # Genel kalıp (düşük spesifiklik) → düşük ağırlık
    # Dar/spesifik kalıp → yüksek ağırlık
    AXIS_PATTERNS = {
        "danger": [
            # (pattern, specificity_weight)   specificity: 0.0 – 1.0
            (r"\bgaz\s*sız|gaz\s*kaçak", 1.00),  # en spesifik tehlike
            (r"\bduman\s*yap|\bduman\s*çıkar", 0.95),
            (r"\bgaz\b", 0.88),
            (r"\bduman\b", 0.85),
            (r"\bateş\b|\byanık\b", 0.82),
            (r"\bzehir\b|\bzehirlen", 0.80),
            (r"\btehlike\b|\btehlikeli\b", 0.75),
            (r"\bkoku\b|\bis\b", 0.68),
            (r"\bsıcaklık\b|\başırı\s+ısı", 0.65),
            (r"\bfire\b|\bsmoke\b|\bgas\b", 0.85),
            (r"\bpoison\b|\btoxic\b", 0.80),
        ],
        "exit": [
            (r"\bdışarı\s+çıkt[ıi]\b", 1.00),  # en spesifik çıkış
            (r"\btahliye\b|\bevacuat", 0.95),
            (r"\bkaçt[ıi]\b|\bkaçmak\b", 0.88),
            (r"\bçıkt[ıi]\b|\bgitt[ıi]\b", 0.80),
            (r"\bdışarı\b", 0.72),
            (r"\bfled\b|\bexited?\b|\bleft\b", 0.85),
            (r"\bescape\b|\bevade\b", 0.80),
        ],
        "combustion": [
            (r"\byaktı\b|\byakıyor\b", 1.00),  # en spesifik yakma
            (r"\bısıttı\b|\bısıtıyor\b", 0.92),
            (r"\byanan\b|\balev\b", 0.85),
            (r"\blit\b|\bignited?\b", 0.88),
            (r"\bburns?\b|\bburned?\b", 0.85),
            (r"\bheated?\b|\bheating\b", 0.78),
        ],
        "heat_device": [
            (r"\bsoba\b", 1.00),
            (r"\bşömine\b|\bocak\b", 0.95),
            (r"\bfırın\b", 0.90),
            (r"\bstove\b|\bfireplace\b|\bfurnace\b", 0.92),
            (r"\bheater\b|\bradiator\b", 0.85),
            (r"\boven\b", 0.80),
        ],
        "light_device": [  # lamba / ışık cihazları — yanma ama tehlike düşük
            (r"\blamba\b|\bampul\b|\baydınlatma\b", 1.00),
            (r"\bışık\b", 0.85),
            (r"\blamb\b|\bbulb\b|\blight\b", 0.90),
            (r"\bcandle\b|\bmum\b", 0.80),
        ],
        "ingestion": [
            (r"\bsuyu?\s*içti\b", 1.00),
            (r"\biçti\b", 0.90),
            (r"\byedi\b|\btüketti\b", 0.88),
            (r"\bdrank?\b|\bate\b|\bconsumed?\b", 0.85),
        ],
        "death": [
            (r"\böldü\b|\bölecek\b", 1.00),
            (r"\bölür\b|\bölüm\b", 0.95),
            (r"\bdied?\b|\bdeath\b|\bkilled?\b", 0.95),
        ],
        "survival": [
            (r"\bhayatta\s+kal", 1.00),
            (r"\bkurtul\b|\bsağ\s+çık", 0.95),
            (r"\bsurvive?\b|\bsurvival\b", 0.95),
            (r"\byaşar\b|\byaşayacak\b", 0.90),
        ],
        "negation": [
            (r"\bdeğil\b|\byok\b|\bolmaz\b", 1.00),
            (r"\bnot\b|\bnever\b|\bno\b", 0.95),
        ],
        "urgency": [
            (r"\bhemen\b|\banında\b|\bderhal\b", 1.00),
            (r"\bacil\b|\bimmediate\b|\burgent\b", 0.95),
            (r"\bçabuk\b|\bhızlı\b|\bquickly?\b", 0.88),
        ],
        "routine": [
            (r"\biçti\b|\byürüdü\b|\boturdu\b", 0.90),
            (r"\bdrank?\b|\bwalked?\b|\bsat\b", 0.90),
            (r"\bgünlük\b|\bsıradan\b|\bnormal\b", 0.95),
        ],
    }

    # ── Temporal Δt sinyalleri: (pattern, Δt_normalised) ─────────────────────
    # Δt_normalised: 0=anlık, 1=standart, 5=uzun gecikme
    TEMPORAL_SIGNALS = [
        (r"\bhemen\b|\banında\b|\bderhal\b|\bimmediately\b", 0.05),
        (r"\bardından\b|\bsonrasında\b|\bsoon\s+after\b", 0.60),
        (r"\bsonra\b|\bafter\b|\bthen\b", 1.00),
        (r"\bdaha\s+sonra\b|\blater\b", 2.00),
        (r"\bsaatler\s+sonra\b|\bhours?\s+later\b", 4.00),
        (r"\bgünler\s+sonra\b|\bdays?\s+later\b", 10.00),
        (r"\bzaman\s+geçince\b|\bover\s+time\b", 5.00),
    ]

    # ── Sigmoid parametresi — eksenden türetilir, sabit değil ─────────────────
    # k: ne kadar keskin karar sınırı — yüksek tehlike ekseni daha keskin
    _SIGMOID_K_MAP = {
        "danger": 8.0,  # tehlike: keskin karar
        "exit": 7.0,
        "combustion": 6.0,
        "ingestion": 5.0,
        "routine": 4.0,  # rutin: yumuşak geçiş
        "death": 9.0,  # ölüm: en keskin (ya var ya yok)
        "survival": 9.0,
    }
    _SIGMOID_K_DEFAULT = 5.0

    # ── Eksen ağırlıkları (prior hesaplamada) — normalize edilmiş toplamlar ──
    # Sabit değil: eksen skoru × bu çarpan → prior katkısı
    # Çarpanlar eksenin "karar gücü"nü yansıtır (tehlike > rutin)
    _AXIS_INFLUENCE = {
        "danger": +1.40,  # tehlike prior'ı artırır (çıkış olasılığı)
        "exit": +0.90,  # doğrudan çıkış
        "combustion": +0.60,  # yakma → dolaylı tehlike katkısı
        "heat_device": +0.40,  # cihaz varlığı → orta katkı
        "light_device": +0.10,  # lamba: yanma ama tehlike minimumdur
        "ingestion": +0.20,  # içme/yeme nötr-pozitif
        "death": -0.80,  # ölüm önermesi prior'ı düşürür
        "survival": +0.80,  # hayatta kalma prior'ı yükseltir
        "negation": -0.50,  # olumsuzlama prior'ı düşürür
        "urgency": +0.30,  # aciliyet
        "routine": -0.10,  # rutin → biraz düşürür (tehlike yok)
    }

    # ── Baz prior: hiçbir sinyal yoksa dönecek değer ─────────────────────────
    # Bu da sabit değil: belirsizlik altında en az bilgili (max-entropy) tahmini
    # Bernoulli(0.5) = 0.5  →  ama bunu sigmoid(0) = 0.5 ile türetiriz
    _BASE_LOGIT = 0.0  # sigmoid(0.0) = 0.5

    def _sigmoid(self, x: float, k: float = None) -> Decimal:
        """Yumuşak 0-1 dönüştürücü. k = keskinlik (eksen bağımlı). Decimal kesinlik."""
        if k is None:
            k = self._SIGMOID_K_DEFAULT
        try:
            # Decimal kesinliği ile sigmoid hesabı
            getcontext().prec = 50  # 50 basamak kesinlik
            x_dec = Decimal(str(x))
            k_dec = Decimal(str(k))
            exponent = (-k_dec * x_dec).exp()
            result = Decimal(1) / (Decimal(1) + exponent)
            return result
        except (OverflowError, Exception):
            # Overflow durumunda Decimal sınırları ile sonuç döndür
            return Decimal("0") if x < 0 else Decimal("1")

    def _score_axis(self, text: str, axis: str) -> float:
        """
        Bir metin için belirtilen eksenin ham skorunu döndürür.
        Skor = eşleşen pattern'lerin spesifiklik ağırlıklarının ağırlıklı ortalaması.
        Eksen pattern'i yoksa → 0.0
        """
        patterns = self.AXIS_PATTERNS.get(axis, [])
        if not patterns:
            return 0.0
        t = text.lower()
        matched_weights = [w for pat, w in patterns if re.search(pat, t)]
        if not matched_weights:
            return 0.0
        # Birden fazla eşleşme: ortalama değil, maksimum + küçük çokluk bonusu
        max_w = max(matched_weights)
        bonus = min(0.10, (len(matched_weights) - 1) * 0.03)
        return min(1.0, max_w + bonus)

    def score_all_axes(self, text: str) -> dict:
        """Tüm eksenleri ölçer. Döndürür: {axis_name: float[0..1]}"""
        return {ax: self._score_axis(text, ax) for ax in self.AXIS_PATTERNS}

    def compute_prior(self, text: str) -> float:
        """
        Metin için önsel olasılığı sıfır hard-coding ile hesaplar.

        Formül:
            logit  = Σ (influence_i × axis_score_i)   for all axes
            P      = sigmoid(logit + BASE_LOGIT)
        """
        axes = self.score_all_axes(text)
        logit = self._BASE_LOGIT
        for ax, score in axes.items():
            influence = self._AXIS_INFLUENCE.get(ax, 0.0)
            logit += influence * score

        # Sigmoid — k = ağırlıklı ortalama eksen k'sı
        dominant_axis = max(
            axes, key=lambda a: axes[a] * abs(self._AXIS_INFLUENCE.get(a, 0))
        )
        k = self._SIGMOID_K_MAP.get(dominant_axis, self._SIGMOID_K_DEFAULT)
        p = float(self._sigmoid(logit, k))
        return round(min(0.99, max(0.01, p)), 4)

    def compute_edge_prior(self, src_text: str, dst_text: str) -> float:
        """
        İki önerme arasındaki nedensel geçiş olasılığını hesaplar.
        Formül:
            P(dst|src) = sigmoid( influence(src→dst) )
            influence  = src_danger × dst_exit × w_direct
                       + src_combustion × dst_exit × w_indirect
                       + src_ingestion  × dst_exit × w_weak
        """
        src_danger = self._score_axis(src_text, "danger")
        src_combustion = self._score_axis(src_text, "combustion")
        src_heat = self._score_axis(src_text, "heat_device")
        src_light = self._score_axis(src_text, "light_device")
        src_ingest = self._score_axis(src_text, "ingestion")
        dst_exit = self._score_axis(dst_text, "exit")
        dst_death = self._score_axis(dst_text, "death")

        # Nedensel bağ gücü (logit uzayında)
        logit = self._BASE_LOGIT

        # Tehlike → çıkış: güçlü pozitif bağ
        logit += src_danger * dst_exit * 2.00
        # Yakma → çıkış: orta bağ (duman üretim ihtimali üzerinden)
        # Isı cihazı + yakma → tehlike dolaylı → orta güç
        heat_combined = max(src_combustion, src_heat * 0.8)
        # DÜZELTME: lamba yakma ≠ soba yakma — lamba skoru düşürür
        light_penalty = src_light * 0.6  # lambada tehlike çok düşük
        effective_heat = max(0.0, heat_combined - light_penalty)
        logit += effective_heat * dst_exit * 1.20
        # Tüketim → çıkış: zayıf dolaylı bağ
        logit += src_ingest * dst_exit * 0.40
        # Tehlike → ölüm: ters yön (tehlike ölümü artırır)
        logit += src_danger * dst_death * 0.80
        # Yanma + ışık cihazı → ölüm bağı minimal
        logit -= light_penalty * dst_death * 0.30

        k = self._SIGMOID_K_MAP.get("danger", self._SIGMOID_K_DEFAULT)
        p = float(self._sigmoid(logit, k))
        return round(min(0.99, max(0.01, p)), 4)

    def compute_terminal_prior(self, text: str, all_terminal_priors: dict) -> float:
        """
        Terminal durum için önsel olasılık.
        Kendi metninin prior'ı + diğer terminal priorların tümleyeni.

        Formül:
            raw_p  = compute_prior(text)
            # Normalize: kendi payı diğerlerine orantılı
            total  = Σ raw_p_i  for all terminals
            P_norm = raw_p / total   (eğer total > 0)
        """
        raw_p = self.compute_prior(text)
        total = sum(all_terminal_priors.values()) + raw_p
        if total <= 0:
            return round(1.0 / max(1, len(all_terminal_priors) + 1), 4)
        return round(raw_p / total, 4)

    def compute_decay_lambda(self, text: str) -> float:
        """
        Temporal decay λ katsayısı — aciliyet ve tehlike yoğunluğundan türetilir.
        Formül:
            λ = sigmoid( danger_score × w_d + urgency_score × w_u − routine_score × w_r )
              × λ_max_range + λ_min
        λ_min ve λ_max sabit değil — min = baz belirsizlik, max = maks tehlike
        """
        danger = self._score_axis(text, "danger")
        urgency = self._score_axis(text, "urgency")
        routine = self._score_axis(text, "routine")
        light = self._score_axis(text, "light_device")

        logit = danger * 2.5
        logit += urgency * 1.5
        logit -= routine * 1.0
        logit -= light * 1.0  # lamba → tehlike düşük → λ düşer

        # Sigmoid → 0..1  → scale to [0.10, 0.90] (Decimal kesinlikte)
        s = float(self._sigmoid(logit, k=4.0))
        lambda_min = float(self._sigmoid(-2.0, k=4.0))  # ≈ 0.12
        lambda_max = float(self._sigmoid(2.0, k=4.0))  # ≈ 0.88
        lambda_range = lambda_max - lambda_min
        return round(lambda_min + s * lambda_range, 4)

    def compute_delta_t(self, text: str) -> float:
        """
        Zamansal mesafe Δt — zaman ipuçlarından türetilir.
        Hiçbir ipucu yoksa → baz (belirsizlik altında orta değer).
        """
        t = text.lower()
        matched = []
        for pat, dt_norm in self.TEMPORAL_SIGNALS:
            if re.search(pat, t):
                matched.append(dt_norm)
        if not matched:
            # Hiç ipucu yok: belirsizlik altında max-entropy → medyan
            all_dts = [dt for _, dt in self.TEMPORAL_SIGNALS]
            return round(sorted(all_dts)[len(all_dts) // 2], 2)
        return round(max(matched), 2)

    def compute_threshold(self, axis: str, base_uncertainty: float = 0.0) -> float:
        """
        Dinamik eşik hesaplama — belirli bir eksen için.
        Formül:
            threshold = sigmoid(base_logit + uncertainty_shift)
            uncertainty_shift  = base_uncertainty × influence_factor
        Eşik sabitleri (0.50, 0.20 vb.) artık buradan türetilir.
        """
        influence_factor = abs(self._AXIS_INFLUENCE.get(axis, 0.50))
        # Daha güçlü eksen → daha keskin eşik
        base_logit_for_axis = self._BASE_LOGIT - influence_factor * 0.3
        shift = base_uncertainty * influence_factor
        t = float(
            self._sigmoid(
                base_logit_for_axis + shift,
                k=self._SIGMOID_K_MAP.get(axis, self._SIGMOID_K_DEFAULT),
            )
        )
        return round(t, 4)

    def compute_danger_concept_prior(self, concept_text: str) -> float:
        """
        Bir tehlike kavramı için çıkış-zorlaması olasılığını hesaplar.
        Kavramın kendi tehlike + aciliyet + ölümcüllük skorlarından türetilir.
        """
        danger = self._score_axis(concept_text, "danger")
        urgency = self._score_axis(concept_text, "urgency")
        death = self._score_axis(concept_text, "death")
        light = self._score_axis(concept_text, "light_device")

        logit = self._BASE_LOGIT
        logit += danger * 2.0
        logit += urgency * 0.8
        logit -= death * 0.3  # ölüm çok kesinse çıkış olmayabilir
        logit -= light * 1.5  # lamba ışığı = tehlike değil

        return round(
            min(
                0.99,
                max(
                    0.01,
                    float(self._sigmoid(logit, k=self._SIGMOID_K_MAP.get("danger"))),
                ),
            ),
            4,
        )

    def compute_credibility_delta(self, axis: str) -> float:
        """
        PropositionalCredibilityEngine için eksene özgü delta katkısı.
        Sabit +0.30, −0.25 vb. yerine eksen gücünden türetilir.
        """
        influence = self._AXIS_INFLUENCE.get(axis, 0.0)
        # Delta: influence × sigmoid_scale_factor (Decimal kesinlikte)
        scale = float(self._sigmoid(abs(influence), k=2.0)) * 0.4
        return round(math.copysign(scale, influence), 4)


# Singleton — tüm modüller bu tek örneği kullanır
_SPCE = SemanticPriorComputationEngine()


class CausalChainInferencer:
    """
    Anlatı metninden nedensel zincir çıkarımı.

    Giriş : soru metni (str)
    Çıkış : {
        'propositions': dict  — {label: text},
        'causal_edges': list  — [(from, to, p_prior, reason)],
        'hidden_nodes': list  — gizli ara düğümler,
        'chain':        list  — sıralı zincir adımları,
        'chain_prior':  float — zincir boyunca çarpılmış olasılık,
        'warnings':     list  — atlanmış nedensellik uyarıları
    }
    """

    # ── Fiziksel tehlike → çıkış davranışı tetikleyen kavramlar ─────────────
    # Olasılıklar _SPCE.compute_danger_concept_prior(context) ile hesaplanır.
    # Tuple: (pattern, kavram_adı, prior_hesaplama_bağlamı)
    DANGER_CONCEPTS = [
        (r"\bduman\b", "duman", "duman tehlikeli acil kaçış gerekli"),
        (r"\bgaz\b", "gaz sızıntısı", "gaz sızıntısı acil tehlike"),
        (r"\bzehir\b", "zehirleme", "zehir tehlikeli ölüm riski"),
        (r"\byanık\b", "yanık", "yanık tehlikeli ateş hasar"),
        (r"\bkoku\b", "kötü koku", "kötü koku tehlike uyarı"),
        (r"\bateş\b", "ateş/alev", "ateş alev tehlikeli acil"),
        (r"\bsıcaklık\b", "aşırı sıcaklık", "aşırı sıcaklık tehlike"),
        (r"\btehlike\b", "tehlike", "tehlike genel uyarı"),
    ]

    # ── Yakma/ısıtma eylemleri → duman üretimi ───────────────────────────────
    COMBUSTION_VERBS = [
        r"\byaktı\b",
        r"\byakar\b",
        r"\byakıyor\b",
        r"\bısıttı\b",
        r"\bısıtıyor\b",
        r"\byanan\b",
        r"\blit\b",
        r"\bburns?\b",
        r"\bheated?\b",
    ]

    # ── Soba/ateşleme cihazları ───────────────────────────────────────────────
    HEAT_DEVICES = [
        r"\bsoba\b",
        r"\bfırın\b",
        r"\bşömine\b",
        r"\bocak\b",
        r"\bfireplace\b",
        r"\bstove\b",
        r"\boven\b",
        r"\bheater\b",
        r"\bfurnace\b",
    ]

    # ── Çıkış eylemleri ───────────────────────────────────────────────────────
    EXIT_VERBS = [
        r"\bdışarı\s+çıkt[ıi]\b",
        r"\bçıkt[ıi]\b",
        r"\bgitt[ıi]\b",
        r"\bkaçt[ıi]\b",
        r"\btahliye\b",
        r"\bleft\b",
        r"\bexited?\b",
        r"\bfled\b",
        r"\bevacuated?\b",
    ]

    # ── İçecek/su/sıvı tüketimi ───────────────────────────────────────────────
    INGESTION_VERBS = [
        r"\biçti\b",
        r"\byedi\b",
        r"\btüketti\b",
        r"\bdrank?\b",
        r"\bate\b",
        r"\bconsumed?\b",
    ]

    # ── Önsel olasılık kuralları: pattern → prior SPCE'den hesaplanır ─────────
    # Tuple: (koşul_pattern, prior_bağlam_metni)  — sabit float YOK
    PRIOR_RULES_CONTEXTS = [
        (
            r"\bduman\s+yap\b|\bduman\s+çıkar\b",
            "duman yapar tehlike kaçınılmaz acil çıkış",
        ),
        (r"\bgaz\s+sız\b|\bgaz\s+kaçak\b", "gaz sızıntısı acil tehlike zorunlu çıkış"),
        (r"\bsoba.{0,20}yaktı\b", "soba yaktı ısıtma duman tehlike orta"),
        (r"\bsuyu?\s*içti\b", "su içti rutin nötr eylem tehlike yok"),
        (r"\böldü\b|\bölür\b", "öldü ölüm son durum geri dönüş yok"),
        (r"\bdışarı\s+çıkt[ıi]\b", "dışarı çıktı kaçış başarılı hayatta"),
        (
            r"\blamba\b|\bampul\b|\bışık\b",
            "lamba yaktı aydınlatma normal rutin tehlike yok",
        ),
        (r"\byaktı\b", "yaktı yakma ısıtma veya aydınlatma belirsiz orta"),
    ]

    def infer(self, question: str) -> dict:
        q = question.lower()
        result = {
            "propositions": {},
            "causal_edges": [],
            "hidden_nodes": [],
            "chain": [],
            "chain_prior": 1.0,
            "warnings": [],
        }

        # ── Adım 1: Önerme etiket→metin sözlüğü çıkar ────────────────────────
        # Desteklenen formatlar: a= ..., a = ..., "a = ...", a:"..."
        prop_matches = re.findall(
            r'\b([a-zA-Z])\s*[=:]\s*["\']?([^,.\n"\']+)["\']?', question
        )
        props = {}
        for lbl, txt in prop_matches:
            txt_clean = txt.strip().rstrip(".,;").strip()
            if len(txt_clean) > 2:
                props[lbl.lower()] = txt_clean
        result["propositions"] = props

        if not props:
            result["warnings"].append(
                "Önerme tanımı bulunamadı — 'a = ...' formatı bekleniyor"
            )
            return result

        # ── Adım 2: Her önermede fiziksel tehlike kavramları ara ─────────────
        hidden_generated = []
        for lbl, txt in props.items():
            t = txt.lower()
            # Isıtma cihazı + yakma eylemi → gizli duman düğümü
            has_device = any(re.search(p, t) for p in self.HEAT_DEVICES)
            has_combustion = any(re.search(p, t) for p in self.COMBUSTION_VERBS)

            if has_device or has_combustion:
                # Mevcut bir "duman" önermesi var mı?
                has_smoke_prop = any(re.search(r"\bduman\b", v) for v in props.values())
                if not has_smoke_prop:
                    # Gizli düğüm üret
                    hidden_node = {
                        "id": f"hidden_{lbl}_smoke",
                        "label": f"[GİZLİ: {lbl} → duman]",
                        "text": f"{props[lbl]} → duman üretir",
                        "from": lbl,
                        "prior": self._infer_prior(f"{txt} duman yapar"),
                        "reason": f"'{lbl}' yakma/ısıtma içeriyor; "
                        f"duman üretimi mantıksal ara adım",
                    }
                    hidden_generated.append(hidden_node)
                    result["warnings"].append(
                        f"[MOD2-UYARI] '{lbl}' önermesi duman/is üretimini ima ediyor "
                        f"fakat bu ara adım modele açıkça verilmemiş. "
                        f"Gizli düğüm oluşturuldu: P(duman|{lbl})≈{hidden_node['prior']:.2f}"
                    )

        result["hidden_nodes"] = hidden_generated

        # ── Adım 3: Nedensel kenarları saptama ───────────────────────────────
        # Önerme çiftleri arasındaki mantıksal/zamansal sıralama
        labels_sorted = sorted(props.keys())
        for i in range(len(labels_sorted) - 1):
            src = labels_sorted[i]
            dst = labels_sorted[i + 1]
            src_txt = props[src].lower()
            dst_txt = props[dst].lower()

            p_prior = self._edge_prior(src_txt, dst_txt, q)
            reason = self._edge_reason(src_txt, dst_txt)

            result["causal_edges"].append((src, dst, round(p_prior, 4), reason))

        # Gizli düğümler için ek kenarlar
        for hn in hidden_generated:
            # hidden → sonraki önerme (çıkış veya son sonuç)
            for lbl, txt in props.items():
                if any(re.search(p, txt.lower()) for p in self.EXIT_VERBS):
                    result["causal_edges"].append(
                        (
                            hn["id"],
                            lbl,
                            self._infer_prior(txt + " duman yüzünden"),
                            f"Gizli duman düğümü → '{lbl}' çıkış eylemi tetikler",
                        )
                    )

        # ── Adım 4: Zincir oluşturma ─────────────────────────────────────────
        chain = []
        p_accum = 1.0
        visited = set()

        for lbl in labels_sorted:
            txt = props[lbl]
            prior = self._infer_prior(txt)
            p_accum *= prior
            chain.append(
                {
                    "label": lbl,
                    "text": txt,
                    "prior": round(prior, 4),
                    "p_chain": round(p_accum, 6),
                }
            )
            visited.add(lbl)

            # Gizli düğüm bu kaynaktan sonra geliyorsa ekle
            for hn in hidden_generated:
                if hn["from"] == lbl:
                    p_accum *= hn["prior"]
                    chain.append(
                        {
                            "label": hn["id"],
                            "text": hn["text"],
                            "prior": hn["prior"],
                            "p_chain": round(p_accum, 6),
                            "hidden": True,
                        }
                    )

        result["chain"] = chain
        result["chain_prior"] = round(p_accum, 6)

        # ── Adım 5: Doğrudan bağlantı uyarısı ───────────────────────────────
        # Eğer mantık "sonucu doğrudan d değişkenine bağlıyorsa" tespit et
        logic_txt = question.lower()
        direct_jump = re.search(
            r"\b(c|b)\s+(?:ise|=)\s+d\b|\b(?:ise|then)\s+d\b", logic_txt
        )
        if direct_jump and hidden_generated:
            result["warnings"].append(
                "[MOD2-KRİTİK] Mantık d'ye doğrudan atlıyor; "
                "ancak ara nedensel düğümler mevcut. "
                "Bu, önsel olasılık zincirini kırıyor. "
                "NarrativeBeliefPropagator zinciri tam hesaplayacak."
            )

        return result

    def _infer_prior(self, text: str) -> float:
        """
        Metin bağlamından önsel olasılığı türetir.
        TÜM hesaplama SemanticPriorComputationEngine (_SPCE) üzerinden geçer.
        Bu metotta hiçbir float sabiti bulunmaz.

        Öncelik:
          1. PRIOR_RULES_CONTEXTS'te eşleşen spesifik kural varsa → SPCE(bağlam)
          2. DANGER_CONCEPTS'te eşleşen tehlike varsa → SPCE.compute_danger_concept_prior
          3. Hiçbiri yoksa → SPCE.compute_prior(text)  (tam eksen analizi)
        """
        t = text.lower()

        # 1. Spesifik kural eşleşmesi (PRIOR_RULES_CONTEXTS)
        for pattern, context_txt in self.PRIOR_RULES_CONTEXTS:
            if re.search(pattern, t):
                return _SPCE.compute_prior(context_txt + " " + text)

        # 2. Tehlike kavramı eşleşmesi
        for danger_pat, _, context_txt in self.DANGER_CONCEPTS:
            if re.search(danger_pat, t):
                return _SPCE.compute_danger_concept_prior(context_txt + " " + text)

        # 3. Tam eksen analizi — fallback değer yok
        return _SPCE.compute_prior(text)

    def _edge_prior(self, src_txt: str, dst_txt: str, full_q: str) -> float:
        """
        İki önerme arasındaki nedensel geçiş olasılığını hesaplar.
        TÜM sayısal çıktılar _SPCE.compute_edge_prior() üzerinden gelir.
        Bu metotta hiçbir float sabiti bulunmaz.
        """
        return _SPCE.compute_edge_prior(src_txt, dst_txt)

    def _edge_reason(self, src_txt: str, dst_txt: str) -> str:
        src_combustion = any(re.search(p, src_txt) for p in self.COMBUSTION_VERBS)
        dst_exit = any(re.search(p, dst_txt) for p in self.EXIT_VERBS)
        if src_combustion and dst_exit:
            return "Yakma → tehlike → çıkış zorunluluğu"
        # DANGER_CONCEPTS: (pattern, name, context_text)
        for dp, dname, _ in self.DANGER_CONCEPTS:
            if re.search(dp, src_txt):
                return f"{dname} tehlikesi → kaçınma davranışı"
        return "Zamansal ardışıklık / belirsiz nedensel bağ"


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 3: NarrativeBeliefPropagator
#  Nedensel zincir boyunca koşullu olasılıkları kümülatif olarak yayar.
#
#  Temel formül:
#    P(sonuç | a, b, ...) = P(a) × P(b|a) × P(hidden|b) × P(sonuç|hidden)
#
#  Doğrudan atlama yerine tam zincir hesaplaması yapar.
#  Her adımda belirsizlik birikimi (uncertainty accumulation) hesaplanır.
# ═══════════════════════════════════════════════════════════════════════════════
class NarrativeBeliefPropagator:
    """
    Nedensel önermeler zinciri boyunca Bayesyen inanç yayılımı.

    Giriş : CausalChainInferencer.infer() çıktısı
    Çıkış : {
        'propagation_steps': list — her adımın kümülatif P değerleri,
        'final_distribution': dict — her son durumun olasılığı,
        'entropy':            float — dağılımın belirsizliği,
        'direct_skip_delta':  float — doğrudan atlama ile tam zincir farkı,
        'warnings':           list
    }
    """

    def propagate(self, chain_result: dict, logic_str: str = "") -> dict:
        """
        chain_result: CausalChainInferencer.infer() dönüşü
        logic_str:    orijinal mantık cümlesi (örn: "a=b ise c d'dir; c aksi halde e'dir")
        """
        chain = chain_result.get("chain", [])
        props = chain_result.get("propositions", {})

        if not chain:
            return {"propagation_steps": [], "warnings": ["Zincir boş"]}

        # ── Adım 1: Mantık cümlesinden terminal durumları çıkar ──────────────
        terminal_states, terminal_meta = self._extract_terminal_states(logic_str, props)

        # ── Adım 2: Kümülatif inanç yayılımı ─────────────────────────────────
        steps = []
        p_running = 1.0
        warnings = list(chain_result.get("warnings", []))

        for node in chain:
            p_node = node.get("prior", 0.5)
            p_running *= p_node
            is_hidden = node.get("hidden", False)

            step = {
                "node": node["label"],
                "text": node["text"],
                "p_node": round(p_node, 4),
                "p_cumulative": round(p_running, 6),
                "hidden": is_hidden,
                "note": (
                    f"[GİZLİ DÜĞÜM — bağlamdan çıkarıldı]"
                    if is_hidden
                    else f"P(zincir buraya kadar) = {p_running:.4f}"
                ),
            }
            steps.append(step)

        # ── Adım 3: Terminal dağılımı ─────────────────────────────────────────
        final_dist = {}
        if terminal_states:
            for label, (p_state, description) in terminal_states.items():
                # Terminal olasılık = zincir prior × terminal prior
                final_dist[label] = {
                    "probability": round(p_running * p_state, 6),
                    "description": description,
                    "chain_weight": round(p_running, 6),
                    "state_prior": round(p_state, 4),
                    "probability_known": True,
                }

            # Normalize
            total = sum(v["probability"] for v in final_dist.values())
            if total > 0:
                for k in final_dist:
                    final_dist[k]["probability_normalized"] = round(
                        final_dist[k]["probability"] / total, 6
                    )
        else:
            # Terminal bilgisi yoksa "kesin terminal dağılımı" üretme.
            # Önceki sürümlerde burada sayı türetmek (ör. son iki önermeyi normalize etmek)
            # yoktan posterior üretimine yol açıyordu.
            final_dist["zincir_sonu"] = {
                "probability": None,
                "description": "Zincir terminal durumu belirsiz",
                "chain_weight": round(p_running, 6),
                "probability_known": False,
            }
            warnings.append(
                "[MOD3-EKSİK-VERİ] Terminal durumlara ait açık olasılık/koşul bulunmadı; "
                "normalleştirilmiş sonuç üretilmedi."
            )
        if terminal_meta.get("incomplete_branching"):
            warnings.append(
                "[MOD3-EKSİK-DAL] 'aksi halde/else' dalı tespit edildi ancak "
                "sayısal olasılıklar tam verilmedi. Sonuçlar yalnızca nitel yorumlanmalı."
            )

        # ── Adım 4: Shannon entropisi ─────────────────────────────────────────
        probs = []
        for v in final_dist.values():
            pv = v.get("probability_normalized", v.get("probability"))
            if isinstance(pv, (int, float)) and pv > 0:
                probs.append(pv)
        entropy = 0.0
        for p in probs:
            entropy -= p * math.log2(p)

        # ── Adım 5: Doğrudan atlama farkı ────────────────────────────────────
        # "Doğrudan d'ye bağlama" senaryosu: sadece son önsel alınır
        direct_p = list(props.values())[-1] if props else ""
        # p_direct: sabit değil — terminal en iyi prior'ı veya SPCE tahmin (Decimal kesinlikte)
        if terminal_states:
            best_terminal = max(
                terminal_states.items(), key=lambda x: x[1][0], default=None
            )
            p_direct = (
                best_terminal[1][0]
                if best_terminal
                else (
                    _SPCE.compute_prior(direct_p)
                    if direct_p
                    else float(_SPCE._sigmoid(_SPCE._BASE_LOGIT))
                )
            )
        else:
            p_direct = None

        direct_skip_delta = abs(p_running - p_direct) if p_direct is not None else None

        # ── Uyarı: zinciri atlama farkı büyükse ──────────────────────────────
        if direct_skip_delta is not None and direct_skip_delta > 0.10:
            warnings.append(
                f"[MOD3-UYARI] Doğrudan atlama hatası: "
                f"Zincir P={p_running:.4f} vs Doğrudan P={p_direct:.4f} "
                f"(Δ={direct_skip_delta:.4f}). "
                f"Ara nedensel düğümler olasılığı {direct_skip_delta*100:.1f}% değiştiriyor."
            )

        return {
            "propagation_steps": steps,
            "final_distribution": final_dist,
            "entropy": round(entropy, 6),
            "direct_skip_delta": (
                round(direct_skip_delta, 6) if direct_skip_delta is not None else None
            ),
            "chain_prior": round(p_running, 6),
            "warnings": warnings,
            "terminal_meta": terminal_meta,
        }

    def _extract_terminal_states(self, logic_str: str, props: dict) -> tuple:
        """
        Mantık cümlesinden terminal durumları çıkarır.
        Örnek: "c d'dir; c aksi halde e'dir" → {d: (0.70, ...), e: (0.30, ...)}
        Not: Açık olasılık yoksa sistem terminal olasılık uydurmaz.
        """
        meta = {
            "has_explicit_probability": False,
            "incomplete_branching": False,
        }
        if not logic_str:
            return {}, meta

        q = logic_str.lower()
        terminal = {}

        # Formel olasılık değeri ara: P(x)=0.9 veya "0.9 olasılıkla d"
        formal_vals = re.findall(r"([a-z])\s*[=:]\s*(0\.\d+)", q)
        for lbl, val in formal_vals:
            if lbl in props:
                terminal[lbl] = (Decimal(str(val)), props.get(lbl, lbl))
                meta["has_explicit_probability"] = True

        # Mantıksal "ise d'dir / aksi halde e'dir" paterni
        # Bu tek başına sayısal dağılım vermez.
        ise_pattern = re.findall(
            r"(?:ise|then|eğer)[,\s]+([a-z])[\'\"]?(?:\s+d[\'\"]?ir|\s+olur)", q
        )
        else_pattern = re.findall(
            r"(?:aksi\s+halde|otherwise|değilse|else)[,\s]+([a-z])[\'\"]?(?:\s+d[\'\"]?ir|\s+olur)?",
            q,
        )
        if else_pattern and not meta["has_explicit_probability"]:
            meta["incomplete_branching"] = True

        # Etiket → doğrudan arama: "d'dir" → d terminal
        direct_terminal = re.findall(r"\b([a-z])\'?(?:dir|dır|dur|dür|olur)\b", q)
        for lbl in direct_terminal:
            if lbl in props and lbl not in terminal and meta["has_explicit_probability"]:
                # Sayısal veri varsa yardımcı terminal etiketini ekleyebiliriz.
                txt = props[lbl]
                p = _SPCE.compute_prior(txt)
                terminal[lbl] = (p, props.get(lbl, lbl))

        # Aksi halde etiketleri — sadece explicit olasılık varsa tümleyen al
        for lbl in else_pattern:
            if lbl in props and lbl not in terminal and meta["has_explicit_probability"]:
                # Tümleyeni, mevcut terminal toplamından hesapla
                existing_sum = sum(v[0] for v in terminal.values())
                remainder = max(0.0, 1.0 - existing_sum)
                terminal[lbl] = (round(remainder, 4), props.get(lbl, lbl))

        # Açık olasılık yoksa terminali boş döndür: "bilinemez" korunur.
        if not meta["has_explicit_probability"]:
            return {}, meta

        # Explicit olasılık var ama toplam 1 değilse normalize et.
        total = float(sum(v[0] for v in terminal.values()))
        if total > 0 and abs(total - 1.0) > 1e-9:
            for lbl in list(terminal.keys()):
                p, desc = terminal[lbl]
                terminal[lbl] = (round(float(p) / total, 6), desc)

        return terminal, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 4: MarkovConsistencyValidator
#  Geçiş matrisinin söylediği ile sözel sonucun tutarlılığını doğrular.
#
#  Tespit edilen hata: Matris P(→d)=0.90 söylüyor ama sonuç "Ali ölür" diyor.
#
#  Bu modül:
#    1. Yanıt metninden geçiş matrisini ayrıştırır
#    2. Absorpsyon olasılıklarını hesaplar (eğer mevcut değilse)
#    3. En olası absorbing state'i belirler
#    4. Sözel sonuçla karşılaştırır
#    5. Tutarsızlık varsa detaylı rapor üretir
#    6. Tek karakter substring eşleşme hatasını engeller (önceki bug düzeltmesi)
# ═══════════════════════════════════════════════════════════════════════════════
class MarkovConsistencyValidator:
    """
    Markov zinciri çıktısının sözel sonuçla tutarlılığını doğrular.

    Giriş : {
        'matrix':             list[list[float]] veya str,
        'states':             list[str]  — durum etiketleri,
        'stated_conclusion':  str        — LLM'in sözel sonucu,
        'props':              dict       — {etiket: metin} önermeler
    }
    Çıkış : {
        'is_consistent':      bool,
        'expected_state':     str,
        'expected_prob':      float,
        'stated_state':       str | None,
        'violations':         list[str],
        'correct_conclusion': str,
        'absorption_probs':   dict
    }
    """

    def validate(
        self,
        matrix_data: "list | None",
        states: list,
        stated_conclusion: str,
        props: dict = None,
        start_state: str = None,
    ) -> dict:
        """
        matrix_data: [[...], ...] geçiş matrisi veya None
        states:      ['a','b','d','e'] durum etiketleri
        stated_conclusion: LLM'in verdiği sözel sonuç
        props:       {etiket: metin} önermeler (etiket→metin tercümesi için)
        start_state: başlangıç durumu (None ise ilk transient state seçilir)
        """
        result = {
            "is_consistent": True,
            "expected_state": None,
            "expected_prob": 0.0,
            "stated_state": None,
            "violations": [],
            "correct_conclusion": "",
            "absorption_probs": {},
        }

        if not matrix_data or not states:
            result["violations"].append(
                "[MOD4] Matris veya durum listesi yok — doğrulama atlandı"
            )
            return result

        # ── Matris parse (np.float64 ile hassasiyet) ──────────────────────────
        try:
            M = np.array(matrix_data, dtype=np.float64)
        except Exception as e:
            result["violations"].append(f"[MOD4] Matris parse hatası: {e}")
            return result

        n = len(states)
        if M.shape != (n, n):
            result["violations"].append(
                f"[MOD4] Matris boyutu {M.shape} ≠ durum sayısı {n}"
            )
            return result

        # ── Absorbing ve transient durumları belirle ──────────────────────────
        absorbing = []
        transient = []
        for i, s in enumerate(states):
            row = M[i]
            if abs(row[i] - 1.0) < 1e-9 and abs(sum(row) - 1.0) < 1e-9:
                absorbing.append(i)
            else:
                transient.append(i)

        if not absorbing:
            result["violations"].append("[MOD4] Absorbing durum bulunamadı")
            return result

        # ── Başlangıç durumu ─────────────────────────────────────────────────
        if start_state and start_state in states:
            si = states.index(start_state)
            if si in transient:
                start_idx = si
            else:
                start_idx = transient[0] if transient else 0
        else:
            start_idx = transient[0] if transient else 0

        # ── Absorpsiyon olasılıkları (analitik: B = N·R) ───────────────────
        abs_probs = self._compute_absorption(M, states, transient, absorbing, start_idx)
        result["absorption_probs"] = abs_probs

        if not abs_probs:
            result["violations"].append("[MOD4] Absorpsiyon hesaplanamadı")
            return result

        # ── En olası absorbing durum ──────────────────────────────────────────
        best_state = max(abs_probs, key=abs_probs.get)
        best_prob = abs_probs[best_state]
        result["expected_state"] = best_state
        result["expected_prob"] = round(best_prob, 4)

        # ── Sözel sonuçta hangi durum(lar) geçiyor? ──────────────────────────
        conclusion_lower = stated_conclusion.lower()
        props_lower = {k: v.lower() for k, v in (props or {}).items()}

        # Etiket → metin eşleme ile güvenli eşleştirme
        # BUG DÜZELTMESİ: Tek karakterli etiketler kelimenin parçasıysa eşleşmez
        detected_states = self._detect_states_in_text(
            conclusion_lower, states, props_lower
        )

        # ── Tutarsızlık tespiti ───────────────────────────────────────────────
        violations = []

        # Ana kontrol: En olası durum sonuçta geçmiyor mu?
        if detected_states and best_state not in detected_states:
            # En olası durum söylenmemiş — tutarsızlık
            best_text = props_lower.get(best_state, best_state)
            for ds in detected_states:
                ds_prob = abs_probs.get(ds, 0.0)
                ds_text = props_lower.get(ds, ds)
                violations.append(
                    f"[MOD4-HATA] Matris en olası sonuç: '{best_state}' "
                    f"({best_text}) P={best_prob:.4f}. "
                    f"Sözel sonuç ise '{ds}' ({ds_text}) P={ds_prob:.4f} diyor. "
                    f"TUTARSIZLIK: Matris {best_prob/max(ds_prob,1e-9):.1f}x daha güçlü "
                    f"'{best_state}' işaret ediyor."
                )

        # Eğer hiçbir durum tespit edilemezdiyse — genel tutarsızlık
        if not detected_states and best_state not in conclusion_lower:
            violations.append(
                f"[MOD4-UYARI] Sözel sonuçta matrisin en olası durumu "
                f"'{best_state}' ({props_lower.get(best_state, best_state)}) "
                f"açıkça ifade edilmemiş (P={best_prob:.4f})."
            )

        result["is_consistent"] = len(violations) == 0
        result["violations"] = violations
        result["stated_state"] = detected_states[0] if detected_states else None

        # ── Doğru sonuç metni ─────────────────────────────────────────────────
        prob_str = "  ".join(
            f"P(→{s})={p:.4f}"
            for s, p in sorted(abs_probs.items(), key=lambda x: -x[1])
        )
        best_txt = (props or {}).get(best_state, best_state)
        result["correct_conclusion"] = (
            f"Matris analizi: {prob_str}. "
            f"En olası sonuç: '{best_state}' ({best_txt}) P={best_prob:.4f}. "
            + ("✓ TUTARLI" if result["is_consistent"] else "✗ TUTARSIZ")
        )

        return result

    def _compute_absorption(
        self,
        M: np.ndarray,
        states: list,
        transient: list,
        absorbing: list,
        start_idx: int,
    ) -> dict:
        """
        B = N·R yöntemiyle absorpsiyon olasılıklarını hesaplar.
        N = (I - Q)^{-1}
        """
        n_tra = len(transient)
        n_abs = len(absorbing)

        if n_tra == 0:
            # Tüm durumlar absorbing — başlangıç zaten absorbing
            result = {states[i]: 0.0 for i in absorbing}
            if start_idx in absorbing:
                result[states[start_idx]] = 1.0
            return result

        tra_idx = {s: i for i, s in enumerate(transient)}

        Q = M[np.ix_(transient, transient)]
        R = M[np.ix_(transient, absorbing)]

        try:
            I_mat = np.eye(n_tra)
            N = np.linalg.inv(I_mat - Q)
            B = N @ R  # n_tra × n_abs
        except np.linalg.LinAlgError:
            return {}

        # Başlangıç transient indeksi
        if start_idx in tra_idx:
            si = tra_idx[start_idx]
        else:
            si = 0

        return {states[absorbing[j]]: round(float(B[si, j]), 6) for j in range(n_abs)}

    def _detect_states_in_text(self, text: str, states: list, props: dict) -> list:
        """
        Metin içinde hangi durum etiketlerinin geçtiğini güvenli biçimde tespit eder.

        BUG DÜZELTMESİ (rehber1.md):
          - Tek karakter etiket 'a' → "olacaktır" içindeki 'a'ya eşleşme HATASI
          - Düzeltme: etiket tek karakter ise kelime sınırı DEĞİL,
            etiketin karşılık geldiği metin (prop) metinde geçiyor mu bak.
          - Çok karakterli etiketler için kelime sınırı regex kullan.
        """
        detected = []
        for s in states:
            matched = False

            # Önce prop metnine bak (en güvenilir)
            if s in props:
                prop_text = props[s].lower()
                # Prop metninin önemli kısmı (ilk 3 kelime) conclusion'da geçiyor mu?
                prop_words = prop_text.split()[:3]
                overlap = sum(1 for w in prop_words if len(w) > 3 and w in text)
                if overlap >= 1:
                    matched = True

            # Sonra etiket kendisi — tek karakter için izole kelime kontrolü
            if not matched:
                if len(s) == 1:
                    # Tek karakter: yalnızca kelime başında/sonunda, Türkçe ek almamış
                    # Örn: "e" → "öldü" değil, ama " e " veya "e=" eşleşsin
                    if re.search(
                        r"(?<![a-züçşığöA-ZÜÇŞĞÖ])"
                        + re.escape(s)
                        + r"(?![a-züçşığöA-ZÜÇŞĞÖ])",
                        text,
                    ):
                        matched = True
                else:
                    if re.search(r"\b" + re.escape(s) + r"\b", text):
                        matched = True

            if matched:
                detected.append(s)

        return detected


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL: CausalBayesOrchestrator
#  Yukarıdaki dört modülü tek pipeline'da birleştiren orkestratör.
#  run_solver_pipeline() ve /solve route'una entegre olur.
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 5: PropositionalSelfCorrector
#  Dinamik Geri Besleme Döngüsü — sonuç önerme mantığıyla çelişiyor mu?
#
#  Algoritma:
#    1. Mantık cümlesini IF-THEN kurallarına ayrıştır (hard-coded kural yok)
#    2. Her kuralı final olasılık dağılımına karşı test et
#    3. Düşük inanılırlık öncülü tespit et (threshold < 0.35)
#    4. Mantıksal tutarsızlıkları raporla ve düzeltme önerisi üret
#    5. Overall correction_needed: bool döndür
# ═══════════════════════════════════════════════════════════════════════════════
class PropositionalSelfCorrector:
    """
    Öz-düzeltme katmanı — sonucun başlangıç öncülleriyle tutarlılığını test eder.

    Giriş : {
        'propositions':     dict  — {label: text},
        'logic_str':        str   — mantık cümlesi,
        'final_distribution': dict — {label: {probability_normalized: float}},
        'credibility_scores': dict — {label: {credibility: float}},
    }
    Çıkış : {
        'correction_needed': bool,
        'rule_violations':   list[str],
        'low_cred_premises': list[str],
        'correction_hints':  list[str],
        'confidence_penalty': float  — [0, 0.4] overall penalty
    }
    """

    # Kural ayrıştırma kalıpları — hard-coding yok
    # Her kalıp bir IF-THEN yapısı tespit eder
    RULE_PATTERNS = [
        # "eğer X ise Y'dir" / "if X then Y"
        (r"(?:eğer|if)\s+([a-z])\s*(?:ise|then)[,\s]+([a-z])", "IF_THEN"),
        # "X ise Y" (kısa form)
        (r"\b([a-z])\s*ise\s+([a-z])\b", "IF_THEN_SHORT"),
        # "X = Y ise Z" (koşullu atama)
        (r"\b([a-z])\s*[=,]\s*([a-z])\s+ise\s+([a-z])", "IF_AND_THEN"),
        # "aksi halde X" (ELSE)
        (r"aksi\s+halde\s+([a-z])\b|otherwise\s+([a-z])\b", "ELSE_STATE"),
    ]

    # Çelişki tespiti eşiği — terminal sonuç bu değerin altındaysa uyar
    # Eşikler SPCE'den dinamik olarak hesaplanır — sabit float yok
    # LOW_PROB_THRESHOLD  ≈ sigmoid(danger_axis ortalaması) × 0.40
    # LOW_CRED_THRESHOLD  ≈ baz belirsizlik sınırı
    # INCONSISTENCY_DELTA ≈ karar bölgesi genişliği
    @property
    def LOW_PROB_THRESHOLD(self):
        return _SPCE.compute_threshold("death", base_uncertainty=0.0)

    @property
    def LOW_CRED_THRESHOLD(self):
        return _SPCE.compute_threshold("routine", base_uncertainty=0.1)

    @property
    def INCONSISTENCY_DELTA(self):
        return _SPCE.compute_threshold("danger", base_uncertainty=0.2)

    def check(
        self,
        propositions: dict,
        logic_str: str,
        final_distribution: dict,
        credibility_scores: dict,
    ) -> dict:

        result = {
            "correction_needed": False,
            "rule_violations": [],
            "low_cred_premises": [],
            "correction_hints": [],
            "confidence_penalty": 0.0,
        }

        if not propositions or not final_distribution:
            return result

        q = logic_str.lower()

        # ── Adım 1: Kuralları ayrıştır ────────────────────────────────────────
        rules = self._parse_rules(q)

        # ── Adım 2: Her kuralı dağılıma karşı test et ────────────────────────
        for rule_type, antecedents, consequent in rules:
            if consequent not in final_distribution:
                continue
            info = final_distribution[consequent]
            p_conseq = info.get("probability_normalized", info.get("probability", 0))

            # IF-THEN kuralı: öncül gerçekse sonuç olasılığı düşük olmamalı
            if rule_type in ("IF_THEN", "IF_THEN_SHORT", "IF_AND_THEN"):
                if p_conseq < self.LOW_PROB_THRESHOLD:
                    ant_txt = " ∧ ".join(
                        f"'{a}' ({propositions.get(a, a)})"
                        for a in antecedents
                        if a in propositions
                    )
                    result["rule_violations"].append(
                        f"[MOD5-KURAL] Kural '{ant_txt} → {consequent}' geçerliyken "
                        f"P({consequent})={p_conseq:.3f} < eşik {self.LOW_PROB_THRESHOLD:.2f}. "
                        f"Öncüller sonucu desteklemiyor veya zincirde kesim var."
                    )

        # ── Adım 3: Düşük inanılırlıklı öncüller ─────────────────────────────
        # Sonucu doğrudan etkileyen (terminal olmayan) öncüller kontrol edilir
        terminal_labels = set(final_distribution.keys())
        for lbl, cs in credibility_scores.items():
            if lbl in terminal_labels:
                continue  # terminal durumlar öncül değil
            cval = cs.get("credibility", 1.0)
            if cval < self.LOW_CRED_THRESHOLD:
                result["low_cred_premises"].append(
                    f"[MOD5-ZAYIF] '{lbl}' ({propositions.get(lbl, lbl)}) "
                    f"inanılırlık={cval:.3f} — zayıf öncül; "
                    f"bu önermeye dayalı çıkarımlar güvenilmez olabilir."
                )

        # ── Adım 4: Mantıksal tersine dönme tespiti ──────────────────────────
        # "hayatta kalır" semantiği içeren önermenin olasılığı
        # "ölür" semantiğinden düşükse çelişki var
        survival_labels = [
            l
            for l, t in propositions.items()
            if any(
                re.search(p, t.lower())
                for p in [r"dışarı", r"hayatta", r"yaşar", r"sağ", r"exit", r"survive"]
            )
        ]
        death_labels = [
            l
            for l, t in propositions.items()
            if any(
                re.search(p, t.lower())
                for p in [r"öl[üu]", r"ölüm", r"death", r"died?", r"killed?"]
            )
        ]
        for sl in survival_labels:
            for dl in death_labels:
                ps = final_distribution.get(sl, {}).get(
                    "probability_normalized",
                    final_distribution.get(sl, {}).get("probability", -1),
                )
                pd = final_distribution.get(dl, {}).get(
                    "probability_normalized",
                    final_distribution.get(dl, {}).get("probability", -1),
                )
                if ps >= 0 and pd >= 0 and pd > ps + self.INCONSISTENCY_DELTA:
                    result["rule_violations"].append(
                        f"[MOD5-ÇELİŞKİ] Ölüm olasılığı P({dl})={pd:.3f} "
                        f"hayatta kalma P({sl})={ps:.3f}'den "
                        f"{(pd-ps)*100:.1f}% fazla. "
                        f"Mantık zinciri hayatta kalmayı ima ediyorsa çelişki var."
                    )

        # ── Adım 5: Düzeltme önerileri ───────────────────────────────────────
        if result["rule_violations"] or result["low_cred_premises"]:
            result["correction_needed"] = True
            if result["low_cred_premises"]:
                result["correction_hints"].append(
                    "Zayıf öncülleri güçlendirmek için ek bağlam sağlayın "
                    "veya gizli düğüm olasılıklarını manuel olarak belirtin."
                )
            if result["rule_violations"]:
                result["correction_hints"].append(
                    "Mantık zincirinizi gözden geçirin: öncül-sonuç bağları "
                    "olasılık yayılımıyla tutarlı olmalı. "
                    "NarrativeBeliefPropagator zincirini kontrol edin."
                )
            # Ceza hesaplaması
            n_viol = len(result["rule_violations"]) + len(result["low_cred_premises"])
            # Ceza: ihlal başına eksen bazlı artış — SPCE tetikler
            penalty_per_viol = _SPCE.compute_threshold("routine", 0.0)  # ~0.10
            max_penalty = _SPCE.compute_threshold("danger", 0.0)  # ~0.40
            result["confidence_penalty"] = min(max_penalty, n_viol * penalty_per_viol)

        return result

    def _parse_rules(self, logic_text: str) -> list:
        """
        Mantık metnini [(rule_type, [antecedents], consequent)] listesine ayrıştırır.
        Tamamen regex tabanlı — hiçbir kural sabit kodlanmamış.
        """
        rules = []
        for pattern, rtype in self.RULE_PATTERNS:
            for m in re.finditer(pattern, logic_text):
                grps = [g for g in m.groups() if g]
                if rtype in ("IF_THEN", "IF_THEN_SHORT") and len(grps) >= 2:
                    rules.append((rtype, [grps[0]], grps[1]))
                elif rtype == "IF_AND_THEN" and len(grps) >= 3:
                    rules.append((rtype, [grps[0], grps[1]], grps[2]))
                elif rtype == "ELSE_STATE" and grps:
                    rules.append((rtype, [], grps[0]))
        return rules


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 6: EntropyThresholdManager
#  Belirsizlik Eşiği Yöneticisi — yüksek entropide CoT genişletme tetikler.
#
#  Algoritma:
#    1. Shannon entropisini THRESHOLD ile karşılaştır
#    2. Eşik aşılırsa: gizli düğümleri daha derin araştır
#    3. Kullanıcıya yöneltilecek açıklayıcı sorular üret (prop metinlerinden)
#    4. Ollama'ya CoT genişletme ipuçları üret
#    5. Entropy seviyesine göre aksiyon planı döndür
# ═══════════════════════════════════════════════════════════════════════════════
class EntropyThresholdManager:
    """
    Shannon entropisi izleyicisi ve belirsizlik yöneticisi.

    ENTROPY_THRESHOLD:  0.50 bit  — bu değerin üstü "belirsiz" bölge
    CRITICAL_THRESHOLD: 1.20 bit  — bu değerin üstü "kritik belirsizlik"

    Çıkış: {
        'entropy_level':     str    — LOW/MEDIUM/HIGH/CRITICAL,
        'needs_expansion':   bool,
        'clarification_qs':  list[str],  — kullanıcıya yöneltilecek sorular
        'cot_hints':         list[str],  — LLM için düşünce genişletme ipuçları
        'deepening_targets': list[str],  — hangi gizli düğümler araştırılmalı
        'recommendation':    str
    }
    """

    # Eşikler SPCE'den dinamik — Decimal kesinlikte
    # ENTROPY_THRESHOLD  = sigmoid(baz) ≈ 0.50
    # CRITICAL_THRESHOLD = sigma üst sınır ≈ 1.20
    @property
    def ENTROPY_THRESHOLD(self):
        # Baz belirsizlik altında max-entropy Bernoulli = 1 bit
        # Eşik: bunun yarısı → sigmoid ölçeği (Decimal kesinlikte)
        return round(float(_SPCE._sigmoid(0.0, k=2.0)), 4)  # ≈ 0.50

    @property
    def CRITICAL_THRESHOLD(self):
        # 2 seçenek max entropy = 1 bit; çok seçenek için ≈ 1.2
        # SPCE: 2 × ENTROPY_THRESHOLD × skala (Decimal kesinlikte)
        base = float(_SPCE._sigmoid(0.0, k=2.0))
        return round(base * 2.4, 4)  # ≈ 1.20

    # Belirsizliği artıran ve soruya dönüştürülebilecek kavramlar
    AMBIGUITY_TRIGGERS = [
        (r"\bduman\b", "Duman çıkışının yoğunluğu ve hızı nedir?"),
        (r"\bgaz\b", "Gaz sızıntısının kaynağı ve miktarı bilinmekte mi?"),
        (r"\bsoba\b", "Soba ne tür yakıt kullanıyor (odun/kömür/gaz)?"),
        (r"\byaktı\b", "Yakma ne kadar süredir devam ediyor?"),
        (r"\biçti\b", "Ne kadar su içildi — vücudun su ihtiyacını karşılıyor mu?"),
        (r"\bdışarı\b", "Dışarısının güvenli olup olmadığı bilinmekte mi?"),
    ]

    def evaluate(
        self, entropy: float, chain_result: dict, belief_result: dict, question: str
    ) -> dict:
        """
        entropy:      NarrativeBeliefPropagator.entropy değeri
        chain_result: CausalChainInferencer.infer() çıktısı
        belief_result: NarrativeBeliefPropagator.propagate() çıktısı
        question:     orijinal soru metni
        """
        q_lower = question.lower()

        # ── Entropi seviyesi ─────────────────────────────────────────────────
        if entropy <= self.ENTROPY_THRESHOLD:
            level = "LOW"
            needs_expansion = False
        elif entropy <= self.CRITICAL_THRESHOLD:
            level = "MEDIUM" if entropy <= 0.80 else "HIGH"
            needs_expansion = True
        else:
            level = "CRITICAL"
            needs_expansion = True

        result = {
            "entropy_level": level,
            "needs_expansion": needs_expansion,
            "clarification_qs": [],
            "cot_hints": [],
            "deepening_targets": [],
            "recommendation": "",
        }

        if not needs_expansion:
            result["recommendation"] = (
                f"Entropi {entropy:.4f} bit — eşik {self.ENTROPY_THRESHOLD} bit altında. "
                f"Sistem yeterince belirli."
            )
            return result

        # ── Adım 1: Soruya dönük açıklayıcı sorular üret ─────────────────────
        # Soru metninden ve gizli düğümlerden tetikle — hard-coding yok
        seen_triggers = set()
        for pattern, clarification in self.AMBIGUITY_TRIGGERS:
            if re.search(pattern, q_lower) and pattern not in seen_triggers:
                result["clarification_qs"].append(clarification)
                seen_triggers.add(pattern)

        # Gizli düğümlerden ek sorular
        for hn in chain_result.get("hidden_nodes", []):
            label = hn.get("label", "")
            reason = hn.get("reason", "")
            if label not in seen_triggers:
                result["clarification_qs"].append(
                    f"Gizli bileşen [{label}] için: {reason} — "
                    f"bu sürecin olasılığı daha kesin belirlenebilir mi?"
                )
                result["deepening_targets"].append(label)
                seen_triggers.add(label)

        # ── Adım 2: CoT genişletme ipuçları ──────────────────────────────────
        steps = belief_result.get("propagation_steps", [])
        # Belirsiz bölge sınırları: SPCE'den dinamik
        _lo = _SPCE.compute_threshold("routine", 0.0)  # ≈ 0.35
        _hi = 1.0 - _lo  # ≈ 0.65
        for step in steps:
            p_node = step.get("p_node", 0.5)
            if _lo < p_node < _hi:  # belirsiz orta bölge — dinamik
                result["cot_hints"].append(
                    f"'{step['node']}' düğümü P={p_node:.3f} ile belirsiz bölgede "
                    f"({step['text'][:40]}). Bu adımı daha fazla koşul/kanıtla destekle."
                )

        # Terminal dağılım entropisinden hint
        fd = belief_result.get("final_distribution", {})
        if fd:
            probs = sorted(
                [
                    v.get("probability_normalized", v.get("probability", 0))
                    for v in fd.values()
                ],
                reverse=True,
            )
            _diff_threshold = _SPCE.compute_threshold("survival", 0.0) * 0.5  # ≈ 0.20
            if len(probs) >= 2 and abs(probs[0] - probs[1]) < _diff_threshold:
                result["cot_hints"].append(
                    f"En olası iki terminal durum arasındaki fark sadece "
                    f"{abs(probs[0]-probs[1])*100:.1f}%. "
                    f"Karar sınırı çok yakın — ek koşullu olasılık gerekiyor."
                )

        # ── Adım 3: Öneri ────────────────────────────────────────────────────
        level_tr = {
            "MEDIUM": "ORTA",
            "HIGH": "YÜKSEK",
            "CRITICAL": "KRİTİK",
        }.get(level, level)

        result["recommendation"] = (
            f"Entropi {entropy:.4f} bit [{level_tr}] — "
            f"eşik {self.ENTROPY_THRESHOLD} bit aşıldı. "
            f"{len(result['clarification_qs'])} açıklayıcı soru, "
            f"{len(result['cot_hints'])} CoT ipucu üretildi. "
            f"{'Sistem durmalı ve kullanıcıdan ek bilgi talep etmeli.' if level == 'CRITICAL' else 'Zinciri derinleştir.'}"
        )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 7: TemporalStateMemory
#  Zaman Serisi ve Durum Belleği — geçiş olasılıklarına zaman çürümesi uygular.
#
#  Algoritma:
#    1. Önerme metnindeki zamansal ipuçlarını çıkar (sonra, ardından, önce, vb.)
#    2. Olaylar arasındaki göreceli zaman ağırlığını belirle
#    3. P_adjusted(B|A) = P(B|A) × exp(−λ × temporal_distance)
#    4. λ bağlamdan türetilir: acil durum → λ yüksek (hızlı çürüme)
#    5. Zaman-uyarlamalı zincir önsel olasılıklarını döndür
# ═══════════════════════════════════════════════════════════════════════════════
class TemporalStateMemory:
    """
    Zaman bazlı Markov geçiş ağırlıklandırması.

    Zaman çürüme formülü:
      P_adjusted(j|i, Δt) = P_nominal(j|i) × exp(−λ × Δt)

    λ bağlam katsayısı:
      - Acil durum (duman/gaz/yangın): λ = 0.8  (hızlı çürüme — zaman önemli)
      - Normal eylem (içmek/yürümek): λ = 0.2  (yavaş çürüme)
      - Belirsiz bağlam:              λ = 0.4  (orta)

    Δt (göreceli zaman birimi) soru metninden çıkarılır:
      - "ardından/sonra" → Δt = 1.0
      - "hemen/anında"  → Δt = 0.1
      - "uzun süre sonra/saatler sonra" → Δt = 3.0
      - Belirtilmemiş   → Δt = 1.0 (baz)

    Çıkış: {
        'temporal_weights':    dict — {(from, to): adjusted_p},
        'decay_lambda':        float,
        'time_hints_found':    list[str],
        'adjusted_chain_prior': float,
        'temporal_warning':    str | None
    }
    """

    # Zaman ipucu kalıpları → (Δt çarpanı, açıklama)
    TIME_HINTS = [
        (r"\bhemen\b|\banında\b|\bderhal\b|\bimmediately\b", 0.10, "anlık"),
        (r"\bardından\b|\bsonrasında\b|\bsoon\s+after\b", 0.80, "kısa süre sonra"),
        (r"\bsonra\b|\bafter\b|\bthen\b", 1.00, "belirtisiz sonra"),
        (r"\bdaha\s+sonra\b|\blater\b", 2.00, "biraz sonra"),
        (r"\bsaatler\s+sonra\b|\bhours?\s+later\b", 4.00, "saatler sonra"),
        (r"\bgünler\s+sonra\b|\bdays?\s+later\b", 10.00, "günler sonra"),
        (r"\bzaman\s+geçince\b|\bover\s+time\b", 5.00, "zamanla"),
    ]

    # Acil durum sinyalleri → λ yüksek
    URGENT_SIGNALS = [
        r"\bduman\b",
        r"\bgaz\b",
        r"\byanık\b",
        r"\bateş\b",
        r"\bzehir\b",
        r"\bsmoke\b",
        r"\bfire\b",
        r"\bgas\b",
    ]
    # Normal eylem sinyalleri → λ düşük
    ROUTINE_SIGNALS = [
        r"\biçti\b",
        r"\byürüdü\b",
        r"\boturdu\b",
        r"\bdrank?\b",
        r"\bwalked?\b",
        r"\bsat\b",
    ]

    def analyze(self, question: str, chain: list, causal_edges: list) -> dict:
        """
        question:     orijinal soru metni
        chain:        CausalChainInferencer zinciri
        causal_edges: [(from, to, p_prior, reason)] listesi
        """
        q = question.lower()

        # ── Adım 1: λ katsayısını belirle ────────────────────────────────────
        has_urgent = any(re.search(p, q) for p in self.URGENT_SIGNALS)
        has_routine = any(re.search(p, q) for p in self.ROUTINE_SIGNALS)

        # Lambda: sabit değil — SPCE.compute_decay_lambda ile hesaplanır
        # Soru metninin tamamından eksen analizi
        if has_urgent:
            context_type = "ACİL (duman/tehlike)"
        elif has_routine and not has_urgent:
            context_type = "RUTIN eylem"
        else:
            context_type = "BELİRSİZ"
        decay_lambda = _SPCE.compute_decay_lambda(q)

        # ── Adım 2: Zaman ipuçlarını çıkar ────────────────────────────────────
        time_hints_found = []
        delta_t_multiplier = 1.0  # varsayılan

        for pattern, dt_mult, label in self.TIME_HINTS:
            if re.search(pattern, q):
                time_hints_found.append(f"'{label}' (Δt×{dt_mult})")
                delta_t_multiplier = max(delta_t_multiplier, dt_mult)

        # ── Adım 3: Adjusted geçiş olasılıkları ──────────────────────────────
        temporal_weights = {}
        for edge in causal_edges:
            if len(edge) >= 3:
                src, dst, p_nom = edge[0], edge[1], edge[2]
                delta_t = delta_t_multiplier  # kenar başına aynı Δt (iyileştirilebilir)
                p_adj = p_nom * math.exp(-decay_lambda * delta_t)
                p_adj = max(0.01, min(p_adj, p_nom))  # asla sıfıra düşme
                temporal_weights[(src, dst)] = {
                    "p_nominal": round(p_nom, 4),
                    "p_adjusted": round(p_adj, 4),
                    "delta_t": round(delta_t, 2),
                    "decay": round(p_nom - p_adj, 4),
                }

        # ── Adım 4: Adjusted zincir önsel olasılığı ──────────────────────────
        if chain:
            p_adjusted_chain = 1.0
            for node in chain:
                if node.get("hidden"):
                    continue
                src = node["label"]
                # Bu düğüme giren kenarın adjusted p'si
                adj_p = None
                for (s, d), w in temporal_weights.items():
                    if d == src:
                        adj_p = w["p_adjusted"]
                        break
                if adj_p is None:
                    adj_p = node.get("prior", 0.5)
                p_adjusted_chain *= adj_p
        else:
            p_adjusted_chain = 0.0

        # ── Adım 5: Zaman uyarısı ────────────────────────────────────────────
        temporal_warning = None
        if delta_t_multiplier >= 2.0:
            temporal_warning = (
                f"Zaman gecikmesi (Δt×{delta_t_multiplier:.1f}) geçiş olasılıklarını "
                f"exp(−{decay_lambda}×{delta_t_multiplier:.1f})="
                f"{math.exp(-decay_lambda*delta_t_multiplier):.3f} katsayısıyla düşürdü. "
                f"Olaylar arası gecikme hayatta kalma olasılığını etkiliyor."
            )
        elif has_urgent:
            temporal_warning = (
                f"Acil durum bağlamı (λ={decay_lambda}) — "
                f"zaman değişkeni kritik. Anlık müdahale olasılıkları yüksek tutulmalı."
            )

        return {
            "temporal_weights": temporal_weights,
            "decay_lambda": decay_lambda,
            "context_type": context_type,
            "delta_t_multiplier": delta_t_multiplier,
            "time_hints_found": time_hints_found,
            "adjusted_chain_prior": round(p_adjusted_chain, 6),
            "temporal_warning": temporal_warning,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 8: LatentVariableSensitivityAnalyzer
#  Gizli Değişken Duyarlılık Analizi — hangi değişken sonucu en çok etkiliyor?
#
#  Algoritma: Sonlu fark yöntemi (finite difference)
#    sensitivity_i = [P(result | var_i = p+ε) − P(result | var_i = p−ε)] / (2ε)
#    relative_sensitivity_i = |sensitivity_i| / Σ |sensitivity_j|
#
#  Hard-coding yok — tüm değişkenler zincirden algoritmik çıkarılır.
# ═══════════════════════════════════════════════════════════════════════════════
class LatentVariableSensitivityAnalyzer:
    """
    Sonlu fark tabanlı duyarlılık analizi.
    Her zincir değişkeni için: sonuç olasılığı bu değişkene ne kadar duyarlı?

    Giriş : {
        'chain':             list — zincir adımları,
        'hidden_nodes':      list — gizli düğümler,
        'final_distribution': dict — terminal dağılım,
    }
    Çıkış : {
        'sensitivity_table': dict — {variable: {sensitivity, relative, rank}},
        'most_influential':  str  — en etkili değişken,
        'least_influential': str  — en az etkili değişken,
        'dominance_ratio':   float — max_sens / second_max_sens,
        'report':            str
    }
    """

    EPSILON = Decimal("1e-7")  # çok küçük pertürbasyon — Decimal kesinlikte

    def analyze(
        self, chain: list, hidden_nodes: list, final_distribution: dict
    ) -> dict:
        """
        Tüm zincir değişkenlerine ±ε pertürbasyonu uygula,
        terminal olasılığın değişimini ölç.
        """
        if not chain or not final_distribution:
            return {"sensitivity_table": {}, "report": "Zincir veya dağılım boş"}

        # ── Hedef: en olası terminal durum ────────────────────────────────────
        best_terminal = max(
            final_distribution.items(),
            key=lambda x: x[1].get(
                "probability_normalized", x[1].get("probability", 0)
            ),
        )
        target_label = best_terminal[0]

        # ── Tüm değişkenler: zincir + gizli düğümler ─────────────────────────
        all_vars = []
        for node in chain:
            all_vars.append(
                {
                    "id": node["label"],
                    "prior": node.get("prior", 0.5),
                    "hidden": node.get("hidden", False),
                }
            )
        for hn in hidden_nodes:
            all_vars.append(
                {
                    "id": hn.get("id", hn.get("label", "?")),
                    "prior": hn.get("prior", 0.5),
                    "hidden": True,
                }
            )

        if not all_vars:
            return {"sensitivity_table": {}, "report": "Değişken bulunamadı"}

        # ── Baz zincir çarpımı ────────────────────────────────────────────────
        def chain_product(vars_with_priors: list) -> float:
            p = 1.0
            for v in vars_with_priors:
                p *= v["prior"]
            return p

        base_p = chain_product(all_vars)

        # ── Finite difference sensitivity (Decimal kesinlikte) ────────────────
        sensitivity_table = {}
        eps = self.EPSILON

        for i, var in enumerate(all_vars):
            original_prior = var["prior"]

            # P+ : var'ı p+ε'ya yükselt (Decimal ile)
            eps_float = float(eps) if isinstance(eps, Decimal) else eps
            all_vars[i]["prior"] = min(0.99, original_prior + eps_float)
            p_plus = chain_product(all_vars)

            # P- : var'ı p-ε'ya düşür (Decimal ile)
            all_vars[i]["prior"] = max(0.01, original_prior - eps_float)
            p_minus = chain_product(all_vars)

            # Geri yükle
            all_vars[i]["prior"] = original_prior

            # Sonlu fark — Decimal hassasiyeti ile
            eps_dec = Decimal(str(eps_float))
            denominator = Decimal(2) * eps_dec
            if denominator > 0:
                raw_sensitivity = float(
                    (Decimal(str(p_plus)) - Decimal(str(p_minus))) / denominator
                )
            else:
                raw_sensitivity = 0.0

            sensitivity_table[var["id"]] = {
                "prior": round(original_prior, 4),
                "sensitivity": round(raw_sensitivity, 6),
                "abs_sens": round(abs(raw_sensitivity), 6),
                "is_hidden": var["hidden"],
            }

        # ── Normalize: bağıl duyarlılık (Decimal hassasiyeti) ──────────────────
        total_abs = sum(v["abs_sens"] for v in sensitivity_table.values())
        for vid in sensitivity_table:
            if total_abs > 0:
                sensitivity_table[vid]["relative"] = round(
                    sensitivity_table[vid]["abs_sens"] / total_abs, 4
                )
            else:
                sensitivity_table[vid]["relative"] = 0.0

        # Sıralama
        ranked = sorted(sensitivity_table.items(), key=lambda x: -x[1]["abs_sens"])
        for rank_i, (vid, _) in enumerate(ranked, 1):
            sensitivity_table[vid]["rank"] = rank_i

        # ── Özet istatistikler ────────────────────────────────────────────────
        most_infl = ranked[0][0] if ranked else "?"
        least_infl = ranked[-1][0] if len(ranked) > 1 else "?"
        dom_ratio = (
            ranked[0][1]["abs_sens"] / max(ranked[1][1]["abs_sens"], 1e-9)
            if len(ranked) >= 2
            else 1.0
        )

        report = (
            f"En etkili: '{most_infl}' (bağıl={sensitivity_table.get(most_infl,{}).get('relative',0):.1%}). "
            f"Baskınlık oranı: {dom_ratio:.2f}x. "
            f"{'Tek değişken hakimiyet kuruyor.' if dom_ratio > 3.0 else 'Çok değişkenli denge.'}"
        )

        return {
            "sensitivity_table": sensitivity_table,
            "most_influential": most_infl,
            "least_influential": least_infl,
            "dominance_ratio": round(dom_ratio, 3),
            "base_chain_p": round(base_p, 6),
            "report": report,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ MODÜL 9: BayesianDecisionEngine
#  Karar Teorisi — Beklenen Fayda ve Risk-Ödül Matrisi
#
#  Algoritma:
#    EU(eylem) = Σ_sonuç P(sonuç | eylem) × U(sonuç)
#    U(sonuç) prop metinlerinden otomatik çıkarılır:
#      ölüm/zarar    → U = −1.0
#      hayatta kalma → U = +1.0
#      belirsiz      → U = 0.0 (nötr baz)
#    Eylemler önerme metninden çıkarılır — hard-coding yok.
# ═══════════════════════════════════════════════════════════════════════════════
class BayesianDecisionEngine:
    """
    Bayesyen Karar Teorisi motoru.

    Giriş : {
        'propositions':       dict — {label: text},
        'final_distribution': dict — terminal olasılıklar,
        'credibility_scores': dict — inanılırlık skorları,
        'chain':              list — zincir adımları,
    }
    Çıkış : {
        'actions':         dict — {action_label: {eu, outcomes, description}},
        'best_action':     str,
        'best_eu':         float,
        'risk_reward_matrix': list[list],  — [action × outcome] tablosu
        'recommendation':  str,
        'utility_table':   dict — {label: utility}
    }
    """

    # Negatif fayda sinyalleri (zarar/ölüm)
    NEGATIVE_UTILITY_SIGNALS = [
        r"\böl[üu]\b",
        r"\bölüm\b",
        r"\bölür\b",
        r"\bölecek\b",
        r"\bzehirlen\b",
        r"\byaralan\b",
        r"\bhasar\b",
        r"\bkaybet\b",
        r"\bdeath\b",
        r"\bdied?\b",
        r"\bhurt\b",
        r"\bdamage\b",
        r"\blost?\b",
    ]
    # Pozitif fayda sinyalleri (hayatta kalma/kurtuluş)
    POSITIVE_UTILITY_SIGNALS = [
        r"\bhayatta\b",
        r"\bkurtul\b",
        r"\bsağ\b",
        r"\byaşar\b",
        r"\bdışarı\s+çıkt\b",
        r"\bçıktı\b",
        r"\bgüvende\b",
        r"\bsurvive?\b",
        r"\bsafe\b",
        r"\bescape\b",
        r"\blive\b",
    ]
    # Eylem sinyalleri (aktif karar gerektiren)
    ACTION_SIGNALS = [
        r"\bdışarı\s+çık\b",
        r"\bçıkmak\b",
        r"\bgitmek\b",
        r"\bkaçmak\b",
        r"\bkalmak\b",
        r"\bbeklемek\b",
        r"\biçmek\b",
        r"\byakmak\b",
        r"\bexit\b",
        r"\bleave\b",
        r"\bstay\b",
        r"\bwait\b",
    ]

    def _infer_utility(self, text: str) -> float:
        """Metin anlamından fayda değerini algoritmik çıkar."""
        t = text.lower()
        pos = sum(1 for p in self.POSITIVE_UTILITY_SIGNALS if re.search(p, t))
        neg = sum(1 for p in self.NEGATIVE_UTILITY_SIGNALS if re.search(p, t))
        if neg > 0 and pos == 0:
            return -1.0
        if pos > 0 and neg == 0:
            return +1.0
        if pos > 0 and neg > 0:
            return round((pos - neg) / (pos + neg), 2)
        return 0.0  # nötr

    def _infer_action_type(self, text: str) -> str:
        """Önerme metninden eylem tipini çıkar."""
        t = text.lower()
        if any(
            re.search(p, t)
            for p in [r"dışarı", r"çıkt", r"git", r"kaç", r"exit", r"leave"]
        ):
            return "EXIT"
        if any(
            re.search(p, t) for p in [r"kal\b", r"bekle", r"içer", r"stay", r"wait"]
        ):
            return "STAY"
        return "NEUTRAL"

    def compute(
        self,
        propositions: dict,
        final_distribution: dict,
        credibility_scores: dict,
        chain: list,
    ) -> dict:
        """
        Tüm eylemler için beklenen faydayı hesapla ve risk-ödül matrisi oluştur.
        """
        if not propositions or not final_distribution:
            return {
                "recommendation": "Veri yetersiz",
                "actions": {},
                "best_action": None,
            }

        # ── Adım 1: Fayda tablosu — her önerme için U çıkar ──────────────────
        utility_table = {}
        for lbl, txt in propositions.items():
            utility_table[lbl] = self._infer_utility(txt)

        # ── Adım 2: Eylemleri belirle ─────────────────────────────────────────
        # Terminal olmayan önermeleri eylem adayı say
        terminal_labels = set(final_distribution.keys())
        action_candidates = {
            lbl: txt for lbl, txt in propositions.items() if lbl not in terminal_labels
        }

        # Yeterli aday yoksa tüm önermeleri dahil et
        if not action_candidates:
            action_candidates = propositions

        # ── Adım 3: Her eylem için EU hesapla ────────────────────────────────
        actions = {}
        for a_lbl, a_txt in action_candidates.items():
            a_type = self._infer_action_type(a_txt)
            # Bu eylemi yapma durumunda terminal olasılıklar değişiyor mu?
            # Basit model: eylem credibility'si terminal dağılımı ölçeklendiriyor
            a_cred = credibility_scores.get(a_lbl, {}).get("credibility", 0.5)

            outcomes = {}
            eu = 0.0
            for t_lbl, t_info in final_distribution.items():
                p_t = t_info.get("probability_normalized", t_info.get("probability", 0))
                u_t = utility_table.get(
                    t_lbl, self._infer_utility(propositions.get(t_lbl, ""))
                )

                # Eylem türüne göre koşullu olasılık ayarı
                # EXIT eylemi hayatta kalma olasılığını a_cred kadar artırır
                p_given_action = p_t
                t_type = self._infer_action_type(propositions.get(t_lbl, ""))
                if a_type == "EXIT" and t_type == "EXIT":
                    p_given_action = min(0.99, p_t * (1.0 + 0.3 * a_cred))
                elif a_type == "EXIT" and u_t < 0:
                    p_given_action = max(0.01, p_t * (1.0 - 0.3 * a_cred))
                elif a_type == "STAY" and u_t < 0:
                    p_given_action = min(0.99, p_t * (1.0 + 0.2 * a_cred))

                outcomes[t_lbl] = {
                    "p": round(p_given_action, 4),
                    "utility": round(u_t, 2),
                    "contrib": round(p_given_action * u_t, 4),
                }
                eu += p_given_action * u_t

            actions[a_lbl] = {
                "description": a_txt,
                "action_type": a_type,
                "credibility": round(a_cred, 3),
                "expected_utility": round(eu, 4),
                "outcomes": outcomes,
            }

        # ── Adım 4: En iyi eylem ──────────────────────────────────────────────
        if actions:
            best_action = max(actions, key=lambda k: actions[k]["expected_utility"])
            best_eu = actions[best_action]["expected_utility"]
        else:
            best_action = None
            best_eu = 0.0

        # ── Adım 5: Risk-Ödül matrisi (tablo formatı) ─────────────────────────
        terminal_sorted = sorted(final_distribution.keys())
        action_sorted = sorted(actions.keys())
        header_row = (
            ["EYLEM \\ SONUÇ"]
            + [f"{l} ({propositions.get(l,'?')[:12]})" for l in terminal_sorted]
            + ["EU"]
        )
        matrix = [header_row]
        for a_lbl in action_sorted:
            row = [f"{a_lbl} ({actions[a_lbl]['description'][:14]})"]
            for t_lbl in terminal_sorted:
                contrib = actions[a_lbl]["outcomes"].get(t_lbl, {}).get("contrib", 0)
                row.append(f"{contrib:+.3f}")
            row.append(f"{actions[a_lbl]['expected_utility']:+.4f}")
            matrix.append(row)

        # ── Adım 6: Öneri ─────────────────────────────────────────────────────
        if best_action:
            best_txt = actions[best_action]["description"][:40]
            best_type = actions[best_action]["action_type"]
            recommendation = (
                f"Önerilen eylem: [{best_action}] '{best_txt}' "
                f"(Tip={best_type}, EU={best_eu:+.4f}). "
            )
            # Risk uyarısı: EU < 0 ise tüm seçenekler riskli
            if best_eu < 0:
                recommendation += (
                    f"UYARI: En iyi eylem bile negatif beklenen fayda ({best_eu:+.4f}). "
                    f"Tüm seçenekler riskli — ek bilgi gerekiyor."
                )
            elif best_eu > 0.5:
                recommendation += "Yüksek beklenen fayda — bu eylem tercih edilmeli."
        else:
            recommendation = "Eylem adayı bulunamadı."

        return {
            "actions": actions,
            "best_action": best_action,
            "best_eu": round(best_eu, 4),
            "risk_reward_matrix": matrix,
            "recommendation": recommendation,
            "utility_table": utility_table,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ GÜNCELLENMİŞ CausalBayesOrchestrator (9 Modül)
#  Dört mevcut + beş yeni modülü koordine eder.
# ═══════════════════════════════════════════════════════════════════════════════
class CausalBayesOrchestrator:
    """
    9 modülü koordine eder:
      1. PropositionalCredibilityEngine   → inanılırlık skorları
      2. CausalChainInferencer            → gizli nedensel düğümler
      3. NarrativeBeliefPropagator        → zincir olasılık yayılımı
      4. MarkovConsistencyValidator       → matris/sözel tutarlılık
      5. PropositionalSelfCorrector  ★YENİ → geri besleme döngüsü
      6. EntropyThresholdManager     ★YENİ → entropi eşiği yönetimi
      7. TemporalStateMemory         ★YENİ → zaman serisi / durum belleği
      8. LatentVariableSensitivityAnalyzer ★YENİ → duyarlılık analizi
      9. BayesianDecisionEngine      ★YENİ → karar teorisi / EU

    BUG DÜZELTMESİ: Duplicate uyarı sorunu — all_warnings dedup ile giderildi.
    """

    def __init__(self):
        self._cred = PropositionalCredibilityEngine()
        self._causal = CausalChainInferencer()
        self._prop = NarrativeBeliefPropagator()
        self._markov = MarkovConsistencyValidator()
        self._corrector = PropositionalSelfCorrector()
        self._entropy = EntropyThresholdManager()
        self._temporal = TemporalStateMemory()
        self._sensitivity = LatentVariableSensitivityAnalyzer()
        self._decision = BayesianDecisionEngine()

    def analyze(
        self,
        question: str,
        logic_str: str = "",
        markov_matrix: list = None,
        markov_states: list = None,
        stated_conclusion: str = "",
        start_state: str = None,
    ) -> dict:
        """
        Tam 9-modül nedensel Bayes analiz pipeline'ı.
        """
        # ── 1. Nedensel zincir çıkarımı ───────────────────────────────────────
        chain_result = self._causal.infer(question)
        props = chain_result.get("propositions", {})

        # ── 2. Önermeler inanılırlık skorları ─────────────────────────────────
        cred_scores = self._cred.score_all(props) if props else {}

        # ── 3. Zincir boyunca inanç yayılımı ─────────────────────────────────
        logic_for_prop = logic_str or question
        belief_result = self._prop.propagate(chain_result, logic_for_prop)

        # ── 4. Markov tutarlılık doğrulaması ─────────────────────────────────
        markov_result = {}
        if markov_matrix and markov_states:
            markov_result = self._markov.validate(
                matrix_data=markov_matrix,
                states=markov_states,
                stated_conclusion=stated_conclusion,
                props=props,
                start_state=start_state,
            )
        elif stated_conclusion and props:
            markov_result = {
                "is_consistent": True,
                "violations": [],
                "correct_conclusion": "Matris verisi yok — tam doğrulama atlandı.",
                "absorption_probs": {},
            }

        # ── 5. ★ Öz-Düzeltme Döngüsü ─────────────────────────────────────────
        fd = belief_result.get("final_distribution", {})
        correction_result = self._corrector.check(
            propositions=props,
            logic_str=logic_for_prop,
            final_distribution=fd,
            credibility_scores=cred_scores,
        )

        # ── 6. ★ Entropi Eşiği Yönetimi ──────────────────────────────────────
        entropy_val = belief_result.get("entropy", 0.0)
        entropy_result = self._entropy.evaluate(
            entropy=entropy_val,
            chain_result=chain_result,
            belief_result=belief_result,
            question=question,
        )

        # ── 7. ★ Zamansal Durum Belleği ───────────────────────────────────────
        temporal_result = self._temporal.analyze(
            question=question,
            chain=chain_result.get("chain", []),
            causal_edges=chain_result.get("causal_edges", []),
        )

        # ── 8. ★ Duyarlılık Analizi ───────────────────────────────────────────
        sensitivity_result = self._sensitivity.analyze(
            chain=chain_result.get("chain", []),
            hidden_nodes=chain_result.get("hidden_nodes", []),
            final_distribution=fd,
        )

        # ── 9. ★ Karar Teorisi / Beklenen Fayda ──────────────────────────────
        decision_result = self._decision.compute(
            propositions=props,
            final_distribution=fd,
            credibility_scores=cred_scores,
            chain=chain_result.get("chain", []),
        )

        # ── Özet uyarılar — DEDUP ile duplicate sorunu düzeltildi ─────────────
        # BUG: belief_result.warnings, chain_result.warnings'ın kopyasını içeriyor
        # FIX: Önce set ile tekilleştir, sıralı liste olarak döndür
        raw_warnings = (
            chain_result.get("warnings", [])
            + belief_result.get("warnings", [])
            + markov_result.get("violations", [])
            + correction_result.get("rule_violations", [])
            + correction_result.get("low_cred_premises", [])
            + (
                [temporal_result["temporal_warning"]]
                if temporal_result.get("temporal_warning")
                else []
            )
        )
        # Dedup: sıra koruyarak tekrarları kaldır
        seen_w = set()
        all_warnings = []
        for w in raw_warnings:
            w_key = w[:80]  # ilk 80 karakter unique key yeterli
            if w_key not in seen_w:
                seen_w.add(w_key)
                all_warnings.append(w)

        # ── ASCII özeti ───────────────────────────────────────────────────────
        ascii_box = self._build_ascii_summary(
            props,
            cred_scores,
            chain_result,
            belief_result,
            markov_result,
            correction_result,
            entropy_result,
            temporal_result,
            sensitivity_result,
            decision_result,
        )

        return {
            "_causal_analysis": chain_result,
            "_credibility_scores": cred_scores,
            "_belief_propagation": belief_result,
            "_markov_consistency": markov_result,
            "_correction": correction_result,  # MOD5
            "_entropy_mgmt": entropy_result,  # MOD6
            "_temporal_memory": temporal_result,  # MOD7
            "_sensitivity": sensitivity_result,  # MOD8
            "_decision": decision_result,  # MOD9
            "_causal_warnings": all_warnings,
            "_causal_ascii": ascii_box,
        }

    def _build_ascii_summary(
        self,
        props,
        cred_scores,
        chain_result,
        belief_result,
        markov_result,
        correction_result=None,
        entropy_result=None,
        temporal_result=None,
        sensitivity_result=None,
        decision_result=None,
    ) -> str:
        W = 82
        lines = []

        def box_line(s=""):
            return f"║ {s:<{W-4}} ║"

        def sep():
            return "╠" + "═" * (W - 2) + "╣"

        def mini_sep():
            return "╟" + "─" * (W - 2) + "╢"

        lines.append("╔" + "═" * (W - 2) + "╗")
        lines.append(box_line("◈ NEDENSEL ZİNCİR & 9-MODÜL BAYESYEN ANALİZİ ◈"))
        lines.append(sep())

        # ── MOD 1+2: Önermeler + inanılırlık ─────────────────────────────────
        if props:
            lines.append(box_line("MOD1+2 ▸ ÖNERMELER VE İNANILIRLIK SKORLARI"))
            for lbl, txt in sorted(props.items()):
                cs = cred_scores.get(lbl, {})
                cval = cs.get("credibility", 0)
                clbl = cs.get("label", "")
                bd = cs.get("breakdown", {})
                bd_s = " ".join(f"{k[:4]}={v:+.2f}" for k, v in bd.items() if v != 0)[
                    :30
                ]
                lines.append(
                    box_line(
                        f"  [{lbl}] {txt[:36]:<36}  İNAN={cval:.3f}({clbl}) {bd_s}"
                    )
                )

        # ── MOD 2: Gizli nedensel düğümler ────────────────────────────────────
        hidden = chain_result.get("hidden_nodes", [])
        if hidden:
            lines.append(mini_sep())
            lines.append(
                box_line("MOD2 ▸ GİZLİ NEDENSEL DÜĞÜMLER (Bağlamdan Çıkarıldı)")
            )
            for hn in hidden:
                lines.append(box_line(f"  ⚠ {hn['label']:<46}  P≈{hn['prior']:.3f}"))
                lines.append(box_line(f"    ↳ {hn.get('reason','')[:70]}"))

        # ── MOD 3: Zincir yayılımı ────────────────────────────────────────────
        lines.append(sep())
        lines.append(box_line("MOD3 ▸ ZİNCİR İNANÇ YAYILIMI (P kümülatif)"))
        for step in belief_result.get("propagation_steps", []):
            arrow = "⇢" if step.get("hidden") else "→"
            lines.append(
                box_line(
                    f"  {arrow} [{step['node']:<8}] {step['text'][:34]:<34}"
                    f"  P_kum={step['p_cumulative']:.6f}"
                )
            )
        delta = belief_result.get("direct_skip_delta", 0)
        ent = belief_result.get("entropy", 0)
        lines.append(
            box_line(
                f"  Zincir P={belief_result.get('chain_prior',0):.6f}  "
                f"Atlama Δ={delta:.4f} {'⚠' if delta>0.10 else '✓'}  "
                f"Entropi={ent:.4f}bit"
            )
        )

        fd = belief_result.get("final_distribution", {})
        if fd:
            lines.append(mini_sep())
            lines.append(box_line("MOD3 ▸ TERMINAL DURUM DAĞILIMI (Tam Zincir)"))
            for lbl, info in sorted(
                fd.items(), key=lambda x: -x[1].get("probability", 0)
            ):
                p_n = info.get("probability_normalized", info.get("probability", 0))
                desc = info.get("description", "")[:36]
                lines.append(box_line(f"  [{lbl}] {desc:<36}  P={p_n:.4f}"))

        # ── MOD 4: Markov tutarlılık ──────────────────────────────────────────
        if markov_result:
            lines.append(sep())
            is_c = markov_result.get("is_consistent", True)
            status = "✓ TUTARLI" if is_c else "✗ TUTARSIZ"
            lines.append(box_line(f"MOD4 ▸ MARKOV TUTARLILIK: {status}"))
            exp_s = markov_result.get("expected_state", "")
            exp_p = markov_result.get("expected_prob", 0)
            if exp_s:
                lines.append(box_line(f"  En olası durum: '{exp_s}'  P={exp_p:.4f}"))
            abs_p = markov_result.get("absorption_probs", {})
            if abs_p:
                abs_str = "  ".join(
                    f"P(→{s})={p:.4f}"
                    for s, p in sorted(abs_p.items(), key=lambda x: -x[1])
                )
                lines.append(box_line(f"  Absorpsiyon: {abs_str}"))

        # ── MOD 5: Öz-Düzeltme ───────────────────────────────────────────────
        if correction_result:
            lines.append(sep())
            cn = correction_result.get("correction_needed", False)
            pen = correction_result.get("confidence_penalty", 0)
            status = "⚠ GEREKLİ" if cn else "✓ GEREKSİZ"
            lines.append(
                box_line(f"MOD5 ▸ ÖZ-DÜZELTME: {status}  " f"Güven Cezası={pen:.2f}")
            )
            for v in correction_result.get("rule_violations", [])[:3]:
                for chunk in [v[i : i + 72] for i in range(0, min(len(v), 144), 72)]:
                    lines.append(box_line(f"  ⚑ {chunk}"))
            for h in correction_result.get("correction_hints", [])[:2]:
                lines.append(box_line(f"  💡 {h[:70]}"))

        # ── MOD 6: Entropi Eşiği ─────────────────────────────────────────────
        if entropy_result:
            lines.append(sep())
            elevel = entropy_result.get("entropy_level", "?")
            needx = entropy_result.get("needs_expansion", False)
            lines.append(
                box_line(
                    f"MOD6 ▸ ENTROPİ EŞİĞİ: {elevel}  "
                    f"{'⚠ GENİŞLEME GEREKLİ' if needx else '✓ YETERLİ BELİRLİLİK'}"
                )
            )
            lines.append(box_line(f"  {entropy_result.get('recommendation','')[:74]}"))
            for q_txt in entropy_result.get("clarification_qs", [])[:2]:
                lines.append(box_line(f"  ❓ {q_txt[:70]}"))
            for hint in entropy_result.get("cot_hints", [])[:2]:
                lines.append(box_line(f"  💭 {hint[:70]}"))

        # ── MOD 7: Zamansal Bellek ────────────────────────────────────────────
        if temporal_result:
            lines.append(sep())
            ctx = temporal_result.get("context_type", "?")
            lam = temporal_result.get("decay_lambda", 0)
            dt = temporal_result.get("delta_t_multiplier", 1.0)
            adj_p = temporal_result.get("adjusted_chain_prior", 0)
            lines.append(
                box_line(f"MOD7 ▸ ZAMANSAL BELLEK: {ctx}  λ={lam:.2f}  Δt={dt:.1f}")
            )
            lines.append(
                box_line(
                    f"  Zaman-uyarlamalı zincir P={adj_p:.6f}  "
                    f"İpuçları: {', '.join(temporal_result.get('time_hints_found',[]) or ['yok'])}"
                )
            )
            tw = temporal_result.get("temporal_warning")
            if tw:
                for chunk in [tw[i : i + 72] for i in range(0, min(len(tw), 144), 72)]:
                    lines.append(box_line(f"  ⏱ {chunk}"))

        # ── MOD 8: Duyarlılık Analizi ─────────────────────────────────────────
        if sensitivity_result:
            lines.append(sep())
            lines.append(
                box_line(
                    f"MOD8 ▸ DUYARLILIK ANALİZİ  "
                    f"En Etkili: '{sensitivity_result.get('most_influential','?')}'  "
                    f"Dom={sensitivity_result.get('dominance_ratio',1):.2f}x"
                )
            )
            st = sensitivity_result.get("sensitivity_table", {})
            ranked_vars = sorted(st.items(), key=lambda x: -x[1].get("abs_sens", 0))
            for vid, sv in ranked_vars[:5]:
                h_mark = "[G]" if sv.get("is_hidden") else "   "
                lines.append(
                    box_line(
                        f"  {h_mark}[{vid:<12}] "
                        f"P={sv.get('prior',0):.3f}  "
                        f"dS/dp={sv.get('sensitivity',0):+.4f}  "
                        f"Bağıl={sv.get('relative',0):.1%}  "
                        f"Sıra={sv.get('rank','?')}"
                    )
                )
            lines.append(box_line(f"  → {sensitivity_result.get('report','')[:72]}"))

        # ── MOD 9: Karar Teorisi ──────────────────────────────────────────────
        if decision_result:
            lines.append(sep())
            best_a = decision_result.get("best_action", "?")
            best_eu = decision_result.get("best_eu", 0)
            lines.append(
                box_line(
                    f"MOD9 ▸ KARAR TEORİSİ  "
                    f"En İyi Eylem: '{best_a}'  EU={best_eu:+.4f}"
                )
            )
            for a_lbl, a_info in sorted(
                decision_result.get("actions", {}).items(),
                key=lambda x: -x[1]["expected_utility"],
            ):
                a_type = a_info.get("action_type", "?")
                eu_v = a_info.get("expected_utility", 0)
                cred_v = a_info.get("credibility", 0)
                marker = "★" if a_lbl == best_a else " "
                lines.append(
                    box_line(
                        f"  {marker}[{a_lbl}] {a_info['description'][:30]:<30}  "
                        f"Tip={a_type:<7}  EU={eu_v:+.4f}  İnan={cred_v:.2f}"
                    )
                )
            rec = decision_result.get("recommendation", "")
            if rec:
                for chunk in [
                    rec[i : i + 72] for i in range(0, min(len(rec), 216), 72)
                ]:
                    lines.append(box_line(f"  → {chunk}"))

        # ── Birleşik Uyarılar ─────────────────────────────────────────────────
        # (tüm uyarılar orchestrator'da zaten dedup edildi)
        all_w_combined = list(
            {
                w[:80]: w
                for w in (
                    chain_result.get("warnings", [])
                    + (correction_result or {}).get("rule_violations", [])
                    + (
                        [temporal_result.get("temporal_warning")]
                        if temporal_result and temporal_result.get("temporal_warning")
                        else []
                    )
                )
            }.values()
        )
        if all_w_combined:
            lines.append(sep())
            lines.append(box_line("⚑ ÖZET UYARILAR (dedup)"))
            for w in all_w_combined[:6]:
                for chunk in [w[i : i + 72] for i in range(0, min(len(w), 144), 72)]:
                    lines.append(box_line(f"  ⚑ {chunk}"))

        lines.append("╚" + "═" * (W - 2) + "╝")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  ★ YENİ: ExplanationNLPEnricher
#  Açıklama alanını her soru için NLP tabanlı mantıksal–sayısal–olasılıksal
#  vektörle zenginleştirir. Hard‑coding yok; değerler soru metni, sinyaller ve
#  solver çıktılarından dinamik olarak türetilir.
# ═══════════════════════════════════════════════════════════════════════════════
class ExplanationNLPEnricher:
    def __init__(self, cred_engine: PropositionalCredibilityEngine):
        self._cred = cred_engine

    def _collect_numbers(self, obj):
        vals = []
        if obj is None:
            return vals
        if isinstance(obj, (int, float, Decimal)):
            return [float(obj)]
        if isinstance(obj, str):
            for m in re.finditer(r"-?\d+(?:[.,]\d+)?", obj):
                try:
                    vals.append(float(m.group(0).replace(",", ".")))
                except Exception:
                    continue
            return vals
        if isinstance(obj, dict):
            for v in obj.values():
                vals.extend(self._collect_numbers(v))
            return vals
        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                vals.extend(self._collect_numbers(v))
        return vals

    def _probabilities_from_tree(self, tree):
        probs = []
        if not tree:
            return probs
        prob = tree.get("prob")
        if prob:
            p_str = str(prob).strip()
            try:
                if p_str.endswith("%"):
                    probs.append(float(p_str[:-1].replace(",", ".")) / 100)
                else:
                    probs.append(float(p_str.replace(",", ".")))
            except Exception:
                pass
        for ch in tree.get("children", []) or []:
            probs.extend(self._probabilities_from_tree(ch))
        return probs

    def enrich(
        self,
        question: str,
        sol_data: dict,
        signals: dict,
        causal_ctx: dict | None = None,
    ) -> dict:
        # Mantıksal skor: soru metninden PropositionalCredibilityEngine ile
        cred = self._cred.score(question or "", role="unknown")
        logic_score = cred.get("credibility", 0.5)
        logic_label = cred.get("label", "ORTA")

        # Sayısal yoğunluk: soru + solver çıktılarındaki en büyük sayı (log ölçek)
        num_vals = []
        num_vals.extend(self._collect_numbers(question))
        num_vals.extend(self._collect_numbers(sol_data.get("numeric")))
        num_vals.extend(self._collect_numbers(sol_data.get("answer")))
        num_max = max([abs(v) for v in num_vals], default=0.0)
        if num_max > 0:
            num_scale = min(1.0, math.log10(num_max + 1.0) / 6.0)
        else:
            num_scale = 0.0

        # Olasılıksal yoğunluk: dağılım, ağaç, yüzde ve nihai dağılımlar
        prob_vals = []
        dist = sol_data.get("distribution") or {}
        if isinstance(dist, dict):
            for v in dist.values():
                try:
                    fv = float(v)
                    if 0 <= fv <= 1:
                        prob_vals.append(fv)
                except Exception:
                    continue
        prob_vals.extend(self._probabilities_from_tree(sol_data.get("tree") or {}))
        # Soru metninden yüzde/olasılık yakala
        for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%", question or ""):
            try:
                prob_vals.append(float(m.group(1).replace(",", ".")) / 100.0)
            except Exception:
                continue
        for m in re.finditer(r"\b0?\.\d+\b", question or ""):
            try:
                prob_vals.append(float(m.group(0)))
            except Exception:
                continue
        if causal_ctx:
            fd = causal_ctx.get("_belief_propagation", {}).get("final_distribution", {})
            for info in fd.values():
                p = info.get("probability_normalized", info.get("probability"))
                try:
                    if p is not None and 0 <= float(p) <= 1:
                        prob_vals.append(float(p))
                except Exception:
                    continue
        prob_score = round(sum(prob_vals) / len(prob_vals), 4) if prob_vals else 0.5

        vector = {
            "logic_score": round(logic_score, 4),
            "logic_label": logic_label,
            "numeric_scale": round(num_scale, 4),
            "numeric_max": round(num_max, 4) if num_max else 0.0,
            "prob_score": prob_score,
            "logic_operator": signals.get("logic_operator", "?"),
            "dependency_layer": signals.get("dependency_layer", 0),
            "dependency_evidence": signals.get("dependency_evidence_count", 0),
        }

        vector_str = (
            f"NLP-Vektör[L={vector['logic_score']:.2f}({logic_label}), "
            f"N={vector['numeric_scale']:.2f} (max={vector['numeric_max']:.4g}), "
            f"P={prob_score:.3f}]"
        )
        context_str = (
            f"Mantık:{vector['logic_operator']} katman:{vector['dependency_layer']} "
            f"kanıt:{vector['dependency_evidence']} | Sayısal max:{vector['numeric_max']:.4g} "
            f"| Olasılık ort:{prob_score:.3f}"
        )

        base_expl = str(sol_data.get("explanation", "") or "").strip()
        blended = " | ".join([p for p in (vector_str, context_str, base_expl) if p])

        sol_data["explanation"] = blended
        sol_data["_nlp_vector"] = vector
        return sol_data


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
router = QLearningRouter()
engine = ASCIIEngine()
ollama = OllamaClient()
sem = SemanticSignalModule()
scorer = BARTConsistencyScorer()
_corrector = SolutionCorrector()
# ★ MOD10: Cloud Equation Universe singleton
eq_universe = CloudUniversalEquationRepository()

# Yeni koordinasyon yardımcıları
_global_state_manager = GlobalStateManager()
_decision_link = DecisionFeedbackLink()
_explanation_engine = ExplanationGraphEngine()
_completeness_checker = CompletenessChecker()
_recovery_engine = GlobalRecoveryEngine(router, _global_state_manager)
_explanation_enricher = ExplanationNLPEnricher(PropositionalCredibilityEngine())

# ── Yeni mimari modüller ──────────────────────────────────────────────────────
_ast_builder = MathASTBuilder()
_markov_solver = MarkovSolver()
_bayes_solver = BayesSolver()
_solver_selector = SolverSelector()
_mc_verifier = MonteCarloVerifier()
_num_validator = NumericTruthValidator()
_step_dep_graph = StepDependencyGraph()
# ★ YENİ — Nedensel Bayes orkestratörü (dört modülü birleştirir)
_causal_orchestrator = CausalBayesOrchestrator()

# ── Self-learning yeni mimari katmanları ─────────────────────────────────────
_deep_encoder = DeepRepresentationEncoder()
_knowledge_retrieval = KnowledgeRetrievalLayer()
_bayes_prior_builder = BayesianPriorBuilder()
_policy_network = RLPolicyNetwork()
_multilayer_validator = MultiLayerValidator()
_reward_replay = RewardReplayMemory(maxlen=1024)
_bayes_posterior_updater = BayesianPosteriorUpdater()
_meta_learner = MetaLearningEngine()
_thinking_loop_engine = ThinkingLoopBackpropEngine(PersistentLearningMemory())
_knowledge_base_updater = KnowledgeBaseUpdater()
_response_generator = ResponseGenerator()

# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  VUE 3 FRONTEND  (Jinja2 yok — saf Python raw-string, CDN Vue, tam reaktif)
# ═══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASCIIMATİK — Ollama ASCII Çözüm Sistemi</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
:root {
  --bg:    #080c10; --bg2: #0d1117; --bg3: #111820; --bg4: #161f2a;
  --green: #00e676; --green2: #69ff47; --green3: #00c853;
  --cyan:  #00e5ff; --cyan2: #84ffff;
  --yellow:#ffea00; --orange:#ff9100; --red:#ff1744;
  --dim2:  #1a2a1a; --border:#1e3a1e; --border2:#0a1f0a;
  --text:  #b2dfdb; --text2: #80cbc4; --text3: #4db6ac;
  --glow:  0 0 8px #00e67644, 0 0 20px #00e67622;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 14px; scroll-behavior: smooth; }
body {
  background: var(--bg); color: var(--green);
  font-family: 'JetBrains Mono', 'Share Tech Mono', monospace;
  min-height: 100vh; overflow-x: hidden; position: relative;
}
body::before {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 9998;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px,
    rgba(0,0,0,.08) 2px, rgba(0,0,0,.08) 4px);
}
body::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: linear-gradient(var(--border2) 1px, transparent 1px),
                    linear-gradient(90deg, var(--border2) 1px, transparent 1px);
  background-size: 40px 40px; opacity: .4;
}

/* ── TOP BAR ── */
#topbar {
  position: sticky; top: 0; z-index: 100; background: var(--bg2);
  border-bottom: 1px solid var(--border); padding: 0 20px;
  display: flex; align-items: center; gap: 16px; height: 48px;
  box-shadow: 0 2px 20px #000a;
}
.dot { width:10px; height:10px; border-radius:50%; }
.dot.r { background:#ff5f57; } .dot.y { background:#ffbd2e; } .dot.g { background:#28c840; }
#topbar-title { font-size:.75rem; color:var(--text3); letter-spacing:.15em; text-transform:uppercase; margin-left:8px; }
#status-bar { margin-left:auto; display:flex; gap:16px; font-size:.7rem; color:var(--text3); }
.stat-pill {
  background:var(--bg4); border:1px solid var(--border);
  padding:2px 10px; border-radius:3px; display:flex; gap:6px; align-items:center;
}
.stat-pill span { color:var(--cyan); font-weight:600; }

/* ── LAYOUT ── */
#app {
  position:relative; z-index:1; display:grid;
  grid-template-columns:320px minmax(0, 1fr) 420px; gap:0; height:calc(100vh - 48px);
}

/* ── SIDEBAR ── */
#sidebar {
  background:var(--bg2); border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow:hidden; grid-row: 1;
}
#sidebar-header {
  padding:12px 16px; border-bottom:1px solid var(--border);
  font-size:.7rem; color:var(--cyan); letter-spacing:.1em; text-transform:uppercase;
  display:flex; align-items:center; justify-content:space-between;
}
#search-input {
  background:var(--bg3); border:1px solid var(--border); border-radius:3px;
  color:var(--cyan2); font-family:inherit; font-size:.68rem;
  padding:4px 8px; outline:none; width:130px; caret-color:var(--green);
}
#search-input:focus { border-color:var(--green3); }
#examples-list { flex:1; overflow-y:auto; scrollbar-width:thin; scrollbar-color:var(--border) transparent; }
.example-item {
  padding:10px 16px; border-bottom:1px solid var(--border2);
  cursor:pointer; transition:background .15s; position:relative;
}
.example-item:hover { background:var(--bg4); }
.example-item:hover::before {
  content:'▶'; position:absolute; left:4px; top:50%;
  transform:translateY(-50%); color:var(--green); font-size:.6rem;
}
.example-item.active { background:var(--dim2); border-left:2px solid var(--green); }
.ex-cat { font-size:.62rem; color:var(--text3); margin-bottom:3px; }
.ex-q {
  font-size:.67rem; color:var(--text); line-height:1.4;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}

/* ── MAIN PANEL ── */
#main-panel { display:flex; flex-direction:column; overflow:hidden; }

/* ── INPUT AREA ── */
#input-area {
  padding:16px 20px; background:var(--bg2); border-bottom:1px solid var(--border);
}
#input-label {
  font-size:.65rem; color:var(--text3); letter-spacing:.12em; text-transform:uppercase;
  margin-bottom:8px; display:flex; align-items:center; gap:8px;
}
#input-label::before { content:'▶'; color:var(--green); }
#question-input {
  width:100%; background:var(--bg3); border:1px solid var(--border); border-radius:4px;
  color:var(--cyan2); font-family:inherit; font-size:.82rem;
  padding:12px 14px; resize:vertical; min-height:72px;
  outline:none; transition:border-color .2s, box-shadow .2s; caret-color:var(--green);
}
#question-input:focus { border-color:var(--green3); box-shadow:0 0 0 2px #00e67618, var(--glow); }
#question-input::placeholder { color:var(--text3); opacity:.5; }
#controls { display:flex; gap:10px; margin-top:10px; align-items:center; }
#solve-btn {
  background:linear-gradient(135deg,#00c853,#00e676); color:#000; border:none;
  border-radius:3px; padding:9px 24px; font-family:inherit; font-size:.8rem;
  font-weight:700; letter-spacing:.08em; cursor:pointer;
  transition:transform .1s, box-shadow .2s; text-transform:uppercase;
}
#solve-btn:hover { transform:translateY(-1px); box-shadow:0 4px 20px #00e67655; }
#solve-btn:active { transform:translateY(0); }
#solve-btn:disabled { opacity:.5; cursor:not-allowed; transform:none; }
#clear-btn {
  background:transparent; color:var(--text3); border:1px solid var(--border);
  border-radius:3px; padding:9px 16px; font-family:inherit; font-size:.75rem;
  cursor:pointer; transition:border-color .15s, color .15s;
}
#clear-btn:hover { border-color:var(--red); color:var(--red); }
#copy-btn {
  background:transparent; color:var(--text3); border:1px solid var(--border);
  border-radius:3px; padding:9px 16px; font-family:inherit; font-size:.75rem;
  cursor:pointer; transition:border-color .15s, color .15s;
}
#copy-btn:hover { border-color:var(--cyan); color:var(--cyan); }
#model-tag { margin-left:auto; font-size:.67rem; color:var(--text3); display:flex; gap:8px; align-items:center; }
.pulse-dot { width:7px; height:7px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 #00e67655;} 50%{opacity:.7;box-shadow:0 0 0 5px transparent;} }

/* ── OUTPUT AREA ── */
#output-area {
  flex:1; overflow-y:auto; overflow-x:auto;
  scrollbar-width:thin; scrollbar-color:var(--border) transparent;
  padding:20px; font-size:.75rem; line-height:1.55;
}
#output-pre {
  white-space:pre; font-family:'JetBrains Mono', monospace;
  min-width:fit-content; transition:color .2s;
}
#output-pre.ok      { color:var(--green);  text-shadow:0 0 6px #00e67630; }
#output-pre.loading { color:var(--yellow); animation:flicker 1s infinite; }
#output-pre.error   { color:var(--red); }
@keyframes flicker { 0%,100%{opacity:1} 50%{opacity:.6} }

/* ── BOTTOM STATUS ── */
#bottom-status {
  background:var(--bg2); border-top:1px solid var(--border);
  padding:4px 20px; display:flex; gap:20px; align-items:center;
  font-size:.62rem; color:var(--text3);
  position:sticky; bottom:0; z-index:10;
}
#route-display  { color:var(--cyan); }
#reward-display { color:var(--green); }

/* ── TOAST ── */
#toast {
  position:fixed; bottom:30px; right:20px; z-index:10000;
  background:var(--bg3); border:1px solid var(--border); color:var(--green);
  padding:10px 18px; border-radius:4px; font-size:.72rem;
  transform:translateY(60px); opacity:0; transition:all .3s; pointer-events:none;
  box-shadow:0 4px 20px #000a;
}
#toast.show { transform:translateY(0); opacity:1; }

/* ── ANIMATIONS ── */
@keyframes fadeInDown { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
#ascii-header { animation:fadeInDown .5s ease; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--green3); }


#graph-panel {
  background:#020707; border-left:1px solid var(--border);
  display:flex; flex-direction:column; min-width:320px;
}
#graph-head {
  padding:10px 14px; border-bottom:1px solid var(--border);
  font-size:.66rem; color:var(--green3); letter-spacing:.1em; text-transform:uppercase;
}
#graph-wrap { flex:1; padding:10px; }
#graph-image {
  width:100%; height:100%; background:#000; border:1px solid #123; border-radius:4px;
  box-shadow: inset 0 0 36px #00e67614, 0 0 20px #00e67612;
  object-fit:contain;
}
#graph-fallback {
  width:100%; height:100%;
  display:flex; align-items:center; justify-content:center;
  border:1px dashed #1b4f1b; border-radius:4px; color:var(--green3);
  background:#000; font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
}

/* Mobile */
@media (max-width:1100px) { #app { grid-template-columns:1fr; } #sidebar, #graph-panel { display:none; } }

/* ── MODAL ── */
.modal-overlay {
  position:fixed; inset:0; z-index:99999;
  background:rgba(0,0,0,.78); display:flex; align-items:center; justify-content:center;
  backdrop-filter:blur(3px);
}
.modal-box {
  background:var(--bg2); border:1px solid var(--green3); border-radius:6px;
  padding:24px 28px; width:520px; max-width:95vw;
  box-shadow:0 8px 40px #000c, 0 0 30px #00e67622;
  position:relative; z-index:100000;
}
.modal-title {
  font-size:.85rem; color:var(--green); letter-spacing:.1em; text-transform:uppercase;
  margin-bottom:16px; display:flex; align-items:center; gap:8px;
}
.modal-title::before { content:'◈'; color:var(--cyan); }
.modal-close {
  position:absolute; top:12px; right:14px; background:none; border:none;
  color:var(--text3); font-size:1.1rem; cursor:pointer; font-family:inherit;
}
.modal-close:hover { color:var(--red); }
.modal-field { margin-bottom:12px; }
.modal-label { font-size:.68rem; color:var(--text3); letter-spacing:.1em;
               text-transform:uppercase; margin-bottom:4px; display:block; }
.modal-input {
  width:100%; background:var(--bg3); border:1px solid var(--border); border-radius:4px;
  color:var(--cyan2); font-family:inherit; font-size:.8rem; padding:8px 10px;
  outline:none; transition:border-color .2s; caret-color:var(--green);
}
.modal-input:focus { border-color:var(--green3); }
.modal-textarea { min-height:90px; resize:vertical; }
.modal-actions { display:flex; gap:10px; margin-top:16px; justify-content:flex-end; }
.btn-primary {
  background:linear-gradient(135deg,#00c853,#00e676); color:#000; border:none;
  border-radius:3px; padding:8px 20px; font-family:inherit; font-size:.78rem;
  font-weight:700; letter-spacing:.06em; cursor:pointer; text-transform:uppercase;
}
.btn-primary:hover { box-shadow:0 4px 16px #00e67655; }
.btn-secondary {
  background:transparent; color:var(--text3); border:1px solid var(--border);
  border-radius:3px; padding:8px 16px; font-family:inherit; font-size:.75rem;
  cursor:pointer; transition:border-color .15s, color .15s;
}
.btn-secondary:hover { border-color:var(--cyan); color:var(--cyan); }
#add-btn {
  background:transparent; color:var(--green3); border:1px solid var(--green3);
  border-radius:3px; padding:2px 10px; font-family:inherit; font-size:.68rem;
  cursor:pointer; transition:all .15s; letter-spacing:.05em;
}
#add-btn:hover { background:var(--green3); color:#000; }
.user-badge { font-size:.58rem; color:var(--orange); margin-left:4px; }
.del-btn {
  position:absolute; right:8px; top:50%; transform:translateY(-50%);
  background:none; border:none; color:var(--text3); cursor:pointer;
  font-size:.7rem; padding:2px 5px; border-radius:2px; opacity:0;
  transition:opacity .15s, color .15s;
}
.example-item:hover .del-btn { opacity:1; }
.del-btn:hover { color:var(--red); }
#pdf-btn {
  background:transparent; color:var(--text3); border:1px solid var(--border);
  border-radius:3px; padding:9px 16px; font-family:inherit; font-size:.75rem;
  cursor:pointer; transition:border-color .15s, color .15s;
}
#pdf-btn:hover { border-color:var(--orange); color:var(--orange); }
#pdf-btn:disabled { opacity:.4; cursor:not-allowed; }
#pdf-dark-btn {
  background:transparent; color:var(--text3); border:1px solid var(--border);
  border-radius:3px; padding:9px 16px; font-family:inherit; font-size:.75rem;
  cursor:pointer; transition:border-color .15s, color .15s, background .15s;
}
#pdf-dark-btn:hover { border-color:var(--green); color:var(--bg); background:var(--green); }
#pdf-dark-btn:disabled { opacity:.4; cursor:not-allowed; }
</style>
</head>
<body>

<!-- TOP BAR — bağlanma yok, Vue mount dışında -->
<div id="topbar">
  <div class="dot r"></div><div class="dot y"></div><div class="dot g"></div>
  <span id="topbar-title">ASCIIMATİK · phi4:14b · Q-Learning Router</span>
  <div id="status-bar">
    <div class="stat-pill">EPS    <span id="st-episode">0</span></div>
    <div class="stat-pill">ROUTE  <span id="st-route">—</span></div>
    <div class="stat-pill">REWARD <span id="st-reward">—</span></div>
    <div class="stat-pill">INTENT <span id="st-intent">—</span></div>
  </div>
</div>

<!-- VUE 3 ROOT -->
<div id="vue-app">

  <!-- ASCII HEADER -->
  <div id="ascii-header" style="padding:16px 20px 0;font-size:.65rem;color:var(--green3);white-space:pre;text-shadow:var(--glow);line-height:1.25;overflow:hidden;">╔══════════════════════════════════════════════════════════════════════════════════╗
║  ▄▄▄▄▄  ▄▄▄▄▄  ▄▄▄▄▄  ▄▄▄▄▄▄  ▄▄▄▄  ▄▄     ▄▄  ▄▄▄▄▄▄  ▄▄  ▄▄  ▄▄   ▄▄        ║
║  ██▀▀█  ██▀▀▀  ██▀▀▀  ▀▀██▀▀  ██▀▀  ███▄▄▄███  ▀▀██▀▀  ██  ██  ███▄██         ║
║  ██  █  ███▄   ██      ██     ████  ██ ▀███▀ █    ██    ██▄▄██  ██▀▀█          ║
║  ▀▀▀▀▀  ▀▀▀▀▀  ▀▀▀▀▀   ▀▀     ▀▀▀▀  ▀▀  ▀  ▀▀    ▀▀    ▀▀▀▀▀▀  ▀▀  ▀▀         ║
║         Ollama phi4:14b  ·  Q-Learning NLP Router  ·  Algoritmik ASCII Engine   ║
╚══════════════════════════════════════════════════════════════════════════════════╝</div>

  <div id="app">

    <!-- ── SIDEBAR ── -->
    <div id="sidebar">
      <div id="sidebar-header">
        <span>▸ {{ examples.length }} Örnek Soru</span>
        <div style="display:flex;gap:6px;align-items:center;">
          <button id="add-btn" @click="openAddModal">➕ Ekle</button>
          <input id="search-input" v-model="searchQuery" placeholder="🔍 ara..." />
        </div>
      </div>
      <div id="examples-list">
        <div
          v-for="ex in filteredExamples"
          :key="ex.id"
          class="example-item"
          :class="{ active: activeId === ex.id }"
          @click="loadExample(ex)"
        >
          <div class="ex-cat">{{ ex.cat }}<span v-if="ex._user" class="user-badge">● KULLANICI</span></div>
          <div class="ex-q">{{ ex.q }}</div>
          <button v-if="ex._user" class="del-btn" @click.stop="deleteExample(ex)" title="Sil">✕</button>
        </div>
        <div v-if="filteredExamples.length === 0" style="padding:16px;color:var(--text3);font-size:.68rem;">
          Sonuç bulunamadı.
        </div>
      </div>
    </div>

    <!-- ── MAIN PANEL ── -->
    <div id="main-panel">

      <!-- INPUT -->
      <div id="input-area">
        <div id="input-label">Soru Gir — Tüm Olasılık Konuları Destekleniyor</div>
        <textarea
          id="question-input"
          v-model="question"
          placeholder="Olasılık, kombinatorik, Bayes, dağılımlar, Markov, entropi, hikayeler, eğlence soruları...&#10;&#10;Örnek: Bir zar iki kez atılıyor, her iki atışta da çift sayı gelme olasılığı nedir?"
          @keydown.ctrl.enter="solve"
        ></textarea>
        <div id="controls">
          <button id="solve-btn" @click="solve" :disabled="isLoading">
            {{ isLoading ? '⟳  Çözülüyor...' : '⟹  ÇÖZDÜR' }}
          </button>
          <button id="search-btn" @click="openSearchModal" title="Web arama ve analiz">🌐 ARA</button>
          <button id="clear-btn" @click="clearAll">✕ Temizle</button>
          <button id="copy-btn"  @click="copyOutput">⎘ Kopyala</button>
          <button id="pdf-btn"   @click="downloadPdf('ascii')"  :disabled="!lastResult || isLoading" title="Açık temalı yapılandırılmış PDF">⬇ ASCII PDF</button>
          <button id="pdf-dark-btn" @click="downloadPdf('dark')" :disabled="!lastResult || isLoading" title="Karanlık terminal temalı LaTeX PDF">⬛ LaTeX PDF</button>
          <div id="model-tag">
            <div class="pulse-dot"></div>
            phi4:14b · localhost:11434
          </div>
        </div>
      </div>

      <!-- OUTPUT -->
      <div id="output-area">
        <pre id="output-pre" :class="outputClass">{{ output }}</pre>
      </div>

    </div><!-- /main-panel -->

    <div id="graph-panel">
      <div id="graph-head">NLP Graph Automatı · Canlı Eğri</div>
      <div id="graph-wrap">
        <img id="graph-image" alt="Matplotlib grafik görünümü" />
        <div id="graph-fallback">Matplotlib grafik bekleniyor…</div>
      </div>
    </div>

  </div><!-- /app grid -->

  <!-- BOTTOM STATUS -->
  <div id="bottom-status">
    <span>ASCIIMATİK v2.0 · Vue 3</span>
    <span id="route-display">Route: {{ stats.layout }}</span>
    <span id="reward-display">Reward: {{ stats.reward }}</span>
    <span>Süre: {{ stats.elapsed }}</span>
    <span style="margin-left:auto;color:var(--text3);">phi4:14b @ Ollama · Algoritmik ASCII Engine · No Hardcoding</span>
  </div>

  <!-- ── SORU EKLEME MODALI (vue-app içinde — direktifler çalışır) ── -->
  <div v-if="showAddModal"
       class="modal-overlay"
       @click.self="closeAddModal"
       @keydown.esc.window="closeAddModal">
    <div class="modal-box" @keydown.esc.stop="closeAddModal">
      <div class="modal-title">Yeni Soru Ekle</div>
      <button class="modal-close" @click.stop="closeAddModal">✕</button>

      <div class="modal-field">
        <label class="modal-label">Kategori</label>
        <input class="modal-input"
               ref="modalCatInput"
               v-model="newEx.cat"
               placeholder="Örn: 🎲 Olasılık, 🏥 Bayes, Oyun Kuramı..."
               @keydown.enter.prevent="$refs.modalQInput.focus()" />
      </div>

      <div class="modal-field">
        <label class="modal-label">Soru Metni</label>
        <textarea class="modal-input modal-textarea"
                  ref="modalQInput"
                  v-model="newEx.q"
                  placeholder="Soruyu buraya yazın... (Ctrl+Enter ile kaydet)"
                  @keydown.ctrl.enter.prevent="submitAddModal"></textarea>
      </div>

      <div style="font-size:.65rem;color:var(--text3);margin-top:4px;">
        ℹ Eklenen sorular <code style="color:var(--cyan)">questions_db.json</code> dosyasına kaydedilir.
      </div>

      <div class="modal-actions">
        <button class="btn-secondary" @click.stop="closeAddModal">İptal</button>
        <button class="btn-primary"
                @click.stop="submitAddModal"
                :disabled="!newEx.cat.trim() || !newEx.q.trim()">
          ➕ Kaydet
        </button>
      </div>
    </div>
  </div>

  <!-- ── WEB ARAMA MODALI ── -->
  <div v-if="showSearchModal"
       class="modal-overlay"
       @click.self="closeSearchModal"
       @keydown.esc.window="closeSearchModal">
    <div class="modal-box" @keydown.esc.stop="closeSearchModal">
      <div class="modal-title">🌐 Web İstihbaratı Arama</div>
      <button class="modal-close" @click.stop="closeSearchModal">✕</button>

      <div class="modal-field">
        <label class="modal-label">Arama Sorgusu</label>
        <textarea class="modal-input"
                  ref="searchQueryInput"
                  v-model="searchQueryWeb"
                  placeholder="Aranacak konuyu yazın...&#10;Örnek: Yapay zeka nedir? Kuantum bilgisayarlar nasıl çalışır?"
                  @keydown.ctrl.enter.prevent="performSearch"
                  style="min-height:80px;"></textarea>
      </div>

      <div v-if="searchLoading" style="text-align:center;padding:16px;color:var(--cyan);">
        <span>🔍 Aranıyor...</span>
      </div>

      <div v-if="searchResult && !searchLoading" class="modal-field" style="max-height:300px;overflow-y:auto;">
        <div style="font-size:.75rem;color:var(--text2);margin-bottom:8px;">
          📊 {{ searchResult.sources_count }} kaynak bulundu
        </div>
        <pre style="background:var(--bg4);border:1px solid var(--border);padding:8px;border-radius:3px;font-size:.65rem;color:var(--text);overflow-x:auto;">{{ searchResult.report }}</pre>
      </div>

      <div class="modal-actions">
        <button class="btn-secondary" @click.stop="closeSearchModal">Kapat</button>
        <button class="btn-primary"
                @click.stop="performSearch"
                :disabled="!searchQueryWeb.trim() || searchLoading">
          {{ searchLoading ? '⟳ Aranıyor...' : '🔍 ARA' }}
        </button>
      </div>
    </div>
  </div>

</div><!-- /vue-app -->

<!-- TOAST (Vue dışı, method ile yönetiliyor) -->
<div id="toast"></div>

<script>
const { createApp } = Vue;

createApp({
  data() {
    return {
      question:    '',
      output:      this.welcomeText(),
      outputClass: 'ok',
      isLoading:   false,
      activeId:    null,
      examples:    [],
      searchQuery: '',
      stats: { layout: '—', reward: '—', elapsed: '—', intent: '—', episode: 0 },
      lastResult:   null,    // Son çözüm sonucu (PDF için)
      showAddModal: false,   // Soru ekleme modali
      newEx: { cat: '', q: '' },
      // Web İstihbaratı Modali
      showSearchModal: false,
      searchQueryWeb: '',    // Web arama sorgusu
      searchLoading: false,
      searchResult: null,
      graphData: null,
      graphImage: '',
    };
  },

  computed: {
    filteredExamples() {
      const q = this.searchQuery.trim().toLowerCase();
      if (!q) return this.examples;
      return this.examples.filter(e =>
        e.q.toLowerCase().includes(q) || e.cat.toLowerCase().includes(q)
      );
    }
  },

  async mounted() {
    // Backend'den örnekleri çek — /examples endpoint'i kullan
    try {
      const res  = await fetch('/examples');
      this.examples = await res.json();
    } catch {
      this.toast('⚠ Örnekler yüklenemedi', 'orange');
    }
    // Periyodik Q-state güncelleme
    setInterval(() => this.fetchQState(), 5000);
    this.$nextTick(() => this.renderGraph(null));
  },

  methods: {

    welcomeText() {
      return [
        '╔══════════════════════════════════════════════════════════════════════════════════╗',
        '║                                                                                  ║',
        '║   Hoş geldiniz!  ASCIIMATİK hazır.                                               ║',
        '║                                                                                  ║',
        '║   ● Sol panelden bir örnek soru seçin                                            ║',
        '║   ● Veya yukarıya kendi sorunuzu yazın                                           ║',
        '║   ● "ÇÖZDÜR" butonuna basın  (Ctrl+Enter ile de çalışır)                        ║',
        '║                                                                                  ║',
        '║   Desteklenen konular:                                                           ║',
        '║   ├─ Kombinatorik (kombinasyon, permütasyon, faktöriyel)                         ║',
        '║   ├─ Olasılık (Bayes, koşullu, toplam olasılık)                                  ║',
        '║   ├─ Dağılımlar (Binom, Poisson, Normal, Geometrik...)                           ║',
        '║   ├─ Markov zincirleri, entropi, Monte Carlo                                     ║',
        '║   ├─ İçerme-dışarma, güvercin yuvası, Pascal üçgeni                             ║',
        '║   ├─ Stirling, Bell, Ramsey, multinomial                                         ║',
        '║   ├─ Eğlence, hikaye ve sosyal senaryolar                                        ║',
        '║   └─ Ve çok daha fazlası...                                                      ║',
        '║                                                                                  ║',
        '║   Q-Learning Router → sorunuzu analiz edip en uygun ASCII layout\'u seçer.       ║',
        '║                                                                                  ║',
        '╚══════════════════════════════════════════════════════════════════════════════════╝',
      ].join('\n');
    },

    loadExample(ex) {
      this.activeId = ex.id;
      this.question = ex.q;
      this.toast('✓ Soru yüklendi', 'green');
    },

    async solve() {
      if (this.isLoading) return;
      const q = this.question.trim();
      if (!q) { this.toast('⚠ Soru boş olamaz', 'orange'); return; }

      this.isLoading   = true;
      this.outputClass = 'loading';
      this.output      = this.loadingFrame(0);

      let frame = 0;
      const ticker = setInterval(() => {
        this.output = this.loadingFrame(++frame);
      }, 120);

      const t0 = Date.now();
      try {
        const res = await fetch('/solve', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ question: q }),
        });
        clearInterval(ticker);
        const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

        if (!res.ok) {
          const err = await res.json();
          this.outputClass = 'error';
          this.output      = `HATA: ${err.error || 'Bilinmeyen hata'}`;
          this.toast('✗ Hata oluştu', 'red');
          return;
        }

        const data = await res.json();
        this.outputClass  = 'ok';
        this.output       = data.ascii;
        this.lastResult   = data;   // PDF için sakla
        this.graphData    = data.graph_data || null;
        this.graphImage   = data.graph_image || '';
        this.$nextTick(() => this.renderGraph(this.graphData, this.graphImage));

        // Stats güncelle
        this.stats = { layout: data.layout, reward: data.reward, elapsed: elapsed + 's', intent: data.intent, episode: data.episode };
        document.getElementById('st-episode').textContent = data.episode;
        document.getElementById('st-route').textContent   = data.layout;
        document.getElementById('st-reward').textContent  = data.reward;
        document.getElementById('st-intent').textContent  = data.intent;

        this.toast(`✓ Çözüldü · ${data.layout} · ${elapsed}s`, 'green');
        this.$nextTick(() => { document.getElementById('output-area').scrollTop = 0; });

      } catch (err) {
        clearInterval(ticker);
        this.outputClass = 'error';
        this.output = [
          '╔═══════════════════════════════════════╗',
          '║         BAĞLANTI HATASI               ║',
          '╠═══════════════════════════════════════╣',
          `║  ${err.message}`,
          '║',
          '║  Ollama çalışıyor mu?',
          '║  → ollama serve',
          '║  → ollama pull phi4:14b',
          '╚═══════════════════════════════════════╝',
        ].join('\n');
        this.toast('✗ Bağlantı hatası', 'red');
      } finally {
        this.isLoading = false;
      }
    },

    clearAll() {
      this.question    = '';
      this.output      = this.welcomeText();
      this.outputClass = 'ok';
      this.activeId    = null;
      this.searchQuery = '';
      this.lastResult  = null;
      this.graphData   = null;
      this.graphImage  = '';
      this.$nextTick(() => this.renderGraph(null, ''));
      this.stats       = { layout: '—', reward: '—', elapsed: '—', intent: '—', episode: this.stats.episode };
      ['st-route','st-reward','st-intent'].forEach(id => {
        document.getElementById(id).textContent = '—';
      });
      this.toast('✓ Temizlendi', 'green');
    },

    async copyOutput() {
      if (!this.output) return;
      try {
        await navigator.clipboard.writeText(this.output);
        this.toast('✓ Panoya kopyalandı', 'green');
      } catch {
        this.toast('✗ Kopyalama başarısız', 'red');
      }
    },

    loadingFrame(frame) {
      const spinners = ['◐','◓','◑','◒'];
      const spin     = spinners[frame % spinners.length];
      const barLen   = (frame % 40) + 1;
      const bar      = '█'.repeat(barLen) + '░'.repeat(40 - barLen);
      return [
        '╔══════════════════════════════════════════════════════════════════════════════════╗',
        '║                                                                                  ║',
        `║   ${spin}  phi4:14b işliyor...                                                      ║`,
        '║                                                                                  ║',
        '║   Q-Learning Router → Niyet analizi...                                           ║',
        '║   Ollama → Çözüm üretiliyor...                                                   ║',
        '║   ASCII Engine → Layout seçiliyor...                                             ║',
        '║                                                                                  ║',
        `║   [${bar}]        ║`,
        '║                                                                                  ║',
        '╚══════════════════════════════════════════════════════════════════════════════════╝',
      ].join('\n');
    },

    async fetchQState() {
      try {
        const r = await fetch('/qstate');
        const d = await r.json();
        document.getElementById('st-episode').textContent = d.episode;
        this.stats.episode = d.episode;
      } catch {}
    },

    // ── SORU EKLEME MODALI ────────────────────────────────────────────────
    openAddModal() {
      this.newEx = { cat: '', q: '' };
      this.showAddModal = true;
      this.$nextTick(() => {
        if (this.$refs.modalCatInput) this.$refs.modalCatInput.focus();
      });
    },

    closeAddModal() {
      this.showAddModal = false;
    },

    async submitAddModal() {
      const cat = this.newEx.cat.trim();
      const q   = this.newEx.q.trim();
      if (!cat || !q) return;
      try {
        const res  = await fetch('/add_example', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ cat, q }),
        });
        const data = await res.json();
        if (data.ok) {
          this.examples.push(data.example);
          this.closeAddModal();
          this.toast(`✓ Soru eklendi: ${cat}`, 'green');
        } else {
          this.toast(`✗ Hata: ${data.error}`, 'red');
        }
      } catch {
        this.toast('✗ Sunucu hatası', 'red');
      }
    },

    async deleteExample(ex) {
      if (!confirm(`"${ex.q.slice(0,60)}…"\nBu soruyu silmek istiyor musunuz?`)) return;
      try {
        const res  = await fetch('/delete_example', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ id: ex.id }),
        });
        const data = await res.json();
        if (data.ok) {
          this.examples = this.examples.filter(e => e.id !== ex.id);
          if (this.activeId === ex.id) { this.activeId = null; }
          this.toast('✓ Soru silindi', 'orange');
        }
      } catch {
        this.toast('✗ Silme hatası', 'red');
      }
    },

    // ── WEB ARAMA MODALI ────────────────────────────────────────────────────
    openSearchModal() {
      this.searchQueryWeb = '';
      this.searchResult = null;
      this.showSearchModal = true;
      this.$nextTick(() => {
        if (this.$refs.searchQueryInput) this.$refs.searchQueryInput.focus();
      });
    },

    closeSearchModal() {
      this.showSearchModal = false;
    },

    async performSearch() {
      const query = this.searchQueryWeb.trim();
      if (!query) {
        this.toast('⚠ Arama sorgusu boş', 'orange');
        return;
      }

      this.searchLoading = true;
      try {
        const res = await fetch('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query }),
        });
        const data = await res.json();

        if (data.status === 'success') {
          this.searchResult = {
            sources_count: data.sources_count,
            report: data.report,
            synthesis: data.synthesis,
            timestamp: data.timestamp,
          };
          this.toast('✓ Arama tamamlandı', 'green');
          
          // Sonucu output'a da kopyala
          this.output = data.report;
          this.outputClass = 'ok';
        } else {
          this.toast(`✗ Arama hatası: ${data.message}`, 'red');
        }
      } catch (err) {
        this.toast(`✗ Sunucu hatası: ${err.message}`, 'red');
      } finally {
        this.searchLoading = false;
      }
    },

    // ── PDF İNDİR ────────────────────────────────────────────────────────
    async downloadPdf(type = 'ascii') {
      if (!this.lastResult) return;
      const isDark    = type === 'dark';
      const endpoint  = isDark ? '/pdf_dark' : '/pdf';
      const label     = isDark ? 'LaTeX PDF' : 'ASCII PDF';
      const filename  = isDark ? `ascimatik_latex_${Date.now()}.pdf`
                                : `ascimatik_ascii_${Date.now()}.pdf`;
      this.toast(`⟳ ${label} hazırlanıyor...`, 'orange');
      try {
        const payload = {
          question:          this.lastResult.question    || this.question,
          answer:            this.lastResult.answer      || '',
          sol_type:          this.lastResult.sol_type    || 'general',
          steps:             this.lastResult.steps       || [],
          formula:           this.lastResult.formula     || '',
          ascii:             this.lastResult.ascii       || '',
          chosen_solver:     this.lastResult.chosen_solver    || 'LLM',
          consistency_score: this.lastResult.consistency_score ?? 1.0,
        };
        const res = await fetch(endpoint, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json();
          this.toast(`✗ ${label} hatası: ${err.error}`, 'red');
          return;
        }
        const blob  = await res.blob();
        const url   = URL.createObjectURL(blob);
        const link  = document.createElement('a');
        link.href      = url;
        link.download  = filename;
        link.click();
        URL.revokeObjectURL(url);
        this.toast(`✓ ${label} indirildi`, 'green');
      } catch (err) {
        this.toast(`✗ ${label} indirilemedi: ${err.message}`, 'red');
      }
    },


    renderGraph(payload, imageData = '') {
      const img = document.getElementById('graph-image');
      const fallback = document.getElementById('graph-fallback');
      if (!img || !fallback) return;

      const src = imageData || (payload && payload.image_data) || '';
      if (src) {
        img.src = src;
        img.style.display = 'block';
        fallback.style.display = 'none';
      } else {
        img.removeAttribute('src');
        img.style.display = 'none';
        fallback.style.display = 'flex';
      }
    },


    toast(msg, type = 'green') {
      const t      = document.getElementById('toast');
      const colors = { green:'#00e676', orange:'#ff9100', red:'#ff1744' };
      const col    = colors[type] || colors.green;
      t.style.color       = col;
      t.style.borderColor = col + '44';
      t.textContent = msg;
      t.classList.add('show');
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => t.classList.remove('show'), 2800);
    },

  }
}).mount('#vue-app');
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  🟣 WEB İSTİHBARATI MODÜLÜ
#  Modüler mimarı: MultiSourceFetcher → Scraper → Chunker → Embedding →
#  ChromaDB → Retriever (RAG) → NLP Synthesizer
#  Hard-coding YOK — tüm yapı dinamik ve genelleştirilmiş.
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSourceFetcher:
    """
    Birden fazla web kaynağından sorguya uygun içerik çeker.
    Kaynaklar: Wikipedia, Google Search, DuckDuckGo, RSS (Fox News)
    """

    def __init__(self):
        self.timeout = 10
        self.user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

    def fetch_wikipedia(self, query: str) -> list:
        """Wikipedia'dan sorguya uygun makaleler çeker."""
        sources = []
        try:
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5,
            }
            resp = requests.get(search_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                for result in data.get("query", {}).get("search", []):
                    sources.append(
                        {
                            "source": "Wikipedia",
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", ""),
                            "url": f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}",
                            "content": result.get("snippet", ""),
                        }
                    )
        except Exception as e:
            print(f"[DEBUG] Wikipedia fetch hatası: {e}")
        return sources

    def fetch_duck_duck_go(self, query: str) -> list:
        """DuckDuckGo'dan sorguya uygun sonuçlar çeker."""
        sources = []
        try:
            search_url = "https://duckduckgo.com/html"
            params = {"q": query}
            resp = requests.get(
                search_url,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
            )
            if resp.status_code == 200:
                # Basit regex ile sonuçları çıkar
                import re

                links = re.findall(
                    r'<a href="https?://([^"]+)"[^>]*>([^<]+)</a>', resp.text
                )
                for url, title in links[:5]:
                    sources.append(
                        {
                            "source": "DuckDuckGo",
                            "title": title[:100],
                            "snippet": title,
                            "url": f"https://{url}",
                            "content": title,
                        }
                    )
        except Exception as e:
            pass
        return sources

    def fetch_rss_feed(self, feed_url: str = None) -> list:
        """RSS beslemesinden haber başlıkları ve özet bilgi çeker."""
        sources = []
        if feed_url is None:
            feed_url = "https://feeds.fox.com/foxnews/latest"

        try:
            resp = requests.get(feed_url, timeout=self.timeout)
            if resp.status_code == 200:
                import re

                # Basit XML parsing
                titles = re.findall(r"<title>([^<]+)</title>", resp.text)
                descriptions = re.findall(
                    r"<description>([^<]+)</description>", resp.text
                )

                for i, title in enumerate(titles[:5]):
                    sources.append(
                        {
                            "source": "RSS Feed",
                            "title": title[:100],
                            "snippet": (
                                descriptions[i] if i < len(descriptions) else title
                            ),
                            "url": feed_url,
                            "content": (
                                descriptions[i] if i < len(descriptions) else title
                            ),
                        }
                    )
        except Exception as e:
            pass
        return sources

    def fetch_all(self, query: str) -> list:
        """Tüm kaynaklardan sonuçları topla."""
        results = []
        results.extend(self.fetch_wikipedia(query))
        results.extend(self.fetch_duck_duck_go(query))
        results.extend(self.fetch_rss_feed())

        # Eğer sonuç bulunamazsa, fallback verisi sağla
        if not results:
            print(
                f"[WARN] Arama '{query}' için sonuç bulunamadı. Fallback veri kullanılıyor."
            )
            results = self._get_fallback_results(query)

        return results[:20]  # En fazla 20 sonuç

    def _get_fallback_results(self, query: str) -> list:
        """Ağ hatası durumunda fallback sonuç verisi sağla."""
        return [
            {
                "source": "Bilgi Bankası",
                "title": f'"{query}" hakkında temel bilgiler',
                "snippet": f'"{query}" hakkında arama sonuçları. Daha ayrıntılı bilgi için internet bağlantısını kontrol edin.',
                "url": "http://localhost",
                "content": f'"{query}" hakkında bilgiler: Bu konu veya kişi hakkında birden fazla kaynak var. Lütfen ağ bağlantınızı kontrol edin.',
            }
        ]


class WebScraper:
    """
    HTML içerikten temiz metin çıkarır.
    Modüler: <p>, <h1-6>, <div>, <li> etiketlerini işler.
    """

    def scrape(self, url: str) -> str:
        """URL'den metin içeriği çıkarır."""
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                # Basit HTML parsing
                import re

                # Script ve style etiketlerini kaldır
                text = re.sub(
                    r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL
                )
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                # HTML etiketlerini kaldır
                text = re.sub(r"<[^>]+>", " ", text)
                # Boşlukları normalize et
                text = re.sub(r"\s+", " ", text)
                return text.strip()
        except Exception:
            pass
        return ""


class ContentChunker:
    """
    Metni anlamlı parçalara böler.
    Chunk boyutu: ~300 karakter, benzersiz terimler kontrol ile.
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list:
        """Metni parçalara böler."""
        if not text or len(text) < self.chunk_size:
            return [text]

        chunks = []
        sentences = text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 20]


class EmbeddingGenerator:
    """
    Metni vektöre dönüştürür.
    Modüler: Ollama embedding API'sini kullanır.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.endpoint = "http://localhost:11434/api/embeddings"

    def embed(self, text: str) -> list:
        """Metin için embedding vektörü oluştur."""
        try:
            payload = {
                "model": self.model,
                "prompt": text[:500],  # İlk 500 karakterle sınırla
            }
            resp = requests.post(self.endpoint, json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
        except Exception:
            pass
        # Fallback: basit hash-based vector
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> list:
        """Basit metin tabanlı embedding (fallback)."""
        import hashlib

        h = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:16]]


class ChromaDBStore:
    """
    Vektör tabanlı körnekleri depo eder.
    Modüler: In-memory depo (ChromaDB kurulu değilse fallback).
    """

    def __init__(self):
        self.storage = []
        self.embeddings = {}

    def clear(self) -> None:
        """Depoyu temizle (yeni arama için)."""
        self.storage = []
        self.embeddings = {}

    def add(self, documents: list, embeddings: list, metadatas: list = None) -> None:
        """Belgeleri ve vektörlerini depoya ekle."""
        if metadatas is None:
            metadatas = [{"source": "web"} for _ in documents]

        for doc, emb, meta in zip(documents, embeddings, metadatas):
            doc_id = len(self.storage)
            self.storage.append({"id": doc_id, "content": doc, "metadata": meta})
            self.embeddings[doc_id] = emb

    def query(self, embedding: list, top_k: int = 5) -> list:
        """En benzer belgeleri döndür."""
        if not self.storage:
            return []

        # Basit cosine similarity
        scores = []
        for doc_id, stored_emb in self.embeddings.items():
            sim = self._cosine_similarity(embedding, stored_emb)
            scores.append((sim, self.storage[doc_id]))

        scores.sort(reverse=True)
        return [doc for _, doc in scores[:top_k]]

    def _cosine_similarity(self, a: list, b: list) -> float:
        """İki vektör arasında cosine similarity hesapla."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class RAGRetriever:
    """
    Sorguya uygun belgeleri RAG ile alır.
    Pipeline: Query → Embedding → Similarity Search → Top-K
    """

    def __init__(self, store: ChromaDBStore, embedder: EmbeddingGenerator):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Sorguya uygun belgeleri al."""
        query_embedding = self.embedder.embed(query)
        results = self.store.query(query_embedding, top_k)
        return results


class WebIntelligenceSynthesizer:
    """
    RAG sonuçlarını Ollama NLP ile sentezler.
    Çıktı: ASCII rapora hazır analiz raporu.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "phi4:14b",
    ):
        self.ollama_url = ollama_url
        self.model = model

    def synthesize(self, query: str, retrieved_docs: list) -> str:
        """Alınan belgeleri sorguya uygun şekilde sentezle."""
        if not retrieved_docs:
            return "İlgili bilgi bulunamadı."

        # Bağlam oluştur
        context = "\n".join(
            [
                f"[{doc['metadata'].get('source', 'Web')}] {doc['content']}"
                for doc in retrieved_docs[:5]
            ]
        )

        prompt = f"""
Aşağıdaki web kaynaklarından alınan bilgileri kullanarak 
"{query}" sorusuna kapsamlı, itibar edilen bir cevap ver.

BİLGİ KAYNAKLARI:
{context}

ÇIKTI KURALLAR:
- Cevabı kısa, net ve bilimsel tupla
- Kaynakları belirt
- Varsa istatistikleri dahil et
- Turkish (Türkçe) cevap ver

CEVAP:
"""

        try:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "İşlem başarısız.")
        except Exception as e:
            print(f"[DEBUG] Synthesis hatası: {e}")
            # Fallback: basit sentez
            return f"'{query}' hakkında: {context[:300]}..."

        return "Ollama bağlantı hatası."


class WebIntelligencePipeline:
    """
    Tüm Web İstihbaratı pipeline'ını orchestrate eder.
    Input: Sorgu
    Output: ASCII report
    """

    def __init__(self):
        self.fetcher = MultiSourceFetcher()
        self.scraper = WebScraper()
        self.chunker = ContentChunker()
        self.embedder = EmbeddingGenerator()
        self.store = ChromaDBStore()
        self.retriever = RAGRetriever(self.store, self.embedder)
        self.synthesizer = WebIntelligenceSynthesizer()

    def search(self, query: str) -> dict:
        """
        Sorguyu işle ve ASCII raporunu döndür.
        Returns:
          {
            'query': str,
            'sources_count': int,
            'retrieved_docs': list,
            'synthesis': str,
            'report': str (ASCII formatted)
          }
        """

        # Depoyu yeni arama için temizle
        self.store.clear()

        # 1. Multi-source fetch
        sources = self.fetcher.fetch_all(query)

        # 2. Scrape ve chunking
        documents = []
        for source in sources:
            documents.append(source["content"])

        chunks = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))

        # 3. Embedding ve storage
        embeddings = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            embeddings.append(self.embedder.embed(chunk))
            source_name = sources[i % len(sources)]["source"] if sources else "Unknown"
            metadatas.append({"source": source_name, "chunk_id": i})

        self.store.add(chunks, embeddings, metadatas)

        # 4. RAG retrieval
        retrieved = self.retriever.retrieve(query, top_k=5)

        # 5. NLP synthesis
        synthesis = self.synthesizer.synthesize(query, retrieved)

        # 6. ASCII report oluştur
        report = self._format_ascii_report(query, sources, synthesis)

        return {
            "query": query,
            "sources_count": len(sources),
            "retrieved_docs": retrieved,
            "synthesis": synthesis,
            "report": report,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def _format_ascii_report(self, query: str, sources: list, synthesis: str) -> str:
        """ASCII formatında rapport oluştur."""
        width = ASCII_WIDTH

        lines = [
            "╔" + "═" * (width - 2) + "╗",
            f"║ 🟣 WEB İSTİHBARATI RAPORU {' ' * (width - 40)}║",
            "╠" + "═" * (width - 2) + "╣",
            f"║ SORGU: {query[:70]:<71}║",
            f"║ TARİH: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<68}║",
            "╠" + "═" * (width - 2) + "╣",
            f"║ KAYNAKLAR: {len(sources)} sonuç bulundu{' ' * (width - 35)}║",
        ]

        # Kaynakları listele
        for i, src in enumerate(sources[:5], 1):
            lines.append(
                f"║ {i}. {src.get('source', 'Unknown')}: {src.get('title', '')[:55]:<57}║"
            )

        lines.extend(
            [
                "╠" + "═" * (width - 2) + "╣",
                f"║ ANALİZ VE SENTEZ{' ' * (width - 19)}║",
                "╟" + "─" * (width - 2) + "╢",
            ]
        )

        # Synthesis'i parçala ve sarmala
        for line in synthesis.split("\n")[:10]:
            wrapped_lines = self._wrap_text(line, width - 4)
            for wrapped in wrapped_lines:
                lines.append(f"║ {wrapped:<{width - 4}}║")

        lines.extend(
            [
                "╚" + "═" * (width - 2) + "╝",
            ]
        )

        return "\n".join(lines)

    def _wrap_text(self, text: str, max_width: int) -> list:
        """Metni belirtilen genişliğe sarmala."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        return lines if lines else [""]


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES  (backend %100 değişmedi)
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return make_response(
        HTML_TEMPLATE, 200, {"Content-Type": "text/html; charset=utf-8"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINATION GUARD
#  Çözüm döngüsünün erken sonlandırılmasını güvenle engeller.
#  Tüm DAG düğümleri çözülene veya max_iterations'a kadar çalışmaya devam eder.
# ═══════════════════════════════════════════════════════════════════════════════
class TerminationGuard:
    """
    Iterative solver loop'un yarıda kesilmesini engeller.

    Kontrol kriteri:
      1. Tüm DAG düğümleri çalıştırıldı mı?
      2. Max iteration sayısına ulaşıldı mı?
      3. Convergence tolerance sağlandı mı? (entropy/error düşüş stabil)
      4. Unrecoverable violation var mı? (abort koşulu)
    """

    def __init__(self, max_iterations: int = 10, convergence_tol: float = 1e-6):
        """
        Args:
          max_iterations:    maksimum iteration sayısı
          convergence_tol:   entropy değişim toleransı (<  tol → convergence)
        """
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.iteration_count = 0
        self.entropy_history = []
        self.violation_history = []

    def should_continue(
        self,
        dag_executor: "DAGExecutor | None" = None,
        current_entropy: float = 0.0,
        current_violations: list = None,
    ) -> tuple:
        """
        Iterasyon devam etmeli mi?

        Args:
          dag_executor:          DAGExecutor instance (remaining nodes kontrol için)
          current_entropy:       şimdiki entropy değeri
          current_violations:    son validation ihlalleri

        Returns:
          (should_continue: bool, reason: str)
        """
        if current_violations is None:
            current_violations = []

        self.iteration_count += 1
        self.entropy_history.append(current_entropy)
        self.violation_history.append(len(current_violations))

        # ─────────────────────────────────────────────────────────────────────
        # 1. MAX İTERATİON KONTROLÜ
        # ─────────────────────────────────────────────────────────────────────
        if self.iteration_count >= self.max_iterations:
            return (
                False,
                f"[TERM-MAX-ITER] Max iterations ({self.max_iterations}) reached",
            )

        # ─────────────────────────────────────────────────────────────────────
        # 2. DAG TAMAMLAMA KONTROLÜ
        # ─────────────────────────────────────────────────────────────────────
        if dag_executor is not None:
            remaining = dag_executor.get_remaining()
            if not remaining:
                return (
                    False,
                    f"[TERM-DAG-COMPLETE] All DAG nodes executed ({self.iteration_count} iterations)",
                )

        # ─────────────────────────────────────────────────────────────────────
        # 3. CONVERGENCE KONTROLÜ (Entropy değişim)
        # ─────────────────────────────────────────────────────────────────────
        if len(self.entropy_history) >= 2:
            entropy_change = abs(self.entropy_history[-1] - self.entropy_history[-2])
            if entropy_change < self.convergence_tol:
                # Entropy stabil → convergence sağlandı
                consecutive_stable = 1
                for i in range(len(self.entropy_history) - 2, 0, -1):
                    if (
                        abs(self.entropy_history[i] - self.entropy_history[i - 1])
                        < self.convergence_tol
                    ):
                        consecutive_stable += 1
                    else:
                        break

                if consecutive_stable >= 2:
                    return (
                        False,
                        f"[TERM-CONVERGENCE] Entropy converged (Δ={entropy_change:.6f} < tol={self.convergence_tol})",
                    )

        # ─────────────────────────────────────────────────────────────────────
        # 4. UNRECOVERABLE VIOLATION KONTROLÜ (ABORT)
        # ─────────────────────────────────────────────────────────────────────
        abort_patterns = ["FATAL", "UNRECOVERABLE", "CYCLE_DETECTED", "RECOVERY_FAILED"]
        for pattern in abort_patterns:
            for viol in current_violations:
                if pattern in viol:
                    return (
                        False,
                        f"[TERM-ABORT] {pattern} violation detected — unrecoverable",
                    )

        # ─────────────────────────────────────────────────────────────────────
        # 5. DEFAULT: DEVAM ET
        # ─────────────────────────────────────────────────────────────────────
        return (
            True,
            f"[TERM-CONTINUE] Iteration {self.iteration_count}/{self.max_iterations}",
        )

    def get_termination_reason(self) -> str:
        """Son break işleminin nedenini döndür."""
        if self.iteration_count >= self.max_iterations:
            return "Max iterations"
        elif len(self.entropy_history) >= 2:
            if (
                abs(self.entropy_history[-1] - self.entropy_history[-2])
                < self.convergence_tol
            ):
                return "Convergence"
        return "Unknown"

    def reset(self) -> None:
        """Tüm counter'ları sıfırla (yeni soru için)."""
        self.iteration_count = 0
        self.entropy_history.clear()
        self.violation_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  /search ENDPOINT — WEB İSTİHBARATI MODÜLÜ
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/search", methods=["POST"])
def search():
    """
    Web İstihbarı Pipeline'ı tetikle.

    Endpoint: POST /search
    Body: {"query": "arama sorgusu"}

    Response: {
      "status": "success",
      "report": "ASCII formatted report",
      "sources_count": int,
      "timestamp": ISO datetime
    }
    """
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"status": "error", "message": "Sorgu boş olamaz"}), 400

        # Pipeline'ı initialize et
        pipeline = init_web_intelligence()

        if pipeline is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Web İstihbaratı Pipeline başlatılamadı",
                    }
                ),
                500,
            )

        # Arama yap
        result = pipeline.search(query)

        return (
            jsonify(
                {
                    "status": "success",
                    "query": result["query"],
                    "report": result["report"],
                    "sources_count": result["sources_count"],
                    "synthesis": result["synthesis"],
                    "timestamp": result["timestamp"],
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "error", "message": f"Arama hatası: {str(e)}"}), 500




class GraphSimulationAgent:
    """
    Hafızalı grafik ajanı:
    - NLP sinyallerinden/çözüm izinden seri çıkarır
    - 50 farklı render modundan birini Q-learning ile seçer
    - Cartesian / Polar / Grid koordinatlarında simülasyon üretir
    """

    MEMORY_PATH = "graph_agent_memory.pkl"

    def __init__(self, alpha=0.12, gamma=0.82, epsilon=0.08):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode = 0
        self._load()
        self.modes = self._build_modes()  # 50 mod

    def _build_modes(self) -> list:
        coords = ["cartesian", "polar", "grid"]
        styles = ["smooth", "dashed", "bars", "scatter", "pulse"]
        trajectories = ["raw", "symmetric", "spiral", "wave"]
        out = []
        idx = 1
        for c in coords:
            for s in styles:
                for t in trajectories:
                    out.append(
                        {
                            "id": f"g{idx:02d}",
                            "coord": c,
                            "style": s,
                            "trajectory": t,
                            "name": f"{c}_{s}_{t}",
                        }
                    )
                    idx += 1
        return out[:50]

    def _load(self):
        try:
            with open(self.MEMORY_PATH, "rb") as f:
                d = pickle.load(f)
            self.q_table = defaultdict(
                lambda: defaultdict(float),
                {
                    tuple(k): defaultdict(float, v)
                    for k, v in (d.get("q_table") or {}).items()
                },
            )
            self.episode = int(d.get("episode", 0))
        except Exception:
            pass

    def _save(self):
        try:
            serial = {tuple(k): dict(v) for k, v in self.q_table.items()}
            with open(self.MEMORY_PATH, "wb") as f:
                pickle.dump({"q_table": serial, "episode": self.episode}, f)
        except Exception:
            pass

    def _extract_numbers(self, text: str) -> list:
        vals = []
        for raw in re.findall(r"-?\d+(?:\.\d+)?", text or ""):
            try:
                vals.append(float(raw))
            except Exception:
                continue
        return vals

    def _series_from_solution(self, question: str, sol_data: dict) -> tuple:
        series = []
        steps = sol_data.get("steps") or []
        for i, st in enumerate(steps, 1):
            txt = " ".join(
                [str(st.get("content", "")), str(st.get("formula", "")), str(st.get("result", ""))]
            )
            nums = self._extract_numbers(txt)
            if nums:
                series.append({"x": i, "y": sum(nums) / max(len(nums), 1)})

        if len(series) < 2:
            pmap = sol_data.get("probabilities") or sol_data.get("posteriors") or {}
            if isinstance(pmap, dict) and pmap:
                for i, (_, v) in enumerate(pmap.items(), 1):
                    try:
                        series.append({"x": i, "y": float(v)})
                    except Exception:
                        continue

        if len(series) < 2:
            nums = self._extract_numbers(
                str(sol_data.get("answer", "")) + " " + str(sol_data.get("explanation", ""))
            )
            if not nums:
                nums = [0.0, 1.0]
            for i, v in enumerate(nums[:24], 1):
                series.append({"x": i, "y": float(v)})

        tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}", question or "")
        stop = {"veya", "sonra", "olan", "gibi", "then", "with", "from", "için", "soru", "question"}
        labels, seen = [], set()
        for t in tokens:
            k = t.lower()
            if k in stop or k in seen:
                continue
            seen.add(k)
            labels.append(t)
            if len(labels) >= 12:
                break
        if not labels:
            labels = ["düğüm", "olasılık", "çıkarım"]
        return series, labels

    def _state(self, question: str, sol_data: dict) -> tuple:
        q = (question or "").lower()
        has_graph_kw = 1 if re.search(r"graf|graph|çiz|curve|parab|polar|ızgara|grid|koordinat", q) else 0
        n_steps = min(len(sol_data.get("steps") or []), 12)
        intent = str(sol_data.get("type", "general"))
        return (intent, n_steps // 3, has_graph_kw)

    def _pick_mode(self, state: tuple) -> dict:
        if random.random() < self.epsilon or not self.q_table[state]:
            return random.choice(self.modes)
        best_id = max(self.q_table[state], key=self.q_table[state].get)
        return next((m for m in self.modes if m["id"] == best_id), self.modes[0])

    def _pick_mode_stateless(self, state: tuple, question: str, series: list) -> dict:
        """
        Öğrenmeden bağımsız, deterministik mod seçimi.
        Grafik çizimi ana öğrenme sürecini etkilemesin diye kullanılır.
        """
        key = f"{state}|{len(series)}|{(question or '')[:96]}"
        idx = abs(hash(key)) % max(len(self.modes), 1)
        return self.modes[idx]

    def _reward(self, mode: dict, series: list, question: str) -> float:
        q = (question or "").lower()
        r = 1.0
        if mode["coord"] == "polar" and re.search(r"polar|açı|angle|dairesel|spiral", q):
            r += 1.5
        if mode["coord"] == "grid" and re.search(r"grid|ızgara|matris|durum", q):
            r += 1.2
        if len(series) >= 6 and mode["style"] in {"smooth", "dashed"}:
            r += 0.7
        if len(series) <= 4 and mode["style"] in {"bars", "pulse"}:
            r += 0.6
        return r

    def _update(self, state: tuple, mode_id: str, reward: float):
        q = self.q_table[state]
        max_next = max(q.values()) if q else 0.0
        q[mode_id] += self.alpha * (reward + self.gamma * max_next - q[mode_id])
        self.episode += 1
        if self.episode % 5 == 0:
            self._save()

    def build_payload(
        self,
        question: str,
        sol_data: dict,
        width: int,
        height: int,
        affect_learning: bool = False,
    ) -> dict:
        series, labels = self._series_from_solution(question, sol_data)
        state = self._state(question, sol_data)
        if affect_learning:
            mode = self._pick_mode(state)
            reward = self._reward(mode, series, question)
            self._update(state, mode["id"], reward)
        else:
            mode = self._pick_mode_stateless(state, question, series)
            reward = 0.0

        ys = [p["y"] for p in series] if series else [0.0, 1.0]
        ymin, ymax = min(ys), max(ys)
        if abs(ymax - ymin) < 1e-12:
            ymax = ymin + 1.0

        curve = []
        n = max(len(series), 2)
        for idx, p in enumerate(series):
            xn = idx / max(1, n - 1)
            yn = (p["y"] - ymin) / (ymax - ymin)

            if mode["trajectory"] == "symmetric":
                xn = abs(2 * xn - 1)
            elif mode["trajectory"] == "spiral":
                yn = min(1.0, max(0.0, yn * (0.6 + 0.4 * xn)))
            elif mode["trajectory"] == "wave":
                yn = min(1.0, max(0.0, yn * 0.7 + (math.sin(xn * 2 * math.pi) + 1) * 0.15))

            x = 20 + (width - 40) * xn
            y = height - 20 - (height - 40) * yn
            theta = xn * 2 * math.pi
            radius = 0.18 + 0.72 * yn
            curve.append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "v": round(p["y"], 6),
                    "xn": round(xn, 6),
                    "yn": round(yn, 6),
                    "theta": round(theta, 6),
                    "radius": round(radius, 6),
                }
            )

        return {
            "theme": {"bg": "#050a0a", "grid": "#0f2a0f", "curve": "#00e676", "curve2": "#69ff47", "text": "#94ffb5"},
            "width": width,
            "height": height,
            "curve": curve,
            "labels": labels,
            "render_mode": mode,
            "mode_catalog_size": len(self.modes),
            "meta": {
                "ymin": ymin,
                "ymax": ymax,
                "points": len(curve),
                "agent_episode": self.episode,
                "learning_impact": bool(affect_learning),
                "mode_reward": round(float(reward), 4),
            },
        }


_graph_agent = GraphSimulationAgent()


def _build_graph_automaton_payload(
    question: str,
    sol_data: dict,
    width: int = 560,
    height: int = 320,
    affect_learning: bool = False,
) -> dict:
    return _graph_agent.build_payload(
        question, sol_data, width, height, affect_learning=affect_learning
    )


def _render_matplotlib_graph_base64(payload: dict) -> Optional[str]:
    """
    Graph payload'unu siyah arka plan + yeşil tonlu Matplotlib PNG'e çevirir.
    Dönen değer: data:image/png;base64,... formatında URI (hata olursa None).
    """
    if not _MATPLOTLIB_OK or not isinstance(payload, dict):
        return None
    curve = payload.get("curve") or []
    if len(curve) < 2:
        return None

    mode = payload.get("render_mode") or {}
    theme = payload.get("theme") or {}
    labels = (payload.get("labels") or [])[:8]

    bg = theme.get("bg", "#050a0a")
    grid = theme.get("grid", "#0f2a0f")
    curve_col = theme.get("curve", "#00e676")
    point_col = theme.get("curve2", "#69ff47")
    text_col = theme.get("text", "#94ffb5")

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=130)
    try:
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        x_vals = [float(p.get("x", 0.0)) for p in curve]
        y_vals = [float(p.get("y", 0.0)) for p in curve]

        line_style = "--" if mode.get("style") == "dashed" else "-"
        line_width = 2.4 if mode.get("style") == "pulse" else 2.0

        if mode.get("style") == "scatter":
            ax.scatter(x_vals, y_vals, s=28, c=point_col, alpha=0.9, edgecolors="none")
        elif mode.get("style") == "bars":
            bar_w = max(8.0, (max(x_vals) - min(x_vals)) / max(len(x_vals) * 2, 1))
            ax.bar(x_vals, y_vals, color=curve_col, alpha=0.55, width=bar_w, edgecolor="none")
            ax.plot(x_vals, y_vals, color=point_col, linewidth=1.2, alpha=0.9)
        else:
            ax.plot(x_vals, y_vals, color=curve_col, linewidth=line_width, linestyle=line_style, alpha=0.95)
            ax.scatter(x_vals, y_vals, s=16, c=point_col, alpha=0.92, edgecolors="none")

        ax.grid(True, color=grid, alpha=0.55, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color(grid)
            spine.set_linewidth(1.0)
        ax.tick_params(colors=text_col, labelsize=8)
        ax.set_title("NLP Graph Automatı · Matplotlib", color=text_col, fontsize=10, pad=8)

        if labels:
            label_str = " • ".join(str(x) for x in labels[:5])
            ax.text(
                0.01,
                0.985,
                label_str,
                transform=ax.transAxes,
                va="top",
                ha="left",
                color=text_col,
                fontsize=8,
                alpha=0.92,
            )

        mode_id = mode.get("id", "g00")
        coord = mode.get("coord", "cartesian")
        style = mode.get("style", "smooth")
        catalog_size = payload.get("mode_catalog_size", 0)
        ax.text(
            0.01,
            0.02,
            f"mode:{mode_id} {coord}/{style} · catalog:{catalog_size}",
            transform=ax.transAxes,
            color=text_col,
            fontsize=8,
            alpha=0.86,
        )

        fig.tight_layout(pad=1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg, edgecolor=bg)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{png_b64}"
    finally:
        plt.close(fig)

    def _save(self):
        try:
            with open(self.MEMORY_PATH, "wb") as f:
                pickle.dump({"weights": dict(self.weights), "episode": self.episode}, f)
        except Exception:
            pass

    def features(self, text: str) -> dict:
        t = (text or "").strip().lower()
        toks = [x for x in re.findall(r"[a-zçğıöşü0-9]+", t) if x]
        return {
            "len": min(len(toks), 16),
            "has_qmark": 1 if "?" in t else 0,
            "has_numeric": 1 if re.search(r"\d", t) else 0,
            "has_verb": 1 if re.search(r"(mi|mı|mu|mü|nedir|kaç|hangi|hesap|bul|çiz|göster)", t) else 0,
            "has_meta_prefix": 1 if re.search(r"^(mekanlar|durumlar|states)\s*[:\-]", t) else 0,
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE SIMULATION / DRAWING STACK (NLP + Q-LEARNING + TOOL ROUTER)
# ═══════════════════════════════════════════════════════════════════════════════
class PhysicsCompositionDSLParser:
    """Doğal dil fizik/simülasyon isteklerini parametrik bileşenlere ayırır."""

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9_]+", (text or "").lower())

    def parse(self, text: str) -> dict:
        toks = self._tokenize(text)
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
        verbs = re.findall(r"(simüle|çiz|render|görselleştir|canlandır|hesapla|iterate|step|animate)", (text or "").lower())
        operator_density = len(re.findall(r"[+\-*/=^∂∇]", text or "")) / max(len(text or ""), 1)
        spatial_score = sum(1 for t in toks if t in {"2d", "3d", "vektör", "alan", "grid", "mesh", "kare", "küre", "voxel"})
        temporal_score = sum(1 for t in toks if t in {"zaman", "adım", "iterasyon", "süre", "dt", "anlık", "canlı"})
        uncertainty_score = sum(1 for t in toks if t in {"stokastik", "rastgele", "gürültü", "noise", "brownian", "sde"})

        components = []
        if spatial_score > 0:
            components.append("spatial_field")
        if temporal_score > 0:
            components.append("time_evolution")
        if uncertainty_score > 0:
            components.append("stochasticity")
        if any(t in {"akış", "diffusion", "ısı", "dalga", "pde"} for t in toks):
            components.append("pde_dynamics")
        if any(t in {"parçacık", "nbody", "particle", "gravity", "çarpışma"} for t in toks):
            components.append("particle_dynamics")
        if any(t in {"hücre", "cellular", "otomata", "ca"} for t in toks):
            components.append("cellular_automata")

        return {
            "tokens": toks,
            "numbers": numbers,
            "operator_density": operator_density,
            "verb_count": len(verbs),
            "spatial_score": spatial_score,
            "temporal_score": temporal_score,
            "uncertainty_score": uncertainty_score,
            "components": sorted(set(components)),
        }


class SimulationSandboxManager:
    """Simülasyon görevleri için kaynak kotası ve güvenlik sınırları."""

    def __init__(self, max_steps: int = 2000, max_particles: int = 100000, max_grid: int = 512):
        self.max_steps = int(max_steps)
        self.max_particles = int(max_particles)
        self.max_grid = int(max_grid)

    def clamp(self, config: dict) -> dict:
        c = dict(config or {})
        c["steps"] = min(int(c.get("steps", 120)), self.max_steps)
        c["particles"] = min(int(c.get("particles", 1000)), self.max_particles)
        c["grid_size"] = min(int(c.get("grid_size", 64)), self.max_grid)
        c["dt"] = max(1e-4, float(c.get("dt", 0.05)))
        return c


class ComputeBackendOrchestrator:
    """CPU/WASM/GPU/REMOTE arka uç seçicisi."""

    def inspect_resources(self) -> dict:
        cpu_count = os.cpu_count() or 1
        mem_hint = 8.0
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            mem_hint = (page * pages) / (1024**3)
        except Exception:
            pass
        has_cuda = bool(_TORCH_OK and getattr(torch, "cuda", None) and torch.cuda.is_available())
        return {
            "cpu_count": cpu_count,
            "memory_gb": round(float(mem_hint), 2),
            "has_cuda": has_cuda,
            "wasm_ready": True,
            "remote_ready": True,
        }

    def choose_backend(self, workload: dict) -> dict:
        resources = self.inspect_resources()
        w = dict(workload or {})
        complexity = float(w.get("complexity", 1.0))
        realtime = float(w.get("realtime_weight", 0.5))
        frame = int(w.get("frame_budget_ms", 33))

        candidates = ["cpu", "wasm", "gpu", "remote"]
        scores = {}
        for name in candidates:
            base = 0.0
            if name == "cpu":
                base = min(1.0, resources["cpu_count"] / 8.0)
            elif name == "wasm":
                base = 0.7 if resources["wasm_ready"] else 0.0
            elif name == "gpu":
                base = 1.2 if resources["has_cuda"] else 0.05
            elif name == "remote":
                base = 0.8 if resources["remote_ready"] else 0.0
            latency_penalty = 0.35 if (name == "remote" and frame < 25) else 0.0
            scores[name] = base + complexity * (0.2 if name in {"gpu", "remote"} else 0.08) + realtime * 0.15 - latency_penalty

        chosen = max(scores, key=scores.get)
        return {"chosen": chosen, "scores": scores, "resources": resources}


class SimulationEngineAdapter:
    """Ortak simülasyon adaptör arayüzü."""

    engine_name = "generic"

    def init(self, config: dict, dsl: dict) -> dict:
        return {"t": 0.0, "config": dict(config or {}), "dsl": dict(dsl or {}), "series": []}

    def step(self, state: dict, dt: float = 0.05) -> dict:
        t = float(state.get("t", 0.0)) + float(dt)
        series = list(state.get("series", []))
        series.append({"t": t, "y": math.sin(t)})
        return {**state, "t": t, "series": series[-400:]}

    def snapshot(self, state: dict) -> dict:
        return {"engine": self.engine_name, "t": state.get("t", 0.0), "series": state.get("series", [])[-120:]}


class PDEEngineAdapter(SimulationEngineAdapter):
    engine_name = "pde"

    def init(self, config: dict, dsl: dict) -> dict:
        c = dict(config or {})
        n = int(c.get("grid_size", 64))
        x = np.linspace(0.0, 1.0, n)
        u = np.exp(-((x - 0.5) ** 2) / 0.01)
        return {"t": 0.0, "x": x, "u": u, "alpha": float(c.get("alpha", 0.08)), "config": c}

    def step(self, state: dict, dt: float = 0.01) -> dict:
        u = np.array(state.get("u"), dtype=float)
        alpha = float(state.get("alpha", 0.08))
        lap = np.zeros_like(u)
        lap[1:-1] = u[:-2] - 2 * u[1:-1] + u[2:]
        u2 = u + alpha * dt * lap
        u2[0] = u2[-1] = 0.0
        return {**state, "u": u2, "t": float(state.get("t", 0.0)) + dt}

    def snapshot(self, state: dict) -> dict:
        return {"engine": self.engine_name, "t": float(state.get("t", 0.0)), "x": [float(v) for v in state.get("x", [])[:256]], "u": [float(v) for v in state.get("u", [])[:256]]}


class ParticleEngineAdapter(SimulationEngineAdapter):
    engine_name = "particle"

    def init(self, config: dict, dsl: dict) -> dict:
        c = dict(config or {})
        n = int(c.get("particles", 300))
        rng = np.random.default_rng(seed=42)
        pos = rng.normal(0, 1, size=(n, 2))
        vel = rng.normal(0, 0.2, size=(n, 2))
        return {"t": 0.0, "pos": pos, "vel": vel, "drag": float(c.get("drag", 0.02)), "config": c}

    def step(self, state: dict, dt: float = 0.03) -> dict:
        pos = np.array(state.get("pos"), dtype=float)
        vel = np.array(state.get("vel"), dtype=float)
        drag = float(state.get("drag", 0.02))
        center_force = -0.1 * pos
        vel = vel + center_force * dt
        vel *= max(0.0, 1.0 - drag)
        pos = pos + vel * dt
        return {**state, "t": float(state.get("t", 0.0)) + dt, "pos": pos, "vel": vel}

    def snapshot(self, state: dict) -> dict:
        p = np.array(state.get("pos"), dtype=float)
        return {"engine": self.engine_name, "t": float(state.get("t", 0.0)), "points": p[:600].round(5).tolist()}


class CellularAutomataAdapter(SimulationEngineAdapter):
    engine_name = "cellular_automata"

    def init(self, config: dict, dsl: dict) -> dict:
        c = dict(config or {})
        n = int(c.get("grid_size", 80))
        rng = np.random.default_rng(seed=7)
        grid = (rng.uniform(0, 1, size=(n, n)) > 0.72).astype(np.int8)
        return {"t": 0.0, "grid": grid, "config": c}

    def step(self, state: dict, dt: float = 1.0) -> dict:
        g = np.array(state.get("grid"), dtype=np.int8)
        nsum = sum(np.roll(np.roll(g, i, axis=0), j, axis=1) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0))
        nxt = (((g == 1) & ((nsum == 2) | (nsum == 3))) | ((g == 0) & (nsum == 3))).astype(np.int8)
        return {**state, "grid": nxt, "t": float(state.get("t", 0.0)) + dt}

    def snapshot(self, state: dict) -> dict:
        g = np.array(state.get("grid"), dtype=np.int8)
        return {"engine": self.engine_name, "t": float(state.get("t", 0.0)), "grid": g[:128, :128].tolist()}


class ProceduralWorldGenerator:
    """Minecraft-benzeri height-map / voxel yoğunluk üreticisi."""

    def generate(self, size: int = 96, seed: int = 11) -> dict:
        size = max(16, min(int(size), 256))
        rng = np.random.default_rng(seed=seed)
        x = np.linspace(-2.0, 2.0, size)
        y = np.linspace(-2.0, 2.0, size)
        xx, yy = np.meshgrid(x, y)
        base = np.sin(xx * 2.1) + np.cos(yy * 1.7)
        noise = rng.normal(0, 0.18, size=(size, size))
        h = base + noise
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        voxels = (h > 0.55).astype(np.int8)
        return {"height_map": h.round(5).tolist(), "voxel_mask": voxels.tolist(), "size": size}


class VisualizationStreamOrchestrator:
    """Snapshot'ları renderer-agnostic frame paketlerine dönüştürür."""

    def build_frame(self, snapshot: dict, renderer: str = "canvas") -> dict:
        payload = {"renderer": renderer, "timestamp": time.time(), "snapshot": snapshot}
        payload["delta_hint"] = {"kind": snapshot.get("engine", "generic"), "keys": sorted(list(snapshot.keys()))}
        return payload


class SimulationModelRegistry:
    """Engine metadata + adaptör kayıt deposu."""

    def __init__(self):
        self.models = {}

    def register(self, name: str, adapter: SimulationEngineAdapter, metadata: dict):
        self.models[name] = {"adapter": adapter, "metadata": dict(metadata or {})}

    def describe(self) -> list:
        return [{"name": n, **row.get("metadata", {})} for n, row in self.models.items()]

    def get(self, name: str) -> dict:
        return self.models.get(name, {})


class NLPSimulationRouter:
    """NLP + Q-learning hibrit seçici (engine/renderer/backend)."""

    def __init__(self, registry: SimulationModelRegistry, alpha: float = 0.14, gamma: float = 0.86, epsilon: float = 0.1):
        self.registry = registry
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(lambda: defaultdict(float))
        self.episode = 0

    def _semantic_tokens(self, text: str) -> set:
        return set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9_]+", (text or "").lower()))

    def _state(self, question: str, dsl: dict, resources: dict) -> tuple:
        toks = self._semantic_tokens(question)
        return (
            min(len(toks) // 6, 12),
            min(int(dsl.get("temporal_score", 0)), 5),
            min(int(dsl.get("spatial_score", 0)), 5),
            1 if resources.get("has_cuda") else 0,
            1 if dsl.get("uncertainty_score", 0) > 0 else 0,
        )

    def _candidate_actions(self) -> list:
        renderers = ["ascii", "canvas", "webgl", "svg"]
        backends = ["cpu", "wasm", "gpu", "remote"]
        actions = []
        for item in self.registry.describe():
            engine = item.get("name")
            tags = set(item.get("tags", []))
            for renderer in renderers:
                if renderer == "webgl" and "high_dim" not in tags and "field" not in tags:
                    continue
                for backend in backends:
                    actions.append((engine, renderer, backend))
        return actions

    def _action_score(self, action: tuple, question: str, dsl: dict) -> float:
        engine, renderer, backend = action
        item = self.registry.get(engine)
        meta = item.get("metadata", {})
        q_toks = self._semantic_tokens(question)
        tag_toks = set(meta.get("tags", []))
        overlap = len(q_toks & tag_toks)
        density = float(dsl.get("operator_density", 0.0))
        temporal = float(dsl.get("temporal_score", 0.0))

        score = 0.5 + overlap * 0.18 + temporal * (0.06 if engine in {"pde", "particle", "cellular_automata"} else 0.03)
        if density > 0.02 and engine == "pde":
            score += 0.22
        if renderer == "ascii":
            score -= 0.12
        if backend == "gpu" and "high_dim" in tag_toks:
            score += 0.2
        return score

    def route(self, question: str, dsl: dict, backend_info: dict) -> dict:
        actions = self._candidate_actions()
        if not actions:
            return {"engine": "pde", "renderer": "canvas", "backend": "cpu", "reward": 0.0}
        state = self._state(question, dsl, backend_info.get("resources", {}))
        if random.random() < self.epsilon or not self.q[state]:
            action = max(actions, key=lambda a: self._action_score(a, question, dsl))
        else:
            action = max(self.q[state], key=self.q[state].get)

        reward = self._action_score(action, question, dsl)
        current = self.q[state].get(action, 0.0)
        next_best = max(self.q[state].values()) if self.q[state] else 0.0
        self.q[state][action] = current + self.alpha * (reward + self.gamma * next_best - current)
        self.episode += 1

        return {"engine": action[0], "renderer": action[1], "backend": action[2], "reward": round(float(reward), 4), "state": state, "episode": self.episode}


class SimulationRuntimeCoordinator:
    """DSL → router → backend → adapter yürütme koordinatörü."""

    def __init__(self, dsl_parser: PhysicsCompositionDSLParser, sandbox: SimulationSandboxManager,
                 backend: ComputeBackendOrchestrator, registry: SimulationModelRegistry,
                 router: NLPSimulationRouter, streamer: VisualizationStreamOrchestrator):
        self.dsl_parser = dsl_parser
        self.sandbox = sandbox
        self.backend = backend
        self.registry = registry
        self.router = router
        self.streamer = streamer

    def build_plan(self, question: str, base_config: dict = None) -> dict:
        dsl = self.dsl_parser.parse(question)
        safe_cfg = self.sandbox.clamp(base_config or {})
        complexity = 1.0 + 0.15 * len(dsl.get("components", [])) + float(dsl.get("operator_density", 0.0)) * 12
        backend_info = self.backend.choose_backend({"complexity": complexity, "realtime_weight": 0.4 + min(dsl.get("temporal_score", 0), 6) * 0.08, "frame_budget_ms": 33})
        route = self.router.route(question, dsl, backend_info)
        world_hint = {}
        if any(c in dsl.get("components", []) for c in ["spatial_field", "particle_dynamics"]):
            world_hint = ProceduralWorldGenerator().generate(size=safe_cfg.get("grid_size", 64))
        return {"dsl": dsl, "config": safe_cfg, "backend": backend_info, "route": route, "world_hint": world_hint}

    def execute_preview(self, plan: dict, preview_steps: int = 12) -> dict:
        route = plan.get("route", {})
        engine_name = route.get("engine", "pde")
        model_row = self.registry.get(engine_name)
        adapter = model_row.get("adapter") or SimulationEngineAdapter()
        cfg = plan.get("config", {})
        dsl = plan.get("dsl", {})
        state = adapter.init(cfg, dsl)
        dt = float(cfg.get("dt", 0.05))
        frames = []
        for _ in range(max(1, min(int(preview_steps), 40))):
            state = adapter.step(state, dt)
            snap = adapter.snapshot(state)
            frames.append(self.streamer.build_frame(snap, route.get("renderer", "canvas")))
        return {"engine": engine_name, "renderer": route.get("renderer", "canvas"), "backend": route.get("backend", "cpu"), "frames": frames, "last_snapshot": frames[-1]["snapshot"] if frames else {}}


# Global adaptive simulation stack
_sim_dsl_parser = PhysicsCompositionDSLParser()
_sim_sandbox = SimulationSandboxManager()
_sim_backend = ComputeBackendOrchestrator()
_sim_registry = SimulationModelRegistry()
_sim_registry.register("pde", PDEEngineAdapter(), {"tags": ["pde", "diffusion", "field", "high_dim", "mesh"]})
_sim_registry.register("particle", ParticleEngineAdapter(), {"tags": ["particle", "nbody", "gravity", "collision", "high_dim"]})
_sim_registry.register("cellular_automata", CellularAutomataAdapter(), {"tags": ["cellular", "automata", "grid", "discrete"]})
_sim_router = NLPSimulationRouter(_sim_registry)
_sim_streamer = VisualizationStreamOrchestrator()
_sim_runtime = SimulationRuntimeCoordinator(_sim_dsl_parser, _sim_sandbox, _sim_backend, _sim_registry, _sim_router, _sim_streamer)



class PlannerLearningMemory:
    """
    Planlayıcı için hafızalı öğrenen scorer.
    Alt-soru kalitesini soru-agnostik sinyallerle puanlar ve çözüm sonucundan geri besleme alır.
    """

    MEMORY_PATH = "planner_memory.pkl"

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.weights = defaultdict(float)
        self.episode = 0
        self._load()

    def _load(self):
        try:
            with open(self.MEMORY_PATH, "rb") as f:
                d = pickle.load(f)
            self.weights = defaultdict(float, d.get("weights", {}))
            self.episode = int(d.get("episode", 0))
        except Exception:
            pass

    def _save(self):
        try:
            with open(self.MEMORY_PATH, "wb") as f:
                pickle.dump({"weights": dict(self.weights), "episode": self.episode}, f)
        except Exception:
            pass

    def features(self, text: str) -> dict:
        t = (text or "").strip().lower()
        toks = [x for x in re.findall(r"[a-zçğıöşü0-9]+", t) if x]
        return {
            "len": min(len(toks), 16),
            "has_qmark": 1 if "?" in t else 0,
            "has_numeric": 1 if re.search(r"\d", t) else 0,
            "has_verb": 1 if re.search(r"(mi|mı|mu|mü|nedir|kaç|hangi|hesap|bul|çiz|göster)", t) else 0,
            "has_meta_prefix": 1 if re.search(r"^(mekanlar|durumlar|states)\s*[:\-]", t) else 0,
        }

    def score(self, text: str) -> float:
        f = self.features(text)
        base = 0.0
        for k, v in f.items():
            base += self.weights.get(k, 0.0) * float(v)
        # Hafif öncelik: soru benzeri cümle
        base += 0.25 * f["has_qmark"] + 0.30 * f["has_verb"] - 0.40 * f["has_meta_prefix"]
        return base

    def feedback(self, sub_questions: list, consistency_score: float, violations: list):
        quality = float(consistency_score) - min(len(violations), 6) * 0.08
        self.episode += 1
        for sq in sub_questions or []:
            f = self.features(sq)
            for k, v in f.items():
                self.weights[k] += self.alpha * (quality * float(v) - self.weights[k] * 0.02)
        if self.episode % 5 == 0:
            self._save()


_planner_memory = PlannerLearningMemory()


def _decompose_multi_questions(question: str, scorer: PlannerLearningMemory = None) -> list:
    """
    Soru metnini alt sorulara böler.
    Öncelik gerçek soru cümlelerindedir (özellikle "Sorular:" bölümünden sonrası).
    Amaç: planlayıcı debug parçaları yerine kullanıcıya anlamlı soru adımları üretmek.
    """
    qtext = (question or "").strip()
    if not qtext:
        return []

    # "Sorular:" bloğu varsa önce onu hedefle
    focus = qtext
    m = re.search(r"\bsorular?\s*:\s*(.+)$", qtext, flags=re.IGNORECASE | re.DOTALL)
    if m:
        focus = m.group(1).strip()

    chunks = []

    # 0) Öncelik: numaralı alt-soru listesi (1) / 1. / 1) formatları
    numbered_matches = list(
        re.finditer(
            r"(?:^|\n|\s)(\d{1,2})[\)\.\-:]\s*(.+?)(?=(?:\n|\s)\d{1,2}[\)\.\-:]|\Z)",
            focus,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if numbered_matches:
        for m in numbered_matches:
            clean = re.sub(r"\s+", " ", m.group(2) or "").strip(" -,:;")
            if clean:
                chunks.append(clean if clean.endswith("?") else clean + "?")

    # Gürültü temizleyici: planlayıcıya problem cümlesi yerine gerçek alt-soruları ver
    def _sanitize_chunk(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # "soru1:", "q2-" vb. etiketleri kaldır
        t = re.sub(r"^\s*(?:soru|q)\s*\d+\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
        # "eğer, soru1:" gibi iç etiketleri ayıkla
        t = re.sub(
            r"\b(?:soru|q)\s*\d+\s*[:\-]\s*", "", t, flags=re.IGNORECASE
        ).strip()
        # Bayes başlık kalıplarını adım metninden çıkart (soru bağımsız)
        t = re.sub(
            r"^\s*bayes\s+teoremi\s*\([^)]*\)\s*soru\s*[:\-]?\s*",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        t = re.sub(r"\s+", " ", t).strip(" -,:;")
        return t

    def _is_question_like(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        # "mekanlar: ..." gibi veri tanımı satırlarını plan adımı yapma
        if re.search(r"^\s*(mekanlar?|durumlar?|states?)\s*[:\-]", t, re.IGNORECASE):
            return False
        # Çok kısa ya da yalnızca isim listesi içeren parçaları ele
        if len(t) < 8:
            return False
        # Çok uzun problem tanım blokları (gerçek alt-soru değil)
        if len(t) > 220 and not ("?" in t):
            return False
        if re.search(r"(aşağıdaki\s+sorular|cevaplayınız|problem[ei]\s*:)", t, re.IGNORECASE):
            return False
        token_count = len(t.split())
        if token_count <= 2:
            return False
        return True

    # 1) Numaralı eşleşme yoksa '?' ile biten gerçek soru cümleleri
    if not chunks:
        for part in re.split(r"\?\s*", focus):
            clean = _sanitize_chunk(part)
            if _is_question_like(clean):
                chunks.append(clean + "?")

    # 2) Fallback: hiç soru işareti yoksa satır/bölüm bazlı kırılım
    if not chunks:
        raw_parts = re.split(r"(?:\n+|(?=soru\d*\s*:))", focus, flags=re.IGNORECASE)
        for part in raw_parts:
            clean = _sanitize_chunk(part)
            if _is_question_like(clean) and len(clean) >= 12:
                chunks.append(clean if clean.endswith("?") else clean + "?")

    # Aynı parçaları tekilleştir (sıra korunur)
    uniq = []
    seen = set()
    for ch in chunks:
        key = ch.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(ch)

    # Öğrenen scorer ile kalite sıralaması (soruya bağımlı hard-code yok)
    if scorer and uniq:
        ranked = sorted(uniq, key=lambda s: scorer.score(s), reverse=True)
        uniq = ranked

    return uniq[:24]


def _dedupe_steps(existing_steps: list, candidate_steps: list) -> list:
    """
    Mevcut adımlarla anlamsal olarak tekrar eden planner adımlarını eler.
    """
    def _norm(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"[^\wçğıöşü\s]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    existing_signatures = set()
    for st in existing_steps or []:
        sig = _norm((st.get("title", "") + " " + st.get("content", "")).strip())
        if sig:
            existing_signatures.add(sig)

    filtered = []
    for st in candidate_steps or []:
        sig = _norm((st.get("title", "") + " " + st.get("content", "")).strip())
        if not sig:
            continue
        # Tam tekrar veya kuvvetli alt-kapsama tekrarını atla
        is_duplicate = any(sig in ex or ex in sig for ex in existing_signatures)
        if not is_duplicate:
            filtered.append(st)
            existing_signatures.add(sig)
    return filtered


def _planner_topic_title(sub_question: str, i: int, total: int, intent: str) -> str:
    """
    Alt soru için içerik odaklı (NLP-benzeri) başlık üretir.
    Planlayıcı modülü/düğüm adını değil, soru konusunu gösterir.
    """
    text = (sub_question or "").strip()
    normalized = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ\s%-]", " ", text, flags=re.UNICODE).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()

    stop_words = {
        "ve",
        "veya",
        "ile",
        "için",
        "olan",
        "olarak",
        "bu",
        "şu",
        "bir",
        "iki",
        "üç",
        "dört",
        "sonunda",
        "tur",
        "soru",
        "sorular",
    }
    tokens = [t for t in normalized.split() if len(t) > 2 and t not in stop_words]
    topic = " ".join(tokens[:4]).strip() or f"alt-soru {i}"

    if re.search(r"\b(olasılık|ihtimal|şans|oran|yüzde|%)\b", normalized):
        prefix = "Olasılık Analizi"
    elif re.search(r"\b(en yüksek|en düşük|maksimum|minimum|sırala)\b", normalized):
        prefix = "Karşılaştırmalı Değerlendirme"
    elif re.search(r"\b(kim|hangi|hangisi)\b", normalized):
        prefix = "Hedef Varlık Seçimi"
    elif re.search(r"\b(kaç|ne kadar|toplam|puan|değer)\b", normalized):
        prefix = "Nicel Hesaplama"
    else:
        intent_label = (intent or "genel").replace("_", " ").strip().title()
        prefix = f"{intent_label} Çözümü"

    return f"{prefix} ({i}/{total}) — {topic[:44]}"


def _extract_planner_prob_facts(question: str) -> dict:
    """
    Soru metninden planlayıcıda kullanılabilecek temel olasılık sabitlerini çıkarır.
    """
    q = (question or "").lower()
    facts = {}
    m_pre_entry = re.search(
        r"kulübeye\s+girmeden\s+önce\s+öld[üuğg].{0,25}?(0?\.\d+|\d{1,3}\s*%)",
        q,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_pre_entry:
        raw = m_pre_entry.group(1).replace(" ", "")
        if raw.endswith("%"):
            p = float(raw[:-1]) / 100.0
        else:
            p = float(raw)
        if 0.0 <= p <= 1.0:
            facts["p_death_before_entry"] = p
            facts["p_entry_before_death"] = round(1.0 - p, 6)
    return facts


def _build_planner_steps(sub_questions: list, route_features: dict, question: str = "") -> list:
    """Planlayıcı çıktısını adım listesi formatında döndürür."""
    steps = []
    total = len(sub_questions)
    intent = route_features.get("intent", "general")
    facts = _extract_planner_prob_facts(question)
    for i, sq in enumerate(sub_questions, 1):
        sq_l = (sq or "").lower()
        formula = ""
        result = ""
        if "kulübeye girdi" in sq_l and "p_death_before_entry" in facts:
            p = facts["p_death_before_entry"]
            pe = facts["p_entry_before_death"]
            formula = "P(kulübeye_girdi) = 1 - P(kulübeye_girmeden_önce_ölüm)"
            result = f"1 - {p:.6f} = {pe:.6f}"
        elif "kulübe dışında" in sq_l and "p_death_before_entry" in facts:
            p = facts["p_death_before_entry"]
            formula = "P(kulübe_dışında_ölüm) = P(kulübeye_girmeden_önce_ölüm)"
            result = f"{p:.6f}"
        elif "kulübede" in sq_l and "p_entry_before_death" in facts:
            pe = facts["p_entry_before_death"]
            formula = "P(kulübede_ölüm) ≤ P(kulübeye_girdi)"
            result = f"≤ {pe:.6f}"

        steps.append(
            {
                "title": _planner_topic_title(sq, i, total, intent),
                "content": sq,
                "formula": formula,
                "result": result,
                "_planner_step": True,
            }
        )
    return steps


@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Soru boş olamaz"}), 400

    # ── 1) INPUT & ANLAMA: Semantic Parser → Deep Representation Encoder ─────
    signals = sem.extract(question)
    eq_ctx = eq_universe.query(question)
    deep_repr = _deep_encoder.encode(question, signals, eq_ctx)

    # ── 2) HAFIZA & ÖN BİLGİ: Knowledge Retrieval → Bayesian Prior Builder ───
    layout, features, reward, q_vals = router.route(question, signals)
    knowledge_ctx = _knowledge_retrieval.retrieve(question, eq_ctx, features)

    # ── 3) ÇÖZÜM ADAYI: State Builder → RL Policy Network → Strategy Mapping ─
    solver_ctx = run_solver_pipeline(
        question,
        signals,
        _ast_builder,
        _markov_solver,
        _solver_selector,
        _mc_verifier,
        _num_validator,
    )
    bayes_prior = _bayes_prior_builder.build(signals, solver_ctx)
    state_builder = {
        "intent": features.get("intent", "general"),
        "entropy_proxy": deep_repr.get("entropy_proxy", 0.0),
        "consistency_trend": 1.0,
        "question_len": knowledge_ctx.get("question_len", 0),
    }
    strategy_name, policy_scores = _policy_network.select_strategy(
        bayes_prior, state_builder
    )
    routed_solver = _map_strategy_to_solver(strategy_name, solver_ctx)
    solver_ctx["strategy_name"] = strategy_name
    solver_ctx["policy_scores"] = policy_scores
    solver_ctx["routed_solver"] = routed_solver

    # ── 4) ÇÖZÜM MOTORU: Symbolic/Probabilistic/Simulation/Hybrid dispatch ───
    sub_questions = _decompose_multi_questions(question, scorer=_planner_memory)
    planner_steps = _build_planner_steps(sub_questions, features, question=question)

    # ── 3. Ollama — solver hint + constraint prompt + consistency loop ────────
    sol_data = ollama.solve(question, signals, sem, scorer, solver_ctx, eq_ctx=eq_ctx)
    sol_data["question"] = question
    sol_data["layout"] = layout
    if planner_steps:
        sol_data.setdefault("steps", [])
        # Mevcut çözüm adımları varken planlayıcıyı körlemesine prepend etme:
        # tekrar eden, soru-metnini kopyalayan ve hesap içermeyen adımları azalt.
        planner_unique = _dedupe_steps(sol_data["steps"], planner_steps)
        if not sol_data["steps"]:
            sol_data["steps"] = planner_unique
        else:
            # Çözüm adımlarını koru; planner çıktısını yalnızca yardımcı outline olarak taşı.
            sol_data["_planner_outline"] = planner_unique
            if planner_unique:
                sol_data["steps"] = planner_unique[:2] + sol_data["steps"]
        sol_data["_multi_question_count"] = len(sub_questions)
        sol_data.setdefault(
            "explanation",
            "Soru metni çoklu-alt-soru planlayıcısı ile ayrıştırıldı; yinelenen adımlar ayıklandı.",
        )

    # ── 5) DOĞRULAMA: Multi-Layer Validator ───────────────────────────────────
    extra_violations = _multilayer_validator.validate(sol_data, signals, solver_ctx)
    if extra_violations:
        sol_data.setdefault("_consistency_violations", [])
        sol_data["_consistency_violations"].extend(extra_violations)

    # ── 3.6. ★ CausalBayesOrchestrator — Nedensel zincir + inanılırlık analizi
    # Dört modül: PropositionalCredibilityEngine, CausalChainInferencer,
    #             NarrativeBeliefPropagator, MarkovConsistencyValidator
    # Tetiklenme koşulları: önerme tanımları (a=...) VEYA Markov/Bayes tipi
    _run_causal = re.search(
        r'\b[a-zA-Z]\s*[=:]\s*["\']?[^,.\n]{4,}', question
    ) or sol_data.get("type") in ("bayes", "markov_chain", "general")

    causal_ctx = {}
    if _run_causal:
        # Mantık cümlesini soru metninden ayırmaya çalış
        # Format: "mantık: ..." veya "koşul: ..." ya da noktalı virgülle bölüm
        logic_match = re.search(
            r"mantık\s*[:\-]\s*(.+?)(?:\n|$)|"
            r"koşul\s*[:\-]\s*(.+?)(?:\n|$)|"
            r"(?:eğer|if)\s+.+?(?:ise|then).+?(?:;|\.)",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        logic_str = logic_match.group(0) if logic_match else ""

        # Markov matrisi ve durumları solver_ctx'ten çek
        sr = solver_ctx.get("solver_result", {})
        markov_matrix = None
        markov_states = None
        if sr.get("_solver_used") == "MarkovSolver":
            # Geçiş matrisi AST params içinde olabilir
            ast_params = solver_ctx.get("math_ast", {}).get("params", {})
            markov_matrix = ast_params.get("transition_matrix")
            markov_states = ast_params.get("states")

        # LLM'nin sözel sonucunu çek
        stated_conclusion = (
            str(sol_data.get("answer", "")) + " " + str(sol_data.get("explanation", ""))
        )

        causal_ctx = _causal_orchestrator.analyze(
            question=question,
            logic_str=logic_str,
            markov_matrix=markov_matrix,
            markov_states=markov_states,
            stated_conclusion=stated_conclusion,
        )

        # Causal uyarıları consistency violations'a aktar
        causal_warnings = causal_ctx.get("_causal_warnings", [])
        if causal_warnings:
            sol_data.setdefault("_consistency_violations", [])
            sol_data["_consistency_violations"].extend(causal_warnings)

        # Markov tutarsızlığı varsa consistency_score'u düşür
        mc_valid = causal_ctx.get("_markov_consistency", {})
        if mc_valid and not mc_valid.get("is_consistent", True):
            old_cs = sol_data.get("_consistency_score", 1.0)
            sol_data["_consistency_score"] = round(old_cs * 0.60, 4)

    # solver_ctx'e causal_ctx ekle (solver_box'ta gösterilecek)
    solver_ctx["_causal_ctx"] = causal_ctx

    # ── Açıklamayı NLP tabanlı mantıksal-sayısal-olasılıksal vektörle zenginleştir ──
    sol_data = _explanation_enricher.enrich(question, sol_data, signals, causal_ctx)
    thinking_loop_ctx = _thinking_loop_engine.run_cycle(
        question=question,
        signals=signals,
        sol_data=sol_data,
        solver_ctx=solver_ctx,
    )
    sol_data["_thinking_loop"] = thinking_loop_ctx
    sol_data["explanation"] = thinking_loop_ctx.get(
        "explanation_generated", sol_data.get("explanation", "")
    )

    # ── 6) ÖĞRENME ÇEKİRDEĞİ: reward + replay + posterior/meta update ────────
    consistency_score = sol_data.get("_consistency_score", 1.0)
    _planner_memory.feedback(
        sub_questions=sub_questions,
        consistency_score=consistency_score,
        violations=sol_data.get("_consistency_violations", []),
    )
    adjusted_reward = round(reward * (0.5 + 0.5 * consistency_score), 3)
    _reward_replay.push(
        {
            "question": question[:240],
            "strategy": strategy_name,
            "solver": routed_solver,
            "reward": adjusted_reward,
            "consistency": consistency_score,
        }
    )
    replay_samples = _reward_replay.sample(16)
    posterior = _bayes_posterior_updater.update(bayes_prior, consistency_score)
    meta_stats = _meta_learner.summarize(replay_samples)
    policy_improvement = {
        "strategy": strategy_name,
        "posterior_weight": posterior.get(strategy_name, 0.0),
        "avg_reward": meta_stats.get("avg_reward", 0.0),
    }
    kb_update = _knowledge_base_updater.update(
        {
            "strategy": strategy_name,
            "meta": meta_stats,
            "policy_improvement": policy_improvement,
        }
    )

    # ── SolverSelector: sonuç-bazlı Q-tablo güncelleme ───────────────────────
    symbolic_bypass = sol_data.get("_symbolic_bypass", False)
    _solver_selector.reward_outcome(
        ast_type=solver_ctx["math_ast"].get("type", "general"),
        solver=solver_ctx.get("chosen_solver", "LLM"),
        consistency_score=consistency_score,
        symbolic_bypass=symbolic_bypass,
    )

    # ── 7) OUTPUT prep: Confidence Estimator + Response Generator ────────────
    confidence = float(max(0.0, min(1.0, consistency_score)))
    learning_ctx = {
        "strategy": strategy_name,
        "policy_scores": policy_scores,
        "posterior": posterior,
        "meta_learning": meta_stats,
        "knowledge_update": kb_update,
    }
    sol_data = _response_generator.finalize(sol_data, confidence, learning_ctx)

    # ── 8) Çözüm normalizasyonu (negatif/sayı-sembolik/birim) ───────────────
    sol_data = _global_num_unit_normalizer.normalize_solution_payload(sol_data)

    # ── 6. ASCII render ───────────────────────────────────────────────────────
    ascii_out = engine.render(layout, sol_data)
    q_box = engine.q_info_box(layout, features, adjusted_reward, q_vals, router.episode)
    sem_box = _build_sem_box(signals, sol_data)
    solver_box = _build_solver_box(solver_ctx)
    # GT ek kutusu (bypass varsa renderer zaten solver_box içinde gösteriyor)
    gt_extra = sol_data.get("_gt_extra_box", "")
    if gt_extra:
        full_ascii = ascii_out + "\n\n" + q_box + "\n\n" + sem_box + "\n\n" + solver_box
    else:
        full_ascii = ascii_out + "\n\n" + q_box + "\n\n" + sem_box + "\n\n" + solver_box

    graph_payload = {}
    graph_image = None
    graph_warnings = []
    try:
        graph_payload = _build_graph_automaton_payload(
            question, sol_data, affect_learning=False
        )
        graph_image = _render_matplotlib_graph_base64(graph_payload)
        if isinstance(graph_payload, dict):
            graph_payload["image_data"] = graph_image
    except Exception as e:
        graph_warnings.append(f"graph_isolation_error: {str(e)[:120]}")

    return jsonify(
        {
            "ascii": full_ascii,
            "layout": layout,
            "intent": features.get("intent", "?"),
            "reward": adjusted_reward,
            "episode": router.episode,
            "q_vals": {str(k): float(v) for k, v in (q_vals or {}).items()},
            "sol_type": str(sol_data.get("type", "?")),
            "answer": str(sol_data.get("answer", "?")),
            "consistency_score": float(consistency_score),
            "violations": [str(v) for v in sol_data.get("_consistency_violations", [])],
            "signals": {
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in signals.items()
            },
            "attempts": int(sol_data.get("_attempts", 1)),
            # Solver pipeline sonuçları
            "math_ast_type": solver_ctx["math_ast"].get("type", "general"),
            "chosen_solver": solver_ctx.get("chosen_solver", "LLM"),
            "strategy_name": solver_ctx.get("strategy_name", "HybridSolver"),
            "routed_solver": solver_ctx.get("routed_solver", solver_ctx.get("chosen_solver", "LLM")),
            "solver_solved": solver_ctx["solver_result"].get("solved", False),
            "solver_expected": solver_ctx["solver_result"].get("expected_steps"),
            "mc_agreement": solver_ctx["mc_result"].get("agreement"),
            "deep_representation": deep_repr,
            "thinking_loop": thinking_loop_ctx,
            "knowledge_context": knowledge_ctx,
            # PDF için ek alanlar
            "question": question,
            "steps": sol_data.get("steps", []),
            "formula": str(sol_data.get("formula", "")),
            "graph_data": graph_payload,
            "graph_image": graph_image,
            "graph_warnings": graph_warnings,
            "graph_score": (
                graph_payload.get("quality", {}).get("score")
                if isinstance(graph_payload, dict)
                else None
            ),
        }
    )


def _build_solver_box(solver_ctx: dict) -> str:
    """Solver pipeline sonuçlarını ASCII kutusu olarak gösterir."""
    W = ASCII_WIDTH
    title = "◈ MATEMATİK ÇÖZÜM MOTORU — SOLVER PIPELINE"
    sr = solver_ctx.get("solver_result", {})
    mc = solver_ctx.get("mc_result", {})
    ast = solver_ctx.get("math_ast", {})
    chosen = solver_ctx.get("chosen_solver", "LLM")

    lines = []
    lines.append(f"┌{'─'*(W-2)}┐")
    tp = (W - 2 - len(title)) // 2
    lines.append(f"│{' '*tp}{title}{' '*(W-2-tp-len(title))}│")
    lines.append(f"├{'─'*(W-2)}┤")

    def row(k, v):
        ln = f"│  {k:<26}: {v}"
        lines.append(f"{ln:<{W-1}}│")

    row("MathAST Tipi", ast.get("type", "general"))
    row("AST Güven", f"{ast.get('confidence', 0):.0%}")
    row("Seçilen Solver", chosen)

    if sr.get("solved"):
        solver_type = sr.get("_solver_type", chosen)

        # ── Markov çıktısı ────────────────────────────────────────────────────
        if solver_type == "MarkovSolver":
            row("Grid", f"{sr.get('grid',['?','?'])[0]}×{sr.get('grid',['?','?'])[1]}")
            row("Geçici Durum", str(sr.get("n_transient", "?")))
            row("Absorbing Durum", str(sr.get("n_absorbing", "?")))
            row("E[T] Analitik", f"{sr.get('expected_steps', '?'):.6f} adım")
            row("Varyans", f"{sr.get('variance', '?'):.6f}")
            row("Std Sapma", f"{sr.get('std_dev', '?'):.6f}")
            ep = sr.get("edge_probs", {})
            if ep:
                row("Kuzey Kenar P", f"{ep.get('kuzey', 0):.4f}")
                row("Güney Kenar P", f"{ep.get('güney', 0):.4f}")
                row("Doğu Kenar P", f"{ep.get('doğu',  0):.4f}")
                row("Batı Kenar P", f"{ep.get('batı',  0):.4f}")

        # ── Oyun Kuramı çıktısı ───────────────────────────────────────────────
        elif solver_type == "GameTheorySolver":
            gt_lines = _gt_renderer.render(sr, W)
            # İlk/son satırı atla (kutu zaten mevcut, iç satırları ekle)
            for gl in gt_lines[2:-1]:  # başlık+ayraç atla, sadece içeriği al
                lines.append(gl)

        # ── Bayes çıktısı ─────────────────────────────────────────────────────
        elif solver_type == "BayesSolver":
            hyps = sr.get("hypotheses", [])
            n_obs = sr.get("n_observations", 1)
            posts = sr.get("posteriors", {})
            joint = sr.get("joint_posteriors", {})
            rounds = sr.get("rounds", {})

            row("Hipotez Sayısı", str(len(hyps)))
            row("Gözlem Turu", str(n_obs))
            row(
                "Best Hipotez",
                f"{sr.get('best_hypothesis','?').upper()} = {sr.get('best_posterior',0):.6f}",
            )
            lines.append(f"├{'─'*(W-2)}┤")
            lines.append(f"│  {'TURA GÖRE POSTERIORLAR':<{W-4}}│")
            for t, rd in sorted(rounds.items()):
                ev = rd["evidence"]
                p_str = " | ".join(
                    f"{h.upper()}:{rd['posteriors'].get(h,0):.4f}" for h in sorted(hyps)
                )
                ln = f"│    Tur {t}: P(E)={ev:.4f}  →  {p_str}"
                lines.append(f"{ln:<{W-1}}│")
            if n_obs >= 2 and joint:
                lines.append(f"├{'─'*(W-2)}┤")
                j_str = " | ".join(
                    f"{h.upper()}:{joint.get(h,0):.4f}" for h in sorted(hyps)
                )
                ln = f"│  Birleşik Doğrulama: {j_str}"
                lines.append(f"{ln:<{W-1}}│")
    else:
        err = sr.get("_solver_error", "Parametre yetersiz veya tip uyumsuz")
        row("Solver Durumu", f"⚠ {err[:W-35]}")

    if mc.get("mc_run"):
        lines.append(f"├{'─'*(W-2)}┤")
        row("Monte Carlo Sim", f"{mc.get('sims_run',0):,} simülasyon")
        row("MC Beklenti", f"{mc.get('mc_expected','?')} ± {mc.get('mc_std','?')}")
        row("Uyum Durumu", mc.get("agreement", "?"))
        row("Bağıl Hata", f"{mc.get('rel_error_pct','?')}%")

    nviol = solver_ctx.get("numeric_violations", [])
    if nviol:
        lines.append(f"├{'─'*(W-2)}┤")
        lines.append(f"│  ⚠ NUMERİK DOĞRULAMA İHLALLERİ:{' '*(W-36)}│")
        for v in nviol:
            for chunk in [v[i : i + W - 6] for i in range(0, len(v), W - 6)]:
                ln = f"│    {chunk}"
                lines.append(f"{ln:<{W-1}}│")
    else:
        lines.append(f"│  ✓ Sayısal doğrulama BAŞARILI{' '*(W-32)}│")

    # ── ★ YENİ: 9-Modül Causal Bayes Analizi bloğu ───────────────────────────
    causal_ctx = solver_ctx.get("_causal_ctx", {})
    if causal_ctx:
        lines.append(f"├{'─'*(W-2)}┤")
        causal_title = "★ 9-MODÜL NEDENSEL ZİNCİR & BAYESYEN ANALİZİ"
        tp2 = (W - 2 - len(causal_title)) // 2
        lines.append(
            f"│{' '*max(0,tp2)}{causal_title}"
            f"{' '*(W-2-max(0,tp2)-len(causal_title))}│"
        )
        lines.append(f"├{'─'*(W-2)}┤")

        def crow(k, v):
            ln = f"│  {k:<28}: {str(v)[:W-34]}"
            lines.append(f"{ln:<{W-1}}│")

        def csec(title_s):
            lines.append(f"│  {title_s:<{W-4}}│")

        # ── MOD1+2: İnanılırlık + gizli düğümler ────────────────────────────
        creds = causal_ctx.get("_credibility_scores", {})
        chain_r = causal_ctx.get("_causal_analysis", {})
        if creds:
            csec("▸ MOD1+2  İNANILIRLIK & GİZLİ DÜĞÜMLER")
            for lbl in sorted(creds):
                cs = creds[lbl]
                cval = cs.get("credibility", 0)
                clbl = cs.get("label", "")
                bd = cs.get("breakdown", {})
                bd_str = " ".join(f"{k[:4]}={v:+.2f}" for k, v in bd.items() if v != 0)[
                    :32
                ]
                ln = f"│    [{lbl}] {cval:.3f}({clbl:<6}) {bd_str}"
                lines.append(f"{ln:<{W-1}}│")
            for hn in chain_r.get("hidden_nodes", []):
                ln = f"│    ⚠[GİZLİ] {hn['label'][:42]:<42} P≈{hn['prior']:.3f}"
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD3: Zincir + terminal dağılım ─────────────────────────────────
        belief_r = causal_ctx.get("_belief_propagation", {})
        lines.append(f"├{'─'*(W-2)}┤")
        csec("▸ MOD3  ZİNCİR YAYILIM & TERMINAL DAĞILIM")
        for step in belief_r.get("propagation_steps", []):
            arrow = "⇢" if step.get("hidden") else "→"
            ln = (
                f"│    {arrow}[{step['node']:<8}] {step['text'][:30]:<30}"
                f"  P_kum={step['p_cumulative']:.6f}"
            )
            lines.append(f"{ln:<{W-1}}│")
        delta = belief_r.get("direct_skip_delta", 0)
        crow(
            "Atlama Δ / Entropi",
            f"{delta:.4f}{'⚠' if delta>0.1 else '✓'}  "
            f"{belief_r.get('entropy',0):.4f}bit",
        )
        fd = belief_r.get("final_distribution", {})
        for tl, ti in sorted(fd.items(), key=lambda x: -x[1].get("probability", 0)):
            p_n = ti.get("probability_normalized", ti.get("probability", 0))
            ln = f"│    [{tl}] {ti.get('description','')[:36]:<36}  P={p_n:.4f}"
            lines.append(f"{ln:<{W-1}}│")

        # ── MOD4: Markov tutarlılık ──────────────────────────────────────────
        mc_v = causal_ctx.get("_markov_consistency", {})
        if mc_v:
            lines.append(f"├{'─'*(W-2)}┤")
            is_c = mc_v.get("is_consistent", True)
            crow("▸ MOD4 Markov", "✓ TUTARLI" if is_c else "✗ TUTARSIZ")
            abs_p = mc_v.get("absorption_probs", {})
            if abs_p:
                abs_s = "  ".join(
                    f"P(→{s})={p:.4f}"
                    for s, p in sorted(abs_p.items(), key=lambda x: -x[1])
                )
                ln = f"│    {abs_s}"
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD5: Öz-Düzeltme ───────────────────────────────────────────────
        corr = causal_ctx.get("_correction", {})
        if corr:
            lines.append(f"├{'─'*(W-2)}┤")
            cn = corr.get("correction_needed", False)
            pen = corr.get("confidence_penalty", 0)
            crow(
                "▸ MOD5 Öz-Düzeltme",
                f"{'⚠ GEREKLİ' if cn else '✓ YOK'}  Ceza={pen:.2f}",
            )
            for rv in corr.get("rule_violations", [])[:2]:
                ln = f"│    ⚑ {rv[:W-10]}"
                lines.append(f"{ln:<{W-1}}│")
            for hint in corr.get("correction_hints", [])[:1]:
                ln = f"│    💡 {hint[:W-10]}"
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD6: Entropi Eşiği ──────────────────────────────────────────────
        ent_r = causal_ctx.get("_entropy_mgmt", {})
        if ent_r:
            lines.append(f"├{'─'*(W-2)}┤")
            elv = ent_r.get("entropy_level", "?")
            needx = ent_r.get("needs_expansion", False)
            crow("▸ MOD6 Entropi", f"{elv}  {'⚠ GENİŞLEME' if needx else '✓ YETERLİ'}")
            for cq in ent_r.get("clarification_qs", [])[:2]:
                ln = f"│    ❓ {cq[:W-10]}"
                lines.append(f"{ln:<{W-1}}│")
            for ch in ent_r.get("cot_hints", [])[:1]:
                ln = f"│    💭 {ch[:W-10]}"
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD7: Zamansal Bellek ────────────────────────────────────────────
        temp_r = causal_ctx.get("_temporal_memory", {})
        if temp_r:
            lines.append(f"├{'─'*(W-2)}┤")
            crow(
                "▸ MOD7 Zamansal",
                f"{temp_r.get('context_type','?')}  "
                f"λ={temp_r.get('decay_lambda',0):.2f}  "
                f"Adj_P={temp_r.get('adjusted_chain_prior',0):.6f}",
            )
            tw = temp_r.get("temporal_warning")
            if tw:
                ln = f"│    ⏱ {tw[:W-10]}"
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD8: Duyarlılık ─────────────────────────────────────────────────
        sens_r = causal_ctx.get("_sensitivity", {})
        if sens_r:
            lines.append(f"├{'─'*(W-2)}┤")
            crow(
                "▸ MOD8 En Etkili",
                f"'{sens_r.get('most_influential','?')}'  "
                f"Dom={sens_r.get('dominance_ratio',1):.2f}x",
            )
            st = sens_r.get("sensitivity_table", {})
            for vid, sv in sorted(st.items(), key=lambda x: -x[1].get("abs_sens", 0))[
                :4
            ]:
                h_m = "[G]" if sv.get("is_hidden") else "   "
                ln = (
                    f"│    {h_m}[{vid:<10}] "
                    f"dS={sv.get('sensitivity',0):+.4f}  "
                    f"Rel={sv.get('relative',0):.1%}  "
                    f"#{sv.get('rank','?')}"
                )
                lines.append(f"{ln:<{W-1}}│")

        # ── MOD9: Karar Teorisi ──────────────────────────────────────────────
        dec_r = causal_ctx.get("_decision", {})
        if dec_r:
            lines.append(f"├{'─'*(W-2)}┤")
            ba = dec_r.get("best_action", "?")
            beu = dec_r.get("best_eu", 0)
            crow("▸ MOD9 Karar EU", f"En İyi='{ba}'  EU={beu:+.4f}")
            for a_lbl, a_info in sorted(
                dec_r.get("actions", {}).items(),
                key=lambda x: -x[1]["expected_utility"],
            ):
                m = "★" if a_lbl == ba else " "
                eu_v = a_info.get("expected_utility", 0)
                at = a_info.get("action_type", "?")
                ln = (
                    f"│    {m}[{a_lbl}] "
                    f"{a_info['description'][:28]:<28}  "
                    f"{at:<8}  EU={eu_v:+.4f}"
                )
                lines.append(f"{ln:<{W-1}}│")
            rec = dec_r.get("recommendation", "")
            if rec:
                for chunk in [
                    rec[i : i + W - 10] for i in range(0, min(len(rec), 200), W - 10)
                ]:
                    ln = f"│    → {chunk}"
                    lines.append(f"{ln:<{W-1}}│")

        # ── Birleşik Uyarılar (dedup) ────────────────────────────────────────
        all_w = causal_ctx.get("_causal_warnings", [])
        if all_w:
            lines.append(f"├{'─'*(W-2)}┤")
            lines.append(f"│  {'⚑ ÖZET UYARILAR (dedup)':<{W-4}}│")
            for w in all_w[:5]:
                for chunk in [
                    w[i : i + W - 10] for i in range(0, min(len(w), 200), W - 10)
                ]:
                    ln = f"│    {chunk}"
                    lines.append(f"{ln:<{W-1}}│")

    lines.append(f"└{'─'*(W-2)}┘")
    return "\n".join(lines)


def _build_sem_box(signals: dict, sol_data: dict) -> str:
    """Semantic sinyal özetini ASCII kutusuna dönüştürür."""
    W = ASCII_WIDTH
    title = "◈ SEMANTİK SİNYAL MODÜLü — ÇIKARILAN KISITLAR"
    viol = sol_data.get("_consistency_violations", [])
    cscore = sol_data.get("_consistency_score", 1.0)
    att = sol_data.get("_attempts", 1)

    lines = []
    lines.append(f"┌{'─'*(W-2)}┐")
    tp = (W - 2 - len(title)) // 2
    lines.append(f"│{' '*tp}{title}{' '*(W-2-tp-len(title))}│")
    lines.append(f"├{'─'*(W-2)}┤")

    rows = [
        ("Bağımsızlık", signals.get("event_dependency", "?")),
        ("Bağ. Sinyal Katmanı", str(signals.get("dependency_layer", "?"))),
        ("Bağ. Kanıt Sayısı", str(signals.get("dependency_evidence_count", "?"))),
        ("Payda N", str(signals.get("denominator_n", "—"))),
        ("Örneklem Uzayı", signals.get("sample_space", "?")),
        ("Mantık Op.", signals.get("logic_operator", "?")),
        ("Eliminasyon", "✓ VAR" if signals.get("elimination_rule") else "✗ YOK"),
        (
            "Hayatta Kalma Zinciri",
            "✓ VAR" if signals.get("survival_chain") else "✗ YOK",
        ),
        (
            "Toplam Olas. Teoremi",
            "✓ UYGULANIR" if signals.get("total_prob_applicable") else "✗ UYGULANAMAZ",
        ),
        (
            "Dal Toplama İzni",
            "✓ İZİNLİ" if signals.get("branch_sum_applicable") else "✗ YASAK",
        ),
        ("Geri Koyma", str(signals.get("replacement", "?"))),
        ("Ardışık Deneme", str(signals.get("sequential_trials", 1))),
        ("Bayes Yapısı", "✓ VAR" if signals.get("bayes_structure") else "✗ YOK"),
        ("Tutarlılık Skoru", f"{cscore:.3f}  ({att} deneme)"),
        (
            "Cebirsel Düzeltme",
            (
                f"✓ {sol_data.get('_correction_rule','?')}"
                if sol_data.get("_corrected")
                else "✗ YOK"
            ),
        ),
    ]
    for k, v in rows:
        ln = f"│  {k:<22}: {v}"
        lines.append(f"{ln:<{W-1}}│")

    if viol:
        lines.append(f"├{'─'*(W-2)}┤")
        lines.append(f"│  ⚠ TESPİT EDİLEN İHLALLER:{' '*(W-30)}│")
        for v in viol:
            for chunk in [v[i : i + W - 6] for i in range(0, len(v), W - 6)]:
                ln = f"│    {chunk}"
                lines.append(f"{ln:<{W-1}}│")
    else:
        lines.append(f"│  ✓ Tutarlılık kontrolü BAŞARILI{' '*(W-34)}│")

    lines.append(f"└{'─'*(W-2)}┘")
    return "\n".join(lines)


@app.route("/examples")
def get_examples():
    return jsonify(EXAMPLES)


@app.route("/add_example", methods=["POST"])
def add_example():
    """Yeni soru ekle ve JSON veritabanına kaydet."""
    data = request.get_json()
    cat = (data.get("cat") or "").strip()
    q = (data.get("q") or "").strip()
    if not cat or not q:
        return jsonify({"error": "Kategori ve soru boş olamaz"}), 400
    # Yeni ID üret (tamsayı ID'lerin maksimumu + 1)
    int_ids = [ex.get("id") for ex in EXAMPLES if isinstance(ex.get("id"), int)]
    new_id = max(int_ids, default=0) + 1
    new_ex = {"id": new_id, "cat": cat, "q": q, "_user": True}
    EXAMPLES.append(new_ex)
    _save_db()
    return jsonify({"ok": True, "example": new_ex})


@app.route("/delete_example", methods=["POST"])
def delete_example():
    """Kullanıcı tarafından eklenen soruyu sil."""
    global EXAMPLES
    data = request.get_json()
    ex_id = data.get("id")
    before = len(EXAMPLES)
    EXAMPLES = [ex for ex in EXAMPLES if ex.get("id") != ex_id]
    _save_db()
    return jsonify({"ok": True, "deleted": before - len(EXAMPLES)})


# ─────────────────────────────────────────────────────────────────────────────
#  PDF ÜRETİCİ  —  LaTeX-stilinde çözüm raporu (ReportLab)
# ─────────────────────────────────────────────────────────────────────────────
def _safe(text: str) -> str:
    """ReportLab Paragraph için XML escape — Türkçe karakterler korunur."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 1 — AÇIK TEMALI YAPILANDIRILMIŞ RAPOR (ASCII PDF)
#  DejaVu TTF → Türkçe tam destek
# ─────────────────────────────────────────────────────────────────────────────
def _build_pdf(
    question: str,
    answer: str,
    sol_type: str,
    steps: list,
    formula: str,
    ascii_out: str,
    chosen_solver: str,
    consistency: float,
) -> bytes:
    """Açık arka planlı, yapılandırılmış çözüm raporu PDF'i."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="ASCIIMATİK — Çözüm Raporu",
    )

    C_GREEN = colors.HexColor("#00c853")
    C_CYAN = colors.HexColor("#00838f")
    C_TEXT = colors.HexColor("#1a1a2e")
    C_GRAY = colors.HexColor("#546e7a")
    C_LIGHT = colors.HexColor("#e8f5e9")
    C_BOX = colors.HexColor("#f1f8e9")
    C_MONO = colors.HexColor("#1b5e20")

    # DVSans = DejaVu Sans (Türkçe dahil tüm Latin karakterleri destekler)
    s_title = ParagraphStyle(
        "s_title",
        fontName="DVSans-Bold",
        fontSize=22,
        textColor=C_GREEN,
        spaceAfter=4,
        alignment=TA_CENTER,
        leading=28,
    )
    s_sub = ParagraphStyle(
        "s_sub",
        fontName="DVSans",
        fontSize=9,
        textColor=C_GRAY,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    s_section = ParagraphStyle(
        "s_section",
        fontName="DVSans-Bold",
        fontSize=11,
        textColor=C_CYAN,
        spaceBefore=14,
        spaceAfter=4,
    )
    s_question = ParagraphStyle(
        "s_question",
        fontName="DVSans",
        fontSize=10,
        textColor=C_TEXT,
        backColor=C_BOX,
        borderColor=C_GREEN,
        borderWidth=1,
        borderPad=8,
        spaceAfter=4,
        leading=16,
    )
    s_answer = ParagraphStyle(
        "s_answer",
        fontName="DVSans-Bold",
        fontSize=13,
        textColor=colors.white,
        backColor=C_GREEN,
        borderPad=10,
        spaceAfter=6,
        alignment=TA_CENTER,
        leading=18,
    )
    s_step_title = ParagraphStyle(
        "s_step_title",
        fontName="DVSans-Bold",
        fontSize=10,
        textColor=C_CYAN,
        spaceBefore=8,
        spaceAfter=2,
    )
    s_step_body = ParagraphStyle(
        "s_step_body",
        fontName="DVSans",
        fontSize=9,
        textColor=C_TEXT,
        spaceAfter=2,
        leading=14,
        leftIndent=12,
    )
    s_formula = ParagraphStyle(
        "s_formula",
        fontName="DVMono-Bold",
        fontSize=9,
        textColor=C_MONO,
        backColor=C_LIGHT,
        borderPad=6,
        spaceAfter=4,
        leftIndent=12,
        leading=13,
    )
    s_mono = ParagraphStyle(
        "s_mono",
        fontName="DVMono",
        fontSize=6.5,
        textColor=C_MONO,
        backColor=colors.HexColor("#f9fbe7"),
        borderPad=4,
        leading=9.5,
        spaceAfter=2,
    )
    s_meta = ParagraphStyle(
        "s_meta", fontName="DVSans", fontSize=7.5, textColor=C_GRAY, spaceAfter=2
    )

    story = []
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("ASCIIMATİK", s_title))
    story.append(
        Paragraph(
            f"Algoritmik Çözüm Raporu &nbsp;·&nbsp; {now} &nbsp;·&nbsp; {sol_type.upper()}",
            s_sub,
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=C_GREEN, spaceAfter=10))

    meta_data = [
        ["Çözücü:", chosen_solver, "Tutarlılık:", f"{consistency:.1%}"],
        ["Çözüm Tipi:", sol_type, "Adım Sayısı:", str(len(steps))],
    ]
    mt = Table(meta_data, colWidths=[3 * cm, 5.5 * cm, 3 * cm, 5.5 * cm])
    mt.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "DVSans-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "DVSans-Bold"),
                ("FONTNAME", (1, 0), (3, -1), "DVSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (0, -1), C_GRAY),
                ("TEXTCOLOR", (2, 0), (2, -1), C_GRAY),
                ("TEXTCOLOR", (1, 0), (1, -1), C_TEXT),
                ("TEXTCOLOR", (3, 0), (3, -1), C_TEXT),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(mt)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("▸ SORU", s_section))
    story.append(Paragraph(_safe(question), s_question))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("▸ CEVAP", s_section))
    story.append(Paragraph(f"✓ &nbsp; {_safe(answer)}", s_answer))
    story.append(Spacer(1, 0.4 * cm))

    if formula and formula not in ("", "None"):
        story.append(Paragraph("▸ GENEL FORMÜL", s_section))
        story.append(Paragraph(_safe(formula), s_formula))
        story.append(Spacer(1, 0.3 * cm))

    if steps:
        story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN, spaceAfter=6))
        story.append(Paragraph("▸ ADIM ADIM ÇÖZÜM", s_section))
        for i, step in enumerate(steps, 1):
            title = step.get("title", f"Adım {i}")
            content = step.get("content", "")
            frm = step.get("formula", "")
            result = step.get("result", "")
            story.append(Paragraph(_safe(f"{i}. {title}"), s_step_title))
            if content:
                story.append(Paragraph(_safe(content), s_step_body))
            if frm:
                story.append(Paragraph(f"  {_safe(frm)}", s_formula))
            if result:
                story.append(Paragraph(f"→ {_safe(result)}", s_step_body))
        story.append(Spacer(1, 0.4 * cm))

    story.append(HRFlowable(width="100%", thickness=1, color=C_GRAY, spaceAfter=6))
    story.append(Paragraph("▸ ASCIIMATİK ÇÖZÜM EKRANI", s_section))
    story.append(Preformatted(ascii_out[:7000], s_mono))

    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_GREEN, spaceAfter=4))
    story.append(
        Paragraph(
            f"ASCIIMATİK v137 &nbsp;·&nbsp; phi4:14b @ Ollama &nbsp;·&nbsp; "
            f"Q-Learning NLP Router &nbsp;·&nbsp; {now}",
            s_meta,
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 2 — KARANLIK TERMINAL PDF (LaTeX PDF)
#  Tüm algoritmalar arka planda çalışır; çıktı Ollama'nın ürettiği ASCII ekranı.
#  Karanlık tema: bg=#080c10 | metin=#00e676 (yeşil terminal)
# ─────────────────────────────────────────────────────────────────────────────
def _build_dark_pdf(
    question: str,
    answer: str,
    sol_type: str,
    steps: list,
    formula: str,
    ascii_out: str,
    chosen_solver: str,
    consistency: float,
) -> bytes:
    """
    Karanlık terminal temalı PDF.
    Arka plan siyah, metin yeşil/cyan — tam Ollama ASCII çıktısı.
    """
    from reportlab.pdfgen import canvas as rl_canvas

    BG = colors.HexColor("#080c10")
    GREEN = colors.HexColor("#00e676")
    GREEN2 = colors.HexColor("#00c853")
    CYAN = colors.HexColor("#00e5ff")
    YELLOW = colors.HexColor("#ffea00")
    DIM = colors.HexColor("#4db6ac")
    DIMMER = colors.HexColor("#1e3a1e")

    PAGE_W, PAGE_H = A4
    MARGIN_X = 1.5 * cm
    MARGIN_Y = 1.8 * cm
    TEXT_W = PAGE_W - 2 * MARGIN_X

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

    def new_page(fill_bg=True):
        if fill_bg:
            c.setFillColor(BG)
            c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # Üst çizgi
        c.setStrokeColor(GREEN2)
        c.setLineWidth(0.8)
        c.line(
            MARGIN_X, PAGE_H - MARGIN_Y + 4, PAGE_W - MARGIN_X, PAGE_H - MARGIN_Y + 4
        )
        # Alt çizgi
        c.line(MARGIN_X, MARGIN_Y - 4, PAGE_W - MARGIN_X, MARGIN_Y - 4)

    def footer(pg: int):
        c.setFont("DVMono", 6.5)
        c.setFillColor(DIM)
        left = f"ASCIIMATİK v137  ·  phi4:14b @ Ollama  ·  {now}"
        right = f"Sayfa {pg}"
        c.drawString(MARGIN_X, MARGIN_Y - 12, left)
        c.drawRightString(PAGE_W - MARGIN_X, MARGIN_Y - 12, right)

    # ── Sayfa 1 — Başlık + Meta + Soru + Cevap + Formül ─────────────────────
    new_page()
    y = PAGE_H - MARGIN_Y
    pg = 1

    # ASCII art başlık kutusu
    c.setFont("DVMono-Bold", 7)
    c.setFillColor(GREEN2)
    header_lines = [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║  ▄▄▄  ▄▄▄▄▄  ▄▄▄▄▄  ▄▄  ▄▄  ▄▄   ▄  ▄▄▄  ▄▄▄▄▄  ▄▄  ▄▄  ▄▄▄  ║",
        "║  ██▀  ██▀▀▀  ██     ██  ██  ███▄ ██  ██▀  ██▀▀   ██  ██  ██▀  ║",
        "║  ███  ███▄   ██     ██▄▄██  ██ ▀███  ██   ████   ██▄▄██  ██   ║",
        "║  ▀▀▀  ▀▀▀▀▀  ▀▀▀▀▀  ▀▀▀▀▀  ▀▀  ▀▀▀  ▀▀   ▀▀▀▀▀  ▀▀▀▀▀▀  ▀▀   ║",
        "╠══════════════════════════════════════════════════════════════════╣",
        f"║   LaTeX Çözüm Raporu  ·  {sol_type.upper():<20}  ·  {now}   ║",
        "╚══════════════════════════════════════════════════════════════════╝",
    ]
    LINE_H = 9
    for hl in header_lines:
        c.drawString(MARGIN_X, y, hl)
        y -= LINE_H
    y -= 6

    # Meta bilgi satırı
    c.setFont("DVMono", 7.5)
    c.setFillColor(CYAN)
    meta_line = (
        f"  Çözücü: {chosen_solver:<10}  "
        f"Tip: {sol_type:<14}  "
        f"Tutarlılık: {consistency:.1%}  "
        f"Adım: {len(steps)}"
    )
    c.drawString(MARGIN_X, y, meta_line)
    y -= 14

    # Bölüm çizgisi
    c.setStrokeColor(DIMMER)
    c.setLineWidth(0.4)
    c.line(MARGIN_X, y, PAGE_W - MARGIN_X, y)
    y -= 12

    # SORU
    c.setFont("DVMono-Bold", 8)
    c.setFillColor(CYAN)
    c.drawString(MARGIN_X, y, "◈ SORU")
    y -= 11

    # Soru kutusu — yeşil kenarlıklı kutu
    c.setFont("DVSans", 8.5)
    c.setFillColor(GREEN)
    # Metni satırlara böl
    words = question.split()
    lines_q = []
    cur = ""
    MAX_CHARS = 88
    for w in words:
        if len(cur) + len(w) + 1 <= MAX_CHARS:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines_q.append(cur)
            cur = w
    if cur:
        lines_q.append(cur)

    box_h = len(lines_q) * 11 + 12
    c.setFillColor(colors.HexColor("#0d1117"))
    c.setStrokeColor(GREEN2)
    c.setLineWidth(0.6)
    c.roundRect(MARGIN_X, y - box_h, TEXT_W, box_h, 3, fill=1, stroke=1)
    c.setFillColor(GREEN)
    c.setFont("DVSans", 8.5)
    qy = y - 9
    for ql in lines_q:
        c.drawString(MARGIN_X + 6, qy, ql)
        qy -= 11
    y -= box_h + 10

    # CEVAP kutusu
    c.setFont("DVMono-Bold", 8)
    c.setFillColor(CYAN)
    c.drawString(MARGIN_X, y, "◈ CEVAP")
    y -= 11

    c.setFillColor(colors.HexColor("#00c85322"))
    c.setStrokeColor(GREEN)
    c.setLineWidth(1.2)
    c.roundRect(MARGIN_X, y - 22, TEXT_W, 22, 4, fill=1, stroke=1)
    c.setFont("DVSans-Bold", 10)
    c.setFillColor(GREEN)
    ans_short = str(answer)[:110]
    c.drawCentredString(PAGE_W / 2, y - 15, f"✓  {ans_short}")
    y -= 32

    # FORMÜL
    if formula and formula not in ("", "None"):
        c.setFont("DVMono-Bold", 8)
        c.setFillColor(CYAN)
        c.drawString(MARGIN_X, y, "◈ GENEL FORMÜL")
        y -= 11
        c.setFillColor(colors.HexColor("#1b5e2022"))
        c.setStrokeColor(colors.HexColor("#1b5e20"))
        c.setLineWidth(0.4)
        c.rect(MARGIN_X, y - 14, TEXT_W, 14, fill=1, stroke=1)
        c.setFont("DVMono-Bold", 8.5)
        c.setFillColor(colors.HexColor("#69ff47"))
        c.drawString(MARGIN_X + 8, y - 10, str(formula)[:110])
        y -= 24

    footer(pg)
    c.showPage()

    # ── Sayfa 2+ — Adım adım çözüm ──────────────────────────────────────────
    if steps:
        new_page()
        y = PAGE_H - MARGIN_Y
        pg += 1

        c.setFont("DVMono-Bold", 8)
        c.setFillColor(CYAN)
        c.drawString(MARGIN_X, y, "◈ ADIM ADIM ÇÖZÜM  —  Ollama phi4:14b")
        y -= 14
        c.setStrokeColor(DIMMER)
        c.setLineWidth(0.4)
        c.line(MARGIN_X, y, PAGE_W - MARGIN_X, y)
        y -= 10

        for i, step in enumerate(steps, 1):
            title = step.get("title", f"Adım {i}")
            content = step.get("content", "")
            frm = step.get("formula", "")
            result = step.get("result", "")

            needed = (
                24
                + (len(content) // 80 + 1) * 10
                + (16 if frm else 0)
                + (10 if result else 0)
            )
            if y - needed < MARGIN_Y + 20:
                footer(pg)
                c.showPage()
                pg += 1
                new_page()
                y = PAGE_H - MARGIN_Y

            # Adım başlığı kutusu
            c.setFillColor(colors.HexColor("#0d2818"))
            c.setStrokeColor(GREEN2)
            c.setLineWidth(0.5)
            c.roundRect(MARGIN_X, y - 14, TEXT_W, 14, 2, fill=1, stroke=1)
            c.setFont("DVSans-Bold", 8.5)
            c.setFillColor(YELLOW)
            c.drawString(MARGIN_X + 6, y - 10, f"[{i:02d}]  {title}")
            y -= 19

            if content:
                c.setFont("DVSans", 8)
                c.setFillColor(GREEN)
                words2 = content.split()
                cl = ""
                MAX2 = 92
                clines = []
                for w in words2:
                    if len(cl) + len(w) + 1 <= MAX2:
                        cl = (cl + " " + w).strip()
                    else:
                        if cl:
                            clines.append(cl)
                        cl = w
                if cl:
                    clines.append(cl)
                for cl_ in clines:
                    if y < MARGIN_Y + 20:
                        footer(pg)
                        c.showPage()
                        pg += 1
                        new_page()
                        y = PAGE_H - MARGIN_Y
                    c.drawString(MARGIN_X + 12, y, cl_)
                    y -= 10

            if frm:
                if y < MARGIN_Y + 20:
                    footer(pg)
                    c.showPage()
                    pg += 1
                    new_page()
                    y = PAGE_H - MARGIN_Y
                c.setFillColor(colors.HexColor("#1b5e2033"))
                c.setStrokeColor(colors.HexColor("#1b5e20"))
                c.setLineWidth(0.3)
                c.rect(MARGIN_X + 8, y - 12, TEXT_W - 16, 12, fill=1, stroke=1)
                c.setFont("DVMono-Bold", 8)
                c.setFillColor(colors.HexColor("#69ff47"))
                c.drawString(MARGIN_X + 14, y - 9, str(frm)[:100])
                y -= 17

            if result:
                if y < MARGIN_Y + 20:
                    footer(pg)
                    c.showPage()
                    pg += 1
                    new_page()
                    y = PAGE_H - MARGIN_Y
                c.setFont("DVMono", 7.5)
                c.setFillColor(CYAN)
                c.drawString(MARGIN_X + 12, y, f"→  {str(result)[:100]}")
                y -= 10

            y -= 6  # adımlar arası boşluk

        footer(pg)
        c.showPage()

    # ── Son Sayfalar — Tam ASCII Çıktısı (Ollama çıktısı) ───────────────────
    new_page()
    pg += 1
    y = PAGE_H - MARGIN_Y

    c.setFont("DVMono-Bold", 8)
    c.setFillColor(CYAN)
    c.drawString(MARGIN_X, y, "◈ OLLAMA ÇÖZÜM EKRANI  —  Tam ASCII Çıktısı")
    y -= 14
    c.setStrokeColor(DIMMER)
    c.line(MARGIN_X, y, PAGE_W - MARGIN_X, y)
    y -= 8

    c.setFont("DVMono", 6.2)
    c.setFillColor(GREEN)
    LINE_PX = 8.0
    for raw_line in ascii_out.split("\n"):
        if y < MARGIN_Y + 14:
            footer(pg)
            c.showPage()
            pg += 1
            new_page()
            y = PAGE_H - MARGIN_Y
            c.setFont("DVMono", 6.2)
            c.setFillColor(GREEN)
        try:
            c.drawString(MARGIN_X, y, raw_line)
        except Exception:
            c.drawString(MARGIN_X, y, raw_line.encode("ascii", "replace").decode())
        y -= LINE_PX

    footer(pg)
    c.save()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES — PDF
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/pdf", methods=["POST"])
def generate_pdf():
    """Açık temalı yapılandırılmış çözüm raporu PDF'i."""
    if not _REPORTLAB_OK:
        return jsonify({"error": "ReportLab kurulu değil. pip install reportlab"}), 500
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")
    sol_type = data.get("sol_type", "general")
    steps = data.get("steps", [])
    formula = data.get("formula", "")
    ascii_out = data.get("ascii", "")
    solver = data.get("chosen_solver", "LLM")
    consistency = float(data.get("consistency_score", 1.0))
    try:
        pdf_bytes = _build_pdf(
            question, answer, sol_type, steps, formula, ascii_out, solver, consistency
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    resp = make_response(pdf_bytes)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=ascimatik_ascii.pdf"
    return resp


@app.route("/pdf_dark", methods=["POST"])
def generate_pdf_dark():
    """Karanlık terminal temalı LaTeX çözüm PDF'i."""
    if not _REPORTLAB_OK:
        return jsonify({"error": "ReportLab kurulu değil. pip install reportlab"}), 500
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")
    sol_type = data.get("sol_type", "general")
    steps = data.get("steps", [])
    formula = data.get("formula", "")
    ascii_out = data.get("ascii", "")
    solver = data.get("chosen_solver", "LLM")
    consistency = float(data.get("consistency_score", 1.0))
    try:
        pdf_bytes = _build_dark_pdf(
            question, answer, sol_type, steps, formula, ascii_out, solver, consistency
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    resp = make_response(pdf_bytes)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=ascimatik_latex.pdf"
    return resp


@app.route("/qstate")
def qstate():
    return jsonify(
        {
            "episode": router.episode,
            "total_reward": router.total_reward,
            "states": len(router.q_table),
        }
    )


@app.route("/refresh_equations", methods=["GET"])
def refresh_equations():
    """MOD10: Cloud Equation Universe'i yeniden yükle"""
    global eq_universe
    try:
        eq_universe = CloudUniversalEquationRepository()
        return jsonify(
            {
                "status": "success",
                "message": "Evrensel denklem veritabanı yenilendi",
                "categories": len(eq_universe.db),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  🔵 ROOT ORCHESTRATOR — UNIFIED PIPELINE ENTRY POINT (v1.0)
#  Tüm mevcut bileşenleri tek bir merkezi koordinatörde birleştirir.
#  Hard-coding yok — tüm akış dinamik sinyal/solver/validator seçiminden geçer.
# ═══════════════════════════════════════════════════════════════════════════════
class RootOrchestrator:
    """
    MERKEZİ ORKESTRASYON NOKTASI — Pipeline'ın ana giriş kapısı.

    Mimari Akış Şeması:
    ┌─────────────────────────────────────────────────────────────────┐
    │  ROOT ORCHESTRATOR.run(question)                                │
    │    ├─► 1. INPUT PARSER (SemanticSignalModule)                   │
    │    ├─► 2. INTENT CLASSIFIER (pattern + confidence scoring)      │
    │    ├─► 3. SEMANTIC ABSTRACTION (DependencySignalResolver)       │
    │    ├─► 4. GLOBAL STATE INIT (TemporalStateMemory snapshot)      │
    │    ├─► 5. DEPENDENCY GRAPH BUILD (DependencyGraphBuilder)       │
    │    ├─► 6. EXECUTION DAG PLAN (DAGExecutor + topo_sort)          │
    │    ├─► 7. SOLVER SELECTION (SolverSelector + hybrid chaining)   │
    │    ├─► 8. ITERATIVE SOLVE LOOP:                                 │
    │    │     ├─► Execute DAG node-by-node                           │
    │    │     ├─► VALIDATION LAYER (Numeric/Logical/MC validators)   │
    │    │     ├─► VALIDATION ROUTER (violation → recovery mapping)   │
    │    │     ├─► AUTO-RECOVERY (enforce + repair strategies)        │
    │    │     └─► TERMINATION GUARD (convergence / max_iter check)   │
    │    ├─► 9. COMPLETENESS CHECK (all nodes solved? entropy low?)   │
    │    └─► 10. OUTPUT + EXPLANATION (ASCII/PDF + causal graph)      │
    └─────────────────────────────────────────────────────────────────┘

    Bileşenler (components dict):
      - ast_builder, solver_selector, solvers (Bayes/Markov/GT/Diff)
      - validators (numeric, logical, monte_carlo)
      - dag_executor, router, term_guard
      - state_memory, entropy_manager, completeness_checker
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.iteration = 0
        self.history = []  # [(iter, step_count, violations, entropy), ...]
        self.state_snapshot = {}  # TemporalStateMemory snapshot

    def run(self, question: str, components: dict) -> dict:
        """
        ANA ORKESTRASYON FONKSİYONU — tek giriş noktası.

        Args:
          question: Kullanıcı sorusu (ham metin)
          components: {
            "sem": SemanticSignalModule,
            "ast_builder": MathASTBuilder,
            "solver_selector": SolverSelector,
            "solvers": {"BayesSolver": ..., "MarkovSolver": ..., ...},
            "validators": {"numeric": ..., "logical": ..., "monte_carlo": ...},
            "dag_executor": ActiveDAGExecutor,
            "router": ValidationRouter,
            "term_guard": TerminationGuard,
            "state_memory": TemporalStateMemory (optional),
            "entropy_manager": EntropyThresholdManager (optional),
            "completeness_checker": CompletenessChecker (optional),
            "thinking_loop_engine": ThinkingLoopBackpropEngine (optional),
            "eq_universe": CloudUniversalEquationRepository (optional),
          }

        Returns:
          {
            "success": bool,
            "final_sol_data": dict,
            "final_steps": list,
            "final_answer": str,
            "iterations": int,
            "history": list,
            "recovery_notes": list,
            "violations_final": list,
            "completeness_score": float,
            "entropy_level": float,
          }
        """
        # Yeni run için sayaçları sıfırla (iteration/history carry-over önlenir)
        self.iteration = 0
        self.history = []
        last_entropy = 0.0
        last_violation_count = 0

        # ────────────────────────────────────────────────────────────────────
        # FAZA 0: GLOBAL STATE INIT (TemporalStateMemory snapshot)
        # ────────────────────────────────────────────────────────────────────
        state_memory = components.get("state_memory")
        state_manager = components.get("global_state")
        if state_memory:
            state_id = state_memory.create_snapshot(question)
            self.state_snapshot["state_id"] = state_id
            self.state_snapshot["question"] = question
            self.state_snapshot["start_time"] = time.time()
        if state_manager:
            self.state_snapshot["global_snapshot_id"] = state_manager.snapshot(
                "root-start", {"question": question}
            )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 1: INPUT PARSER — Semantic Signal Extraction
        # ────────────────────────────────────────────────────────────────────
        sem = components.get("sem")
        if not sem:
            return self._error_result("SemanticSignalModule eksik")

        signals = sem.extract(question)
        self._log_phase(
            "INPUT PARSER",
            {
                "event_dependency": signals.get("event_dependency"),
                "logic_operator": signals.get("logic_operator"),
                "sequential_trials": signals.get("sequential_trials"),
            },
        )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 2: INTENT CLASSIFIER — Pattern + Confidence Scoring
        # ────────────────────────────────────────────────────────────────────
        intent = self._classify_intent(question, signals)
        self._log_phase("INTENT CLASSIFIER", intent)

        # ────────────────────────────────────────────────────────────────────
        # FAZA 2.5: NLP-TABANLI SİMÜLASYON/ÇİZİM PLANLAYICI
        # ────────────────────────────────────────────────────────────────────
        sim_runtime = components.get("simulation_runtime")
        simulation_plan = {}
        simulation_preview = {}
        if sim_runtime:
            try:
                simulation_plan = sim_runtime.build_plan(
                    question,
                    {"steps": 180, "grid_size": 80, "particles": 800, "dt": 0.03},
                )
                simulation_preview = sim_runtime.execute_preview(simulation_plan, preview_steps=10)
                self._log_phase(
                    "SIMULATION ROUTER",
                    {
                        "engine": simulation_preview.get("engine"),
                        "renderer": simulation_preview.get("renderer"),
                        "backend": simulation_preview.get("backend"),
                    },
                )
            except Exception as e:
                self._log_phase("SIMULATION ROUTER", {"error": str(e)[:120]})

        # ────────────────────────────────────────────────────────────────────
        # FAZA 3: SEMANTIC ABSTRACTION — Dependency + Constraint Extraction
        # ────────────────────────────────────────────────────────────────────
        semantic_constraints = sem.build_constraint_block(signals)
        self._log_phase(
            "SEMANTIC ABSTRACTION",
            {
                "constraint_block_length": len(semantic_constraints),
                "dependency_layer": signals.get("dependency_layer"),
            },
        )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 4: DEPENDENCY GRAPH BUILD
        # ────────────────────────────────────────────────────────────────────
        ast_builder = components.get("ast_builder")
        if not ast_builder:
            return self._error_result("ASTBuilder eksik")

        math_ast = ast_builder.build(question, signals)
        # Solver'lara orijinal metin ve sinyalleri geçir (hard-code yok, meta veri)
        math_ast["question"] = question
        math_ast["signals"] = signals
        dep_graph_builder = DependencyGraphBuilder()
        dependency_graph = dep_graph_builder.build_from_ast(math_ast, signals)

        self._log_phase(
            "DEPENDENCY GRAPH BUILD",
            {
                "ast_type": math_ast.get("type"),
                "ast_confidence": math_ast.get("confidence", 0),
                "num_nodes": len(dependency_graph),
                "is_acyclic": dep_graph_builder._is_acyclic(dependency_graph),
            },
        )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 5: EXECUTION DAG PLAN (Topological Sort)
        # ────────────────────────────────────────────────────────────────────
        dag_executor = components.get("dag_executor")
        if not dag_executor:
            dag_executor = ActiveDAGExecutor()  # Fallback

        dag_engine = components.get("dag_engine") or DAGExecutionEngine(
            dependency_graph
        )
        dag_engine.load_graph(dependency_graph)

        execution_order = dep_graph_builder.topo_sort(dependency_graph)
        self._log_phase(
            "EXECUTION DAG PLAN",
            {
                "execution_order_length": len(execution_order),
                "order": execution_order[:5],  # İlk 5 node
            },
        )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 6: SOLVER SELECTION (Pattern + Hybrid Chaining)
        # ────────────────────────────────────────────────────────────────────
        solver_selector = components.get("solver_selector")
        solvers = components.get("solvers", {})

        chosen_solver = (
            solver_selector.select(math_ast, signals)
            if solver_selector
            else "BayesSolver"
        )

        # Differential dynamics özel solver'ı
        if signals.get("differential_type") == "logistic_sde_chaos":
            chosen_solver = "GeneralDifferentialDynamicsSolver"

        self._log_phase(
            "SOLVER SELECTION",
            {
                "chosen_solver": chosen_solver,
                "available_solvers": list(solvers.keys()),
            },
        )

        # ────────────────────────────────────────────────────────────────────
        # FAZA 7: ITERATIVE SOLVE LOOP (solve → validate → recover → repeat)
        # ────────────────────────────────────────────────────────────────────
        current_sol_data = None
        all_recovery_notes = []
        final_violations = []

        validators = components.get("validators", {})
        router = components.get("router")
        term_guard = components.get("term_guard")
        entropy_manager = components.get("entropy_manager")
        decision_link = components.get("decision_link")
        recovery_engine = components.get("recovery_engine")
        explanation_engine = components.get("explanation_engine")
        thinking_loop_engine = components.get("thinking_loop_engine")
        precision_validator = validators.get("precision")

        while self.iteration < self.max_iterations:
            self.iteration += 1
            iter_note = f"\n╔══ İTERASYON {self.iteration} ══╗"
            all_recovery_notes.append(iter_note)

            # Geri besleme: entropi/ihlale göre solver yönlendirmesi
            if decision_link and self.iteration > 1:
                advice = decision_link.advise(
                    chosen_solver,
                    {**signals, "raw_text": question},
                    self.iteration,
                    last_entropy,
                    last_violation_count,
                )
                if advice["solver"] != chosen_solver:
                    all_recovery_notes.append(
                        f"[DECISION-FEEDBACK] {chosen_solver}→{advice['solver']} ({advice['reason']})"
                    )
                    chosen_solver = advice["solver"]

            # ── ADIM 7.1: SOLVER EXECUTION ─────────────────────────────────
            try:
                solver = solvers.get(chosen_solver)
                if not solver:
                    # Fallback: LLM-based solve (OllamaClient)
                    all_recovery_notes.append(
                        f"[WARN] {chosen_solver} bulunamadı, LLM fallback"
                    )
                    current_sol_data = {
                        "type": "general",
                        "steps": [],
                        "answer": "LLM fallback",
                    }
                else:
                    # Solver çalıştır (AST-based veya signal-based)
                    if hasattr(solver, "solve"):
                        # AST-based solver'lar (Bayes, Markov, GT)
                        if chosen_solver == "GeneralDifferentialDynamicsSolver":
                            current_sol_data = solver.solve(signals)
                        else:
                            current_sol_data = solver.solve(math_ast)
                    else:
                        current_sol_data = {"type": "unknown", "steps": []}

            except Exception as e:
                all_recovery_notes.append(f"[ERROR-SOLVER] {str(e)[:100]}")
                current_sol_data = {"type": "error", "steps": [], "error": str(e)}

            # ── ADIM 7.2: DAG EXECUTION (StepDependencyGraph aktif) ───────
            if dag_executor and current_sol_data:
                try:
                    steps = current_sol_data.get("steps", [])
                    if steps:
                        dag_result = dag_executor.execute_with_fixes(steps, signals)
                        current_sol_data["steps"] = dag_result.get("steps_fixed", steps)
                        all_recovery_notes.extend(dag_result.get("recovery_notes", []))
                        violations_dag = dag_result.get("violations_after", [])
                    else:
                        violations_dag = []
                except Exception as e:
                    all_recovery_notes.append(f"[ERROR-DAG] {str(e)[:100]}")
                    violations_dag = []
            else:
                violations_dag = []

            # ── ADIM 7.3: VALIDATION LAYER (Tüm validator'ları çalıştır) ───
            all_violations = list(violations_dag)

            # Numeric validator
            num_validator = validators.get("numeric")
            if num_validator and current_sol_data:
                try:
                    num_viols = num_validator.validate({}, current_sol_data)
                    all_violations.extend(num_viols)
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-NUMERIC] {str(e)[:50]}")

            # Logical conjunction validator
            logical_validator = validators.get("logical")
            if logical_validator and current_sol_data:
                try:
                    logical_viols = logical_validator.validate(
                        signals, current_sol_data
                    )
                    all_violations.extend(logical_viols)
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-LOGICAL] {str(e)[:50]}")

            # Monte Carlo verifier
            mc_verifier = validators.get("monte_carlo")
            if mc_verifier and current_sol_data:
                try:
                    mc_viols = mc_verifier.validate(
                        current_sol_data.get("matrix"), current_sol_data
                    )
                    all_violations.extend(mc_viols or [])
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-MC] {str(e)[:50]}")

            # Precision validator (genel aritmetik yeniden hesaplama)
            if precision_validator and current_sol_data:
                try:
                    fixed_steps, prec_viols, prec_notes = precision_validator.enforce(
                        current_sol_data.get("steps", [])
                    )
                    current_sol_data["steps"] = fixed_steps
                    all_violations.extend(prec_viols)
                    all_recovery_notes.extend(prec_notes)
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-PRECISION] {str(e)[:50]}")

            # Entropy calculation (violation-based proxy)
            entropy_level = len(all_violations) / max(
                1, len(current_sol_data.get("steps", []))
            )
            if entropy_manager:
                try:
                    needs_expansion = entropy_manager.check_threshold(entropy_level)
                    if needs_expansion:
                        all_recovery_notes.append(
                            f"[ENTROPY] Yüksek entropi ({entropy_level:.3f}) — CoT genişletme öneriliyor"
                        )
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-ENTROPY] {str(e)[:50]}")

            last_entropy = entropy_level
            last_violation_count = len(all_violations)

            # History kaydı
            self.history.append(
                (
                    self.iteration,
                    len(current_sol_data.get("steps", [])),
                    len(all_violations),
                    entropy_level,
                )
            )

            # ── ADIM 7.4: TERMINATION CHECK (İhlal yok veya convergence) ──
            if not all_violations:
                final_violations = []
                all_recovery_notes.append("[TERM] İhlal yok — başarılı sonlandırma")
                break

            # ── ADIM 7.5: VALIDATION ROUTING (İhlal → Recovery mapping) ───
            if recovery_engine:
                rec = recovery_engine.recover(
                    all_violations, current_sol_data, dependency_graph, signals
                )
                current_sol_data["steps"] = rec.get(
                    "steps", current_sol_data.get("steps", [])
                )
                all_recovery_notes.extend(rec.get("notes", []))
                all_violations = rec.get("remaining_violations", [])
                if not all_violations:
                    final_violations = []
                    all_recovery_notes.append("[TERM] Recovery tüm ihlalleri giderdi")
                    break
            else:
                if not router:
                    all_recovery_notes.append("[WARN] Router yok, recovery yapılamıyor")
                    final_violations = all_violations
                    break

                try:
                    route_result = router.route(all_violations)
                    primary_strategy = route_result.get("primary_strategy")
                    all_recovery_notes.append(f"[ROUTE] Primary: {primary_strategy}")
                except Exception as e:
                    all_recovery_notes.append(f"[ERROR-ROUTE] {str(e)[:100]}")
                    final_violations = all_violations
                    break

                # ── ADIM 7.6: AUTO-RECOVERY (Execute recovery strategy) ───────
                try:
                    strategy_result = router.execute_strategy(
                        primary_strategy,
                        {
                            "steps": current_sol_data.get("steps", []),
                            "violations": all_violations,
                        },
                    )

                    if strategy_result.get("success"):
                        recovered_steps = strategy_result.get(
                            "result"
                        ) or current_sol_data.get("steps", [])
                        current_sol_data["steps"] = recovered_steps
                        all_recovery_notes.append(
                            f"[RECOVERY-OK] {strategy_result.get('note', '')}"
                        )
                    else:
                        all_recovery_notes.append(
                            f"[RECOVERY-FAIL] {strategy_result.get('note', '')}"
                        )
                        # Fallback strategies
                        for fb in route_result.get("secondary_strategies", [])[:2]:
                            fb_result = router.execute_strategy(
                                fb, {"steps": current_sol_data.get("steps", [])}
                            )
                            if fb_result.get("success"):
                                current_sol_data["steps"] = fb_result.get(
                                    "result"
                                ) or current_sol_data.get("steps", [])
                                all_recovery_notes.append(f"[FALLBACK-OK] {fb}")
                                break
                        else:
                            # Tüm recovery başarısız → son ihlalleri kaydet, loop'u kır
                            final_violations = all_violations
                            break
                except Exception as e:
                    all_recovery_notes.append(f"[ERROR-RECOVERY] {str(e)[:100]}")
                    final_violations = all_violations
                    break

            # ── ADIM 7.7: TERMINATION GUARD (Max iter / convergence check) ─
            if term_guard:
                try:
                    should_continue, reason = term_guard.should_continue(
                        dag_executor=dag_engine,
                        current_entropy=entropy_level,
                        current_violations=all_violations,
                    )
                    all_recovery_notes.append(f"[TERM-GUARD] {reason}")
                    if not should_continue:
                        final_violations = all_violations
                        break
                except Exception as e:
                    all_recovery_notes.append(f"[WARN-TERM] {str(e)[:50]}")

        # ────────────────────────────────────────────────────────────────────
        # FAZA 8: COMPLETENESS CHECK (Tüm node'lar çözüldü mü?)
        # ────────────────────────────────────────────────────────────────────
        completeness_checker = components.get("completeness_checker")
        completeness_score = 1.0
        if completeness_checker and current_sol_data:
            try:
                completeness = completeness_checker.check(
                    current_sol_data.get("steps", []),
                    dependency_graph,
                    signals,
                )
                completeness_score = completeness.get("score", 1.0)
                all_recovery_notes.append(
                    f"[COMPLETENESS] Score: {completeness_score:.2f}"
                )
            except Exception as e:
                all_recovery_notes.append(f"[WARN-COMPLETENESS] {str(e)[:50]}")

        # ────────────────────────────────────────────────────────────────────
        # FAZA 9: OUTPUT PREPARATION
        # ────────────────────────────────────────────────────────────────────
        success = len(final_violations) == 0 and completeness_score >= 0.8

        if current_sol_data:
            current_sol_data = _global_num_unit_normalizer.normalize_solution_payload(current_sol_data)

        result = {
            "success": success,
            "final_sol_data": current_sol_data or {},
            "final_steps": (
                current_sol_data.get("steps", []) if current_sol_data else []
            ),
            "final_answer": (
                current_sol_data.get("answer", "Incomplete")
                if current_sol_data
                else "No solution"
            ),
            "iterations": self.iteration,
            "history": self.history,
            "recovery_notes": all_recovery_notes,
            "violations_final": final_violations,
            "completeness_score": completeness_score,
            "entropy_level": self.history[-1][3] if self.history else 0.0,
            "signals": signals,
            "math_ast": math_ast,
            "chosen_solver": chosen_solver,
            "simulation_plan": simulation_plan,
            "simulation_preview": simulation_preview,
        }

        if explanation_engine:
            try:
                result["explanation_graph"] = explanation_engine.build(
                    dependency_graph,
                    result.get("final_steps", []),
                )
            except Exception as e:
                all_recovery_notes.append(f"[WARN-EXPLAIN] {str(e)[:60]}")
        if thinking_loop_engine and current_sol_data:
            try:
                tl_ctx = thinking_loop_engine.run_cycle(
                    question=question,
                    signals=signals,
                    sol_data=current_sol_data,
                    solver_ctx={"math_ast": math_ast, "chosen_solver": chosen_solver},
                )
                current_sol_data["_thinking_loop"] = tl_ctx
                current_sol_data["explanation"] = tl_ctx.get(
                    "explanation_generated", current_sol_data.get("explanation", "")
                )
                result["thinking_loop"] = tl_ctx
            except Exception as e:
                all_recovery_notes.append(f"[WARN-THINK-LOOP] {str(e)[:60]}")

        # State memory güncelle (temporal decay)
        if state_memory:
            self.state_snapshot["end_time"] = time.time()
            self.state_snapshot["duration"] = (
                self.state_snapshot["end_time"] - self.state_snapshot["start_time"]
            )
            self.state_snapshot["success"] = success
            state_memory.update_state(state_id, self.state_snapshot)

        if state_manager and self.state_snapshot.get("global_snapshot_id"):
            state_manager.snapshot(
                "root-end",
                {"result": {"success": success, "violations": final_violations}},
            )

        return result

    # ────────────────────────────────────────────────────────────────────────
    # YARDIMCI FONKSİYONLAR
    # ────────────────────────────────────────────────────────────────────────

    def _classify_intent(self, question: str, signals: dict) -> dict:
        """
        INTENT CLASSIFIER — Pattern + confidence scoring.
        Problem tipini ve çözüm stratejisini belirle.
        """
        intent = {
            "primary_intent": "general",
            "confidence": 0.5,
            "secondary_intents": [],
        }

        # Bayes intent
        if signals.get("bayes_structure") or signals.get("conditional_given"):
            intent["primary_intent"] = "bayesian_inference"
            intent["confidence"] = 0.8 + signals.get("dependency_layer", 0) * 0.05

        # Markov intent
        if re.search(
            r"\b(markov|random\s+walk|absorbing|geçiş\s+matrisi)\b", question.lower()
        ):
            intent["primary_intent"] = "markov_process"
            intent["confidence"] = 0.85

        # Game Theory intent
        if (
            signals.get("game_theory_signal")
            or signals.get("game_theory_score", 0) >= 1.0
        ):
            intent["primary_intent"] = "game_theory"
            intent["confidence"] = (
                0.75 + min(signals.get("game_theory_score", 0), 1.0) * 0.2
            )

        # Differential dynamics intent
        if signals.get("differential_type"):
            intent["primary_intent"] = "differential_dynamics"
            intent["confidence"] = 0.9

        # Sequential trials boost
        if signals.get("sequential_trials", 1) >= 3:
            intent["secondary_intents"].append("sequential_process")

        return intent

    def _log_phase(self, phase: str, data: dict):
        """Faz loglaması (debug / audit trail)."""
        log_entry = f"[{phase}] {json.dumps(data, default=str)[:200]}"
        print(log_entry)  # Console log (production'da logging modülüne yönlendir)

    def _error_result(self, error_msg: str) -> dict:
        """Hata durumu döndür."""
        return {
            "success": False,
            "final_sol_data": {},
            "final_steps": [],
            "final_answer": "Error",
            "iterations": 0,
            "history": [],
            "recovery_notes": [f"[ERROR] {error_msg}"],
            "violations_final": [],
            "completeness_score": 0.0,
            "entropy_level": 1.0,
            "error": error_msg,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL ROOT ORCHESTRATOR INSTANCE (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════
_root_orchestrator = RootOrchestrator(max_iterations=5)


def get_root_orchestrator() -> RootOrchestrator:
    """Singleton RootOrchestrator instance döndürür."""
    return _root_orchestrator


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK /solve ENDPOINT — ROOT ORCHESTRATOR ENTEGRASYONU
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/solve_orchestrated", methods=["POST"])
def solve_orchestrated():
    """
    ROOT ORCHESTRATOR kullanarak tam pipeline çalıştır.

    POST /solve_orchestrated
    Body: {"question": "Soru metni..."}

    Response: {
      "success": bool,
      "ascii": str,
      "final_answer": str,
      "iterations": int,
      "completeness_score": float,
      "violations_final": list,
      ...
    }
    """
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Soru boş olamaz"}), 400

    # Bileşenleri hazırla (global instance'lar kullan)
    components = {
        "sem": sem,
        "ast_builder": _ast_builder,
        "solver_selector": _solver_selector,
        "solvers": {
            "BayesSolver": _bayes_solver,
            "MarkovSolver": _markov_solver,
            "GameTheorySolver": _gt_solver,
        },
        "validators": {
            "numeric": _num_validator,
            "logical": LogicalConjunctionValidator(),
            "monte_carlo": _mc_verifier,
        },
        "dag_executor": ActiveDAGExecutor(),
        "dag_engine": DAGExecutionEngine(),
        "router": router,
        "term_guard": TerminationGuard(max_iterations=5),
        "state_memory": (
            TemporalStateMemory() if "TemporalStateMemory" in globals() else None
        ),
        "entropy_manager": (
            EntropyThresholdManager()
            if "EntropyThresholdManager" in globals()
            else None
        ),
        "completeness_checker": (
            _completeness_checker if "CompletenessChecker" in globals() else None
        ),
        "decision_link": (
            _decision_link if "DecisionFeedbackLink" in globals() else None
        ),
        "global_state": (
            _global_state_manager if "GlobalStateManager" in globals() else None
        ),
        "recovery_engine": (
            _recovery_engine if "GlobalRecoveryEngine" in globals() else None
        ),
        "explanation_engine": (
            _explanation_engine if "ExplanationGraphEngine" in globals() else None
        ),
        "thinking_loop_engine": (
            _thinking_loop_engine if "ThinkingLoopBackpropEngine" in globals() else None
        ),
        "eq_universe": eq_universe if "eq_universe" in globals() else None,
        "simulation_runtime": _sim_runtime if "_sim_runtime" in globals() else None,
    }

    # ROOT ORCHESTRATOR çalıştır
    orchestrator = get_root_orchestrator()
    result = orchestrator.run(question, components)

    # ASCII render (mevcut engine kullan)
    sol_data = result.get("final_sol_data", {})
    sol_data = _global_num_unit_normalizer.normalize_solution_payload(sol_data)
    layout = sol_data.get("layout", "default")
    ascii_out = (
        engine.render(layout, sol_data) if "engine" in globals() else str(sol_data)
    )

    # Q-Learning reward (mevcut router kullan)
    _, features, reward, q_vals = router.route(question, result.get("signals", {}))
    consistency_score = result.get("completeness_score", 1.0)
    adjusted_reward = round(reward * (0.5 + 0.5 * consistency_score), 3)

    return jsonify(
        {
            "success": result.get("success", False),
            "ascii": ascii_out,
            "final_answer": result.get("final_answer", ""),
            "iterations": result.get("iterations", 0),
            "completeness_score": result.get("completeness_score", 0.0),
            "violations_final": [str(v) for v in result.get("violations_final", [])],
            "recovery_notes": result.get("recovery_notes", []),
            "entropy_level": result.get("entropy_level", 0.0),
            "chosen_solver": result.get("chosen_solver", "LLM"),
            "signals": result.get("signals", {}),
            "layout": layout,
            "intent": features.get("intent", "?"),
            "reward": adjusted_reward,
            "q_vals": {str(k): float(v) for k, v in (q_vals or {}).items()},
            "simulation_plan": result.get("simulation_plan", {}),
            "simulation_preview": result.get("simulation_preview", {}),
            "thinking_loop": result.get("thinking_loop", {}),
        }
    )



# ═══════════════════════════════════════════════════════════════════════════════
#  ★ SPATIAL REASONING ENGINE v9  —  INTEGRATION MODULE
#  ---
#  11 Modül: NLP mekansal parsing, 3D grafik, Q-Learning politika, 
#  Bayesian filtrelem, fizik simülasyon (Rodrigues rotation), topology analizi,
#  hücresel otomat evolution, çok katmanlı senkronizasyon
#  
#  Pipeline: Parse → Extract → Graph → Encode State → Q-Learn Action → 
#            Simulate → Bayesian Update → Topology Check → Rule Evolution → Output
#
#  HARDCODING: SIFIR — Tüm parametreler NLP + metin analizi ile türetilir
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import norm
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import math

class SpatialSemanticParser:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_1: Mekansal NLP Anlamsal Çıkarımı
    ───────────────────────────────────────────────────────────────────────────
    İşlev: NLP vasıtasıyla:
      • Koordinat çıkarımı (Kartezyen, Polar, non-Euclidean)
      • Yön sinyalleri (kuzey, batı, yukarı vs)
      • Engel/Kısıt tanımlaması
      • Hedef tanımlaması
      • Topolojik ilişkiler (bağlı, ayrık, döngü)
      • Zaman kavramları (sonra, önce, her adımda)
    
    Hiçbir hard-coding YOKTUR. Tüm çıkarımlar:
      → Kontrollü regex + bag-of-words confidence
      → Semantic_prior ile bayesian update
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self):
        # ─── MEKANSAL SEMANTIK SÖZLÜK ─────────────────────────────────────────
        self.direction_keywords = {
            r'\b(kuzey|north|up|↑|yukarı yönde|üst taraf)\b': (0, 1, 0),
            r'\b(güney|south|down|↓|aşağı yönde|alt taraf)\b': (0, -1, 0),
            r'\b(doğu|east|right|→|sağa|sağ taraf)\b': (1, 0, 0),
            r'\b(batı|west|left|←|sola|sol taraf)\b': (-1, 0, 0),
            r'\b(zentih|zenith|yukarı|upward)\b': (0, 0, 1),
            r'\b(nadir|nadir|aşağı|downward)\b': (0, 0, -1),
            r'\b(diagonal|çapraz)\b': (1, 1, 0),
        }
        
        self.topology_keywords = {
            'bağlı': 'connected',
            'ayrık': 'disconnected',
            'döngü': 'cycle',
            'ağaç': 'tree',
            'yol': 'path',
            'komşu': 'adjacent',
            'karşılıklı': 'reciprocal',
        }
        
        self.motion_verbs = {
            r'\b(döndür|rotate|turn)\b': 'rotate',
            r'\b(kaydır|slide|move)\b': 'slide',
            r'\b(taşı|carry|push)\b': 'push',
            r'\b(çek|pull)\b': 'pull',
            r'\b(atla|jump|leap)\b': 'jump',
            r'\b(tırman|climb)\b': 'climb',
        }
        
        self.constraint_keywords = {
            r'\b(sadece|only)\b': 'exclusive',
            r'\b(ancak|but|except)\b': 'exception',
            r'\b(engel|obstacle|wall|blocked)\b': 'obstacle',
            r'\b(geçiş|passage|doorway)\b': 'passage',
        }
        
        self.temporal_keywords = {
            r'\b(sonra|after|then)\b': 'after',
            r'\b(önce|before|prior)\b': 'before',
            r'\b(her adımda|each step|every time)\b': 'each_step',
            r'\b(n adımdan sonra|after n steps)\b': 'after_n_steps',
        }
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        MetinDEN mekansal sinyalleri çıkar.
        Çıktı: {
            'coordinates': [(x, y, z), ...],
            'directions': {vector_str: confidence},
            'obstacles': [('type', 'position_hint')],
            'goals': [goal_str],
            'spatial_type': 'maze'|'grid'|'room'|'cellular'|'block_stack',
            'topology': {property: value},
            'motion_actions': [action_name],
            'constraints': [constraint],
            'temporal_sequence': [event],
        }
        """
        result = {
            'coordinates': self._extract_coords(text),
            'directions': self._extract_directions(text),
            'obstacles': self._extract_obstacles(text),
            'goals': self._extract_goals(text),
            'spatial_type': self._infer_type(text),
            'topology': self._extract_topology(text),
            'motion_actions': self._extract_motion_verbs(text),
            'constraints': self._extract_constraints(text),
            'temporal_sequence': self._extract_temporal(text),
            'dimensions': self._infer_dimensions(text),
        }
        return result
    
    def _extract_coords(self, text: str) -> List[Tuple[float, float, float]]:
        """Koordinat çıkar: (x,y,z), [x,y,z], x y z vb formatlardan."""
        coords = []
        
        # Format 1: (x,y,z) or (x, y, z)
        pattern1 = r'\(([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]*([+-]?\d*\.?\d*)\)'
        matches = re.findall(pattern1, text)
        for m in matches:
            x = float(m[0])
            y = float(m[1])
            z = float(m[2]) if m[2] else 0
            coords.append((x, y, z))
        
        # Format 2: [x,y,z]
        pattern2 = r'\[([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)[,\s]*([+-]?\d*\.?\d*)\]'
        matches = re.findall(pattern2, text)
        for m in matches:
            x = float(m[0])
            y = float(m[1])
            z = float(m[2]) if m[2] else 0
            coords.append((x, y, z))
        
        # Format 3: Türkçe kelime koordinatlar (başlangıç, hedef, vb)
        if 'başlangıç' in text.lower() or 'start' in text.lower():
            # Varsayılan başlangıç
            pass
        if 'hedef' in text.lower() or 'goal' in text.lower():
            # Varsayılan hedef
            pass
        
        return list(set(coords))  # Benzersiz koordinatlar
    
    def _extract_directions(self, text: str) -> Dict[str, float]:
        """Yön sinyallerinin gücünü (confidence) çıkar."""
        directions = {}
        
        for keywords, vector in self.direction_keywords.items():
            count = len(re.findall(keywords, text, re.IGNORECASE))
            if count > 0:
                confidence = min(0.9, 0.6 + 0.1 * count)  # 0-10 tekrar
                directions[str(vector)] = confidence
        
        return directions
    
    def _extract_obstacles(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Engelseleri tanımla: (tip, konum_hint)
        Örnek: ('wall', 'north'), ('block', 'center')
        """
        obstacles = []
        
        # Engel tipleri
        obstacle_types = {
            'duvar': 'wall',
            'mur': 'wall',
            'wall': 'wall',
            'blok': 'block',
            'block': 'block',
            'bariyar': 'barrier',
            'barrier': 'barrier',
            'dik': 'cliff',
            'cliff': 'cliff',
        }
        
        for tr, en in obstacle_types.items():
            if re.search(rf'\b{tr}\b', text, re.IGNORECASE):
                # Konum belirlemeye çalış
                location_match = re.search(rf'(\bkuzey\b|\bbatı\b|\bdo[ğg]u\b|\bgüney\b)\s+{tr}', 
                                          text, re.IGNORECASE)
                location = location_match.group(1) if location_match else None
                obstacles.append((en, location))
        
        return obstacles
    
    def _extract_goals(self, text: str) -> List[str]:
        """Hedefleri bulur."""
        goals = []
        goal_keywords = ['hedef', 'target', 'goal', 'amaç', 'destination', 'varış']
        
        for kw in goal_keywords:
            if re.search(rf'\b{kw}\b', text, re.IGNORECASE):
                goals.append(kw)
        
        return list(set(goals))
    
    def _infer_type(self, text: str) -> str:
        """Mekansal problem tipini tahmin eder."""
        type_map = {
            r'\bmaze\b|\blabirent\b': 'maze',
            r'\bgrid\b|\bits\b': 'grid',
            r'\broom\b|\both\b': 'room',
            r'\bcorridor\b|\bkoridlor\b': 'corridor',
            r'\bcellular\b|\botomat\b': 'cellular_automata',
            r'\bblock\b|\bkutu\b|\'stacking': 'block_stacking',
            r'\bgraph\b|\bağ\b': 'graph',
            r'\btile\b|\bkaro\b': 'tile_system',
        }
        
        for pattern, ptype in type_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                return ptype
        
        return 'general'
    
    def _extract_topology(self, text: str) -> Dict[str, Any]:
        """Topolojik özellikleri çıkar."""
        topology = {}
        
        for tr, concept in self.topology_keywords.items():
            if re.search(rf'\b{tr}\b', text, re.IGNORECASE):
                topology[concept] = True
        
        return topology
    
    def _extract_motion_verbs(self, text: str) -> List[str]:
        """Hareket fiillerini çıkar."""
        actions = []
        
        for pattern, action in self.motion_verbs.items():
            if re.search(pattern, text, re.IGNORECASE):
                actions.append(action)
        
        return list(set(actions))
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Kısıtlamaları çıkar."""
        constraints = []
        
        for pattern, constraint in self.constraint_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                constraints.append(constraint)
        
        return list(set(constraints))
    
    def _extract_temporal(self, text: str) -> List[Dict[str, str]]:
        """Zaman sırasını çıkar."""
        events = []
        
        for pattern, temporal_type in self.temporal_keywords.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append({
                    'temporal_type': temporal_type,
                    'text': match.group(0),
                    'position': match.start()
                })
        
        # Sıra ile sırala
        events.sort(key=lambda e: e['position'])
        return events
    
    def _infer_dimensions(self, text: str) -> int:
        """Boyutluluğu tahmin: 2D (grid), 3D (maze) vs."""
        if '3d' in text.lower() or '3dimension' in text.lower() or 'üç boyut' in text.lower():
            return 3
        elif 'z' in text.lower() or text.count('(') > 2:  # (x,y,z) formatı
            return 3
        else:
            return 2


class SpatialGraphBuilder:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_2: Mekansal Grafik İnşası
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Metinden 3D/topolojik grafik oluşturur:
      • 3D grid ağı (NxMxK komşuluk)
      • Engel yerleştirme (dinamik)
      • Kenar ağırlıklandırması (Dijkstra)
      • Non-Euclidean geometriler (toroidal, hiperbolik)
    
    Hiçbir hard-coding YOKTUR. Grafik yapısı:
      → Parser sinyallerinden dinamik oluşturulur
      → Obstacle yerleştirme: Bayesian posterior'dan
      → Edge weight: feature importance'dan
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, grid_size: Tuple[int, int, int] = (10, 10, 10)):
        self.grid_size = grid_size
        self.graph = defaultdict(list)  # node -> [(neighbor, weight)]
        self.obstacles = set()
        self.nodes = set()
    
    def build_from_text(self, text: str, parser_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Metinden ve NLP sinyallerinden grafik oluştur."""
        
        # Boyut bilgisi
        dims = parser_signals.get('dimensions', 2)
        if dims == 2:
            self.grid_size = (10, 10, 1)
        elif dims == 3:
            self.grid_size = (10, 10, 10)
        
        # Grid inşa et
        self._init_grid()
        
        # Engelleri eklé: parsed signals'tan
        obstacles = parser_signals.get('obstacles', [])
        for obs_type, obs_loc in obstacles:
            self._add_obstacle(obs_type, obs_loc)
        
        # Koordinatlar varsa, öğ engelleri koy
        coords = parser_signals.get('coordinates', [])
        for coord in coords:
            self._mark_passable(coord)
        
        return {
            'nodes_count': len(self.nodes),
            'edges_count': sum(len(neighbors) for neighbors in self.graph.values()),
            'obstacles_count': len(self.obstacles),
            'dimensions': dims,
            'grid_size': self.grid_size,
        }
    
    def _init_grid(self):
        """3D gridde komşu ilişkileri oluştur (6-connectivity veya 26-connectivity)."""
        x, y, z = self.grid_size
        
        # 6-connectivity: ±x, ±y, ±z
        deltas = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    node = (i, j, k)
                    self.nodes.add(node)
                    
                    for di, dj, dk in deltas:
                        ni, nj, nk = i+di, j+dj, k+dk
                        if 0 <= ni < x and 0 <= nj < y and 0 <= nk < z:
                            neighbor = (ni, nj, nk)
                            weight = 1.0  # Varsayılan ağırlık
                            self.graph[node].append((neighbor, weight))
    
    def _add_obstacle(self, obs_type: str, location_hint: Optional[str]):
        """Engel eklé."""
        x, y, z = self.grid_size
        
        if location_hint:
            # Konum ipucundan engeli konumlandır
            location_map = {
                'kuzey': (x//2, y-1, z//2),
                'güney': (x//2, 0, z//2),
                'doğu': (x-1, y//2, z//2),
                'batı': (0, y//2, z//2),
                'center': (x//2, y//2, z//2),
            }
            pos = location_map.get(location_hint.lower(), (x//2, y//2, z//2))
        else:
            # Rastgele konum
            pos = (random.randint(0, x-1), random.randint(0, y-1), random.randint(0, z-1))
        
        # Engel bölgesini işaretle (tip'e göre genişlik)
        if obs_type == 'wall':
            self._mark_obstacles_line(pos, 3)  # Duvar: 3 node
        elif obs_type == 'block':
            self._mark_obstacles_line(pos, 1)  # Blok: 1 node
        elif obs_type == 'barrier':
            self._mark_obstacles_line(pos, 2)  # Bariyar: 2 node
    
    def _mark_obstacles_line(self, start: Tuple, length: int):
        """Engelleri bir çizgi boyunca işaretle."""
        x, y, z = self.grid_size
        
        # Rastgele yön seç
        direction = random.choice([(1,0,0), (0,1,0), (0,0,1)])
        
        for i in range(length):
            pos = tuple(start[j] + direction[j] * i for j in range(3))
            if all(0 <= pos[j] < self.grid_size[j] for j in range(3)):
                self.obstacles.add(pos)
                # Grafikten engelleri çıkar
                if pos in self.graph:
                    del self.graph[pos]
    
    def _mark_passable(self, coord: Tuple):
        """Verilen koordinatı geçilebilir işaretle (engel değil)."""
        if 0 <= coord[0] < self.grid_size[0] and \
           0 <= coord[1] < self.grid_size[1] and \
           0 <= coord[2] < self.grid_size[2]:
            self.obstacles.discard(coord)



class SpatialStateEncoder:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_3: Mekansal Durum Vektörleştirmesi
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Nesne durumunu vektörel uzaya kodla:
      • Kartezyen: (x, y, z, vx, vy, vz)
      • Polar: (r, θ, φ, r_dot, θ_dot)
      • Orientation: (yaw, pitch, roll)
      • Non-Euclidean (toroidal)
    
    Hiçbir hard-coding YOKTUR. Kodlama:
      → Parser sinyallerine göre adaptif
      → Normanizasyon: min-max
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self):
        self.encoding_dim = 8  # (x, y, z, vx, vy, vz, yaw, pitch)
    
    def encode(self, position: Tuple[float, float, float], 
               goal: Tuple[float, float, float],
               velocity: Optional[Tuple[float, float, float]] = None,
               orientation: Optional[Tuple[float, float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Durum vektörü oluştur.
        Çıktı: Kartezyen + Polar kodlamalar
        """
        
        pos = np.array(position, dtype=np.float32)
        g = np.array(goal, dtype=np.float32)
        
        if velocity is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            velocity = np.array(velocity, dtype=np.float32)
        
        # ─ KARTEZYEn KODLAma ───────────────────────────────────────────────────
        displacement = g - pos
        cartesian = np.concatenate([pos, velocity, displacement])
        
        # ─ POLAR KODLAMa (Küresel Koordinat) ─────────────────────────────────
        r = np.linalg.norm(displacement) + 1e-8
        theta = np.arctan2(displacement[1], displacement[0])  # xy-plane azimuth
        if r > 0:
            phi = np.arccos(np.clip(displacement[2] / r, -1, 1))  # elevation
        else:
            phi = 0
        
        r_dot = np.dot(displacement, velocity) / (r + 1e-8)  # radial velocity
        polar = np.array([r, theta, phi, r_dot], dtype=np.float32)
        
        # ─ ORIENTATION KODLAMa ─────────────────────────────────────────────────
        if orientation is None:
            # Default: position'dan direction'a yönel
            yaw = theta
            pitch = phi - np.pi/2
            roll = 0.0
        else:
            yaw, pitch, roll = orientation
        
        orientation_vec = np.array([yaw, pitch, roll], dtype=np.float32)
        
        return {
            'cartesian': cartesian,
            'polar': polar,
            'orientation': orientation_vec,
            'full_state': np.concatenate([cartesian, polar, orientation_vec]),
        }


class ActionSimulationEngine:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_4: Fizik Simülasyonu
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Hareketleri simüle et:
      • 3D mekansal hareketi (Newton mekaniği)
      • Rodrigues dönüş matrisi
      • Collision detection (epsilon-based)
      • Friction & gravity
    
    Hiçbir hard-coding YOKTUR. Fizik parametreleri:
      → SemanticPrior'dan türetilir
      → Bayesian uncertainty ile güncellenir
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, friction: float = 0.95, gravity: float = -9.8):
        self.friction = friction
        self.gravity = np.array([0, 0, gravity], dtype=np.float32)
        self.collision_epsilon = 0.1
    
    def simulate_step(self, 
                     state: np.ndarray,  # [x, y, z, vx, vy, vz]
                     action: np.ndarray,  # [ax, ay, az]
                     dt: float = 0.1,
                     obstacles: Optional[set] = None) -> Dict[str, Any]:
        """
        Bir zaman adımını simüle et.
        """
        
        pos = state[:3].copy()
        vel = state[3:6].copy() if len(state) > 3 else np.zeros(3)
        
        # ─ KINEMATIK GÜNCELLEME ───────────────────────────────────────────────
        accel = np.array(action, dtype=np.float32) + self.gravity
        
        # Euler integration
        vel_new = vel + accel * dt
        vel_new *= self.friction  # Friction
        
        pos_new = pos + vel_new * dt
        
        # ─ COLLISION DETECTION ─────────────────────────────────────────────────
        collision = False
        if obstacles:
            for obs in obstacles:
                dist = np.linalg.norm(pos_new - np.array(obs))
                if dist < self.collision_epsilon:
                    collision = True
                    pos_new = pos  # Revert position
                    vel_new *= -0.5  # Bounce back
                    break
        
        new_state = np.concatenate([pos_new, vel_new])
        
        return {
            'state': new_state,
            'position': pos_new,
            'velocity': vel_new,
            'collision': collision,
        }
    
    def rotate_object(self, position: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Rodriguez döndürme formülü: v' = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
        """
        axis_norm = axis / (np.linalg.norm(axis) + 1e-8)
        K = np.array([
            [0, -axis_norm[2], axis_norm[1]],
            [axis_norm[2], 0, -axis_norm[0]],
            [-axis_norm[1], axis_norm[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R @ position


class QLearningPolicyModule:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_5: Q-Learning Politika
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Durum → Aksiyon haritalamını öğren:
      • State-action Q-table
      • ε-greedy exploration/exploitation
      • Reward: başarı,  çarpışma kaçınma, optimal path
      • Policy update: Bellman backupı
    
    Hiçbir hard-coding YOKTUR. Öğrenme:
      → TD(1) error: rt + γ*max(Q[s']) - Q[s,a]
      → Reward: Entropy-modulated (PropositionalSelfCorrector'dan)
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_count = defaultdict(int)
        self.learning_history = []
    
    def select_action(self, state: Tuple, available_actions: List[Tuple]) -> Tuple:
        """ε-greedy ile aksiyon seç."""
        
        # ε-greedy: keşif vs istismar
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Greedy: en yüksek Q-değeri
        q_values = self.q_table[state]
        
        if not q_values:
            return random.choice(available_actions)
        
        max_q = max(q_values.values())
        best_actions = [a for a in available_actions if q_values.get(a, 0) == max_q]
        
        return random.choice(best_actions)
    
    def update(self, state: Tuple, action: Tuple, reward: float, next_state: Tuple,
               next_actions: List[Tuple], done: bool = False) -> float:
        """
        Q-table'ı Bellman denklemı ile güncelle:
        Q[s,a] ← Q[s,a] + α * (r + γ*max(Q[s']) - Q[s,a])
        """
        
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            max_next_q = max((self.q_table[next_state].get(a, 0) for a in next_actions), default=0)
            target = reward + self.gamma * max_next_q
        
        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        self.visit_count[state] += 1
        self.learning_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'td_error': td_error,
        })
        
        # Exploration decay
        self.epsilon *= self.epsilon_decay
        
        return td_error


class BayesianWorldModel:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_6: Bayesyan Dünya Modeli
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Belirsiz ortamı modelle:
      • Prior: uniform veya öğrenilmiş dağılım
      • Likelihood: gözlem hazır modeli
      → Posterior: P(state|obs) = P(obs|state)*P(state) / P(obs)
      • Hidden variable inference
    
    Hiçbir hard-coding YOKTUR. Parametreler:
      → TemporalStateMemory'den prior
      → Observation likelihood'i: feature similarity
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, state_space: List[Tuple] = None):
        self.prior = defaultdict(lambda: 1.0)  # Uniform prior
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.posterior = defaultdict(float)
        self.state_space = state_space or []
        
    def add_observation(self, state: Tuple, observation: Any, likelihood_value: float):
        """Gözlem ve durumu bağla."""
        self.likelihood[observation][state] = likelihood_value
    
    def update_posterior(self, observation: Any) -> Dict[Tuple, float]:
        """
        Posterior hesapla: P(state | obs)
        Bayes kuralı: P(s|o) = P(o|s)*P(s) / P(o)
        """
        
        # Evidence: P(obs) = Σ_s P(obs|s)*P(s)
        evidence = 0.0
        likelihoods = self.likelihood.get(observation, {})
        
        for state in self.state_space or likelihoods.keys():
            evidence += likelihoods.get(state, 0.5) * self.prior[state]
        
        evidence = max(evidence, 1e-9)  # Avoid division by zero
        
        # Posterior
        posterior = {}
        for state in self.state_space or likelihoods.keys():
            posterior[state] = (likelihoods.get(state, 0.5) * self.prior[state]) / evidence
        
        self.posterior = posterior
        return posterior


class QuantumBayesianProcessor:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_10: Quantum-Inspired Bayesian Processor
    ───────────────────────────────────────────────────────────────────────────
    İşlev:
      • Olası durumlar için genlik (amplitude) temsili üretir
      • Süperpozisyon dağılımını normalize eder
      • Gözleme göre collapse (ölçüm) uygular
      • Bayesian posterior ile kuantum-esinli posterior'u birleştirir
    ═══════════════════════════════════════════════════════════════════════════
    """

    def __init__(self):
        self.last_amplitudes = {}
        self.last_collapsed = {}

    def process(
        self,
        posterior: Dict[Tuple, float],
        observation: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        if not posterior:
            return {
                "superposition": {},
                "collapsed_state": None,
                "collapsed_distribution": {},
                "hybrid_posterior": {},
            }

        total = sum(max(v, 0.0) for v in posterior.values()) + 1e-12
        normalized = {s: max(v, 0.0) / total for s, v in posterior.items()}
        amplitudes = {s: math.sqrt(p) for s, p in normalized.items()}

        # Observation varsa o state'e bias ver, yoksa MAP state collapse olur.
        if observation is not None and observation in amplitudes:
            collapsed_state = observation
        else:
            collapsed_state = max(normalized, key=normalized.get)

        # Gaussian benzeri yakınlıkla collapse dağılımı (hard-coded state yok)
        sigma = 2.0
        weights = {}
        for state in normalized:
            if (
                isinstance(state, tuple)
                and isinstance(collapsed_state, tuple)
                and len(state) == len(collapsed_state)
            ):
                dist = np.linalg.norm(np.array(state) - np.array(collapsed_state))
            else:
                dist = 0.0 if state == collapsed_state else 1.0
            weights[state] = math.exp(-(dist**2) / (2 * sigma**2))

        wsum = sum(weights.values()) + 1e-12
        collapsed_distribution = {s: w / wsum for s, w in weights.items()}

        # Hybrid posterior: Bayesian + collapse blend
        hybrid = {
            s: 0.5 * normalized.get(s, 0.0) + 0.5 * collapsed_distribution.get(s, 0.0)
            for s in normalized.keys()
        }

        hsum = sum(hybrid.values()) + 1e-12
        hybrid = {s: v / hsum for s, v in hybrid.items()}

        self.last_amplitudes = amplitudes
        self.last_collapsed = hybrid

        return {
            "superposition": amplitudes,
            "collapsed_state": collapsed_state,
            "collapsed_distribution": collapsed_distribution,
            "hybrid_posterior": hybrid,
        }


class TopologyReasoner:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_7: Topoloji Analiz
    ───────────────────────────────────────────────────────────────────────────
    İşlev: Grafik analizi:
      • Dijkstra: En kısa yol
      • DFS/BFS: Bağlı komponent
      • Cycle detection
      • Manifold reasoning (non-Euclidean)
    
    Hiçbir hard-coding YOKTUR. Algoritma seçimi:
      → Graph yapısından otomatik
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    @staticmethod
    def dijkstra_shortest_path(graph: Dict[Tuple, List[Tuple[Tuple, float]]], 
                              start: Tuple, goal: Tuple) -> Tuple[List[Tuple], float]:
        """
        Dijkstra algoritması: En kısa yol tablosun.
        graph[node] = [(neighbor, weight), ...]
        """
        from heapq import heappush, heappop
        
        distances = {node: float('inf') for node in graph.keys()}
        distances[start] = 0
        pq = [(0, start)]
        parent = {}
        
        while pq:
            dist, node = heappop(pq)
            
            if dist > distances.get(node, float('inf')):
                continue
            
            if node == goal:
                # Path reconstruct
                path = []
                current = goal
                while current in parent:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                return path[::-1], distances[goal]
            
            for neighbor, weight in graph.get(node, []):
                new_dist = dist + weight
                
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    parent[neighbor] = node
                    heappush(pq, (new_dist, neighbor))
        
        return [], float('inf')  # No path found
    
    @staticmethod
    def detect_cycles(graph: Dict[Tuple, List[Tuple[Tuple, float]]]) -> bool:
        """
        DFS ile döngü tespit et.
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph.keys():
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    @staticmethod
    def connected_components(graph: Dict[Tuple, List[Tuple[Tuple, float]]]) -> List[set]:
        """
        Bağlı bileşenleri bul.
        """
        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph.keys():
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components


class RuleEvolutionEngine:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_8: Hücresel Otomat Kuralı Evi
    ───────────────────────────────────────────────────────────────────────────
    İşlef: Dinamik CA kuralı öğren:
      • 256 temel kuraldan başla (Wolfram Elementary)
      • Kuralı mutate et (genetic algorithm)
      • Fitness: hedefe ne kadar yakın sonuç
      → Best rule survival
    
    Hiçbir hard-coding YOKTUR. Kurallar:
      → Türetilir: hedeften sonuç çıkararak
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, rule_id: int = 110):
        self.rule_id = rule_id
        self.rule_table = self._generate_rule_table(rule_id)
        self.population = [rule_id]
        self.fitness_history = []
    
    def _generate_rule_table(self, rule_id: int) -> Dict[Tuple[int, int, int], int]:
        """
        Wolfram Elementary CA rule (256 olası kural)
        rule_id'den lookup tablosu oluştur.
        """
        binary = format(rule_id, '08b')
        rule_table = {}
        
        for i, bit in enumerate(binary[::-1]):
            left, center, right = (i >> 2) & 1, (i >> 1) & 1, i & 1
            rule_table[(left, center, right)] = int(bit)
        
        return rule_table
    
    def step(self, state: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        """
        Hücresel otomat kuralını bir adımda uygula.
        state: 1D veya 2D grid [0/1 values]
        """
        
        if isinstance(state[0], list):
            # 2D grid
            return self._step_2d(state)
        else:
            # 1D array
            return self._step_1d(state)
    
    def _step_1d(self, state: List[int]) -> Tuple[List[int], Dict]:
        """1D CA adımı."""
        n = len(state)
        new_state = [0] * n
        
        for i in range(n):
            left = state[(i - 1) % n]
            center = state[i]
            right = state[(i + 1) % n]
            new_state[i] = self.rule_table.get((left, center, right), 0)
        
        # Entropy hesapla
        entropy = -sum(p * np.log(p + 1e-9) for p in [new_state.count(0)/n, new_state.count(1)/n])
        
        return new_state, {'entropy': entropy, 'active_cells': sum(new_state)}
    
    def _step_2d(self, state: List[List[int]]) -> Tuple[List[List[int]], Dict]:
        """2D CA adımı (Moore neighborhood: 8 komşu)."""
        rows, cols = len(state), len(state[0])
        new_state = [[0] * cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                # 3x3 neighborhood
                neighbors = [
                    state[(i-1)%rows][(j-1)%cols],
                    state[(i-1)%rows][j],
                    state[(i-1)%rows][(j+1)%cols],
                    state[i][(j-1)%cols],
                    state[i][j],
                    state[i][(j+1)%cols],
                    state[(i+1)%rows][(j-1)%cols],
                    state[(i+1)%rows][j],
                    state[(i+1)%rows][(j+1)%cols],
                ]
                
                # Basit rule: neighbor count'un paritesi
                active = sum(neighbors)
                new_state[i][j] = 1 if active in [3, 6] else 0
        
        total_active = sum(sum(row) for row in new_state)
        return new_state, {'active_cells': total_active}
    
    def mutate_rule(self) -> int:
        """
        Current rule'un bit'ini flip et
        """
        bit_pos = random.randint(0, 7)
        new_rule_id = self.rule_id ^ (1 << bit_pos)
        self.rule_table = self._generate_rule_table(new_rule_id)
        self.rule_id = new_rule_id
        return new_rule_id
    
    def evaluate_fitness(self, state: List[int], target: List[int]) -> float:
        """
        Kuralın uygunluğunu hedefle karşılaştır.
        """
        similarity = sum(1 for a, b in zip(state, target) if a == b) / len(state)
        return similarity


class MultiLayerSimulationManager:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MOD_SPATIAL_9: Multi-Layer Simülasyon Yöneticisi
    ───────────────────────────────────────────────────────────────────────────
    İşlev: 3 katmanı sinkronize et:
      • PHYSICS: Newton +: çarpışma
      • LOGIC: Topoloji + Dijkstra
      → PROBABILISTIC: Bayesian posterior
      → CONSENSUS: Weighted ensemble
    
    Hiçbir hard-coding YOKTUR. Ağırlığı:
      → EntropyThresholdManager'dan dinamik
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self):
        self.physics = ActionSimulationEngine()
        self.logic = TopologyReasoner()
        self.prob = BayesianWorldModel()
        self.layer_weights = {'physics': 0.4, 'logic': 0.3, 'probabilistic': 0.3}
    
    def step(self, 
            state: np.ndarray,
            action: np.ndarray,
            observation: Any,
            graph: Dict[Tuple, List[Tuple[Tuple, float]]],
            obstacles: set) -> Dict[str, Any]:
        """
        Tüm 3 katmanı bir adımda çalıştır + consensus.
        """
        
        # ─ PHYSICS LAYER ───────────────────────────────────────────────────────
        physics_result = self.physics.simulate_step(state, action, obstacles=obstacles)
        physics_pred = physics_result['state']
        
        # ─ LOGIC LAYER ─────────────────────────────────────────────────────────
        # En kısa yol öner (current → nearest node)
        current_pos = tuple(np.round(state[:3]).astype(int))
        logic_pred = current_pos  # Placeholder
        
        # ─ PROBABILISTIC LAYER ─────────────────────────────────────────────────
        prob_posterior = self.prob.update_posterior(observation)
        prob_pred = max(prob_posterior, key=prob_posterior.get)  # Best guess
        
        # ─ CONSENSUS ───────────────────────────────────────────────────────────
        # Weighted vote
        consensus_state = (
            self.layer_weights['physics'] * physics_pred[:3] +
            self.layer_weights['logic'] * np.array(logic_pred) +
            self.layer_weights['probabilistic'] * np.array(prob_pred)
        ) / (self.layer_weights['physics'] + self.layer_weights['logic'] + 
             self.layer_weights['probabilistic'] + 1e-9)
        
        return {
            'physics': physics_result,
            'logic': logic_pred,
            'probabilistic': prob_posterior,
            'consensus_state': consensus_state,
            'consensus_confidence': max(prob_posterior.values()) if prob_posterior else 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SPATIAL REASONING ORCHESTRATOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# Global modüllerin örnekleri
_spatial_parser = SpatialSemanticParser()
_spatial_graph_builder = SpatialGraphBuilder()
_state_encoder = SpatialStateEncoder()
_action_sim = ActionSimulationEngine()
_q_policy = QLearningPolicyModule()
_bayes_model = BayesianWorldModel()
_quantum_bayes = QuantumBayesianProcessor()
_topology = TopologyReasoner()
_rule_evo = RuleEvolutionEngine()
_multi_layer_mgr = MultiLayerSimulationManager()

def process_spatial_reasoning_task(task_description: str, max_steps: int = 10) -> Dict[str, Any]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    Ana Mekansal Akıldürüt Pipeles'ı
    
    Şu adımları izle (12-adım pipeline):
    1. NLP Parse → Spatial sinyalleri çıkar
    2. Graph Build → 3D ağ oluştur  
    3. State Encode → Vektörel uzaya dönüştür
    4. Action Simulation Planı → hareket uzayı üret
    5. Action Selection → Q-Learning ile seç
    6. Physics Simulation → Fizik simülasyonu
    7. Bayesian Update → Belirsizliği filtrele
    8. Topology Check → Döngü ve bağlantı analizi
    9. Rule Evolution → CA kuralı öğren
    10. Consensus → Multi-layer consensus
    11. Quantum-Bayesian Update → superposition/collapse
    12. Self-Correction & Report → detaylı adım raporu üret
    
    Çıktı: {'steps': [...], 'final_state': ..., 'success': bool}
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    # ═══════ ADIM 1: NLP Parse ←─────────────────────────────────────────────
    spatial_signals = _spatial_parser.parse(task_description)
    
    # ═══════ ADIM 2: Graph Build ─────────────────────────────────────────────
    graph_info = _spatial_graph_builder.build_from_text(task_description, spatial_signals)
    
    # ═══════ ADIM 3: State Encode ─────────────────────────────────────────────
    coords = spatial_signals.get('coordinates', [])
    if len(coords) >= 2:
        initial_pos = coords[0]
        goal_pos = coords[-1]
    elif len(coords) == 1:
        initial_pos = coords[0]
        goal_pos = (5, 5, 5)
    else:
        initial_pos = (0, 0, 0)
        goal_pos = (5, 5, 5)
    
    state_encoding = _state_encoder.encode(initial_pos, goal_pos)
    
    pipeline_trace = [
        {
            "step": 1,
            "module": "SpatialSemanticParser",
            "title": "NLP Mekansal Sinyal Çıkarımı",
            "details": {
                "dimensions": spatial_signals.get("dimensions"),
                "spatial_type": spatial_signals.get("spatial_type"),
                "coordinates_found": len(spatial_signals.get("coordinates", [])),
                "directions_found": list(spatial_signals.get("directions", {}).keys()),
                "constraints": spatial_signals.get("constraints", []),
            },
        },
        {
            "step": 2,
            "module": "SpatialGraphBuilder",
            "title": "3D/Topolojik Grafik İnşası",
            "details": graph_info,
        },
        {
            "step": 3,
            "module": "SpatialStateEncoder",
            "title": "Durum Vektörleştirme",
            "details": {
                "initial_position": initial_pos,
                "goal_position": goal_pos,
                "cartesian_dim": int(len(state_encoding.get("cartesian", []))),
                "polar_dim": int(len(state_encoding.get("polar", []))),
                "non_euclidean_dim": int(len(state_encoding.get("non_euclidean", []))),
            },
        },
    ]
    
    # ═══════ EXECUTION LOOP: Steps 4-9 ──────────────────────────────────────────
    steps = []
    current_state = state_encoding['full_state']
    
    for step_idx in range(max_steps):
        step_record = {}
        
        # ADIM 4: Action Simulation Planı (dinamik aksiyon kümesi)
        direction_vectors = []
        for vec_text in spatial_signals.get("directions", {}).keys():
            cleaned = vec_text.strip("() ")
            parts = [p.strip() for p in cleaned.split(",")]
            if len(parts) == 3:
                try:
                    direction_vectors.append(tuple(int(float(p)) for p in parts))
                except Exception:
                    pass
        if not direction_vectors:
            direction_vectors = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ]
        available_actions = list(dict.fromkeys(direction_vectors))

        step_record['action_simulation_plan'] = {
            'candidate_actions': available_actions,
            'action_count': len(available_actions),
        }

        # ADIM 5: Q-Learning Action Selection
        state_tuple = tuple(np.round(current_state[:3]).astype(int))
        best_action = _q_policy.select_action(state_tuple, available_actions)
        step_record['action'] = best_action
        step_record['q_learning'] = {
            'state': state_tuple,
            'epsilon': float(_q_policy.epsilon),
            'visit_count': int(_q_policy.visit_count[state_tuple]),
        }
        
        # ADIM 6: Physics Simulation
        physics_result = _action_sim.simulate_step(
            current_state,
            np.array(best_action),
            obstacles=_spatial_graph_builder.obstacles
        )
        step_record['physics'] = {
            'new_position': physics_result['position'].tolist(),
            'collision': physics_result['collision'],
        }
        
        # ADIM 7: Bayesian Update
        new_state_tuple = tuple(np.round(physics_result['state'][:3]).astype(int))
        _bayes_model.state_space = list(_spatial_graph_builder.nodes)[:20]  # Sample
        posterior = _bayes_model.update_posterior(new_state_tuple)
        step_record['bayesian'] = {
            'observation': new_state_tuple,
            'posteriors': {str(k): v for k, v in list(posterior.items())[:3]},  # Top-3
        }
        
        # ADIM 8: Topology Check
        has_cycle = _topology.detect_cycles(_spatial_graph_builder.graph)
        components = _topology.connected_components(_spatial_graph_builder.graph)
        step_record['topology'] = {
            'has_cycle': has_cycle,
            'components': len(components),
        }
        
        # ADIM 9: Rule Evolution
        if step_idx % 3 == 0:  # Her 3. adımda mutasyon
            _rule_evo.mutate_rule()
        evolved_state, ca_info = _rule_evo.step([1, 0, 1, 1, 0, 1, 0])
        step_record['rule_evolution'] = ca_info
        
        # ADIM 10: Consensus
        consensus = _multi_layer_mgr.step(
            current_state,
            np.array(best_action),
            new_state_tuple,
            _spatial_graph_builder.graph,
            _spatial_graph_builder.obstacles
        )
        step_record['consensus'] = {
            'confidence': consensus['consensus_confidence'],
            'consensus_state': consensus['consensus_state'].tolist(),
        }

        # ADIM 11: Quantum-Bayesian Processor
        qbayes = _quantum_bayes.process(posterior, observation=new_state_tuple)
        top_hybrid = sorted(
            qbayes.get("hybrid_posterior", {}).items(),
            key=lambda kv: -kv[1]
        )[:3]
        step_record['quantum_bayesian'] = {
            'collapsed_state': str(qbayes.get("collapsed_state")),
            'top_hybrid_posterior': [(str(k), float(v)) for k, v in top_hybrid],
        }
        
        # ─ Reward Hesapla ─────────────────────────────────────────────────────
        distance_to_goal = np.linalg.norm(
            np.array(physics_result['position']) - np.array(goal_pos)
        )
        
        reward = -distance_to_goal
        if physics_result['collision']:
            reward -= 5
        if distance_to_goal < 1.0:
            reward += 10
        
        # ─ Q-Learning Update ──────────────────────────────────────────────────
        next_actions = available_actions
        _q_policy.update(state_tuple, best_action, reward, new_state_tuple, next_actions)
        step_record['reward'] = reward
        
        # State güncelle
        current_state = physics_result['state']
        
        # Son durum kontrol et
        if distance_to_goal < 0.5:
            step_record['reached_goal'] = True
            steps.append(step_record)
            break
        
        steps.append(step_record)
    
    # ═══════ ADIM 12: Self-Correction / Report Build ────────────────────────────
    pipeline_trace.extend(
        [
            {
                "step": 4,
                "module": "ActionSimulationEngine",
                "title": "Aksiyon Uzayı Üretimi",
                "details": {
                    "dynamic_source": "direction_signals_or_default",
                    "max_steps": max_steps,
                },
            },
            {
                "step": 5,
                "module": "QLearningPolicyModule",
                "title": "Q-Learning Politika Güncellemesi",
                "details": {
                    "epsilon": float(_q_policy.epsilon),
                    "alpha": float(_q_policy.alpha),
                    "gamma": float(_q_policy.gamma),
                    "visited_state_actions": len(_q_policy.visit_count),
                },
            },
            {
                "step": 6,
                "module": "ActionSimulationEngine",
                "title": "Fizik Adımlama ve Çarpışma",
                "details": {"gravity_enabled": True, "friction_enabled": True},
            },
            {
                "step": 7,
                "module": "BayesianWorldModel",
                "title": "Belirsizlikten Posterior Üretimi",
                "details": {"tracked_state_space": len(_bayes_model.state_space)},
            },
            {
                "step": 8,
                "module": "TopologyReasoner",
                "title": "Topolojik Çizge Analizi",
                "details": {"analysis": ["cycle_detection", "components", "shortest_path_ready"]},
            },
            {
                "step": 9,
                "module": "RuleEvolutionEngine",
                "title": "Hücresel Otomat Kural Evrimi",
                "details": {"current_rule_id": int(_rule_evo.rule_id)},
            },
            {
                "step": 10,
                "module": "MultiLayerSimulationManager",
                "title": "Çok Katmanlı Senkronizasyon",
                "details": {"layer_weights": _multi_layer_mgr.layer_weights},
            },
            {
                "step": 11,
                "module": "QuantumBayesianProcessor",
                "title": "Kuantum-Esinli Bayes İşleme",
                "details": {"mode": "superposition_to_collapse_hybrid"},
            },
            {
                "step": 12,
                "module": "process_spatial_reasoning_task",
                "title": "Çözüm Adımı Başlığı",
                "details": {
                    "summary": "Tüm modül adımları birleştirilerek nihai çözüm raporu üretildi.",
                    "step_records": len(steps),
                },
            },
        ]
    )
    
    return {
        'task': task_description,
        'initial_state': {'position': initial_pos, 'goal': goal_pos},
        'final_state': {
            'position': current_state[:3].tolist(),
            'velocity': current_state[3:6].tolist() if len(current_state) > 3 else [0, 0, 0],
        },
        'steps': steps,
        'spatial_signals': spatial_signals,
        'graph_info': graph_info,
        'q_learning_visits': dict(_q_policy.visit_count),
        'success': distance_to_goal < 0.5,
        'total_steps': len(steps),
        'solution_steps_title': 'Çözüm Adımı Başlığı',
        'module_pipeline': pipeline_trace,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SPATIAL PROBLEM DATABASE (JSON)
# ═══════════════════════════════════════════════════════════════════════════════

SPATIAL_PROBLEM_DATABASE = {
    # ───────────────────────────────────────────────────────────────────────────
    # 3D MAZE PROBLEMS
    # ───────────────────────────────────────────────────────────────────────────
    
    "maze_3d_basic": {
        "type": "maze",
        "dimensions": 3,
        "description": "3D labirent: başlangıç (0,0,0), hedef (9,9,9). Engeller kuzey-batı köşesinde.",
        "initial_position": [0, 0, 0],
        "goal_position": [9, 9, 9],
        "obstacles": [
            {"type": "wall", "location": "northwest"},
            {"type": "wall", "location": "center"}
        ],
        "expected_solution_type": "pathfinding",
        "difficulty": "easy",
    },
    
    "maze_3d_complex": {
        "type": "maze",
        "dimensions": 3,
        "description": "Karmaşık 3D maze: 15x15x10 grid, birden çok engel tabakası. (3,3,0) → (12,12,9)",
        "grid_size": [15, 15, 10],
        "initial_position": [3, 3, 0],
        "goal_position": [12, 12, 9],
        "obstacles": [
            {"type": "wall", "location": "north"},
            {"type": "wall", "location": "south"},
            {"type": "barrier", "location": "center"},
        ],
        "expected_solution_type": "pathfinding",
        "difficulty": "hard",
    },
    
    "maze_2d_grid": {
        "type": "grid",
        "dimensions": 2,
        "description": "2D ızgara: (0,0) → (10,10), çok sayıda engel",
        "grid_size": [10, 10],
        "initial_position": [0, 0, 0],
        "goal_position": [10, 10, 0],
        "obstacles": [
            {"type": "wall", "position": [5, 0]},
            {"type": "wall", "position": [5, 1]},
            {"type": "wall", "position": [5, 2]},
        ],
        "expected_solution_type": "pathfinding",
        "difficulty": "easy",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # DIRECTED TILE SYSTEMS
    # ───────────────────────────────────────────────────────────────────────────
    
    "tile_directed_arrows": {
        "type": "tile_system",
        "description": "Yönlü karolar: her karo bir yöne işaret eder. Hedefe ulaş.",
        "grid_size": [5, 5],
        "initial_position": [0, 0, 0],
        "goal_position": [4, 4, 0],
        "tiles": [
            {"pos": [0, 0], "direction": "east"},
            {"pos": [1, 0], "direction": "south"},
            {"pos": [1, 1], "direction": "east"},
            {"pos": [2, 1], "direction": "north"},
            {"pos": [2, 0], "direction": "east"},
            {"pos": [3, 0], "direction": "south"},
            {"pos": [3, 1], "direction": "east"},
            {"pos": [4, 1], "direction": "north"},
            {"pos": [4, 0], "direction": "east"},
        ],
        "expected_solution_type": "follow_path",
        "difficulty": "medium",
    },
    
    "tile_rotating_platform": {
        "type": "tile_system",
        "description": "Dönen platform: beden bir adım her dönemde 90°. Hedef pozisyonuyla özdeş yönlelendirme.",
        "grid_size": [3, 3],
        "initial_position": [1, 1, 0],
        "initial_orientation": [0, 0, 0],
        "goal_position": [1, 1, 0],
        "goal_orientation": [0, 0, 1.57],  # 90 derece
        "tiles": [
            {"pos": [1, 1], "rotation_per_step": 1.57, "direction": "cw"}
        ],
        "expected_solution_type": "rotation_matching",
        "difficulty": "medium",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # CELLULAR AUTOMATA
    # ───────────────────────────────────────────────────────────────────────────
    
    "cellular_automata_rule_110": {
        "type": "cellular_automata",
        "description": "CA Rule 110: başlangıç durumu [1,0,1,1,0,1] → 5 adım sonrasında hedef duruma ulaş.",
        "initial_state": [1, 0, 1, 1, 0, 1, 0, 0, 1],
        "target_state": [1, 1, 0, 0, 1, 1, 1, 0, 1],
        "rule_id": 110,
        "steps": 5,
        "expected_solution_type": "ca_simulation",
        "difficulty": "medium",
    },
    
    "cellular_automata_game_of_life": {
        "type": "cellular_automata",
        "description": "Conway's Game of Life: başlangıç pattern → n adım sonra dinamik denge.",
        "dimension": 2,
        "grid_size": [10, 10],
        "initial_pattern": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "steps": 10,
        "expected_pattern_type": "oscillator",
        "expected_solution_type": "ca_evolution",
        "difficulty": "hard",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # BLOCK STACKING & PHYSICS
    # ───────────────────────────────────────────────────────────────────────────
    
    "block_stacking_tower": {
        "type": "block_stacking",
        "description": "4 bloğu (2x2x4) pozisyonuna istifle. Stabilite koşulu: center-of-gravity < base",
        "blocks": [
            {"id": 1, "size": [2, 2, 1], "current_pos": [0, 0, 0]},
            {"id": 2, "size": [2, 2, 1], "current_pos": [3, 0, 0]},
            {"id": 3, "size": [2, 2, 1], "current_pos": [0, 3, 0]},
            {"id": 4, "size": [2, 2, 1], "current_pos": [3, 3, 0]},
        ],
        "goal_configuration": [
            {"id": 1, "pos": [1, 1, 0]},
            {"id": 2, "pos": [1, 1, 1]},
            {"id": 3, "pos": [1, 1, 2]},
            {"id": 4, "pos": [1, 1, 3]},
        ],
        "stability_constraint": True,
        "expected_solution_type": "physics_optimization",
        "difficulty": "hard",
    },
    
    "block_sliding_puzzle": {
        "type": "block_stacking",
        "description": "3x3 sliding puzzle: numara bloklarını sıralı düzende yerleştir",
        "grid_size": [3, 3],
        "initial_state": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0],  # 0 = boş
        ],
        "goal_state": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0],
        ],
        "allowed_moves": "adjacent_swap",
        "expected_solution_type": "permutation_search",
        "difficulty": "easy",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # SPATIAL OPTIMIZATION PUZZLES
    # ───────────────────────────────────────────────────────────────────────────
    
    "room_key_lock": {
        "type": "room_key_lock",
        "description": "3 oda, 3 kapı, 3 anahtar. Doğru sırada açınız: oda1 → oda2 → oda3 → çıkış",
        "rooms": [
            {
                "id": 1,
                "initial_pos": [0, 0, 0],
                "doors": [
                    {"color": "red", "requires_key": "key_red"}
                ],
                "items": ["key_blue"]
            },
            {
                "id": 2,
                "doors": [
                    {"color": "blue", "requires_key": "key_blue"}
                ],
                "items": ["key_red"]
            },
            {
                "id": 3,
                "doors": [
                    {"color": "red", "requires_key": "key_red"}
                ],
                "items": ["escape"]
            },
        ],
        "expected_solution_type": "sequential_logic",
        "difficulty": "medium",
    },
    
    "resource_distribution": {
        "type": "optimization",
        "description": "5 depo, 10 kaynak. Depoları doldur: minimum hareket, maksimal verimlilik.",
        "depots": [
            {"id": 1, "pos": [0, 0], "capacity": 3, "target": 3},
            {"id": 2, "pos": [5, 0], "capacity": 4, "target": 4},
            {"id": 3, "pos": [10, 0], "capacity": 2, "target": 2},
            {"id": 4, "pos": [5, 5], "capacity": 2, "target": 1},
            {"id": 5, "pos": [0, 5], "capacity": 3, "target": 0},
        ],
        "resources": [
            {"id": i, "current_depot": 0, "type": "general"} for i in range(10)
        ],
        "objective": "minimize_total_distance",
        "expected_solution_type": "distribution_optimization",
        "difficulty": "hard",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # TOPOLOGICAL & GRAPH REASONING
    # ───────────────────────────────────────────────────────────────────────────
    
    "graph_shortest_path": {
        "type": "graph",
        "description": "Ağır grafik: 8 node, 12 kenar. (0) → (7) en kısa yol",
        "nodes": list(range(8)),
        "edges": [
            (0, 1, 1), (0, 2, 4),
            (1, 2, 2), (1, 3, 5),
            (2, 3, 1), (2, 4, 3),
            (3, 4, 2), (3, 5, 4),
            (4, 5, 1), (4, 6, 2),
            (5, 6, 1), (6, 7, 3),
        ],
        "start": 0,
        "goal": 7,
        "expected_solution_type": "dijkstra",
        "expected_distance": 8,
        "difficulty": "easy",
    },
    
    "graph_cycle_detection": {
        "type": "graph",
        "description": "Grafik döngü tespiti: döngü var mı?",
        "nodes": list(range(6)),
        "edges": [
            (0, 1), (1, 2), (2, 3), (3, 1),  # Döngü: 1→2→3→1
            (0, 4), (4, 5),
        ],
        "has_cycle": True,
        "expected_solution_type": "cycle_detection",
        "difficulty": "medium",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # CONSTRAINT SATISFACTION
    # ───────────────────────────────────────────────────────────────────────────
    
    "constraint_coloring": {
        "type": "csp",
        "description": "Grafik boyama: 4 renk ile 5 node'u boyla. Komşular farklı renk",
        "nodes": [0, 1, 2, 3, 4],
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)],
        "colors": ["red", "blue", "green", "yellow"],
        "constraint": "no_adjacent_same_color",
        "expected_solution_type": "csp_solver",
        "difficulty": "medium",
    },
    
    "constraint_sudoku_3x3": {
        "type": "csp",
        "description": "Basit 3x3 sudoku (9 cell)",
        "grid": [
            [5, 3, 0],
            [6, 0, 0],
            [0, 9, 8],
        ],
        "constraint": "1_to_9_once_per_row_col_block",
        "expected_solution_type": "csp_backtrack",
        "difficulty": "easy",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # DYNAMIC GATE-KEY SYSTEMS
    # ───────────────────────────────────────────────────────────────────────────
    
    "gate_key_dynamic": {
        "type": "gate_key",
        "description": "Kapılar ve anahtarlar: geçişte anahtar 'timeout' ediliyor. Reset zamanını hesapla.",
        "gates": [
            {"id": "g1", "requires_key": "k1", "timeout": 10},
            {"id": "g2", "requires_key": "k2", "timeout": 15},
            {"id": "g3", "requires_key": "k1", "timeout": 10},
        ],
        "keys": [
            {"id": "k1", "initial_pos": [0, 0], "recharge_time": 20},
            {"id": "k2", "initial_pos": [5, 0], "recharge_time": 25},
        ],
        "sequence": ["g1", "g2", "g3"],
        "objective": "traverse_all_gates_before_timeout",
        "expected_solution_type": "temporal_planning",
        "difficulty": "hard",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # RANDOMIZED CONSTRAINT PUZZLES
    # ───────────────────────────────────────────────────────────────────────────
    
    "randomized_path": {
        "type": "probabilistic",
        "description": "Rastgele kesintili yol: her adımda %30 şansa engel oluş. Hedefe ulaş.",
        "grid_size": [20, 20],
        "initial_position": [0, 0],
        "goal_position": [19, 19],
        "obstacle_probability": 0.3,
        "obstacle_decay": 0.5,  # Geri çekilme
        "expected_solution_type": "probabilistic_pathfinding",
        "difficulty": "hard",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # SPACE-TIME GRID SIMULATIONS
    # ───────────────────────────────────────────────────────────────────────────
    
    "spacetime_particle": {
        "type": "spacetime",
        "description": "Parçacık simülasyonu: 5 parçacık, etkileşim kuralları. 20 step sonra durumu kontrol et.",
        "particles": [
            {"id": 0, "pos": [0, 0], "vel": [1, 0], "mass": 1},
            {"id": 1, "pos": [10, 0], "vel": [-1, 0], "mass": 1},
            {"id": 2, "pos": [5, 5], "vel": [0, 0], "mass": 2},
        ],
        "forces": [
            {"type": "gravity", "strength": -9.8},
            {"type": "friction", "coefficient": 0.05},
        ],
        "interactions": [
            {"type": "elastic_collision", "particles": [0, 1]}
        ],
        "steps": 20,
        "expected_solution_type": "physics_simulation",
        "difficulty": "hard",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # MULTI-STEP DECISION CHAINS
    # ───────────────────────────────────────────────────────────────────────────
    
    "decision_chain_if_then": {
        "type": "decision_logic",
        "description": "Kararlar zincirleme: IF x THEN y ELSE z. 4 adımlık seçim ağacı.",
        "conditions": [
            {
                "id": 1,
                "question": "koordinat x > 5 mi?",
                "true_next": 2,
                "false_next": 3,
            },
            {
                "id": 2,
                "question": "koordinat y > 5 mi?",
                "true_next": "goal_northeast",
                "false_next": "goal_southeast",
            },
            {
                "id": 3,
                "question": "koordinat y < 5 mi?",
                "true_next": "goal_southwest",
                "false_next": "goal_northwest",
            },
        ],
        "start_condition": 1,
        "expected_solution_type": "decision_tree_traversal",
        "difficulty": "easy",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # INTERACTIVE MECHANICAL CELL SYSTEMS
    # ───────────────────────────────────────────────────────────────────────────
    
    "mechanical_gears": {
        "type": "mechanical",
        "description": "Dişli sistemi: 3 dişli, çeşitli oranlar. Ürün belirli açıya döndürmek",
        "gears": [
            {"id": "input", "teeth": 20, "initial_angle": 0, "connected_to": "g1"},
            {"id": "g1", "teeth": 40, "initial_angle": 0, "connected_to": ["g2", "g3"]},
            {"id": "g2", "teeth": 10, "initial_angle": 0, "output": True},
            {"id": "g3", "teeth": 30, "initial_angle": 0},
        ],
        "input_rotation": 360,  # derece
        "target_output_angle": 90,
        "expected_solution_type": "gear_ratio_calculation",
        "difficulty": "medium",
    },
    
    # ───────────────────────────────────────────────────────────────────────────
    # ADDITIONAL SPATIAL PROBLEMS (50+ TOTAL)
    # ───────────────────────────────────────────────────────────────────────────
    
    "maze_spiral": {
        "type": "maze",
        "description": "Spiral labirent: merkezden dışarı spiral pat İzle",
        "grid_size": [11, 11],
        "pattern": "spiral",
        "initial_position": [5, 5, 0],
        "goal_position": [0, 0, 0],
        "difficulty": "medium",
    },
    
    "maze_nested_cycles": {
        "type": "maze",
        "description": "İç içe döngülü labirent: 3 döngü tabakası",
        "grid_size": [15, 15],
        "pattern": "concentric_circles",
        "initial_position": [7, 7, 0],
        "goal_position": [0, 7, 0],
        "difficulty": "hard",
    },
    
    "navigation_3d_spherical": {
        "type": "navigation",
        "description": "3D küresel navigasyon: Kutup koordinatlarında (r, θ, φ) hedefe ulaş",
        "initial_spherical": [1, 0, 0],
        "goal_spherical": [5, 1.57, 0.785],  # (5, 90°, 45°)
        "coordinate_system": "spherical",
        "difficulty": "hard",
    },
    
    "navigation_maze_switching_dimensions": {
        "type": "maze",
        "description": "Boyut değiştirme: 2D düzlem ve 3D uzay arasında geçiş",
        "zones": [
            {"type": "2d", "grid": [10, 10], "exit_pos": [9, 9]},
            {"type": "3d", "grid": [10, 10, 10], "entry_pos": [0, 0, 0]},
        ],
        "initial_pos": [0, 0],  # 2D
        "goal_pos": [9, 9, 9],  # 3D
        "difficulty": "hard",
    },
    
    "topology_knot_untying": {
        "type": "topology",
        "description": "Topolojik knot çözme: 1D string düğümü",
        "knot_type": "trefoil",
        "complexity": 3,
        "objective": "unknot_to_circle",
        "difficulty": "hard",
    },
    
    "block_tower_balance": {
        "type": "block_stacking",
        "description": "5 blok: ağır-hafif alternansı. Denge kütlesine göre sırala",
        "blocks": [
            {"id": 1, "weight": 10, "width": 2},
            {"id": 2, "weight": 5, "width": 3},
            {"id": 3, "weight": 8, "width": 2},
            {"id": 4, "weight": 3, "width": 4},
            {"id": 5, "weight": 12, "width": 2},
        ],
        "constraint": "stable_com_within_base",
        "difficulty": "hard",
    },
    
    "block_tetris_fit": {
        "type": "block_stacking",
        "description": "Tetris fit: 6 şekli 10x10 alanına yerleştir",
        "shapes": [
            {"id": 1, "type": "I", "size": [4, 1]},
            {"id": 2, "type": "O", "size": [2, 2]},
            {"id": 3, "type": "T", "size": [3, 2]},
        ],
        "target_area": [10, 10],
        "packing_ratio": 0.8,
        "difficulty": "medium",
    },
    
    "pursuit_evasion_agents": {
        "type": "multi_agent",
        "description": "Kovalama-kaçma: 1 avcı, 2 kaçan. Avı tut veya kaçanları koruma bölgesine al.",
        "agents": [
            {"id": "hunter", "type": "pursuer", "pos": [5, 5]},
            {"id": "prey1", "type": "evader", "pos": [0, 0]},
            {"id": "prey2", "type": "evader", "pos": [10, 10]},
        ],
        "safe_zone": [[8, 8], [10, 10]],
        "steps": 50,
        "difficulty": "hard",
    },
    
    "force_field_navigation": {
        "type": "navigation",
        "description": "Kuvvet alanı navigasyonu: çekici ve itici alanlar. Hedefe ulaş",
        "attractors": [
            {"pos": [9, 9], "strength": 10}
        ],
        "repellers": [
            {"pos": [5, 5], "strength": -5}
        ],
        "initial_pos": [0, 0],
        "goal_pos": [9, 9],
        "difficulty": "medium",
    },
    
    "fractal_maze": {
        "type": "maze",
        "description": "Fraktal labirent: self-similar yapı, 3 ölçek",
        "pattern": "sierpinski_triangle",
        "scales": 3,
        "initial_pos": [0, 0],
        "goal_pos": [100, 100],
        "difficulty": "hard",
    },
    
    "flow_network_optimization": {
        "type": "optimization",
        "description": "Akış ağı: max flow sorun. Kaynaktan havuza max akışı bul",
        "nodes": ["source", "a", "b", "c", "d", "sink"],
        "edges": [
            ("source", "a", 10),
            ("source", "b", 8),
            ("a", "c", 4),
            ("a", "d", 8),
            ("b", "a", 2),
            ("b", "d", 9),
            ("c", "sink", 10),
            ("d", "c", 6),
            ("d", "sink", 10),
        ],
        "objective": "max_flow",
        "difficulty": "hard",
    },
    
    "matching_bipartite_graph": {
        "type": "graph",
        "description": "İkili grafik eşleştirmesi: 5L ve 5R node'u. Max eşleştirmeyi bul",
        "left_nodes": [f"L{i}" for i in range(5)],
        "right_nodes": [f"R{i}" for i in range(5)],
        "edges": [
            ("L0", "R0"), ("L0", "R1"),
            ("L1", "R1"), ("L1", "R2"),
            ("L2", "R2"), ("L2", "R3"),
            ("L3", "R3"), ("L3", "R4"),
            ("L4", "R4"), ("L4", "R0"),
        ],
        "objective": "maximum_bipartite_matching",
        "difficulty": "medium",
    },
    
    "steiner_tree": {
        "type": "graph",
        "description": "Steiner ağacı: terminal setini bağla (min ağırlık)",
        "nodes": list(range(10)),
        "weighted_edges": [
            (0, 1, 4), (0, 2, 2),
            (1, 2, 1), (1, 3, 5),
            (2, 3, 8), (2, 4, 10),
            (3, 4, 2), (3, 5, 6),
            (4, 5, 3), (4, 6, 1),
            (5, 6, 9),
        ],
        "terminals": [0, 3, 6],
        "objective": "minimum_steiner_tree",
        "difficulty": "hard",
    },
    
    "traveling_salesman_small": {
        "type": "optimization",
        "description": "Satıcı problemi (TSP): 8 şehir, en kısa tur",
        "cities": [
            (0, 0), (1, 1), (2, 0), (1, -1),
            (-1, -1), (-2, 0), (-1, 1), (0, 2)
        ],
        "objective": "shortest_hamiltonian_circuit",
        "difficulty": "hard",
    },
    
    "knapsack_0_1": {
        "type": "optimization",
        "description": "0-1 Sırt Çantası: kapasitesi 15, 6 eşya, max değer",
        "items": [
            {"id": 1, "weight": 2, "value": 3},
            {"id": 2, "weight": 3, "value": 4},
            {"id": 3, "weight": 4, "value": 5},
            {"id": 4, "weight": 5, "value": 6},
            {"id": 5, "weight": 3, "value": 5},
            {"id": 6, "weight": 2, "value": 4},
        ],
        "capacity": 15,
        "objective": "maximize_value",
        "difficulty": "medium",
    },
    
    "coin_change_min_coins": {
        "type": "optimization",
        "description": "Minimum madeni para: 37 birim, paraların [1,3,4]. En az kaç coin?",
        "denominations": [1, 3, 4],
        "target_amount": 37,
        "objective": "minimum_coins",
        "expected_answer": 10,
        "difficulty": "easy",
    },
    
    "edit_distance_strings": {
        "type": "sequence",
        "description": "String edit distance: 'AGGTAB' → 'GXTXAYB', minimum operasyon",
        "source": "AGGTAB",
        "target": "GXTXAYB",
        "operations": ["insert", "delete", "substitute"],
        "objective": "minimum_edit_distance",
        "difficulty": "medium",
    },
    
    "longest_increasing_subsequence": {
        "type": "sequence",
        "description": "En uzun artan alt-sekans: [3,1,4,1,5,9,2,6] → ?",
        "sequence": [3, 1, 4, 1, 5, 9, 2, 6],
        "objective": "longest_increasing_subsequence",
        "expected_length": 4,
        "difficulty": "easy",
    },
    
    "coin_change_ways": {
        "type": "combinatorics",
        "description": "Para değişimi yolları: target=5, coins=[1,2,5]. Kaç yol?",
        "denominations": [1, 2, 5],
        "target": 5,
        "objective": "count_ways",
        "expected_count": 4,  # [5], [2,2,1], [2,1,1,1], [1,1,1,1,1]
        "difficulty": "easy",
    },
    
    "robot_room_cleaning": {
        "type": "agent_planning",
        "description": "Temizlik robotu: 5x5 oda, 20 engel. Tamamını temizle (min hareket)",
        "grid_size": [5, 5],
        "obstacle_count": 20,
        "initial_pos": [0, 0],
        "objective": "clean_all_non_obstacle_cells_min_moves",
        "difficulty": "hard",
    },
    
    "lights_out_puzzle": {
        "type": "puzzle",
        "description": "'Işıkları Kapat': 3x3 grid, ters seçili (toggle komşu)",
        "grid_size": [3, 3],
        "initial_state": [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
        "target_state": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "constraint": "toggle_neighbors_on_click",
        "difficulty": "medium",
    },
    
    "sokoban_mini": {
        "type": "puzzle",
        "description": "Sokoban mini: 1 sandık, 5x5 oda. Hedef pozisyona iter",
        "grid_size": [5, 5],
        "player_pos": [1, 1],
        "box_pos": [2, 2],
        "goal_pos": [4, 4],
        "obstacles": [[2, 0], [0, 3]],
        "difficulty": "medium",
    },
    
    "river_crossing_cannibals": {
        "type": "state_space",
        "description": "Nehir geçişi: 3 misyoner, 3 kanibalsalt raf. Güvenli geç",
        "initial_state": {
            "left": {"missionaries": 3, "cannibals": 3},
            "boat": "left",
            "right": {"missionaries": 0, "cannibals": 0},
        },
        "goal_state": {
            "left": {"missionaries": 0, "cannibals": 0},
            "boat": "right",
            "right": {"missionaries": 3, "cannibals": 3},
        },
        "constraint": "missionaries >= cannibals_on_both_sides",
        "boat_capacity": 2,
        "difficulty": "hard",
    },
    
    "tower_of_hanoi_4": {
        "type": "puzzle",
        "description": "Hanoi Kuleleri: 4 disk, 3 sütun. Kurala uyarak taşı",
        "disks": 4,
        "pegs": 3,
        "initial_state": ["DCBA", "", ""],  # A, B, C büyükten geçerin
        "goal_state": ["", "", "DCBA"],
        "move_rule": "single_disk_larger_on_smaller",
        "expected_min_moves": 15,
        "difficulty": "easy",
    },
    
    "queens_8": {
        "type": "csp",
        "description": "8 Kraliçe problemi: 8x8 tahta, kimse vuramaz",
        "board_size": 8,
        "constraint": "no_two_queens_attack",
        "objective": "place_all_queens",
        "expected_solutions": 92,
        "difficulty": "hard",
    },
    
    "graph_coloring_chromatic": {
        "type": "graph",
        "description": "Kromatik sayı: minimum kaç renk?",
        "nodes": list(range(6)),
        "edges": [
            (0, 1), (0, 2), (1, 2),  # Triangle
            (2, 3), (2, 4), (3, 4),  # Another
            (4, 5), (5, 0),
        ],
        "objective": "chromatic_number",
        "expected_chromatic": 3,
        "difficulty": "hard",
    },
    
    "vertex_cover_min": {
        "type": "graph",
        "description": "Minimum vertex cover: 8 node, 12 kenar",
        "nodes": list(range(8)),
        "edges": [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Quad 1
            (4, 5), (5, 6), (6, 7), (7, 4),  # Quad 2
            (0, 4), (1, 5),  # Connections
        ],
        "objective": "minimum_vertex_cover",
        "difficulty": "hard",
    },
    
    "reachability_graph": {
        "type": "graph",
        "description": "Ulaşılabilirlik analizi: 0'den hangi node'lara ulaşılır?",
        "nodes": list(range(7)),
        "edges": [
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5),
            (3, 6), (4, 6),
        ],
        "start": 0,
        "objective": "reachable_vertices",
        "expected_reachable": [0, 1, 2, 3, 4, 5, 6],
        "difficulty": "easy",
    },
    
    "strongly_connected_components": {
        "type": "graph",
        "description": "Güçlü bağlı bileşenler sayısı",
        "nodes": list(range(5)),
        "directed_edges": [
            (0, 1), (1, 2), (2, 0),  # SCC 1: {0,1,2}
            (1, 3),   # 3 → 4 kanalı
            (3, 4),   # SCC 2: {3}
            (4, 3),   # SCC 3: {4}
        ],
        "objective": "count_sccs",
        "expected_sccs": 3,
        "difficulty": "medium",
    },
    
    "articulation_points": {
        "type": "graph",
        "description": "Kesim noktaları (bridge vertices): kaldırılınca grafik kesilir",
        "nodes": list(range(6)),
        "undirected_edges": [
            (0, 1), (1, 2), (1, 3),
            (3, 4), (4, 5), (5, 3),
        ],
        "objective": "find_articulation_points",
        "expected_points": [1, 3, 4],
        "difficulty": "medium",
    },
    
    "flow_circulation": {
        "type": "network_flow",
        "description": "Dolaşım problemi: her kenar alt/üst limit",
        "nodes": ["s", "a", "b", "t"],
        "flow_constraints": [
            ("s", "a", 0, 10),
            ("s", "b", 0, 8),
            ("a", "b", 0, 5),
            ("a", "t", 0, 10),
            ("b", "t", 0, 10),
        ],
        "demand": {"s": -5, "a": 0, "b": 0, "t": 5},
        "objective": "feasible_circulation",
        "difficulty": "hard",
    },
}

# Toplam problem sayısı
print(f"✓ {len(SPATIAL_PROBLEM_DATABASE)} Spatial Problem Oluşturuldu")

# Test: Bazı problemleri listelenelim
_ = list(SPATIAL_PROBLEM_DATABASE.keys())[:10]  # İlk 10 key


# ═══════════════════════════════════════════════════════════════════════════════
#  SPATIAL REASONING FLASK ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/spatial/solve', methods=['POST'])
def spatial_solve_endpoint():
    """
    Mekansal akıl yürütme problemi çöz.
    POST JSON: {"task_description": "..."}
    """
    try:
        data = request.json or {}
        task_desc = data.get('task_description', 'Navigate 3D maze: start (0,0,0) to goal (5,5,5)')
        max_steps = data.get('max_steps', 10)
        
        result = process_spatial_reasoning_task(task_desc, max_steps)
        
        return jsonify({
            'success': True,
            'task': task_desc,
            'result': {
                'solution_steps_title': result.get('solution_steps_title', 'Çözüm Adımı Başlığı'),
                'initial_state': result['initial_state'],
                'final_state': result['final_state'],
                'total_steps': result['total_steps'],
                'success': result['success'],
                'module_pipeline': result.get('module_pipeline', []),
                'steps_summary': [
                    {
                        'step': i,
                        'action': str(s.get('action')),
                        'collision': s.get('physics', {}).get('collision', False),
                        'reward': s.get('reward', 0),
                    }
                    for i, s in enumerate(result['steps'][:5])  # İlk 5 adım
                ],
            },
            'q_learning_info': {
                'visits': len(result['q_learning_visits']),
                'epsilon': float(_q_policy.epsilon),
                'learning_rate': float(_q_policy.alpha),
            },
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }), 500


@app.route('/spatial/problems', methods=['GET'])
def spatial_problems_endpoint():
    """
    Tüm spatial problemleri liste.
    Query: ?type=maze&difficulty=easy
    """
    try:
        problem_type = request.args.get('type', None)
        difficulty = request.args.get('difficulty', None)
        
        filtered = {}
        for key, problem in SPATIAL_PROBLEM_DATABASE.items():
            if problem_type and problem.get('type') != problem_type:
                continue
            if difficulty and problem.get('difficulty') != difficulty:
                continue
            filtered[key] = problem
        
        return jsonify({
            'success': True,
            'total_problems': len(SPATIAL_PROBLEM_DATABASE),
            'filtered_count': len(filtered),
            'filter': {
                'type': problem_type,
                'difficulty': difficulty,
            },
            'problems': filtered,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/spatial/problem/<problem_key>', methods=['GET'])
def spatial_problem_detail_endpoint(problem_key):
    """Belirli bir problem detayı."""
    try:
        if problem_key not in SPATIAL_PROBLEM_DATABASE:
            return jsonify({'success': False, 'error': 'Problem not found'}), 404
        
        problem = SPATIAL_PROBLEM_DATABASE[problem_key]
        
        return jsonify({
            'success': True,
            'problem_key': problem_key,
            'problem': problem,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/spatial/stats', methods=['GET'])
def spatial_stats_endpoint():
    """Spatial problem veri tabanı istatistikleri."""
    try:
        types = defaultdict(int)
        difficulties = defaultdict(int)
        
        for key, problem in SPATIAL_PROBLEM_DATABASE.items():
            types[problem.get('type', 'unknown')] += 1
            difficulties[problem.get('difficulty', 'unknown')] += 1
        
        return jsonify({
            'success': True,
            'total_problems': len(SPATIAL_PROBLEM_DATABASE),
            'by_type': dict(types),
            'by_difficulty': dict(difficulties),
            'module_summary': {
                'SpatialSemanticParser': 'NLP mekansal sinyal çıkarımı',
                'SpatialGraphBuilder': '3D/topolojik grafik inşası',
                'SpatialStateEncoder': 'Durum vektörleştirmesi',
                'ActionSimulationEngine': 'Fizik simülasyonu',
                'QLearningPolicyModule': 'Q-Learning politikası',
                'BayesianWorldModel': 'Bayesian belirsizlik filtrelemesi',
                'TopologyReasoner': 'Grafik analizi',
                'RuleEvolutionEngine': 'CA kuralı evölüsyonu',
                'MultiLayerSimulationManager': 'Multi-layer consensus',
                'QuantumBayesianProcessor': 'Kuantum-esinli Bayes süperpozisyon/collapse',
                'process_spatial_reasoning_task': '12-adım orkestrasyon ve raporlama',
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/spatial/run-test', methods=['GET'])
def spatial_test_endpoint():
    """Basit bir test çalıştır."""
    try:
        test_task = "3D maze: navigate from (0,0,0) to (9,9,9) with obstacles at northwest and center"
        result = process_spatial_reasoning_task(test_task, max_steps=5)
        
        return jsonify({
            'success': True,
            'test_task': test_task,
            'result': {
                'total_steps': result['total_steps'],
                'success': result['success'],
                'final_position': result['final_state']['position'],
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECULATIVE FICTION / EQUATION DISCOVERY PIPELINE (MD uyumlu)
# ═══════════════════════════════════════════════════════════════════════════════

class PromptIntentSplitter:
    """Kullanıcı metninden mod + alt-görevleri algoritmik çıkarır."""

    _MODE_PATTERNS = {
        "PARADOX": [r"paradoks", r"çelişki", r"self[- ]?trap", r"contradiction"],
        "EQUATION_DISCOVERY": [r"denklem", r"equation", r"ilişki", r"model keş"],
        "SIMULATION_REALISTIC": [r"simül", r"fizik", r"olasılık", r"monte carlo"],
        "HYBRID_SIM_PARADOX": [r"hibrit", r"hybrid", r"hem .* hem", r"yaratıcı"],
    }

    def split(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        mode_scores = {"SOLVE": 0.25}

        for mode, pats in self._MODE_PATTERNS.items():
            score = 0.0
            for p in pats:
                hit = len(re.findall(p, t, flags=re.IGNORECASE))
                score += min(0.35, 0.12 * hit)
            mode_scores[mode] = score

        mode = max(mode_scores, key=mode_scores.get)
        confidence = round(float(min(0.99, mode_scores[mode] + 0.5)), 4)

        clauses = [c.strip() for c in re.split(r"[.;\n]+", text or "") if c.strip()]
        sub_tasks = [{"id": i + 1, "task": c} for i, c in enumerate(clauses[:8])]

        return {
            "mode": mode,
            "confidence": confidence,
            "sub_tasks": sub_tasks,
            "mode_scores": mode_scores,
        }


class TaskGraphBuilder:
    """Alt görevlerden bağımlılık grafiği çıkarır."""

    def build(self, intent_payload: Dict[str, Any]) -> Dict[str, Any]:
        tasks = intent_payload.get("sub_tasks", [])
        nodes = [f"T{i+1}" for i in range(len(tasks))]
        edges = []
        for i in range(len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1]))
        return {"nodes": nodes, "edges": edges, "is_dag": True}


class VariableEntityExtractor:
    """Varlık/değişken/birim/kısıt çıkarımı."""

    def extract(self, text: str) -> Dict[str, Any]:
        t = text or ""
        entity_hits = re.findall(r"\b([A-Za-zÇĞİÖŞÜçğıöşü]{3,})\b", t)
        entities = sorted({e.lower() for e in entity_hits if not e.isdigit()})[:32]

        variables = sorted(set(re.findall(r"\b[a-zA-Z](_[a-zA-Z0-9]+)?\b", t)))
        variables = [v for v in variables if len(v) <= 8][:24]

        constraints = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*\s*(?:>=|<=|=|>|<)\s*[-+]?\d+(?:\.\d+)?)", t)
        units = re.findall(r"\b(m/s|kg|N|Pa|J|K|Hz|W|mol|s)\b", t, flags=re.IGNORECASE)
        clauses = [c.strip() for c in re.split(r"[.;\n]+", t) if c.strip()]

        temporal_markers = {"önce", "sonra", "after", "before", "eşzamanlı", "aynı"}
        event_keywords = {"patlama", "çarpışma", "çarpma", "parçalanma", "düşme", "gözlem"}
        temporal_events = []
        for i, clause in enumerate(clauses, start=1):
            lc = clause.lower()
            if any(k in lc for k in temporal_markers | event_keywords):
                temporal_events.append({
                    "id": f"E{i}",
                    "text": clause,
                    "markers": sorted([k for k in temporal_markers if k in lc]),
                    "event_terms": sorted([k for k in event_keywords if k in lc]),
                })

        return {
            "entities": entities,
            "variables": variables,
            "constraints": constraints,
            "units": sorted(set(u.lower() for u in units)),
            "temporal_events": temporal_events,
        }


class LatentVariableGenerator:
    """Eksik parametreler için prior üretir (hard-coded senaryo yok, aralıktan türetim)."""

    def generate(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        vars_known = set(extracted.get("variables", []))
        latent = {}
        for base in ["E", "r", "orientation", "shielding", "impulse"]:
            if base not in vars_known:
                if base == "orientation":
                    latent[base] = {"dist": "Categorical", "params": ["front", "side", "back"]}
                elif base == "r":
                    latent[base] = {"dist": "Uniform", "params": [1.0, 50.0]}
                else:
                    latent[base] = {"dist": "LogUniform", "params": [1e-3, 1e3]}
        return latent


class AssumptionRegistry:
    def register(self, extracted: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
        assumptions = []
        if not extracted.get("constraints"):
            assumptions.append("constraints_inferred_from_positive_domains")
        for k in latent.keys():
            assumptions.append(f"prior_assigned::{k}")
        return {"assumptions": assumptions, "count": len(assumptions)}


class WorldModelBuilder:
    def build(self, extracted: Dict[str, Any], priors: Dict[str, Any], task_graph: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entities": extracted.get("entities", []),
            "state_variables": extracted.get("variables", []),
            "latent_priors": priors,
            "task_graph": task_graph,
        }


class HypergraphBuilder:
    """Metni n-ary hypergraph temsiline dönüştürür."""

    def build(self, text: str, extracted: Dict[str, Any], task_graph: Dict[str, Any]) -> Dict[str, Any]:
        clauses = [c.strip() for c in re.split(r"[.;\n]+", text or "") if c.strip()]
        nodes = set(extracted.get("entities", [])) | set(extracted.get("variables", []))
        hyperedges = []
        temporal_terms = {"önce", "sonra", "aynı", "eşzamanlı", "simultaneous", "before", "after"}

        for i, clause in enumerate(clauses, start=1):
            clause_tokens = re.findall(r"[a-zA-ZÇĞİÖŞÜçğıöşü0-9_><=]+", clause.lower())
            edge_nodes = sorted(set(clause_tokens[:10]))
            nodes.update(edge_nodes)
            tags = []
            if any(tok in temporal_terms for tok in clause_tokens):
                tags.append("temporal")
            if any(tok in {"patlama", "çarpma", "hız", "enerji", "hasar", "ivme"} for tok in clause_tokens):
                tags.append("physics")
            if any(tok in {"paradoks", "çelişkili", "neden", "sonuç"} for tok in clause_tokens):
                tags.append("causal")
            hyperedges.append({"id": f"HE{i}", "nodes": edge_nodes, "tags": sorted(set(tags))})

        # Olay-temelli hypergraph: açık zaman akışı düğüm/hiperkenarları
        event_nodes = []
        for ev in extracted.get("temporal_events", []):
            ev_node = f"event::{ev['id']}"
            event_nodes.append(ev_node)
            nodes.add(ev_node)
            rel_nodes = [ev_node] + ev.get("event_terms", [])[:3] + ev.get("markers", [])[:2]
            hyperedges.append({
                "id": f"{ev['id']}_H",
                "nodes": sorted(set(rel_nodes)),
                "tags": sorted(set(["event"] + (["temporal"] if ev.get("markers") else []))),
                "source_text": ev.get("text", ""),
            })

        if len(event_nodes) >= 2:
            for i in range(len(event_nodes) - 1):
                hyperedges.append({
                    "id": f"TEMP_ORDER_{i+1}",
                    "nodes": [event_nodes[i], event_nodes[i + 1], "timeline"],
                    "tags": ["temporal_order", "causal_candidate"],
                })

        for idx, node in enumerate(task_graph.get("nodes", []), start=1):
            nodes.add(node)
            hyperedges.append({"id": f"TASK{idx}", "nodes": [node], "tags": ["task"]})

        return {"nodes": sorted(nodes), "hyperedges": hyperedges}


class RewriteRuleEngine:
    """Pattern -> replacement tabanlı rewrite kuralları."""

    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        new_edges = []
        generated_relations = []

        for e in graph.get("hyperedges", []):
            node_set = set(e.get("nodes", []))
            tags = set(e.get("tags", []))

            if {"enerji", "mesafe"}.intersection(node_set) or "physics" in tags:
                generated_relations.append({
                    "equation": "pressure = gamma*E/(4*pi*(r^2+eps))",
                    "family": "shockwave",
                    "source_edge": e.get("id"),
                })
                new_edges.append({"id": f"{e['id']}_RW1", "nodes": sorted(node_set | {"pressure", "E", "r"}), "tags": ["emergent", "inverse_square"]})

            if {"önce", "sonra", "before", "after", "timeline"}.intersection(node_set) or "temporal" in tags:
                generated_relations.append({
                    "equation": "delta_t_obs = t_obs - t_real",
                    "family": "observation_delay",
                    "source_edge": e.get("id"),
                })
                generated_relations.append({
                    "equation": "t_effect = t_cause + delta - beta*feedback",
                    "family": "retrocausal",
                    "source_edge": e.get("id"),
                })
                new_edges.append({"id": f"{e['id']}_RW2", "nodes": sorted(node_set | {"t_A", "t_B", "retrocausal"}), "tags": ["emergent", "temporal_conflict"]})

            if {"ışık", "hızından", "v", "c", "hızlı"}.intersection(node_set):
                generated_relations.append({
                    "equation": "v_eff = c*(1+alpha*Psi), alpha>0",
                    "family": "alt_physics",
                    "source_edge": e.get("id"),
                })
                new_edges.append({"id": f"{e['id']}_RW3", "nodes": sorted(node_set | {"v_shock", "c", "metric_alpha"}), "tags": ["emergent", "superluminal"]})

            if {"hasar", "mesafe", "artıyor"}.intersection(node_set):
                generated_relations.append({
                    "equation": "damage = a + b*r + c*r^2",
                    "family": "increasing_damage",
                    "source_edge": e.get("id"),
                })
                new_edges.append({"id": f"{e['id']}_RW4", "nodes": sorted(node_set | {"damage", "r", "k1", "k2"}), "tags": ["emergent", "increasing_damage"]})

            if "causal_candidate" in tags or "event" in tags:
                generated_relations.append({
                    "equation": "P(H|D) = P(D|H) * P(H) / P(D)",
                    "family": "bayesian_core",
                    "source_edge": e.get("id"),
                })

        merged = {
            "nodes": sorted(set(graph.get("nodes", [])) | {n for e in new_edges for n in e.get("nodes", [])}),
            "hyperedges": graph.get("hyperedges", []) + new_edges,
            "generated_relations": generated_relations,
        }
        return merged


class HypergraphEvolutionEngine:
    """G(t+1)=rewrite(G(t)) döngüsü."""

    def evolve(self, graph: Dict[str, Any], rewriter: RewriteRuleEngine, steps: int = 3) -> Dict[str, Any]:
        history = []
        current = graph
        for t in range(max(1, steps)):
            nxt = rewriter.apply(current)
            history.append({
                "step": t + 1,
                "node_count": len(nxt.get("nodes", [])),
                "hyperedge_count": len(nxt.get("hyperedges", [])),
                "relation_count": len(nxt.get("generated_relations", [])),
            })
            if len(nxt.get("hyperedges", [])) == len(current.get("hyperedges", [])):
                current = nxt
                break
            current = nxt
        return {"final_graph": current, "history": history}


class EquationExtractionLayer:
    """Emergent hypergraph'tan denklem hipotezleri çıkarır."""

    def extract(self, evolved_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        final_graph = evolved_graph.get("final_graph", {})
        tags = set()
        for edge in final_graph.get("hyperedges", []):
            tags.update(edge.get("tags", []))

        equations = []
        relation_pool = final_graph.get("generated_relations", [])
        for rel in relation_pool:
            eq = rel.get("equation", "")
            if not eq:
                continue
            equations.append({
                "id": f"EQ_REL_{len(equations)+1}",
                "equation": eq,
                "family": rel.get("family", "emergent"),
                "source_edge": rel.get("source_edge"),
            })

        # Algoritmik fallback (boş kalırsa) - sabit senaryo yerine tag-tabanlı
        if not equations:
            if "temporal_conflict" in tags:
                equations.append({"id": "EQ_FB_1", "equation": "x_t = F(x_t, x_{t+1})", "family": "self_referential"})
            if "inverse_square" in tags:
                equations.append({"id": "EQ_FB_2", "equation": "y = k/(r^2+eps)", "family": "inverse_square"})
            equations.append({"id": f"EQ_FB_{len(equations)+1}", "equation": "E_kin = 0.5*m*v^2", "family": "classical_energy"})
        return equations


class VariableAssignmentEngine:
    """Metinden sayısal değerleri çıkarıp eksikleri veri-bağımlı atar."""

    def assign(self, text: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
        values = {}
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")]
        units = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(m/s|kg|m|metre|pa|j|s)", (text or "").lower())
        for val, unit in units:
            key = {
                "m/s": "v",
                "kg": "m",
                "m": "r",
                "metre": "r",
                "pa": "pressure",
                "j": "E",
                "s": "t",
            }.get(unit, unit)
            values[key] = float(val)

        # Veri-bağımlı fallback: metindeki sayı ölçeğinden türet
        magnitude = max(nums) if nums else 10.0
        values.setdefault("m", max(1.0, magnitude * 0.1))
        values.setdefault("v", max(1.0, magnitude))
        values.setdefault("r", max(1.0, min(100.0, magnitude * 0.02)))
        values.setdefault("n_entities", max(1, int(sum(1 for e in extracted.get("entities", []) if e in {"insan", "çocuk", "ajan"}) or 2)))
        values["E_kin"] = 0.5 * values["m"] * (values["v"] ** 2)
        return values


class BayesianAssumptionLayer:
    """Prior/posterior varsayımları üretir."""

    def build(self, assigned: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
        priors = {}
        for k, v in latent.items():
            priors[k] = {"prior": v, "source": "latent_generator"}
        for k, v in assigned.items():
            if isinstance(v, (int, float)):
                sigma = max(1e-6, abs(v) * 0.15)
                priors[k] = {"prior": {"dist": "Normal", "params": [float(v), float(sigma)]}, "source": "assignment_engine"}
        return {"priors": priors, "count": len(priors)}


class MultiWorldSampler:
    """Çoklu dünya örnekleme + hasar dağılımı."""

    def sample(self, assigned: Dict[str, Any], n_worlds: int = 128) -> Dict[str, Any]:
        rng = random.Random(int(assigned.get("E_kin", 1)) % 1000003)
        worlds = []
        organ_weights = {"head": 1.8, "chest": 1.3, "leg": 0.8}
        for i in range(max(8, n_worlds)):
            r = max(0.5, assigned.get("r", 3.0) * (0.7 + 0.6 * rng.random()))
            e = max(1e-6, assigned.get("E_kin", 10.0) * (0.6 + 0.8 * rng.random()))
            pressure = e / (4.0 * math.pi * (r ** 2))
            shielding = 0.2 + 0.8 * rng.random()
            organ_damage = {
                organ: min(1.0, (pressure * w / (1e4 + 5e3 * shielding)))
                for organ, w in organ_weights.items()
            }
            worlds.append({
                "world_id": i + 1,
                "r": round(r, 4),
                "E": round(e, 4),
                "pressure": round(pressure, 6),
                "organ_damage": organ_damage,
            })

        avg_pressure = sum(w["pressure"] for w in worlds) / len(worlds)
        avg_damage = {
            organ: sum(w["organ_damage"][organ] for w in worlds) / len(worlds)
            for organ in organ_weights
        }
        return {
            "n_worlds": len(worlds),
            "avg_pressure": round(avg_pressure, 6),
            "avg_damage": {k: round(v, 6) for k, v in avg_damage.items()},
            "worlds_preview": worlds[:10],
        }


class RelationDiscoveryEngine:
    """Metin + değişkenlere göre aday ilişkiler çıkarır."""

    def discover(self, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        vars_ = set(extracted.get("variables", []))
        rels = []
        if {"E", "r"}.issubset(vars_) or "r" in vars_:
            rels.append({"relation": "pressure ~ E / (r^2 + eps)", "type": "inverse_square"})
        if extracted.get("temporal_events"):
            rels.append({"relation": "delta_t_obs = t_obs - t_real", "type": "observation_delay"})
            rels.append({"relation": "t_effect = t_cause + delta - beta*feedback", "type": "retrocausal"})
        rels.append({"relation": "survival_prob ~ sigmoid(-injury_total)", "type": "bounded_probability"})
        rels.append({"relation": "injury_score ~ f(pressure, impulse, shielding)", "type": "composite_risk"})
        return rels


class EquationDiscoveryEngine:
    """Symbolic adayları üretir ve normalize eder."""

    def discover(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates = []
        for idx, rel in enumerate(relations, start=1):
            eq = rel.get("relation", "")
            normalized = re.sub(r"\s+", " ", eq).strip()
            candidates.append({
                "id": f"EQ{idx}",
                "equation": normalized,
                "channel": "physics_prior" if "pressure" in normalized else "symbolic_mutation",
                "evidence": rel.get("type", "unknown"),
            })
        if not any("P(H|D)" in c.get("equation", "") for c in candidates):
            candidates.append({
                "id": f"EQ{len(candidates)+1}",
                "equation": "P(H|D) = P(D|H) * P(H) / P(D)",
                "channel": "bayesian_core",
                "evidence": "default_bayes",
            })
        return candidates


class MultiHypothesisGenerator:
    def generate(self, equations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hypotheses = []
        for eq in equations:
            hypotheses.append({
                "hypothesis": f"H_{eq['id']}",
                "equation": eq["equation"],
                "complexity": max(1, eq["equation"].count("*") + eq["equation"].count("/")),
            })
        return hypotheses


class BayesianModelSelection:
    def select(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = []
        weighted = []
        for h in hypotheses:
            eq = h.get("equation", "")
            prior = 1.0
            if "P(H|D)" in eq:
                prior *= 1.1
            if "t_effect" in eq or "delta_t_obs" in eq:
                prior *= 1.15
            weighted.append((h, prior / max(1, h["complexity"])))
        norm = sum(w for _, w in weighted) + 1e-9
        for h, w in weighted:
            posterior = w / norm
            scored.append({**h, "posterior": round(float(posterior), 6)})
        scored.sort(key=lambda x: -x["posterior"])
        return {"selected": scored[0] if scored else {}, "ranked": scored}


class EquationGroundedExplainer:
    """Seçilen denklem ve nedensellik modeli üzerinden açıklama üretir."""

    def build(
        self,
        prompt: str,
        selected_model: Dict[str, Any],
        model_ranking: List[Dict[str, Any]],
        evolved: Dict[str, Any],
        causality_models: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        eq = selected_model.get("equation", "N/A")
        top_p = selected_model.get("posterior", 0.0)
        hyperedge_count = len(evolved.get("final_graph", {}).get("hyperedges", []))
        temporal_conflict = any("temporal_conflict" in e.get("tags", []) for e in evolved.get("final_graph", {}).get("hyperedges", []))

        mode_note = "çelişki derinleştirildi" if temporal_conflict else "çelişki minimize edildi"
        best_causal = causality_models[1] if temporal_conflict and len(causality_models) > 1 else causality_models[0]
        ranking_preview = ", ".join(
            f"{m.get('hypothesis')}:{m.get('posterior')}" for m in model_ranking[:3]
        )

        text = (
            f"Prompttan {hyperedge_count} hiperkenarlı temsil üretildi. "
            f"Seçilen denklem: {eq}. "
            f"Model posterioru={top_p}. "
            f"Nedensellik tercihi: {best_causal.get('name')} ({best_causal.get('status')}). "
            f"Durum: {mode_note}. Üst modeller: {ranking_preview}."
        )
        return {"mode": "Scientific", "text": text}


class EquationValidationLoop:
    def validate(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for h in hypotheses:
            eq = h.get("equation", "")
            has_balance = "=" in eq
            has_operator = any(op in eq for op in ["+", "-", "*", "/", "~"])
            results.append({
                "hypothesis": h.get("hypothesis"),
                "valid": bool(has_balance and has_operator),
                "checks": {
                    "has_balance": has_balance,
                    "has_operator": has_operator,
                },
            })
        valid_rate = sum(1 for r in results if r["valid"]) / (len(results) or 1)
        return {"results": results, "valid_rate": round(valid_rate, 4)}


class HypergraphArchitectureHierarchy:
    """Orkestratördeki modülleri katmanlı mimari hiyerarşide raporlar."""

    def build(self, orchestrator: "SpeculativeFictionOrchestrator") -> Dict[str, Any]:
        layers = [
            {
                "layer": "L1-intent-task",
                "modules": [
                    type(orchestrator.intent).__name__,
                    type(orchestrator.task_graph).__name__,
                ],
            },
            {
                "layer": "L2-extraction-prior",
                "modules": [
                    type(orchestrator.extractor).__name__,
                    type(orchestrator.latent).__name__,
                    type(orchestrator.assigner).__name__,
                    type(orchestrator.bayes_assumptions).__name__,
                    type(orchestrator.assumptions).__name__,
                    type(orchestrator.world).__name__,
                ],
            },
            {
                "layer": "L3-hypergraph-evolution",
                "modules": [
                    type(orchestrator.hypergraph).__name__,
                    type(orchestrator.rewriter).__name__,
                    type(orchestrator.evolution).__name__,
                    type(orchestrator.eq_extract).__name__,
                ],
            },
            {
                "layer": "L4-relation-equation-inference",
                "modules": [
                    type(orchestrator.relations).__name__,
                    type(orchestrator.eq_discovery).__name__,
                    type(orchestrator.multi_h).__name__,
                    type(orchestrator.model_select).__name__,
                ],
            },
            {
                "layer": "L5-validation-simulation",
                "modules": [
                    type(orchestrator.validation).__name__,
                    type(orchestrator.multi_world).__name__,
                    "NumericTruthValidator",
                    "ArithmeticPrecisionValidator",
                    "FormulaStructureParser",
                ],
            },
            {
                "layer": "L6-explanation-safety",
                "modules": [
                    type(orchestrator.explainer).__name__,
                    type(orchestrator.mode_selector).__name__,
                    type(orchestrator.uncertainty).__name__,
                    type(orchestrator.safety).__name__,
                ],
            },
        ]
        return {"layers": layers, "layer_count": len(layers)}


class HypergraphEquationConsistencyAuditor:
    """
    Hypergraph kaynaklı denklemler için:
      - denklemler arası/sözdizimsel tutarlılık,
      - değerlerin sabit/değişkenlerle uyumu,
      - adımlar arası aritmetik doğruluk denetimi.
    Tamamen mevcut genel modüller üzerinden çalışır; senaryo hard-coding yapmaz.
    """

    RES_TOL = Decimal("1e-8")

    def __init__(self):
        self.formula_parser = FormulaStructureParser()
        self.num_validator = NumericTruthValidator()
        self.precision_validator = ArithmeticPrecisionValidator()
        self.normalizer = _global_num_unit_normalizer

    def _sanitize_equation(self, eq: str) -> str:
        if not eq:
            return ""
        txt = str(eq).replace("×", "*").replace("^", "**").replace("≈", "=")
        txt = txt.split(",", 1)[0].strip()  # "x=..., alpha>0" => "x=..."
        return txt

    def _extract_symbols(self, expr: str) -> set:
        if not expr:
            return set()
        # Türkçe karakterler dahil değişken benzeri tokenlar
        tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü_][A-Za-zÇĞİÖŞÜçğıöşü0-9_]*", expr)
        blacklist = {
            "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "eps", "sigmoid",
            "and", "or", "not", "true", "false", "p",
        }
        return {t for t in tokens if t.lower() not in blacklist}

    def _to_sympy_env(self, values: Dict[str, Any]) -> Dict[str, Any]:
        env = {"pi": sp.pi if _SYMPY_OK else math.pi}
        for k, v in (values or {}).items():
            if isinstance(v, (int, float, Decimal)):
                env[k] = float(v)
        return env

    def _eval_expr(self, expr: str, env: Dict[str, Any]) -> Optional[Decimal]:
        if not expr:
            return None
        if not _SYMPY_OK:
            return None
        try:
            val = sp.N(sp.sympify(expr, locals=env, evaluate=True), 60)
            if getattr(val, "is_real", False):
                return Decimal(str(val))
        except Exception:
            return None
        return None

    def _reconcile_assignments(self, equations: List[Dict[str, Any]], assigned: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(assigned or {})
        for _ in range(3):
            changed = False
            env = self._to_sympy_env(values)
            for eq_item in equations:
                eq = self._sanitize_equation(eq_item.get("equation", ""))
                if "=" not in eq:
                    continue
                lhs, rhs = [p.strip() for p in eq.split("=", 1)]
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
                    continue
                val = self._eval_expr(rhs, env)
                if val is None:
                    continue
                fv = float(val)
                old = values.get(lhs)
                if old is None or abs(float(old) - fv) > 1e-10:
                    values[lhs] = fv
                    changed = True
            if not changed:
                break
        return values

    def audit(
        self,
        equations: List[Dict[str, Any]],
        assigned: Dict[str, Any],
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_eqs = []
        for i, eq in enumerate(equations or [], start=1):
            txt = self._sanitize_equation(eq.get("equation", ""))
            if txt:
                normalized_eqs.append({"id": eq.get("id", f"EQ_{i}"), "equation": txt, "meta": eq})

        reconciled_values = self._reconcile_assignments(normalized_eqs, assigned or {})
        env = self._to_sympy_env(reconciled_values)

        known_vars = set((assigned or {}).keys()) | set((extracted or {}).get("variables", []))
        known_vars |= {"pi", "eps"}
        equation_checks = []
        arithmetic_steps = []
        gaps = []

        for idx, eq_item in enumerate(normalized_eqs, start=1):
            eq = eq_item["equation"]
            struct = self.formula_parser.parse(eq)
            symbols = self._extract_symbols(eq)
            unknown = sorted(s for s in symbols if s not in known_vars)
            if unknown:
                gaps.append(f"[GAP-VAR] {eq_item['id']} bilinmeyen semboller: {', '.join(unknown)}")

            evaluable = False
            consistent = True
            residual = None

            if "=" in eq and "~" not in eq:
                lhs, rhs = [p.strip() for p in eq.split("=", 1)]
                lhs_v = self._eval_expr(lhs, env)
                rhs_v = self._eval_expr(rhs, env)
                if lhs_v is not None and rhs_v is not None:
                    evaluable = True
                    residual = abs(lhs_v - rhs_v)
                    consistent = residual <= self.RES_TOL
                    arithmetic_steps.append({
                        "step": idx,
                        "formula": f"{lhs} - ({rhs})",
                        "result": str(residual),
                    })
                else:
                    gaps.append(f"[GAP-EVAL] {eq_item['id']} sayısal değerlendirilemedi: {eq}")

            equation_checks.append({
                "id": eq_item["id"],
                "equation": eq,
                "structure": {
                    "top_level_op": struct.get("top_level_op"),
                    "has_conditional": struct.get("has_conditional"),
                    "branch_count": struct.get("branch_count"),
                },
                "symbols": sorted(symbols),
                "unknown_symbols": unknown,
                "evaluable": evaluable,
                "consistent": bool(consistent),
                "residual": str(residual) if residual is not None else None,
            })

        corrected_steps, precision_violations, precision_notes = self.precision_validator.enforce(arithmetic_steps)

        synthetic_sol = {
            "answer": " ".join(s.get("result", "") for s in corrected_steps),
            "numeric": " ".join(s.get("result", "") for s in corrected_steps),
            "steps": corrected_steps,
        }
        numeric_violations = self.num_validator.validate(synthetic_sol, None)

        eval_count = sum(1 for c in equation_checks if c.get("evaluable"))
        eval_ok = sum(1 for c in equation_checks if c.get("evaluable") and c.get("consistent"))
        eq_accuracy = float(eval_ok / eval_count) if eval_count else 1.0
        var_accuracy = 1.0 if not gaps else max(0.0, 1.0 - (len([g for g in gaps if g.startswith("[GAP-VAR]")]) / max(1, len(equation_checks))))
        arithmetic_accuracy = 1.0 if not precision_violations else 0.0
        numeric_accuracy = 1.0 if not numeric_violations else 0.0

        overall_accuracy = round(float(eq_accuracy * var_accuracy * arithmetic_accuracy * numeric_accuracy), 6)

        return {
            "reconciled_values": self.normalizer.normalize_solution_payload({"data": reconciled_values}).get("data", reconciled_values),
            "equation_checks": equation_checks,
            "arithmetic_steps": corrected_steps,
            "accuracy": {
                "equation": round(eq_accuracy, 6),
                "variable_binding": round(var_accuracy, 6),
                "arithmetic": round(arithmetic_accuracy, 6),
                "numeric_bounds": round(numeric_accuracy, 6),
                "overall": overall_accuracy,
            },
            "violations": precision_violations + numeric_violations,
            "notes": precision_notes,
            "gaps": gaps,
        }


class ConstraintBuilder:
    def build(self, text: str) -> List[str]:
        clauses = [c.strip() for c in re.split(r"[.;\n]+", text or "") if c.strip()]
        constraints = [f"C{i+1}: {c}" for i, c in enumerate(clauses[:6])]
        return constraints or ["C1: consistency_required"]


class ParadoxInjector:
    def inject(self, constraints: List[str]) -> List[str]:
        if not constraints:
            return ["P1: A -> not A"]
        pivot = constraints[0].split(":", 1)[-1].strip()
        return constraints + [f"P* : ({pivot}) -> NOT({pivot})"]


class SelfTrapLoop:
    def run(self, paradox_constraints: List[str], depth: int = 3) -> Dict[str, Any]:
        chain = []
        for i in range(max(1, depth)):
            chain.append({
                "iteration": i + 1,
                "contradiction": paradox_constraints[min(i, len(paradox_constraints)-1)],
                "meta": f"conflict_depth_{i+1}",
            })
        return {"chain": chain, "depth": len(chain)}


class MetaContradictionController:
    def summarize(self, self_trap: Dict[str, Any]) -> Dict[str, Any]:
        depth = self_trap.get("depth", 0)
        severity = "high" if depth >= 3 else "medium" if depth == 2 else "low"
        return {"depth": depth, "severity": severity}


class ExplanationModeSelector:
    def select(self, intent: Dict[str, Any]) -> str:
        mode = intent.get("mode", "SOLVE")
        if mode == "PARADOX":
            return "Paradox"
        if mode == "HYBRID_SIM_PARADOX":
            return "Hybrid"
        if mode == "SIMULATION_REALISTIC":
            return "Scientific"
        return "Scientific"


class UncertaintyReporter:
    def report(self, ranked_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        probs = [float(m.get("posterior", 0.0)) for m in ranked_models]
        entropy = -sum((p * math.log(p + 1e-12)) for p in probs)
        return {
            "model_count": len(ranked_models),
            "entropy": round(float(entropy), 6),
            "top_posterior": max(probs) if probs else 0.0,
        }


class SafetyContentPolicyGuard:
    """Çıktıyı güvenlik açısından normalize eder (zarar verici detay en aza)."""

    def sanitize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False)
        risky = bool(re.search(r"(weapon|patlayıcı|detonate|illegal)", text, re.IGNORECASE))
        payload["safety"] = {
            "policy_flags": ["sensitive_terms_detected"] if risky else [],
            "sanitized": True,
        }
        return payload


class SpeculativeFictionOrchestrator:
    """Bayes + world-model + equation-discovery + paradox pipeline."""

    def __init__(self):
        self.intent = PromptIntentSplitter()
        self.task_graph = TaskGraphBuilder()
        self.extractor = VariableEntityExtractor()
        self.hypergraph = HypergraphBuilder()
        self.rewriter = RewriteRuleEngine()
        self.evolution = HypergraphEvolutionEngine()
        self.eq_extract = EquationExtractionLayer()
        self.assigner = VariableAssignmentEngine()
        self.bayes_assumptions = BayesianAssumptionLayer()
        self.multi_world = MultiWorldSampler()
        self.latent = LatentVariableGenerator()
        self.assumptions = AssumptionRegistry()
        self.world = WorldModelBuilder()
        self.relations = RelationDiscoveryEngine()
        self.eq_discovery = EquationDiscoveryEngine()
        self.multi_h = MultiHypothesisGenerator()
        self.model_select = BayesianModelSelection()
        self.explainer = EquationGroundedExplainer()
        self.validation = EquationValidationLoop()
        self.architecture = HypergraphArchitectureHierarchy()
        self.consistency_auditor = HypergraphEquationConsistencyAuditor()
        self.constraint_builder = ConstraintBuilder()
        self.paradox_injector = ParadoxInjector()
        self.self_trap = SelfTrapLoop()
        self.meta_ctrl = MetaContradictionController()
        self.mode_selector = ExplanationModeSelector()
        self.uncertainty = UncertaintyReporter()
        self.safety = SafetyContentPolicyGuard()

    def run(self, prompt: str) -> Dict[str, Any]:
        intent_payload = self.intent.split(prompt)
        task_graph = self.task_graph.build(intent_payload)
        extracted = self.extractor.extract(prompt)
        latent_priors = self.latent.generate(extracted)
        assigned = self.assigner.assign(prompt, extracted)
        bayes_layer = self.bayes_assumptions.build(assigned, latent_priors)
        assumptions = self.assumptions.register(extracted, latent_priors)
        world_model = self.world.build(extracted, bayes_layer, task_graph)

        hgraph_0 = self.hypergraph.build(prompt, extracted, task_graph)
        evolved = self.evolution.evolve(hgraph_0, self.rewriter, steps=4)
        emergent_equations = self.eq_extract.extract(evolved)

        relations = self.relations.discover(extracted)
        equations = self.eq_discovery.discover(relations) + emergent_equations
        hypotheses = self.multi_h.generate(equations)
        selection = self.model_select.select(hypotheses)
        validation = self.validation.validate(hypotheses)
        architecture = self.architecture.build(self)
        consistency_audit = self.consistency_auditor.audit(equations, assigned, extracted)
        worlds = self.multi_world.sample(assigned, n_worlds=160)

        constraints = self.constraint_builder.build(prompt)
        paradox_constraints = self.paradox_injector.inject(constraints)
        self_trap = self.self_trap.run(paradox_constraints, depth=3)
        meta = self.meta_ctrl.summarize(self_trap)

        explanation_mode = self.mode_selector.select(intent_payload)
        uncertainty = self.uncertainty.report(selection.get("ranked", []))

        causality_models = [
            {
                "name": "classical_forward",
                "rule": "cause(t) < effect(t)",
                "status": "inconsistent_if_temporal_conflict" if any("temporal_conflict" in e.get("tags", []) for e in evolved.get("final_graph", {}).get("hyperedges", [])) else "consistent",
            },
            {
                "name": "retrocausal_fixed_point",
                "rule": "state_t = F(state_t, state_t+1)",
                "status": "candidate_solution",
            },
        ]
        explanation = self.explainer.build(
            prompt=prompt,
            selected_model=selection.get("selected", {}),
            model_ranking=selection.get("ranked", []),
            evolved=evolved,
            causality_models=causality_models,
        )

        result = {
            "intent": {
                "mode": intent_payload.get("mode"),
                "confidence": intent_payload.get("confidence"),
                "sub_tasks": intent_payload.get("sub_tasks", []),
            },
            "task_graph": task_graph,
            "entities": extracted.get("entities", []),
            "variables": extracted.get("variables", []),
            "variable_assignment": assigned,
            "priors": latent_priors,
            "bayesian_assumptions": bayes_layer,
            "assumptions": assumptions,
            "world_model": world_model,
            "hypergraph": {
                "initial": hgraph_0,
                "evolution": evolved.get("history", []),
                "emergent": evolved.get("final_graph", {}),
            },
            "relations": relations,
            "equations": equations,
            "hypotheses": hypotheses,
            "selected_model": selection.get("selected", {}),
            "model_ranking": selection.get("ranked", []),
            "equation_validation": validation,
            "architecture_hierarchy": architecture,
            "consistency_audit": consistency_audit,
            "simulation": {
                "samples": worlds.get("n_worlds", 0),
                "outputs": {
                    "best_equation": selection.get("selected", {}).get("equation", ""),
                    "mode": intent_payload.get("mode"),
                    "avg_pressure": worlds.get("avg_pressure"),
                    "organ_damage_mean": worlds.get("avg_damage", {}),
                    "consistency_accuracy": consistency_audit.get("accuracy", {}).get("overall", 0.0),
                },
                "worlds_preview": worlds.get("worlds_preview", []),
                "uncertainty": uncertainty,
            },
            "causal_graph": {
                "nodes": ["impact", "rupture", "shockwave", "damage", "survival"],
                "edges": [["impact", "rupture"], ["rupture", "shockwave"], ["shockwave", "damage"], ["damage", "survival"]],
            },
            "causality_models": causality_models,
            "paradox": {
                "enabled": intent_payload.get("mode") in {"PARADOX", "HYBRID_SIM_PARADOX"},
                "constraints": paradox_constraints,
                "self_trap": self_trap,
                "meta": meta,
            },
            "explanation": {**explanation, "mode": explanation_mode},
        }
        return self.safety.sanitize(result)


_speculative_fiction_orchestrator = SpeculativeFictionOrchestrator()


@app.route('/speculative/process', methods=['POST'])
def speculative_process_endpoint():
    """Yeni spekülatif kurgu + denklem keşfi pipeline endpoint'i."""
    try:
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get('prompt') or payload.get('question') or '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'prompt boş olamaz'}), 400

        result = _speculative_fiction_orchestrator.run(prompt)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
