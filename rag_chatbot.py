"""
rag_chatbot.py — DNEXT Intelligence SA · Advanced RAG Chatbot
═══════════════════════════════════════════════════════════════
Per-step latency timing visible in the debug panel:
  - Query embedding time
  - Dense retrieval time
  - Sparse retrieval time
  - RRF fusion + dedup + compression time
  - Context building time
  - LLM generation time (first token + total)
  - Total pipeline time
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
import psycopg2.pool
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_URL              = os.environ.get("DATABASE_URL", "")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
NVIDIA_NIM_API_KEY  = os.environ.get("NVIDIA_NIM_API_KEY", "")
NVIDIA_NIM_ENDPOINT = os.environ.get("NVIDIA_NIM_ENDPOINT", "")
LOCAL_MODEL_PATH    = os.environ.get("LOCAL_MODEL_PATH", "")

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best quality · 128k context",
    "mixtral-8x7b-32768":      "Efficient · 32k context",
    "llama3-8b-8192":          "Fastest · 8k context",
}
NVIDIA_MODELS = {
    "meta/llama-3.1-70b-instruct": "High quality · 128k context",
}

DEFAULT_TOP_K           = 6
RRF_K                   = 60
MAX_FINAL_DOCS          = 5
MAX_CONTEXT_CHARS       = 1000
CONFIDENCE_THRESHOLD    = 0.20
COMPRESSION_SENTENCES   = 3
SIMILARITY_DEDUP_THRESH = 0.92
EMBED_MODEL             = "sentence-transformers/all-MiniLM-L6-v2"

COMMODITY_SYNONYMS: dict[str, list[str]] = {
    "wheat":  ["winter wheat", "spring wheat", "durum", "grain"],
    "corn":   ["maize", "yellow corn", "coarse grain"],
    "soy":    ["soybean", "soybeans", "soya", "oilseed"],
    "canola": ["rapeseed", "oilseed rape"],
    "palm":   ["palm oil", "CPO"],
    "sugar":  ["raw sugar", "white sugar"],
    "coffee": ["arabica", "robusta"],
    "rice":   ["paddy", "milled rice"],
    "barley": ["feed barley", "malting barley"],
}

_TEMPORAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blast\s+week\b",           re.I), "last_week"),
    (re.compile(r"\blast\s+month\b",          re.I), "last_month"),
    (re.compile(r"\blast\s+(\d+)\s+days?\b",  re.I), "last_n_days"),
    (re.compile(r"\blast\s+(\d+)\s+months?\b",re.I), "last_n_months"),
    (re.compile(r"\blast\s+year\b",           re.I), "last_year"),
    (re.compile(r"\bsince\s+(\w+)\b",         re.I), "since_month"),
    (re.compile(r"\bin\s+(20\d{2})\b",        re.I), "in_year"),
    (re.compile(r"\b(Q[1-4])\s+(20\d{2})\b", re.I), "quarter_year"),
    (re.compile(r"\brecent(?:ly)?\b",         re.I), "recent"),
    (re.compile(r"\blatest\b",                re.I), "recent"),
    (re.compile(r"\btoday\b",                 re.I), "today"),
    (re.compile(r"\bthis\s+week\b",           re.I), "this_week"),
    (re.compile(r"\bthis\s+month\b",          re.I), "this_month"),
    (re.compile(r"\bthis\s+year\b",           re.I), "this_year"),
]
_MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}

SYSTEM_PROMPT = """You are an expert agricultural commodity market analyst assistant for DNEXT Intelligence SA.
Answer questions about commodity markets, crop conditions, weather impacts, trade flows, and price movements
using ONLY the provided article excerpts as your source of truth.

If the user asks something unrelated to agricultural commodities, politely decline.

Rules:
- Ground every claim in the provided articles. Never hallucinate facts or prices.
- If articles lack sufficient information, say so explicitly.
- Always cite the source name and publication date for every factual claim.
- Be concise and professional — your audience is commodity analysts and traders.
- When discussing sentiment, reference the FinBERT scores shown in the context.
- If multiple articles contradict each other, flag the discrepancy explicitly."""

SUGGESTED_QUESTIONS = [
    "What is the current wheat outlook in Ukraine?",
    "How has corn sentiment changed over the last 3 months?",
    "What are the latest USDA reports saying about soybean supply?",
    "Which regions have the most negative commodity sentiment?",
    "What weather events are impacting crop production?",
    "Compare barley and wheat price trends in 2024",
]


# ─────────────────────────────────────────────────────────────────────────────
# TIMING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def ms(t_start: float) -> int:
    return round((time.perf_counter() - t_start) * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    [data-testid="stSidebar"] { background: #194B64 !important; }
    [data-testid="stSidebar"] * { color: #cde8f0 !important; }
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #7DC8C8 !important; font-size: 0.68rem !important;
        text-transform: uppercase; letter-spacing: 0.09em;
    }
    [data-testid="stSidebar"] hr { border-color: #27607d !important; }
    .main .block-container {
        background: #eef3f7; padding: 1rem 1.5rem 2rem; max-width: 100% !important;
    }
    .chat-header {
        background: white; border-radius: 12px 12px 0 0;
        padding: 1rem 1.4rem 0.8rem; border-bottom: 1px solid #e8eff4; margin-bottom: 0;
    }
    .chat-header-title { font-family: 'DM Serif Display',serif; font-size:1.05rem; color:#194B64; }
    .chat-header-sub   { font-size:0.7rem; color:#7a9fb0; margin-top:0.1rem; }

    .source-card {
        background:#f4f8fb; border-radius:8px; padding:0.6rem 0.85rem;
        margin-bottom:0.4rem; border-left:3px solid transparent; font-size:0.78rem;
    }
    .source-card.positive { border-left-color:#1a5a3a; }
    .source-card.negative { border-left-color:#c0392b; }
    .source-card.neutral  { border-left-color:#8a8a5a; }
    .source-title { font-weight:500; color:#194B64; margin-bottom:0.2rem; line-height:1.3; }
    .source-meta  { display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center; }
    .sent-pill    { display:inline-block; border-radius:10px; padding:1px 8px; font-size:0.65rem; font-weight:500; }
    .sent-positive { background:#e2f5ec; color:#1a5a3a; }
    .sent-negative { background:#fce8e8; color:#8b2020; }
    .sent-neutral  { background:#f0f0e8; color:#5a5a3a; }
    .retriever-pill { display:inline-block; border-radius:10px; padding:1px 8px; font-size:0.65rem; background:#ddeaf8; color:#1a3a7a; }
    .chunk-pill     { display:inline-block; border-radius:10px; padding:1px 8px; font-size:0.65rem; background:#f0e8f8; color:#4a1a7a; }
    .sim-bar-wrap { flex:1; min-width:60px; background:#dde8f0; border-radius:10px; height:4px; margin-top:2px; }
    .sim-bar { height:4px; border-radius:10px; background:#197DAF; }

    .hint-row {
        font-size:0.7rem; color:#7a9fb0; background:#f4f8fb;
        border-radius:6px; padding:4px 10px; margin-bottom:0.4rem; display:inline-block;
    }
    .msg-meta { font-size:0.64rem; color:#a0b8c4; margin-top:0.3rem; display:flex; gap:0.5rem; }
    .meta-badge { background:#eef3f7; border-radius:10px; padding:1px 7px; font-size:0.63rem; color:#7a9fb0; }

    .debug-section { margin-bottom: 0.6rem; }
    .debug-section-title {
        font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: #7DC8C8; font-weight: 600; margin-bottom: 0.25rem;
        border-bottom: 1px solid #dde8f0; padding-bottom: 0.15rem;
    }
    .debug-table { font-size:0.72rem; font-family:monospace; color:#4a6a7a; line-height:2; }
    .debug-key   { color:#7DC8C8; font-weight:500; display:inline-block; min-width:190px; }
    .debug-value { color:#2a4a5a; }
    .t-bar-wrap  { display:inline-block; width:80px; background:#dde8f0;
                   border-radius:4px; height:6px; vertical-align:middle; margin:0 6px; }
    .t-bar       { height:6px; border-radius:4px; }
    .t-bar.fast  { background:#1a5a3a; }
    .t-bar.mid   { background:#197DAF; }
    .t-bar.slow  { background:#c0392b; }
    .t-ms        { font-size:0.7rem; color:#7a9fb0; }

    .empty-state { text-align:center; padding:2.5rem 1rem; color:#7a9fb0; }
    .empty-state-icon  { font-size:2.5rem; margin-bottom:0.6rem; opacity:0.5; }
    .empty-state-title { font-family:'DM Serif Display',serif; font-size:1.1rem; color:#194B64; margin-bottom:0.3rem; }

    [data-testid="stChatInput"] textarea {
        font-family:'DM Sans',sans-serif !important;
        font-size:0.875rem !important; border-radius:10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "chat_history":  [],
        "retrieval_log": [],
        "debug_log":     [],
        "latency_log":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION POOL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pool(db_url: str) -> psycopg2.pool.ThreadedConnectionPool:
    return psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=db_url)

def get_conn(db_url: str):
    return get_pool(db_url).getconn()

def release_conn(db_url: str, conn):
    get_pool(db_url).putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_corpus_stats(db_url: str) -> tuple[int, int]:
    conn = None
    try:
        conn = get_conn(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM enriched_articles WHERE embedding IS NOT NULL;")
            n_emb = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM enriched_articles;")
            n_tot = cur.fetchone()[0]
        return n_emb, n_tot
    except Exception:
        return -1, 0
    finally:
        if conn: release_conn(db_url, conn)

@st.cache_data(ttl=300, show_spinner=False)
def load_sources(db_url: str) -> list[str]:
    conn = None
    try:
        conn = get_conn(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT source FROM enriched_articles "
                        "WHERE embedding IS NOT NULL ORDER BY source;")
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return ["ADMISI", "Brownfield", "Mecardo", "USDA", "AHDB"]
    finally:
        if conn: release_conn(db_url, conn)

@st.cache_data(ttl=300, show_spinner=False)
def chunks_table_exists(db_url: str) -> bool:
    conn = None
    try:
        conn = get_conn(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'article_chunks'
                );
            """)
            return cur.fetchone()[0]
    except Exception:
        return False
    finally:
        if conn: release_conn(db_url, conn)


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data(ttl=3600, show_spinner=False)
def embed_text(text: str) -> list[float]:
    return load_embedder().encode(text, normalize_embeddings=True).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_temporal(query: str) -> tuple[Optional[str], Optional[str]]:
    today = datetime.utcnow().date()
    for pattern, tag in _TEMPORAL_PATTERNS:
        m = pattern.search(query)
        if not m: continue
        if tag == "today":
            return today.isoformat(), today.isoformat()
        if tag == "this_week":
            return (today - timedelta(days=today.weekday())).isoformat(), today.isoformat()
        if tag == "this_month":
            return today.replace(day=1).isoformat(), today.isoformat()
        if tag == "this_year":
            return today.replace(month=1, day=1).isoformat(), today.isoformat()
        if tag == "last_week":
            end = today - timedelta(days=today.weekday() + 1)
            return (end - timedelta(days=6)).isoformat(), end.isoformat()
        if tag == "last_month":
            first = today.replace(day=1)
            last  = first - timedelta(days=1)
            return last.replace(day=1).isoformat(), last.isoformat()
        if tag == "last_year":
            y = today.year - 1; return f"{y}-01-01", f"{y}-12-31"
        if tag == "recent":
            return (today - timedelta(days=30)).isoformat(), None
        if tag == "last_n_days":
            return (today - timedelta(days=int(m.group(1)))).isoformat(), None
        if tag == "last_n_months":
            return (today - timedelta(days=int(m.group(1))*30)).isoformat(), None
        if tag == "in_year":
            y = m.group(1); return f"{y}-01-01", f"{y}-12-31"
        if tag == "since_month":
            mn = _MONTH_MAP.get(m.group(1).lower())
            if mn:
                yr = today.year if mn <= today.month else today.year - 1
                return f"{yr}-{mn:02d}-01", None
        if tag == "quarter_year":
            q = int(m.group(1)[1]); y = m.group(2)
            ms_n = (q-1)*3+1; me = q*3
            import calendar
            ld = calendar.monthrange(int(y), me)[1]
            return f"{y}-{ms_n:02d}-01", f"{y}-{me:02d}-{ld:02d}"
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REWRITER
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    cleaned = query
    for pattern, _ in _TEMPORAL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    for canonical, synonyms in COMMODITY_SYNONYMS.items():
        if canonical in cleaned.lower() or any(s in cleaned.lower() for s in synonyms):
            additions = [s for s in synonyms if s not in cleaned.lower()][:3]
            if additions:
                cleaned += " " + " ".join(additions)
            break
    return cleaned.strip() or query


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE QUERY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_sparse_query(query: str) -> str:
    """
    Extract meaningful content words and build an OR-based tsquery with prefix
    matching. OR logic ensures partial matches surface; ts_rank handles sorting.
    The :* suffix enables prefix matching (e.g. "wheat:*" matches "wheaten").
    """
    stop = {
        "what","is","the","are","how","has","have","been","will","would",
        "could","should","tell","me","about","give","show","latest","recent",
        "current","outlook","situation","update","news","report","data",
        "trend","trends","compare","comparison","between","and","or","in",
        "on","at","of","for","a","an","to","from","with","over","last",
        "this","that","their","there","which","when","where","why","any",
        "get","use","using","used","into","out","its","our","was","were",
    }
    tokens = [
        t for t in re.sub(r"[^\w\s]", " ", query.lower()).split()
        if t not in stop and len(t) > 2
    ]
    if not tokens:
        return ""
    return " | ".join(f"{t}:*" for t in tokens[:6])


# ─────────────────────────────────────────────────────────────────────────────
# DENSE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _dense_retrieve(query, db_url, top_k, source_filter, date_from, date_to,
                    use_chunks: bool = False) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    vec_str = "[" + ",".join(str(x) for x in embed_text(query)) + "]"
    filters: list[str] = []; params: list = []

    if use_chunks:
        filters = ["c.embedding IS NOT NULL"]
        if source_filter:
            filters.append(f"a.source IN ({','.join(['%s']*len(source_filter))})"); params.extend(source_filter)
        if date_from:
            filters.append("a.published_date >= %s"); params.append(date_from)
        if date_to:
            filters.append("a.published_date <= %s"); params.append(date_to)
        where = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT DISTINCT ON (a.id)
                a.id, a.title, a.source, a.published_date, a.url,
                COALESCE(a.summary,LEFT(a.content,{MAX_CONTEXT_CHARS})) AS summary,
                LEFT(a.content,800) AS content,
                COALESCE(a.overall_sentiment,'neutral') AS overall_sentiment,
                COALESCE(a.sentiment_score,0.0) AS sentiment_score,
                a.commodities, c.chunk_text AS matched_chunk, c.chunk_index,
                1-(c.embedding <=> %s::vector) AS similarity
            FROM article_chunks c JOIN enriched_articles a ON c.article_id=a.id
            {where} ORDER BY a.id, c.embedding <=> %s::vector LIMIT %s;
        """
    else:
        filters = ["embedding IS NOT NULL"]
        if source_filter:
            filters.append(f"source IN ({','.join(['%s']*len(source_filter))})"); params.extend(source_filter)
        if date_from:
            filters.append("published_date >= %s"); params.append(date_from)
        if date_to:
            filters.append("published_date <= %s"); params.append(date_to)
        where = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT id, title, source, published_date, url,
                   COALESCE(summary,LEFT(content,{MAX_CONTEXT_CHARS})) AS summary,
                   LEFT(content,800) AS content,
                   COALESCE(overall_sentiment,'neutral') AS overall_sentiment,
                   COALESCE(sentiment_score,0.0) AS sentiment_score,
                   commodities, NULL AS matched_chunk, NULL AS chunk_index,
                   1-(embedding <=> %s::vector) AS similarity
            FROM enriched_articles {where}
            ORDER BY embedding <=> %s::vector LIMIT %s;
        """

    conn = None
    try:
        conn = get_conn(db_url)
        with conn.cursor() as cur:
            cur.execute(sql, [vec_str] + params + [vec_str, top_k])
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in rows: r["_retriever"] = "dense"
        return rows, ms(t0)
    except Exception:
        return [], ms(t0)
    finally:
        if conn: release_conn(db_url, conn)


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_retrieve(query, db_url, top_k, source_filter, date_from, date_to) -> tuple[list[dict], int]:
    t0 = time.perf_counter()

    tsquery_str = _build_sparse_query(query)
    if not tsquery_str:
        return [], ms(t0)

    conditions = []
    params = []
    if source_filter:
        conditions.append(f"source IN ({','.join(['%s'] * len(source_filter))})")
        params.extend(source_filter)
    if date_from:
        conditions.append("published_date >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("published_date <= %s")
        params.append(date_to)

    filter_clause = ("AND " + " AND ".join(conditions)) if conditions else ""

    # Uses idx_enriched_search_gin (stored search_tsv column) — no recomputation overhead
    sql = f"""
        SELECT id, title, source, published_date, url,
               COALESCE(summary, LEFT(content, {MAX_CONTEXT_CHARS})) AS summary,
               LEFT(content, 800) AS content,
               COALESCE(overall_sentiment, 'neutral') AS overall_sentiment,
               COALESCE(sentiment_score, 0.0) AS sentiment_score,
               commodities,
               ts_rank(search_tsv, to_tsquery('english', %s)) AS similarity
        FROM enriched_articles
        WHERE search_tsv @@ to_tsquery('english', %s)
        {filter_clause}
        ORDER BY similarity DESC
        LIMIT %s;
    """
    # params: [tsquery for rank, tsquery for WHERE, ...filters, limit]
    params_full = [tsquery_str, tsquery_str] + params + [top_k]

    conn = None
    try:
        conn = get_conn(db_url)
        with conn.cursor() as cur:
            cur.execute(sql, params_full)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in rows:
            r["_retriever"] = "sparse"
        return rows, ms(t0)
    except Exception as e:
        print(f"Sparse search error: {e}")
        return [], ms(t0)
    finally:
        if conn:
            release_conn(db_url, conn)


# ─────────────────────────────────────────────────────────────────────────────
# RRF FUSION
# Equal weights (2.0/2.0): articles confirmed by BOTH retrievers float to top.
# Raise dense_weight if sparse quality degrades; raise sparse_weight to reward
# exact keyword matches more.
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fusion(dense, sparse, k=RRF_K, top_n=MAX_FINAL_DOCS,
               dense_weight=2.0, sparse_weight=2.0) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    scores: dict[str, float] = {}; docs: dict[str, dict] = {}
    for rank, d in enumerate(dense, 1):
        did = str(d["id"])
        scores[did] = scores.get(did, 0) + dense_weight / (k + rank)
        docs[did] = d
    for rank, d in enumerate(sparse, 1):
        did = str(d["id"])
        scores[did] = scores.get(did, 0) + sparse_weight / (k + rank)
        if did not in docs: docs[did] = d
        else: docs[did]["_retriever"] = "hybrid"
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for did in ranked[:top_n]:
        d = docs[did].copy(); d["_rrf_score"] = scores[did]; result.append(d)
    return result, ms(t0)


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP + COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(articles: list[dict]) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    def ws(t): return set(re.sub(r"[^\w\s]", "", str(t).lower()).split())
    kept = []
    for c in articles:
        tc = ws(c.get("title", ""))
        if not any(
            len(tc & ws(e.get("title", ""))) / max(len(tc | ws(e.get("title", ""))), 1)
            >= SIMILARITY_DEDUP_THRESH for e in kept
        ):
            kept.append(c)
    return kept, ms(t0)


def compress_articles(articles: list[dict], query: str) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    result = []
    for article in articles:
        content = str(article.get("content", "") or article.get("summary", "") or "")
        if not content or content == "nan" or len(content) <= MAX_CONTEXT_CHARS:
            result.append(article)
            continue
        try:
            import nltk
            try:
                sents = nltk.sent_tokenize(content)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                sents = nltk.sent_tokenize(content)
        except ImportError:
            sents = [s.strip() for s in content.split(".") if s.strip()]

        if len(sents) <= COMPRESSION_SENTENCES:
            result.append(article)
            continue

        qterms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        def score(s):
            st_ = set(re.sub(r"[^\w\s]", "", s.lower()).split())
            return len(qterms & st_) * min(len(st_) / 10.0, 1.0)

        top = sorted(
            sorted(enumerate(sents), key=lambda x: score(x[1]), reverse=True)[:COMPRESSION_SENTENCES]
        )
        r = article.copy()
        r["summary"]     = " … ".join(sents[i] for i, _ in top)
        r["_compressed"] = True
        result.append(r)
    return result, ms(t0)


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_context(articles: list[dict]) -> tuple[str, int]:
    t0 = time.perf_counter()
    blocks = []
    for i, a in enumerate(articles, 1):
        date_str = str(a.get("published_date", ""))[:10]
        sent     = a.get("overall_sentiment", "neutral").upper()
        score    = float(a.get("sentiment_score", 0.0) or 0.0)
        summary  = str(a.get("summary", "") or "").strip()
        if len(summary) > MAX_CONTEXT_CHARS:
            summary = summary[:MAX_CONTEXT_CHARS] + "…"
        comm_tags = ""
        raw_comms = a.get("commodities")
        if raw_comms and str(raw_comms) not in ("", "null", "None"):
            try:
                comms = raw_comms if isinstance(raw_comms, list) else json.loads(raw_comms)
                comm_tags = " | ".join(
                    f"{c['name']}: {c.get('sentiment','?')} ({c.get('score',0):.2f})"
                    for c in comms[:4]
                )
            except Exception:
                pass
        matched = str(a.get("matched_chunk", "") or "").strip()
        chunk_section = f"Matched passage: {matched[:300]}\n" if matched and matched not in summary else ""
        blocks.append(
            f"[Article {i}]\n"
            f"Source: {a.get('source','')} | Date: {date_str} | "
            f"Sentiment: {sent} ({score:.2f}) | Via: {a.get('_retriever','dense')}\n"
            + (f"Commodities: {comm_tags}\n" if comm_tags else "")
            + f"URL: {a.get('url','')}\n"
            + chunk_section
            + f"Summary: {summary}"
        )
    return "\n\n---\n\n".join(blocks), ms(t0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRIEVE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(question, db_url, top_k=DEFAULT_TOP_K,
             source_filter=None, date_from=None, date_to=None,
             use_chunks=False) -> tuple[list[dict], str, dict]:
    pipeline_start = time.perf_counter()
    debug: dict    = {"timing_ms": {}}
    T = debug["timing_ms"]

    # ── Step 1: Temporal parsing ──────────────────────────────────────────────
    t0 = time.perf_counter()
    auto_from, auto_to = parse_temporal(question)
    eff_from = date_from or auto_from
    eff_to   = date_to   or auto_to
    T["1_temporal_parse"] = ms(t0)
    debug["temporal"] = {
        "auto_from": auto_from, "auto_to": auto_to,
        "effective_from": eff_from, "effective_to": eff_to,
    }

    # ── Step 2: Query rewriting ───────────────────────────────────────────────
    t0 = time.perf_counter()
    rewritten = rewrite_query(question)
    T["2_query_rewrite"] = ms(t0)
    debug["rewritten_query"] = rewritten

    # ── Step 3: Embedding ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    _vec = embed_text(rewritten)
    T["3_embedding"] = ms(t0)
    debug["embedding_cached"] = T["3_embedding"] < 10

    # ── Step 4: Dense + sparse retrieval (sequential — faster on local DB) ────
    # ThreadPoolExecutor removed: local PostgreSQL has ~0ms network latency so
    # thread overhead costs more than the parallelism saves.
    t0 = time.perf_counter()

    dense_r,  dense_ms  = _dense_retrieve(rewritten, db_url, top_k,
                                           source_filter, eff_from, eff_to, use_chunks)
    sparse_r, sparse_ms = _sparse_retrieve(rewritten, db_url, top_k,
                                            source_filter, eff_from, eff_to)

    T["4a_dense_retrieval"]  = dense_ms
    T["4b_sparse_retrieval"] = sparse_ms
    T["4_retrieval_wall"]    = ms(t0)
    debug["dense_count"]     = len(dense_r)
    debug["sparse_count"]    = len(sparse_r)
    debug["used_chunks"]     = use_chunks
    debug["sparse_query"]    = _build_sparse_query(rewritten)  # visible in debug panel

    # Confidence gate on dense similarity
    top_sim = max((float(r.get("similarity", 0) or 0) for r in dense_r), default=0.0)
    debug["top_similarity"] = round(top_sim, 4)
    debug["confidence_ok"]  = top_sim >= CONFIDENCE_THRESHOLD

    if not dense_r and not sparse_r:
        T["TOTAL_RETRIEVAL"] = ms(pipeline_start)
        return [], "", debug

    # ── Step 5: RRF fusion ────────────────────────────────────────────────────
    fused, T["5_rrf_fusion"] = rrf_fusion(dense_r, sparse_r)
    debug["fusion_count"]    = len(fused)

    # ── Step 6: Deduplication ─────────────────────────────────────────────────
    deduped, T["6_deduplication"] = deduplicate(fused)
    debug["dedup_count"]          = len(deduped)

    # ── Step 7: Contextual compression ────────────────────────────────────────
    compressed, T["7_compression"] = compress_articles(deduped, question)
    debug["compressed_count"]      = sum(1 for a in compressed if a.get("_compressed"))

    # ── Step 8: Context building ──────────────────────────────────────────────
    context, T["8_context_build"] = build_context(compressed)

    T["TOTAL_RETRIEVAL"] = ms(pipeline_start)
    return compressed, context, debug


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def stream_answer(question, context, history, placeholder,
                  backend, model, temperature, max_tokens) -> tuple[str, float, dict]:
    timing: dict = {}
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Articles:\n\n{context}\n\n---\n\nQuestion: {question}"},
    ]

    if backend == "Groq" and GROQ_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
            t_api_start = time.perf_counter()
            stream = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature, stream=True,
            )
            full = ""; first_token_recorded = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta and not first_token_recorded:
                    timing["llm_time_to_first_token_ms"] = ms(t_api_start)
                    first_token_recorded = True
                full += delta
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)
            timing["llm_total_generation_ms"] = ms(t_api_start)
            timing["llm_tokens_approx"]       = len(full.split())
            timing["llm_backend"]             = f"Groq / {model}"
            return full, time.perf_counter() - t_api_start, timing
        except Exception as e:
            st.warning(f"Groq error: {e}")

    if backend == "NVIDIA NIM" and NVIDIA_NIM_API_KEY and NVIDIA_NIM_ENDPOINT:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=NVIDIA_NIM_API_KEY, base_url=NVIDIA_NIM_ENDPOINT)
            t_api_start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            ans = resp.choices[0].message.content
            placeholder.markdown(ans)
            timing["llm_total_generation_ms"] = ms(t_api_start)
            timing["llm_backend"]             = f"NIM / {model}"
            return ans, time.perf_counter() - t_api_start, timing
        except Exception as e:
            st.warning(f"NIM error: {e}")

    if backend == "Local" and LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=LOCAL_MODEL_PATH, n_ctx=4096, n_threads=4, verbose=False)
            hist_t = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
            t_api_start = time.perf_counter()
            out = llm(
                f"SYSTEM: {SYSTEM_PROMPT}\n\n{hist_t}\n\nCONTEXT:\n{context}\n\nUSER: {question}\nASSISTANT:",
                max_tokens=max_tokens, temperature=temperature, stop=["USER:"],
            )
            ans = out["choices"][0]["text"].strip()
            placeholder.markdown(ans)
            timing["llm_total_generation_ms"] = ms(t_api_start)
            timing["llm_backend"]             = f"Local / {os.path.basename(LOCAL_MODEL_PATH)}"
            return ans, time.perf_counter() - t_api_start, timing
        except Exception as e:
            ans = f"Local model error: {e}"; placeholder.markdown(ans)
            return ans, 0.0, timing

    ans = "⚠️ No LLM configured. Set `GROQ_API_KEY` in your `.env` file."
    placeholder.markdown(ans)
    return ans, 0.0, timing


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_source_cards(articles):
    for j, a in enumerate(articles, 1):
        sim   = float(a.get("similarity", 0) or a.get("_rrf_score", 0))
        date  = str(a.get("published_date", ""))[:10]
        sent  = a.get("overall_sentiment", "neutral")
        ret   = a.get("_retriever", "dense")
        url   = a.get("url", "")
        title = str(a.get("title", ""))[:90]
        link  = (f'<a href="{url}" target="_blank" style="color:#197DAF;text-decoration:none">'
                 f'{title} ↗</a>') if url and url != "nan" else title
        ret_label   = {"dense":"🔵 vector","sparse":"🟡 keyword","hybrid":"🟢 hybrid"}.get(ret, ret)
        chunk_badge = (f'<span class="chunk-pill">chunk {a.get("chunk_index","")+1}</span>'
                       if a.get("matched_chunk") else "")
        # Build full HTML outside the f-string — no conditionals inside markdown
        compr_badge = '<span style="font-size:0.7rem;color:#a0b8c4"> · ✂️ compressed</span>' if a.get("_compressed") else ""

        card_html = (
            f'<div class="source-card {sent}">'
            f'<div class="source-title">{link}</div>'
            f'<div class="source-meta">'
            f'<span style="font-size:0.7rem;color:#7a9fb0">{a.get("source","")} · {date}</span>'
            f'<span class="sent-pill sent-{sent}">{sent}</span>'
            f'<span class="retriever-pill">{ret_label}</span>'
            f'{chunk_badge}'
            f'{compr_badge}'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:0.4rem;margin-top:0.3rem">'
            f'<div class="sim-bar-wrap">'
            f'<div class="sim-bar" style="width:{min(sim*100,100):.0f}%"></div>'
            f'</div>'
            f'<span style="font-size:0.65rem;color:#a0b8c4;white-space:nowrap">{sim:.3f}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


def _timing_bar_html(value_ms: int, max_ms: int = 2000) -> str:
    pct = min(value_ms / max_ms * 100, 100)
    cls = "fast" if value_ms < 100 else ("mid" if value_ms < 500 else "slow")
    return (
        f'<div class="t-bar-wrap"><div class="t-bar {cls}" style="width:{pct:.0f}%"></div></div>'
        f'<span class="t-ms">{value_ms} ms</span>'
    )


def render_debug(debug: dict):
    T   = debug.get("timing_ms", {})
    t   = debug.get("temporal", {})
    llm = debug.get("llm_timing", {})

    def row(label, value, timing_key=None):
        bar = _timing_bar_html(T.get(timing_key, 0)) if timing_key and timing_key in T else ""
        return (
            f'<span class="debug-key">{label}</span>'
            f'<span class="debug-value">{value}</span>'
            f'{bar}<br>'
        )

    embed_note = " (cache hit ⚡)" if debug.get("embedding_cached") else " (computed)"
    query_html = (
        '<div class="debug-section">'
        '<div class="debug-section-title">Query Processing</div>'
        '<div class="debug-table">'
        + row("Rewritten query",    debug.get("rewritten_query", "—"))
        + row("Sparse tsquery",     debug.get("sparse_query", "—"))
        + row("Temporal auto",      f"{t.get('auto_from','—')} → {t.get('auto_to','—')}")
        + row("Temporal applied",   f"{t.get('effective_from','—')} → {t.get('effective_to','—')}")
        + row("Temporal parse",     "", "1_temporal_parse")
        + row("Query rewrite",      "", "2_query_rewrite")
        + row("Embedding" + embed_note, "", "3_embedding")
        + '</div></div>'
    )

    mode = "chunks" if debug.get("used_chunks") else "articles"
    retrieval_html = (
        '<div class="debug-section">'
        '<div class="debug-section-title">Retrieval</div>'
        '<div class="debug-table">'
        + row("Mode",           mode)
        + row("Dense hits",     debug.get("dense_count", 0),  "4a_dense_retrieval")
        + row("Sparse hits",    debug.get("sparse_count", 0), "4b_sparse_retrieval")
        + row("Total retrieval wall", "",                      "4_retrieval_wall")
        + row("After RRF",      debug.get("fusion_count", 0), "5_rrf_fusion")
        + row("After dedup",    debug.get("dedup_count", 0),  "6_deduplication")
        + row("Compressed",     f"{debug.get('compressed_count', 0)} articles", "7_compression")
        + row("Context build",  "",                            "8_context_build")
        + row("Top similarity", f"{debug.get('top_similarity', 0):.4f} "
              f"(threshold {CONFIDENCE_THRESHOLD})")
        + row("Confidence",     "✅ pass" if debug.get("confidence_ok") else "❌ fail — answer suppressed")
        + '</div></div>'
    )

    gen_html = ""
    if llm:
        gen_html = (
            '<div class="debug-section">'
            '<div class="debug-section-title">LLM Generation</div>'
            '<div class="debug-table">'
            + row("Backend",         llm.get("llm_backend", "—"))
            + (row("Time to first token", "", "__ttft__") if "llm_time_to_first_token_ms" in llm else "")
            + row("Total generation", "", "__gen__")
            + row("Approx tokens",   llm.get("llm_tokens_approx", "—"))
            + '</div></div>'
        )
        gen_html = gen_html.replace(
            row("Time to first token", "", "__ttft__"),
            f'<span class="debug-key">Time to first token</span>'
            f'<span class="debug-value"></span>'
            f'{_timing_bar_html(llm.get("llm_time_to_first_token_ms", 0), 3000)}<br>'
        ).replace(
            row("Total generation", "", "__gen__"),
            f'<span class="debug-key">Total generation</span>'
            f'<span class="debug-value"></span>'
            f'{_timing_bar_html(llm.get("llm_total_generation_ms", 0), 8000)}<br>'
        )

    total_ms  = T.get("TOTAL_RETRIEVAL", 0)
    gen_ms    = llm.get("llm_total_generation_ms", 0) if llm else 0
    grand_ms  = total_ms + gen_ms
    total_html = (
        '<div class="debug-section">'
        '<div class="debug-section-title">Pipeline Total</div>'
        '<div class="debug-table">'
        + f'<span class="debug-key">Retrieval pipeline</span>'
          f'<span class="debug-value"></span>{_timing_bar_html(total_ms, 3000)}<br>'
        + f'<span class="debug-key">LLM generation</span>'
          f'<span class="debug-value"></span>{_timing_bar_html(gen_ms, 8000)}<br>'
        + f'<span class="debug-key">Grand total</span>'
          f'<span class="debug-value"></span>{_timing_bar_html(grand_ms, 10000)}<br>'
        + '</div></div>'
    )

    st.markdown(query_html + retrieval_html + gen_html + total_html,
                unsafe_allow_html=True)


def export_text(history):
    lines = ["DNEXT Commodity Intelligence Chat",
             f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
             "="*60, ""]
    for m in history:
        lines += [f"[{'You' if m['role']=='user' else 'Assistant'}]", m["content"], ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_chatbot(db_url: str = DB_URL):
    apply_styles()
    init_session()

    if not db_url:
        st.error("❌ `DATABASE_URL` not set."); return

    n_embedded, n_total = load_corpus_stats(db_url)
    if n_embedded < 0:
        st.error("DB connection failed — check DATABASE_URL."); return
    if n_embedded == 0:
        st.info("💡 No embeddings yet. Run: `python pipeline.py --no-scrape`"); return

    all_sources = load_sources(db_url)
    use_chunks  = chunks_table_exists(db_url)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.75rem 0 1.2rem;text-align:center">
            <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                        color:#7DC8C8;letter-spacing:0.06em">DNEXT</div>
            <div style="font-size:0.6rem;color:#5a8fa0;text-transform:uppercase;
                        letter-spacing:0.14em">Commodity Intelligence</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<div style="font-size:0.62rem;color:#7DC8C8;text-transform:uppercase;'
                    'letter-spacing:0.1em;margin-bottom:0.4rem">🔍 Retrieval</div>',
                    unsafe_allow_html=True)
        chat_sources = st.multiselect("Sources", all_sources, default=all_sources, key="chat_sources")
        top_k        = st.slider("Candidates per retriever", 3, 12, DEFAULT_TOP_K, key="top_k_slider")
        show_sources = st.checkbox("Show source cards", value=True,  key="show_sources")
        show_debug   = st.checkbox("Show debug panel",  value=False, key="show_debug")

        st.markdown('<div style="font-size:0.62rem;color:#7DC8C8;text-transform:uppercase;'
                    'letter-spacing:0.1em;margin:0.8rem 0 0.4rem">📅 Date override</div>',
                    unsafe_allow_html=True)
        manual_from = st.date_input("From", value=None, key="manual_from")
        manual_to   = st.date_input("To",   value=None, key="manual_to")
        st.markdown("---")

        st.markdown('<div style="font-size:0.62rem;color:#7DC8C8;text-transform:uppercase;'
                    'letter-spacing:0.1em;margin-bottom:0.4rem">🤖 Generation</div>',
                    unsafe_allow_html=True)
        available_backends = (
            (["Groq"]       if GROQ_API_KEY       else []) +
            (["NVIDIA NIM"] if NVIDIA_NIM_API_KEY else []) +
            (["Local"]      if LOCAL_MODEL_PATH    else [])
        )
        if not available_backends:
            st.error("⚠️ No LLM configured"); backend = "None"; model = ""
        else:
            backend = st.selectbox("Backend", available_backends, key="backend_sel")
            if backend == "Groq":
                model = st.selectbox("Model", list(GROQ_MODELS.keys()),
                                     format_func=lambda x: f"{x} — {GROQ_MODELS[x]}",
                                     key="model_groq")
            elif backend == "NVIDIA NIM":
                model = st.selectbox("Model", list(NVIDIA_MODELS.keys()), key="model_nim")
            else:
                model = os.path.basename(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else "local"
                st.caption(f"Model: {model}")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05, key="temp_slider")
        max_tokens  = st.slider("Max tokens",  256, 2048, 1024, 128,  key="maxtok_slider")
        st.markdown("---")

        if GROQ_API_KEY:   st.success("🚀 Groq ready")
        if use_chunks:     st.success("📦 Chunk retrieval active")
        else:              st.info("📄 Article-level retrieval")

        st.markdown(f"""
        <div style="font-size:0.68rem;color:#5a8fa0;line-height:2;margin-top:0.5rem">
        🔵 Dense · pgvector cosine<br>🟡 Sparse · inverted index (search_tsv)<br>
        🟢 RRF · equal-weight fusion<br>🕐 Auto temporal detection<br>
        ✂️ Contextual compression<br>⚡ Token streaming<br>
        🔧 Per-step latency timing
        </div>
        <div style="margin-top:0.8rem;font-size:0.68rem;color:#5a8fa0">
        <b style="color:#7DC8C8">{n_embedded:,}</b> embedded ·
        <b style="color:#7DC8C8">{int(n_embedded/max(n_total,1)*100)}%</b> coverage
        </div>""", unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.markdown("---")
            st.download_button(
                "⬇ Export conversation",
                data=export_text(st.session_state.chat_history),
                file_name=f"dnext_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True,
            )
            if st.button("🗑 Clear conversation", use_container_width=True):
                for k in ("chat_history","retrieval_log","debug_log","latency_log"):
                    st.session_state[k] = []
                st.rerun()

    # ── Chat header ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-title">🌾 DNEXT Commodity Chat</div>
        <div class="chat-header-sub">
            Hybrid RAG · Dense + Sparse + RRF ·
            {"Chunk" if use_chunks else "Article"} retrieval ·
            {n_embedded:,} articles · Streaming · Per-step timing
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-title">Ask about commodity markets</div>
            <div style="font-size:0.8rem;color:#7a9fb0">
                Grounded in {n_embedded:,} articles · Enable "Show debug panel" to see per-step timing
            </div>
        </div>""", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 3]:
                if st.button(q, key=f"sug_{i}", use_container_width=True):
                    st.session_state._pending_q = q
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🌾"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                turn = i // 2
                lat  = st.session_state.latency_log[turn]  if turn < len(st.session_state.latency_log)  else 0
                dbg  = st.session_state.debug_log[turn]     if turn < len(st.session_state.debug_log)     else {}
                arts = st.session_state.retrieval_log[turn] if turn < len(st.session_state.retrieval_log) else []
                T    = dbg.get("timing_ms", {})
                ret_icon = "🟢" if dbg.get("sparse_count", 0) > 0 else "🔵"
                st.markdown(
                    f'<div class="msg-meta">'
                    f'<span class="meta-badge">{ret_icon} {len(arts)} sources</span>'
                    f'<span class="meta-badge">⚡ {lat:.1f}s total</span>'
                    f'<span class="meta-badge">🔍 {T.get("4_retrieval_wall",0)}ms retrieval</span>'
                    f'<span class="meta-badge">🧠 {dbg.get("llm_timing",{}).get("llm_total_generation_ms",0)}ms generation</span>'
                    f'</div>', unsafe_allow_html=True,
                )
                if show_sources and arts:
                    with st.expander(f"📚 {len(arts)} articles", expanded=False):
                        render_source_cards(arts)
                if show_debug and dbg:
                    with st.expander("🔧 Debug & Timing", expanded=False):
                        render_debug(dbg)

    # ── Input ─────────────────────────────────────────────────────────────────
    question = None
    if hasattr(st.session_state, "_pending_q"):
        question = st.session_state._pending_q
        del st.session_state._pending_q
    question = st.chat_input("Ask about commodity markets…") or question

    if question:
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant", avatar="🌾"):
            hint_ph   = st.empty()
            answer_ph = st.empty()

            with st.spinner("🔍 Searching…"):
                articles, context, debug = retrieve(
                    question, db_url, top_k=top_k,
                    source_filter=chat_sources if set(chat_sources) != set(all_sources) else None,
                    date_from=manual_from.isoformat() if manual_from else None,
                    date_to=manual_to.isoformat()     if manual_to   else None,
                    use_chunks=use_chunks,
                )

            t = debug.get("temporal", {})
            hints = []
            if t.get("auto_from") or t.get("auto_to"):
                hints.append(f"🕐 {t.get('auto_from','?')} → {t.get('auto_to') or 'now'}")
            rw = debug.get("rewritten_query", "")
            if rw and rw.lower().strip() != question.lower().strip():
                hints.append(f"🔄 <em>{rw}</em>")
            if debug.get("embedding_cached"):
                hints.append("⚡ embedding cached")
            if debug.get("sparse_count", 0) > 0:
                hints.append(f"🟡 {debug['sparse_count']} keyword hits")
            if hints:
                hint_ph.markdown(
                    " &nbsp;·&nbsp; ".join(f'<span class="hint-row">{h}</span>' for h in hints),
                    unsafe_allow_html=True,
                )

            if not articles:
                answer = "I couldn't find relevant articles for that question."; latency = 0.0; llm_timing = {}
                answer_ph.markdown(answer)
            elif not debug.get("confidence_ok", True):
                answer = (f"⚠️ Best similarity: **{debug.get('top_similarity',0):.3f}** "
                          f"(threshold {CONFIDENCE_THRESHOLD}). "
                          "The corpus may not contain information on this topic.")
                latency = 0.0; llm_timing = {}
                answer_ph.markdown(answer)
            else:
                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                    if m["role"] in ("user", "assistant")
                ][-6:]
                answer, latency, llm_timing = stream_answer(
                    question, context, history_for_llm,
                    answer_ph, backend, model, temperature, max_tokens,
                )

            debug["llm_timing"] = llm_timing

            T = debug.get("timing_ms", {})
            ret_icon = "🟢" if debug.get("sparse_count", 0) > 0 else "🔵"
            st.markdown(
                f'<div class="msg-meta">'
                f'<span class="meta-badge">{ret_icon} {len(articles)} sources</span>'
                f'<span class="meta-badge">⚡ {latency:.1f}s total</span>'
                f'<span class="meta-badge">🔍 {T.get("4_retrieval_wall",0)}ms retrieval</span>'
                f'<span class="meta-badge">🧠 {llm_timing.get("llm_total_generation_ms",0)}ms generation</span>'
                f'</div>', unsafe_allow_html=True,
            )

            if show_sources and articles:
                with st.expander(f"📚 {len(articles)} articles", expanded=True):
                    render_source_cards(articles)
            if show_debug:
                with st.expander("🔧 Debug & Timing", expanded=True):
                    render_debug(debug)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.retrieval_log.append(articles)
        st.session_state.debug_log.append(debug)
        st.session_state.latency_log.append(latency)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="DNEXT · Commodity Chat",
        page_icon="🌾", layout="wide",
        initial_sidebar_state="expanded",
    )
    render_chatbot(db_url=DB_URL)
