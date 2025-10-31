# search_api.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, json, time, threading, requests
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Any
from PIL import Image

import numpy as np
import faiss
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from tqdm import tqdm


TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
GME_MODEL  = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
IMAGE_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# helpers
def slug_from_model_str(model_str: str) -> str:
    # e.g., "sentence-transformers/all-mpnet-base-v2" -> "all-mpnet-base-v2"
    return model_str.split("/")[-1].lower()

def detect_model_type(model_str: str) -> str:
    s = model_str.lower()
    if "dinov3" in s:
        return "dinov3"
    elif "sentence-transformers" in s:
        return "st"
    elif "gme" in s or "qwen2-vl" in s:
        return "gme"
    # default to sentence-transformers
    return "st"

def find_index_files(index_dir: Path) -> Tuple[Path, Path]:
    idx = sorted(index_dir.glob("*_faiss.index"))
    meta = sorted(index_dir.glob("*_meta.arrow"))
    if not idx or not meta:
        raise FileNotFoundError(f"Could not find *_faiss.index or *_meta.arrow in {index_dir}")
    return idx[0], meta[0]

def get_value_from_row(row_tbl: pa.Table, candidate_cols: List[str], default=""):
    for c in candidate_cols:
        if c in row_tbl.column_names:
            return row_tbl.column(c).to_pylist()[0]
    return default

def warm_file_with_progress(path: Path, chunk_bytes: int = 64 * 1024 * 1024):
    """
    Reads the file sequentially to warm the OS page cache and report progress.
    This gives a real progress bar before faiss.read_index() or Arrow reading.
    """
    size = path.stat().st_size
    print(f">> Warming file cache for [{path}] ({size / (1024**3):.2f} GB)...", flush=True)
    with open(path, "rb", buffering=0) as f, tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            pbar.update(len(chunk))
    print("* Warm complete.", flush=True)

# Local Search API Wrapper
# --- Simple local search pipeline (text / image / region) ---
class LocalSearchPipeline:
    """
    Callable wrapper so you can use it like: results = search_pipeline(input_query, k=5)
    - input_query: str (TEXT) or List[str] (IMAGE/REGION paths)
    - returns: List[Dict] with at least {"title", "content", "score", "id"}
    """
    def __init__(
        self,
        base_url: str,
        collection: str,
        index_name: Optional[str] = None,
        pooling: str = "mean",
        text_model: str = "sentence-transformers/all-mpnet-base-v2",
        image_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url   = base_url.rstrip("/")
        self.collection = collection
        self.index_name = index_name
        self.pooling    = pooling
        self.text_model = text_model
        self.image_model= image_model
        self.timeout    = timeout
        self.session    = session or requests.Session()
        try:
            # ignore env-provided proxies for this session
            self.session.trust_env = False
        except Exception:
            pass
        self.session.proxies = {}  # and explicitly clear any proxies

        # Make sure process-level NO_PROXY covers localhost too (harmless if already set)
        os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

        # best-effort health check (don’t crash if it fails)
        try:
            self.session.get(self.base_url + "/health", timeout=5)
        except Exception:
            pass

    def __call__(self, query: Union[str, List[str]], k: int = 5) -> List[Dict[str, Any]]:
        # Decide qtype + payload
        if isinstance(query, str):
            qtype = "text"
            payload = {
                "qtype": qtype,
                "query": query,
                "top_k": int(k),
                "collection": self.collection,
                "index_name": self.index_name,
                "pooling": self.pooling,
            }
        elif isinstance(query, (list, tuple)):
            image_paths = [str(p) for p in query if p]
            if not image_paths:
                return []

            # Heuristic: treat cropped region files as REGION, else IMAGE
            is_region = any("cropped" in os.path.basename(p) or "region" in os.path.basename(p).lower()
                            for p in image_paths)
            qtype = "region" if is_region else "image"
            payload = {
                "qtype": qtype,
                "image_paths": image_paths,
                "top_k": int(k),
                "collection": self.collection,
                "index_name": self.index_name,
                "pooling": self.pooling,
            }
        else:
            raise TypeError(f"Unsupported query type for LocalSearchPipeline: {type(query)}")

        resp = self.session.post(self.base_url + "/search", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])

        # normalize to a simple list of dicts
        results = []
        for h in hits:
            results.append({
                "id": h.get("id"),
                "row": h.get("row"),
                "score": h.get("score"),
                "title": h.get("title") or "",
                "content": h.get("passage") or "",
                "image_path": h.get("image_path") or "",
                "image_url": h.get("image_url") or "",
                "index_dir": data.get("index_dir"),
            })
        return results



# -----------------------------
# ID Mapper
# -----------------------------

class IdMapper:
    def __init__(self, index_dir: Path, ids: np.ndarray):
        self.index_dir = index_dir
        self.ids = ids
        self.n = len(ids)
        self._aligned = self._check_aligned()
        self._sorted_order = None
        self._sorted_ids = None
        if not self._aligned:
            self._load_or_build()

    def _check_aligned(self) -> bool:
        if self.n == 0:
            return True
        if self.ids[0] != 0 or self.ids[-1] != self.n - 1:
            return False
        k = min(1000, self.n)
        return np.all(self.ids[:k] == np.arange(k))

    def _load_or_build(self):
        map_path = self.index_dir / "id2row.argsort.int32.npy"
        if map_path.exists():
            self._sorted_order = np.load(map_path, mmap_mode="r")
            self._sorted_ids = self.ids[self._sorted_order]
        else:
            so = np.argsort(self.ids).astype(np.int32)
            np.save(map_path, so)
            self._sorted_order = so
            self._sorted_ids = self.ids[self._sorted_order]

    def row_for(self, idv: int) -> int:
        if self._aligned:
            return int(idv)
        pos = int(np.searchsorted(self._sorted_ids, idv))
        if pos >= self.n or self._sorted_ids[pos] != idv:
            raise KeyError(f"id {idv} not found")
        return int(self._sorted_order[pos])

# -----------------------------
# Search Engine
# -----------------------------

class SearchEngine:
    def __init__(
        self,
        base_dir: Path,
        collection: str,
        model_str: str,
        qtype: str = None,                  # "text" | "image" | "fused"
        index_name: Optional[str] = None,
        gpu_index_mode: str = "none",  # "none" | "replicate"
        faiss_threads: int = 8,
        pooling: str = "mean",  # "mean" | "cls"
        device: Optional[str] = None,
    ):
        self.base_dir = base_dir
        self.collection = collection
        self.model_str = model_str
        self.model_type = detect_model_type(model_str)
        self.pooling = pooling
        self.gpu_index_mode = gpu_index_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Resolve index dir
        coll_dir = base_dir / collection
        if index_name:
            self.index_dir = coll_dir / index_name
        else:
            slug = slug_from_model_str(model_str)
            self.index_dir = coll_dir / slug
            if self.index_dir is None:
                raise FileNotFoundError(f"Index directory not found for {model_str}. Looked under {coll_dir}")
        if self.index_dir is None or not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found for {model_str}. Looked under {coll_dir}")

        # Find files
        self.faiss_path, self.meta_path = find_index_files(self.index_dir)

        # faiss index: warm + load with logs
        print(f">> Loading FAISS index from [ {self.faiss_path} ]...", flush=True)
        warm_file_with_progress(self.faiss_path)  # progress before deserialization
        self.cpu_index = faiss.read_index(str(self.faiss_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        faiss.omp_set_num_threads(faiss_threads)
        print(f"* Loaded FAISS index with {self.cpu_index.ntotal} vectors", flush=True)

        # Optionally move to gpus
        self.index = self.cpu_index
        if self.gpu_index_mode != "none" and torch.cuda.is_available():
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True  # halve VRAM
            co.shard = self.gpu_index_mode == "shard"  # shard across GPUs
            self.index = faiss.index_cpu_to_all_gpus(self.cpu_index, co)

        # Arrow metadata: warm + batch progress
        print(f">> Loading meta files from [ {self.meta_path} ]...", flush=True)
        warm_file_with_progress(self.meta_path)  # progress of file read
        mm = pa.memory_map(str(self.meta_path), 'r')
        self.meta_reader: pa_ipc.RecordBatchFileReader = pa_ipc.open_file(mm)

        num_batches = self.meta_reader.num_record_batches
        batches = []
        for i in tqdm(range(num_batches), desc="Reading Arrow batches"):
            batches.append(self.meta_reader.get_batch(i))
        self.meta = pa.Table.from_batches(batches)
        print(f"* Loaded metadata with {self.meta.num_rows} rows in {num_batches} batches", flush=True)

        # id -> row mapper
        ids = self.meta.column("id").to_numpy(zero_copy_only=False)
        self.id_mapper = IdMapper(self.index_dir, ids)

        # model + tokenizer
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = None
        self.image_processor = None
        if self.model_type == "dinov3":
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_str, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_str, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(self.model_str, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()

        # Lock for Faiss if needed
        self._search_lock = threading.Lock()

    @torch.inference_mode()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        if self.model_type == "gme":
            embs = self.model.get_text_embeddings(texts=texts)
        else:
            tok = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            out = self.model(**tok)
            if self.pooling == "cls":
                embs = out.last_hidden_state[:, 0, :]
            else:
                last = out.last_hidden_state  # [B,T,H]
                mask = tok.attention_mask.unsqueeze(-1)  # [B,T,1]
                summed = (last * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embs = summed / counts
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.detach().cpu().float().numpy()

    @torch.inference_mode()
    def embed_image(self, image_paths: List[str]) -> np.ndarray:
        if self.model_type == "gme":
            embs = self.model.get_image_embeddings(images=image_paths)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        elif self.model_type == "dinov3":
            pil_images = []
            for p in image_paths:
                image_file_exts = ["jpg", "png", "jpeg", "JPEG", "JPG", "PNG", "webp"]  # prevent oven/infoseek image paths from causing err
                if not p.endswith(tuple(image_file_exts)):
                    p = [f"{p}.{ext}" for ext in image_file_exts if os.path.exists(f"{p}.{ext}")][0]
                #     print(f"** >> (ext) path: {p}")
                # print(f"** >> path: {p}")
                with Image.open(p) as im:
                    pil_images.append(im.convert("RGB"))
            inputs = self.image_processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embs = getattr(outputs, "pooler_output", None)
            if embs is None and self.pooling == "mean":
                embs = outputs.last_hidden_state.mean(dim=1)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        else:
            raise NotImplementedError("self.model_type should be one of ['gme', 'dinov3']")

        return embs.detach().cpu().float().numpy()

    @torch.inference_mode()
    def embed_fused(self, texts: List[str], image_paths: List[str]) -> np.ndarray:
        if self.model_type != "gme":
            raise ValueError("Fused queries are only supported with GME models.")
        embs = self.model.get_fused_embeddings(texts=texts, images=image_paths)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.detach().cpu().float().numpy()

    def _format_hit(self,
                    idx: int,
                    score: float,
                    max_passage_chars: int = 1200) -> Dict:
        row_idx = self.id_mapper.row_for(idx)
        row = self.meta.slice(row_idx, 1)
        title = get_value_from_row(row, ["title", "wikipedia_title", "doc_title"], default="")
        passage = get_value_from_row(row, ["content", "passage", "wikipedia_content", "text", "chunk"], default="")
        summary = get_value_from_row(row, ["summary", "wikipedia_summary", "doc_summary"], default="")
        image_path = get_value_from_row(row, ["image_path"], default="")
        image_url  = get_value_from_row(row, ["image_url"], default="")

        return {
            "id": int(idx),
            "row": int(row_idx),
            "score": float(score),
            "title": title,
            "passage": passage[:max_passage_chars],
            "image_path": image_path,
            "image_url": image_url
        }

    def search(self, qtype: str, query: str = "", image_paths: Optional[List[str]] = None, top_k: int = 5) -> Dict:
        if top_k <= 0:
            top_k = 5

        if qtype == "text":
            q = self.embed_text([query])
        elif qtype in ["image", "region"]:
            if not image_paths:
                raise HTTPException(status_code=400, detail="image_paths is required for qtype='image' or qtype='region'")
            q = self.embed_image(image_paths)
        elif qtype == "fused":
            if not image_paths or not query:
                raise HTTPException(status_code=400, detail="fused requires both text and image_paths")
            q = self.embed_fused([query], image_paths)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown qtype '{qtype}'")

        top_k = min(top_k, self.index.ntotal if hasattr(self.index, "ntotal") else top_k)

        with self._search_lock:
            scores, ids = self.index.search(q, top_k)

        hits = [self._format_hit(int(idx), float(score)) for idx, score in zip(ids[0], scores[0])]
        return {
            "count": len(hits),
            "top_k": top_k,
            "model_str": self.model_str,
            "index_dir": str(self.index_dir),
            "qtype": qtype,
            "pooling": self.pooling,
            "hits": hits
        }

# -----------------------------
# FastAPI app + registry
# -----------------------------

class EngineRegistry:
    def __init__(self, base_dir: Path, default_collection: str, gpu_index_mode: str, faiss_threads: int, device: Optional[str]):
        self.base_dir = base_dir
        self.default_collection = default_collection
        self.gpu_index_mode = gpu_index_mode
        self.faiss_threads = faiss_threads
        self.device = device
        self._cache: Dict[Tuple[str, str, str, str], SearchEngine] = {}
        self._lock = threading.Lock()

    def list_indexes(self, collection: Optional[str] = None) -> List[str]:
        cdir = self.base_dir / (collection or self.default_collection)
        if not cdir.exists():
            return []
        return [p.name for p in cdir.iterdir() if p.is_dir()]

    def get(
        self,
        model_str: str,
        qtype: Optional[str] = None,
        collection: Optional[str] = None,
        index_name: Optional[str] = None,
        pooling: str = "mean",
    ) -> SearchEngine:
        coll = collection or self.default_collection
        model_id = slug_from_model_str(model_str)
        key = (coll, model_id)
        if key in self._cache:  # the cached engine (TEXT or IMAGE) is returned here
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            eng = SearchEngine(
                base_dir=self.base_dir,
                collection=coll,
                model_str=model_str,
                qtype=qtype,
                index_name=index_name,
                gpu_index_mode=self.gpu_index_mode,
                faiss_threads=self.faiss_threads,
                pooling=pooling,
                device=self.device,
            )
            self._cache[key] = eng
            return eng

# Request/Response models

class SearchRequest(BaseModel):
    # model_str is now optional; server will override based on qtype
    model_str: Optional[str] = None
    qtype: str = "text"     # "text" | "image" | "region" | "fused"
    query: Optional[str] = ""        # for text/fused
    image_paths: Optional[List[str]] = None  # for image/fused/region
    top_k: int = 5
    collection: Optional[str] = None
    index_name: Optional[str] = None  # override directory name if needed
    pooling: Optional[str] = None     # "mean" or "cls" (must match how index was built)

app = FastAPI(title="Local FAISS Search API", version="1.0.0")
REGISTRY: EngineRegistry = None  # set in main()

@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}

@app.get("/collections")
def collections():
    return {"collections": [REGISTRY.default_collection], "indexes": REGISTRY.list_indexes()}

def _model_for_qtype(qtype: str) -> str:
    q = (qtype or "").lower()
    if q in ("image", "region"):
        return IMAGE_MODEL
    elif q == "fused":
        return GME_MODEL
    return TEXT_MODEL

@app.post("/search")
def search(req: SearchRequest):
    pooling = (req.pooling or "mean").lower()
    if pooling not in ("mean", "cls"):
        raise HTTPException(status_code=400, detail="pooling must be 'mean' or 'cls'")

    # Choose model by qtype (server-enforced)
    model_str = _model_for_qtype(req.qtype)

    engine = REGISTRY.get(
        model_str=model_str,
        collection=req.collection,
        index_name=req.index_name,
        pooling=pooling,
    )
    result = engine.search(
        qtype=req.qtype,
        query=req.query or "",
        image_paths=req.image_paths,
        top_k=req.top_k,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path, help="Base dir holding collections (e.g., .../data/infoseek)")
    parser.add_argument("--image_encoder", required=True, type=str, help="gme or dinov3")
    parser.add_argument("--text_encoder", required=True, type=str, help="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--collection", required=True, help="Collection folder (e.g., wiki6m_index or wiki10k_index)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7000, type=int)
    parser.add_argument("--gpu-index", default="none", choices=["none", "replicate", "shard"], help="Move FAISS index to GPUs")
    parser.add_argument("--faiss-threads", default=8, type=int)
    parser.add_argument("--device", default=None, help="Torch device (e.g., cuda, cuda:0, cpu). Default auto.")
    args = parser.parse_args()

    # Ensure unbuffered output for uvicorn runs too
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    global REGISTRY
    REGISTRY = EngineRegistry(
        base_dir=args.base_dir,
        default_collection=args.collection,
        gpu_index_mode=args.gpu_index,
        faiss_threads=args.faiss_threads,
        device=args.device,
    )

    # Preload engines so we see loading progress at startup (+ put the search engine intances into REGISTRY._cache )
    print("\n=== Preloading TEXT engine (mpnet) ===", flush=True)
    REGISTRY.get(model_str=args.text_encoder, collection=args.collection, pooling="mean")
    print("=== TEXT engine ready ===\n", flush=True)

    print("\n=== Preloading IMAGE/REGION engine (GME Qwen2-VL) ===", flush=True)
    REGISTRY.get(model_str=args.image_encoder, collection=args.collection)
    print("=== IMAGE/REGION engine ready ===\n", flush=True)

    # Start API server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
