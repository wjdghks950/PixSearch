'''
Build a FAISS index for Wikipedia using encoder of choice (text or image)
----------------------------------------------------------------------

- Reads a JSONL dump where each row contains:
  { "wikipedia_title", "wikipedia_content", "wikipedia_image_url", "wikipedia_summary" }

- Splits wikipedia_content into passages (token, sentence, or paragraph level).

- Encodes passages with the General Multimodal Embedding (GME) (text-only mode).

- Stores embeddings in a FAISS IndexIDMap2 (inner-product / cosine).

- Saves accompanying metadata (title, passage text, offsets) as a
  memory-mapped Arrow table for later lookup.

'''
import argparse, json, os, re, sys, time, uuid, pathlib, itertools
import signal, functools, shutil, tempfile, requests, hashlib
from urllib.parse import urlparse
from typing import List, Dict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
import faiss
import pyarrow as pa
import pyarrow.feather as feather
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from utils.utils import save_faiss_checkpoint


# image caching helpers for wikipedia image url downloading
def _safe_filename_from_url(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    ext = os.path.splitext(urlparse(url).path)[-1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".jpg"
    return f"{h}{ext}"

def _cache_wikipedia_image(url: str, images_dir: Path, timeout: float = 20.0) -> Path | None:
    try:
        images_dir.mkdir(parents=True, exist_ok=True)
        fp = images_dir / _safe_filename_from_url(url)
        if fp.exists() and fp.stat().st_size > 0:
            return fp
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(fp, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return fp if fp.stat().st_size > 0 else None
    except Exception:
        return None


# chunking helpers
_SENT_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')

def chunk_paragraph(text: str) -> List[str]:
    """Split on blank lines (paragraphs)."""
    return [p.strip() for p in text.splitlines() if p.strip()]

def chunk_sentence(text: str, max_sent: int = 5) -> List[str]:
    """Group consecutive sentences so that each chunk less than or equal to max_sent sentences."""
    sents = [s.strip() for s in _SENT_SPLIT_REGEX.split(text) if s.strip()]
    return [" ".join(sents[i : i + max_sent])
            for i in range(0, len(sents), max_sent)]

def chunk_sections(text: str) -> List[str]:
    # chunks are delineated by '##'
    sections = [t.strip() for t in text.split("##")]
    return sections

def chunk_fixed_tokens(text: str,
                       tokenizer,
                       chunk_tokens: int = 128,
                       overlap: int = 32) -> List[str]:
    """Sliding-window token chunking with overlap."""
    words = text.split()
    stride = chunk_tokens - overlap
    idx = 0
    out = []
    while idx < len(words):
        out.append(" ".join(words[idx : idx + chunk_tokens]))
        idx += stride
    return out

# calculate stats helper
def run_length_stats(args, tokenizer, rank, world_size, device):
    # figure out how many docs this rank will read
    input_corpus_fp = Path(args.input_jsonl)
    num_lines = sum(1 for _ in input_corpus_fp.open("r", encoding="utf-8"))

    if args.limit_total > 0:
        base = args.limit_total // world_size
        extra = args.limit_total % world_size
        local_limit = base + (1 if rank < extra else 0)
        total_for_bar = local_limit
    else:
        total_for_bar = (num_lines // world_size) + (1 if rank < (num_lines % world_size) else 0)

    pbar = tqdm(total=total_for_bar, position=rank, disable=rank != 0, desc=f"[rank {rank}] docs")

    # accumulators
    local_cnt = 0
    local_sum = 0
    local_sum_sq = 0
    local_min = None
    local_max = None

    # histogram accumulators: 0-99, 100-199, ..., 900-999, 1000+
    NBINS = 11
    local_hist = np.zeros(NBINS, dtype=np.int64)

    processed = 0
    with open(args.input_jsonl, 'r', encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp):
            if line_no % world_size != rank:
                continue
            if args.limit_total > 0 and processed >= total_for_bar:
                break
            processed += 1

            doc = json.loads(raw_line)
            content = doc.get("wikipedia_content", "")

            # chunk exactly the same way your indexer would
            if args.chunk_scheme == "paragraph":
                chunks = chunk_paragraph(content)
            elif args.chunk_scheme == "sentence":
                chunks = chunk_sentence(content, args.sent_per_chunk)
            elif args.chunk_scheme == "section":
                chunks = chunk_sections(content)
            else:
                chunks = chunk_fixed_tokens(
                    content, tokenizer,
                    chunk_tokens=args.chunk_tokens,
                    overlap=args.overlap_tokens
                )

            if chunks:
                # tokenize WITHOUT padding/truncation; count true tokenized length (incl. specials)
                enc = tokenizer(chunks, add_special_tokens=True, padding=False, truncation=False)
                lens = np.fromiter((len(ids) for ids in enc["input_ids"]), dtype=np.int64, count=len(chunks))
                if lens.size:
                    # stats
                    local_cnt    += int(lens.size)
                    s             = int(lens.sum())
                    local_sum    += s
                    local_sum_sq += int((lens.astype(np.int64) ** 2).sum())
                    mn, mx        = int(lens.min()), int(lens.max())
                    local_min     = mn if local_min is None else min(local_min, mn)
                    local_max     = mx if local_max is None else max(local_max, mx)

                    # histogram: 100-token bins, overflow at 1000+
                    idx = lens // 100
                    idx = np.minimum(idx, NBINS - 1)  # clamp >=1000 to last bin
                    np.add.at(local_hist, idx, 1)

            pbar.update(1)

    pbar.close()

    # reduce across ranks
    cnt_sum_sumsq = torch.tensor(
        [local_cnt, local_sum, local_sum_sq],
        device=device, dtype=torch.float64
    )
    # for min/max, handle empty-rank edge cases
    min_t = torch.tensor([local_min if local_min is not None else float('inf')], device=device, dtype=torch.float64)
    max_t = torch.tensor([local_max if local_max is not None else float('-inf')], device=device, dtype=torch.float64)
    # histogram
    hist_t = torch.tensor(local_hist, device=device, dtype=torch.long)

    if dist.is_initialized():
        dist.all_reduce(cnt_sum_sumsq, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(hist_t, op=dist.ReduceOp.SUM)

    total_cnt = int(cnt_sum_sumsq[0].item())
    total_sum = float(cnt_sum_sumsq[1].item())
    total_sq  = float(cnt_sum_sumsq[2].item())
    gmin      = int(min_t.item()) if np.isfinite(min_t.item()) else 0
    gmax      = int(max_t.item()) if np.isfinite(max_t.item()) else 0
    hist      = hist_t.detach().cpu().numpy()

    if rank == 0:
        if total_cnt == 0:
            print("[length-stats] No chunks found.")
            return

        mean = total_sum / total_cnt
        var  = max(0.0, (total_sq / total_cnt) - (mean * mean))
        std  = var ** 0.5

        print(f"[length-stats] scheme={args.chunk_scheme}  chunks={total_cnt:,}  "
              f"min={gmin}  max={gmax}  avg={mean:.2f}  std={std:.2f}")

        # ---- draw and save histogram bar chart ----
        labels = [f"{i*100}-{i*100+99}" for i in range(10)] + ["1000+"]
        x = np.arange(len(labels))

        plt.figure(figsize=(10, 4.5))
        plt.bar(x, hist)                          # no explicit colors/styles
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.xlabel("Tokens per chunk")
        plt.ylabel("Count")
        plt.title("Chunk length distribution")
        plt.tight_layout()

        # save under output_dir / model_name
        model_name = args.model_str.split("/")[-1].lower()
        out_dir = Path(args.output_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = Path(f"./length_hist_{args.chunk_scheme}.png")
        plt.savefig(out_png, dpi=160)
        plt.close()

        # optional: also dump CSV counts for inspection
        out_csv = Path(f"./length_hist_{args.chunk_scheme}.csv")
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("bin,count\n")
            for lbl, c in zip(labels, hist.tolist()):
                f.write(f"{lbl},{c}\n")

        print(f"[length-stats] saved chart → {out_png}")
        print(f"[length-stats] saved counts → {out_csv}")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# resume helpers
def load_state(state_path):
    if state_path.exists():
        with state_path.open() as f:
            return json.load(f)
    return {'doc_offset': 0, 'next_local_id': 0}

def save_checkpoint(rank, gpu_index,
                    meta_batches, meta_path,
                    index_path, state_path,
                    processed, next_local_id):
    cpu_idx = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_idx, str(index_path))

    if meta_batches:
        new_meta = pa.concat_tables(meta_batches)
        if meta_path.exists():
            old_meta = feather.read_table(str(meta_path))
            new_meta = pa.concat_tables([old_meta, new_meta])
        feather.write_feather(new_meta, str(meta_path))
        meta_batches.clear()

    with state_path.open("w") as f:
        json.dump(
            {
                "doc_offset": processed,
                "next_local_id": next_local_id,
                "index_ntotal": int(cpu_idx.ntotal)}, f)

    if rank == 0:
        print(f"[rank {rank}] [v] checkpoint @ {processed:,} docs, {int(cpu_idx.ntotal):,} vectors")


# encoder and indexing pipeline
@torch.no_grad()
def encode_passages(args,
                    passages: List[str],
                    tokenizer,
                    model,
                    device,
                    max_length: int = 512) -> np.ndarray:
    """Batch-encode a list of passages → (N, D) float32 numpy array."""
    encs = []
    bs = args.batch_size                     # Adjust depending on GPU RAM
    for i in range(0, len(passages), bs):
        batch = passages[i : i + bs]
        tok = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt').to(device)
        with torch.no_grad():
            if "gme" in args.model_str:
                embs = model.get_text_embeddings(texts=batch)
            elif "sentence-transformers" in args.model_str:  # NOTE: Refer to 'https://huggingface.co/learn/llm-course/en/chapter5/6' for more details
                out = model(**tok)
                if "all-mpnet-base" in args.model_str:
                    embs = mean_pooling(out, tok["attention_mask"])  # mean pooling
                else:
                    embs = out.last_hidden_state[:, 0, :]  # [CLS] pooling
            elif "clip" in args.model_str:
                out = model(**tok)
                if hasattr(out, "pooler_output"):            # CLIP-style
                    embs = out.pooler_output
                else:                                        # Generic — take [CLS]
                    embs = out.last_hidden_state[:, 0, :]

        embs = torch.nn.functional.normalize(embs, p=2, dim=1)  # cosine
        encs.append(embs.cpu().float().numpy())

    return np.vstack(encs)


def encode_images(
                  args,
                  images: List[Image.Image],
                  tokenizer,
                  model,
                  device) -> np.ndarray:
    encs = []
    embs = model.get_image_embeddings(images=images)
    encs.append(embs.cpu().float().numpy())
    return np.vstack(encs)

@torch.no_grad()
def encode_images_dinov3(
    args,
    pil_images: list[Image.Image],
    image_processor,
    image_model,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    outs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = image_processor(images=batch, return_tensors="pt").to(device)
        outputs = image_model(**inputs)
        feats = getattr(outputs, "pooler_output", None)
        if feats is None:  # fallback to mean if pooler not present
            breakpoint()
            feats = outputs.last_hidden_state.mean(dim=1)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        outs.append(feats.detach().cpu().float().numpy())
    return np.vstack(outs)

def build_index(d: int,
                index_type: str = "FlatIP",
                hnsw_m: int = 64):
    """Return a FAISS index given embedding dimension."""
    if index_type == "HNSW":
        idx = faiss.IndexHNSWFlat(d, hnsw_m)
        idx.hnsw.efConstruction = 200
        return idx
    # Default: exact flat inner-product (cosine)
    return faiss.IndexFlatIP(d)


def main(args):
    # rank/world discovery
    rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    if world_size > 1 and not dist.is_initialized():
        # dist.init_process_group(backend="nccl", init_method="env://")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=10)  # NOTE: InT234224331 creasing the process-group timeout to avoid last minute termination
        )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        print(f">> Launching {world_size} processes…")

    print(f"Loading [{args.model_str}] on {device} …")
    is_image_mode = "dino" in args.model_str.lower()
    if is_image_mode:
        image_processor = AutoImageProcessor.from_pretrained(args.model_str)
        model = AutoModel.from_pretrained(
            args.model_str,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_str,
            use_fast=True,
            trust_remote_code=True
        )
        if args.print_length_stats:
            if rank == 0:
                print("[length-stats] Token-length stats requested.")
            run_length_stats(args, tokenizer, rank, world_size, device)
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            return

        model = AutoModel.from_pretrained(
            args.model_str,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device).eval()


    # output paths
    model_name = args.model_str.split("/")[-1].lower()
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / f"wiki6m_faiss_{rank}.index"
    meta_path = out_dir / f"wiki6m_rank_{rank}.arrow"
    state_path = out_dir / f"wiki6m_state_rank_{rank}.json"
    images_dir = out_dir / "images"

    # if resume, revive the state
    state = load_state(state_path) if args.resume else {"doc_offset": 0, "next_local_id": 0}
    processed = state["doc_offset"]  # how many docs this 'rank' processed
    next_local_id = state["next_local_id"]  # running chunk-id counter

    # ----  load existing index / meta if resuming --------------------------
    if args.resume and index_path.exists():
        cpu_idx = faiss.read_index(str(index_path))
        dim = cpu_idx.d
        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), rank, cpu_idx)
        if meta_path.exists():
            pass  # keep previously saved rows out of RAM; re-open on flush
    else:  # peek embedding dim. and create GPU index
        if is_image_mode:
            dummy = Image.new("RGB", (64, 64), color=(255, 255, 255))
            dim = encode_images_dinov3(args, [dummy], image_processor, model, device, batch_size=1).shape[1]
        else:
            dim = encode_passages(args, ["hello"], tokenizer, model, device).shape[1]
        cpu_index = build_index(dim, args.index_type)
        gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), rank, cpu_index)
        gpu_index = faiss.IndexIDMap2(gpu_index)

    meta_batches = []

    # read the corpus and process shards; calculate the number of lines
    input_corpus_fp = Path(args.input_jsonl)
    num_lines = sum(1 for _ in input_corpus_fp.open("r", encoding="utf-8"))

    if args.limit_total > 0:
        base = args.limit_total // world_size
        extra = args.limit_total % world_size
        local_limit = base + (1 if rank < extra else 0)
        if rank == 0:
            print(f"[LIMIT] Processing at most {args.limit_total:,} documents total -> this rank will read ≤ {local_limit:,}")
        pbar = tqdm(total=local_limit, initial=processed, position=rank, disable=rank != 0, desc=f"[rank {rank}] passages")
    else:
        per_rank_total = (num_lines // world_size) + (1 if rank < (num_lines % world_size) else 0)
        pbar = tqdm(total=per_rank_total, initial=processed, position=rank, disable=rank != 0, desc=f"[rank {rank}] passages")

    # main loop
    skip_docs = processed
    with open(args.input_jsonl, 'r', encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp):
            if line_no % world_size != rank:
                continue
            if skip_docs:
                skip_docs -= 1
                continue
            if args.limit_total > 0 and processed >= local_limit:
                break
            processed += 1

            doc = json.loads(raw_line)

            if is_image_mode:
                url = doc.get("wikipedia_image_url", None)
                if not url or (isinstance(url, str) and url.strip().lower() == "null"):
                    pbar.update(1)
                    continue

                if not args.cache_wiki_images:
                    pbar.update(1)
                    continue

                local_img = _cache_wikipedia_image(url, images_dir, timeout=20.0)
                if not (local_img and local_img.exists() and local_img.stat().st_size > 0):
                    pbar.update(1)
                    continue

                # Load, embed, and index one image embedding per document
                try:
                    with Image.open(local_img) as im:
                        pil_img = im.convert("RGB")
                    img_emb = encode_images_dinov3(args, [pil_img], image_processor, model, device, batch_size=max(1, args.batch_size))
                    ids = np.array([next_local_id], dtype=np.int64) * world_size + rank  # ID scheme: reuse your per-rank counter (doc-level now)
                    next_local_id += 1
                    gpu_index.add_with_ids(img_emb.astype(np.float32), ids)

                    # Minimal doc-level metadata for linking
                    meta_batches.append(pa.table({
                        "id": ids,
                        "wikidata_id": [doc.get("wikidata_id", "")],
                        "title": [doc.get("wikipedia_title", "")],
                        "image_path": [str(local_img)],
                        "image_url": [url],
                        "summary": [doc.get("wikipedia_summary", "")],
                        "content": [doc.get("wikipedia_content", "")]
                    }))
                except Exception:
                    # bad/corrupt image – skip
                    pass
            else:
                content = doc["wikipedia_content"]
                title = doc["wikipedia_title"]
                # Choose chunking scheme
                if args.chunk_scheme == "paragraph":
                    chunks = chunk_paragraph(content)
                elif args.chunk_scheme == "sentence":
                    chunks = chunk_sentence(content, args.sent_per_chunk)
                elif args.chunk_scheme == "section":
                    chunks = chunk_sections(content)
                else:
                    chunks = chunk_fixed_tokens(
                        content, tokenizer,
                        chunk_tokens=args.chunk_tokens,
                        overlap=args.overlap_tokens
                    )

                if not chunks:
                    pbar.update(1)
                    continue

                # encode the text chunk with model
                embs = encode_passages(args, chunks, tokenizer, model, device)
                ids = np.arange(next_local_id, next_local_id + len(chunks), dtype=np.int64) * world_size + rank
                next_local_id += len(chunks)
                gpu_index.add_with_ids(embs, ids)

                meta_batches.append(pa.table({
                    "id": ids,  # NOTE: think about the possibility of indexing chunk-wise and appending to each chunk from the same document
                    "wikidata_id": [doc.get("wikidata_id", "")] * len(chunks),
                    "title": [doc.get("wikipedia_title", "")] * len(chunks),
                    "passage": chunks
                }))

            if processed % args.checkpoint_step == 0:
                save_checkpoint(rank, gpu_index, meta_batches,
                                meta_path, index_path, state_path,
                                processed, next_local_id)

            pbar.update(1)

        save_checkpoint(rank, gpu_index, meta_batches,
                        meta_path, index_path, state_path,
                        processed, next_local_id)

    pbar.close()

    # synchronise and merge (rank‑0) ----------------------------------------
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("** Merging partial indexes …")
        merged = faiss.IndexIDMap2(build_index(dim, args.index_type))
        meta_tables = []

        for r in range(world_size):
            part_idx  = faiss.read_index(str(out_dir / f"wiki6m_faiss_{r}.index"))
            merged.merge_from(part_idx, 0)

            part_meta = feather.read_table(str(out_dir / f"wiki6m_rank_{r}.arrow"))
            meta_tables.append(part_meta)

        faiss.write_index(merged, str(out_dir / "wiki6m_faiss.index"))
        feather.write_feather(pa.concat_tables(meta_tables), str(out_dir / "wiki6m_meta.arrow"))
        print("[ v ] Merge complete — all done.")

    if dist.is_initialized():
        dist.destroy_process_group()


# cli parse
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_str",
                    choices=[
                              "sentence-transformers/all-mpnet-base-v2",
                              "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                              "facebook/dinov3-vitb16-pretrain-lvd1689m",
                              "facebook/dinov2-base"
                            ],
                    default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
                )
    p.add_argument("--input_jsonl", default="/home/jeonghkim/projects/region-mm-retrieve/data/infoseek/Wiki6M_ver_1_0.jsonl")
    p.add_argument("--output_dir",  default="/home/jeonghkim/projects/region-mm-retrieve/data/infoseek/wiki6m_index")
    p.add_argument("--chunk_scheme", choices=["section", "paragraph", "sentence", "token"], default="section")
    p.add_argument("--chunk_tokens", type=int, default=128,
                   help="Token window size when chunk_scheme=token")
    p.add_argument("--overlap_tokens", type=int, default=32)
    p.add_argument("--sent_per_chunk", type=int, default=3,
                   help="Sentences per chunk when chunk_scheme=sentence")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--limit_total", type=int, default=0)
    p.add_argument("--index_type", choices=["FlatIP", "HNSW"], default="FlatIP")
    p.add_argument("--resume", action="store_true", help="Resume from a previous run")
    p.add_argument("--cache_wiki_images", action="store_true", help="Cache images from Wikipedia")
    p.add_argument("--checkpoint_step", type=int, default=1000, help="Save index / metadata / state after this many docs per rank")
    p.add_argument("--print_length_stats", action="store_true", help="Print average token length per chunk across the entire dataset")

    args = p.parse_args()
    main(args)
