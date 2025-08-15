#!/usr/bin/env python3
"""
Minimal, dependency-light baseline for natural-language scene localization
with query variants, fast-mode tokenization, and dataclass enhancements.
"""

import argparse
import json
import os
import math
import itertools
import re
import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Set

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import CLIPProcessor, CLIPModel

# -------------------------- helpers --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def union_box(a, b):
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3])
    ]

def enlarge(box, w, h, ratio=0.06):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    bw2, bh2 = bw*(1+ratio), bh*(1+ratio)
    x1n, y1n = max(0, cx - bw2/2), max(0, cy - bh2/2)
    x2n, y2n = min(w-1, cx + bw2/2), min(h-1, cy + bh2/2)
    return [float(x1n), float(y1n), float(x2n), float(y2n)]

def crop_from_box(img: Image.Image, box, pad_ratio=0.02):
    w, h = img.size
    x1, y1, x2, y2 = enlarge(box, w, h, ratio=pad_ratio)
    return img.crop((x1, y1, x2, y2))

def center(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def l2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------- query parsing --------------------
STOPWORDS = set("""
a an the of on in at to with for from by and or not be is are was were been being as into near over under between behind front next this that those these multiple group many
""".split())

V_PEOPLE = {"talk","talking","chat","chatting","converse","conversation","argue","arguing",
            "hug","hugging","handshake","shake","shaking","kiss","kissing","fight","fighting",
            "stand","standing","queue","line","meeting","meet"}

V_HO = {"sell","selling","buy","buying","pay","paying","give","giving","receive","receiving",
        "hold","holding","carry","carrying","snatch","snatching","steal","stealing","offer","offering",
        "serve","serving","show","showing","look","looking","point","pointing"}

N_SYNONYMS = {
    "vendor":"person","seller":"person","customer":"person","man":"person","woman":"person",
    "boy":"person","girl":"person","people":"person","crowd":"person",
    "vegetable":"vegetables","veggies":"vegetables","fruit":"fruits",
    "stall":"cart","trolley":"cart","handcart":"cart","pushcart":"cart","table":"table",
    "bag":"bag","money":"money","cash":"money","note":"money","notes":"money",
    "chain":"jewelry","jewellery":"jewelry","jewel":"jewelry"
}

SYN_QUERY = {
    "buying": ["purchasing","paying for"],
    "selling": ["trading","giving"],
    "talking": ["chatting","speaking"],
    "man": ["person","guy"],
    "woman": ["person","lady"]
}

def tokenize_query(query: str):
    q = re.sub(r"[^a-zA-Z0-9\s]", " ", query.lower())
    toks = [t for t in q.split() if t and t not in STOPWORDS]
    verbs = {t for t in toks if t in V_PEOPLE or t in V_HO}
    nouns = []
    for t in toks:
        tt = N_SYNONYMS.get(t, t)
        if tt not in verbs:
            nouns.append(tt)
    if "person" not in nouns:
        nouns.append("person")
    return verbs, list(dict.fromkeys(nouns))

def generate_query_variants(query: str):
    variants = {query}
    for k, syns in SYN_QUERY.items():
        if k in query:
            for s in syns:
                variants.add(query.replace(k, s))
    return list(variants)

# ---------------- dataclasses ----------------
@dataclass
class Det:
    box: List[float]
    score: float
    label: str

@dataclass
class Candidate:
    box: List[float]
    kind: str            # "single" or "pair"
    labels: List[str]
    from_indices: List[int]
    det_score: float
    clip_score: float = -1.0

# ---------- OWLv2 detector (fast mode) ----------
class OwlV2Detector:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # enable fast tokenization
        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble", use_fast=True
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)

    @torch.no_grad()
    def detect(self, image: Image.Image, labels: List[str], score_thresh=0.12, max_per_label=20) -> List[Det]:
        inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_thresh
        )[0]
        dets = []
        for score, lab_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            lab = labels[int(lab_idx)]
            dets.append(Det([float(v) for v in box], float(score), lab))
        # top-k per label
        out = []
        for lab in set(d.label for d in dets):
            grp = [d for d in dets if d.label == lab]
            grp.sort(key=lambda d: d.score, reverse=True)
            out.extend(grp[:max_per_label])
        return out

# --------------- CLIP ranker ----------------
class ClipRanker:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

    @torch.no_grad()
    def score(self, image_crops: List[Image.Image], text: str) -> List[float]:
        if not image_crops:
            return []
        inputs = self.processor(text=[text], images=image_crops, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        sims = (img_emb @ txt_emb.T).squeeze(-1).float().cpu().tolist()
        return sims

# ---------- candidate building & soft-NMS ----------
def build_candidates(img: Image.Image, dets: List[Det], verbs: Set[str], nouns: List[str]) -> List[Candidate]:
    W, H = img.size
    persons = [(i, d) for i, d in enumerate(dets) if d.label == "person"]
    objects = [(i, d) for i, d in enumerate(dets) if d.label != "person"]
    singles = [Candidate(d.box, "single", [d.label], [i], d.score) for i, d in enumerate(dets)
               if d.label in nouns or d.label == "person"]
    pairs = []
    # person-object interactions
    if any(v in V_HO for v in verbs):
        wanted = [n for n in nouns if n != "person"]
        for ip, pd in persons:
            for io, od in objects:
                if wanted and od.label not in wanted:
                    continue
                if l2(center(pd.box), center(od.box)) < 0.6 * math.hypot(W, H):
                    pairs.append(Candidate(
                        union_box(pd.box, od.box), "pair", ["person", od.label], [ip, io],
                        det_score=min(pd.score, od.score)
                    ))
    # group interactions (person-person)
    is_grp = any(v in V_PEOPLE for v in verbs) or "multiple" in nouns
    if is_grp and len(persons) > 1:
        for (i, a), (j, b) in itertools.combinations(persons, 2):
            if l2(center(a.box), center(b.box)) < 0.5 * math.hypot(W, H):
                pairs.append(Candidate(
                    union_box(a.box, b.box), "pair", ["person", "person"], [i, j],
                    det_score=min(a.score, b.score)
                ))
    return singles + pairs

def soft_nms(candidates: List[Candidate], sigma=0.6, score_threshold=0.12) -> List[Candidate]:
    cands = sorted(candidates, key=lambda c: c.det_score, reverse=True)
    keep = []
    while cands:
        best = cands.pop(0)
        keep.append(best)
        new = []
        for c in cands:
            if iou_xyxy(best.box, c.box) > 0.35 and set(best.labels) == set(c.labels):
                c.det_score *= math.exp(-(iou_xyxy(best.box, c.box)**2)/sigma)
            if c.det_score > score_threshold:
                new.append(c)
        cands = new
    return keep

# ----------- drawing ----------------
def draw_boxes(img: Image.Image, boxes: List[Tuple[List[float], str]], title: str = None):
    vis = img.copy()
    drw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    for box, txt in boxes:
        x1, y1, x2, y2 = box
        drw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=3)
        if txt:
            ty = y1-18 if y1>18 else y1+3
            drw.text((x1+3, ty), txt, fill=(255,0,0), font=font)
    if title:
        drw.text((8,8), title, fill=(0,0,0), font=font)
    return vis

# --------------------- main ---------------------
def main():
    IMAGE_PATH = "snatching.jpg.webp"
    QUERY = "men snatching a chain"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SCORE_THRESH = 0.12
    MAX_LABELS = 8

    variants = generate_query_variants(QUERY)

    with Image.open(IMAGE_PATH) as im:
        img = im.convert("RGB").copy()

    detector = OwlV2Detector(device=DEVICE)
    all_dets = []
    # detect for each query variant
    for q in variants:
        verbs, nouns = tokenize_query(q)
        labels = list(dict.fromkeys(nouns))[:MAX_LABELS]
        if "person" not in labels:
            labels.insert(0, "person")
        dets = detector.detect(img, labels=labels, score_thresh=SCORE_THRESH)
        all_dets.extend(dets)

    # dedupe exact duplicates
    unique = {(tuple(d.box), d.label): d for d in all_dets}.values()

    verbs, nouns = tokenize_query(QUERY)  # use original for candidate logic
    candidates = build_candidates(img, list(unique), verbs, nouns)
    candidates = soft_nms(candidates, sigma=0.6, score_threshold=SCORE_THRESH)

    # CLIP re-ranking
    ranker = ClipRanker(device=DEVICE)
    crops = [crop_from_box(img, c.box, pad_ratio=0.04) for c in candidates]
    scores = ranker.score(crops, QUERY)
    for c, s in zip(candidates, scores):
        c.clip_score = s

    candidates.sort(key=lambda c: (c.clip_score, c.det_score), reverse=True)
    final = candidates[0] if candidates else None

    # save
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"output_{ts}"
    ensure_dir(outdir)
    vis_items = [(c.box, f"#{i+1} {','.join(c.labels)} cl={c.clip_score:.2f}") for i,c in enumerate(candidates)]
    vis_all = draw_boxes(img, vis_items)
    vis_all.save(os.path.join(outdir, "candidates.jpg"), quality=95)

    if final:
        vis_final = draw_boxes(img, [(final.box, "FINAL")])
        vis_final.save(os.path.join(outdir, "final.jpg"), quality=95)
        crop = crop_from_box(img, final.box, pad_ratio=0.04)
        crop.save(os.path.join(outdir, "final_crop.jpg"), quality=95)

    report = {
        "image": IMAGE_PATH,
        "query": QUERY,
        "variants": variants,
        "labels_used": labels,
        "num_detections": len(all_dets),
        "num_candidates": len(candidates),
        "final_box": final.box if final else None
    }
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[DONE] Outputs in {outdir}")

if __name__ == "__main__":
    main()