"""
3FA AUTO TRACE - Vector Engine V2
Drop-in replacement for apps/api/app/tracer.py

Uses OpenCV only:
- 4x internal supersampling
- transparency/white-background handling
- Otsu + adaptive + edge masks
- morphology for clean logo shapes
- connected-component filtering
- conservative contour simplification
- curve-aware SVG paths
- EPS export
"""

from __future__ import annotations
import base64
import math
from typing import Any, Dict, List

import cv2
import numpy as np


SCALE = 4.0
MIN_AREA_REL = 0.000035


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Imej tidak dapat dibaca.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _composite_white(img: np.ndarray) -> np.ndarray:
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    out = bgr * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (max(32, int(w * SCALE)), max(32, int(h * SCALE))),
        interpolation=cv2.INTER_LANCZOS4,
    )


def _candidate_masks(img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)

    _, otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    edge = cv2.Canny(gray, 45, 130)
    edge = cv2.morphologyEx(
        edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

    result = []
    for mask in (otsu, adaptive, edge):
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        result.append(mask)
    return result


def _score(mask: np.ndarray) -> float:
    h, w = mask.shape
    area = float(h * w)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    good = [c for c in contours if cv2.contourArea(c) > area * MIN_AREA_REL]
    if not good:
        return -1e9

    fill = sum(cv2.contourArea(c) for c in good) / area
    if fill > 0.92:
        return -1e6

    return 3.0 * (1.0 - abs(fill - 0.22)) + min(len(good), 12) / 12.0


def _best_mask(masks: List[np.ndarray]) -> np.ndarray:
    return max(masks, key=_score)


def _clean_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    min_area = h * w * MIN_AREA_REL

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)

    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255

    return clean


def _simplify(c: np.ndarray) -> np.ndarray:
    if len(c) < 6:
        return c

    p = cv2.arcLength(c, True)

    # Very conservative reduction: preserve logo corners and curves.
    a = max(0.30, p * 0.00030)
    c1 = cv2.approxPolyDP(c, a, True)

    p2 = cv2.arcLength(c1, True)
    a2 = max(0.40, p2 * 0.00055)
    c2 = cv2.approxPolyDP(c1, a2, True)

    return c2 if len(c2) >= 3 else c1


def _fmt(v: float) -> str:
    if abs(v) < 0.0001:
        v = 0.0
    return f"{v:.2f}"


def _path(points: np.ndarray) -> str:
    pts = points.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return ""

    # Keep sharp corners; use quadratic curves on gentle sections.
    out = [f"M {_fmt(pts[0,0])} {_fmt(pts[0,1])}"]
    n = len(pts)

    for i in range(1, n + 1):
        prev = pts[(i - 1) % n]
        cur = pts[i % n]
        nxt = pts[(i + 1) % n]

        a = prev - cur
        b = nxt - cur
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        sharp = False
        if na > 1e-6 and nb > 1e-6:
            angle = math.degrees(
                math.acos(np.clip(np.dot(a, b) / (na * nb), -1, 1))
            )
            sharp = angle < 125

        if sharp:
            out.append(f"L {_fmt(cur[0])} {_fmt(cur[1])}")
        else:
            mid = (cur + nxt) / 2.0
            out.append(
                f"Q {_fmt(cur[0])} {_fmt(cur[1])} "
                f"{_fmt(mid[0])} {_fmt(mid[1])}"
            )

    out.append("Z")
    return " ".join(out)


def _scale_path(path: str, scale: float) -> str:
    t = path.split()
    out = []
    i = 0

    while i < len(t):
        cmd = t[i]
        out.append(cmd)

        if cmd in ("M", "L"):
            out += [_fmt(float(t[i+1]) * scale),
                    _fmt(float(t[i+2]) * scale)]
            i += 3
        elif cmd == "Q":
            out += [_fmt(float(t[i+1]) * scale),
                    _fmt(float(t[i+2]) * scale),
                    _fmt(float(t[i+3]) * scale),
                    _fmt(float(t[i+4]) * scale)]
            i += 5
        else:
            i += 1

    return " ".join(out)


def _eps_from_paths(paths: List[str], width: int, height: int) -> bytes:
    # EPS uses the same vector geometry. Quadratic SVG segments are
    # represented as short line segments for maximum compatibility.
    lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {width} {height}",
        "%%Creator: 3FA AUTO TRACE Vector Engine V2",
        "1 setlinejoin",
        "1 setlinecap",
        "0 0 0 setrgbcolor",
    ]

    for p in paths:
        tokens = p.split()
        i = 0
        while i < len(tokens):
            cmd = tokens[i]

            if cmd in ("M", "L"):
                x = float(tokens[i+1])
                y = height - float(tokens[i+2])
                lines.append(
                    f"{x:.2f} {y:.2f} "
                    + ("moveto" if cmd == "M" else "lineto")
                )
                i += 3
            elif cmd == "Q":
                # Conservative EPS fallback: line to curve midpoint/end.
                x = float(tokens[i+3])
                y = height - float(tokens[i+4])
                lines.append(f"{x:.2f} {y:.2f} lineto")
                i += 5
            elif cmd == "Z":
                lines.append("closepath fill")
                i += 1
            else:
                i += 1

    lines.append("showpage")
    lines.append("%%EOF")
    return ("\n".join(lines) + "\n").encode("utf-8")


def trace_image(data: bytes) -> Dict[str, Any]:
    original = _decode(data)
    source_h, source_w = original.shape[:2]

    if source_w < 2 or source_h < 2:
        raise ValueError("Saiz imej tidak sah.")

    work = _upscale(_composite_white(original))
    mask = _best_mask(_candidate_masks(work))
    mask = _clean_components(mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    canvas_area = mask.shape[0] * mask.shape[1]
    contours = [
        c for c in contours
        if cv2.contourArea(c) >= canvas_area * MIN_AREA_REL
    ]

    if not contours:
        raise ValueError(
            "Trace gagal. Cuba imej dengan latar lebih jelas/kontras."
        )

    paths = []
    for c in contours:
        # Convert from internal 4x coordinates back to source coordinates.
        pts = c.astype(np.float32) / SCALE
        p = _path(_simplify(pts))
        if p:
            paths.append(_scale_path(p, 4.0))

    if not paths:
        raise ValueError("Trace gagal menghasilkan vector path.")

    out_w = int(source_w * 4)
    out_h = int(source_h * 4)

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{out_w}" height="{out_h}" '
        f'viewBox="0 0 {out_w} {out_h}">',
        '<g fill="#000000" fill-rule="evenodd">'
    ]
    svg.extend(f'<path d="{p}"/>' for p in paths)
    svg.extend(["</g>", "</svg>"])
    svg_text = "\n".join(svg)

    eps = _eps_from_paths(paths, out_w, out_h)
    eps_b64 = base64.b64encode(eps).decode("ascii")

    # Quality is a diagnostic score, not a claim of true source resolution.
    quality = 88
    if source_w * source_h < 150_000:
        quality -= 5
    if len(contours) > 40:
        quality -= 8
    quality = max(1, min(99, quality))

    return {
        "width": out_w,
        "height": out_h,
        "source_width": source_w,
        "source_height": source_h,
        "contours": len(contours),
        "paths": len(paths),
        "quality": quality,
        "svg": svg_text,
        "eps_base64": eps_b64,
        "engine": "3FA Vector Engine V2",
    }
