import base64
import io
import cv2
import numpy as np
from PIL import Image

def _svg_path_from_contour(contour):
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return ""
    x0, y0 = pts[0]
    d = [f"M {x0:.2f} {y0:.2f}"]
    for x, y in pts[1:]:
        d.append(f"L {x:.2f} {y:.2f}")
    d.append("Z")
    return " ".join(d)

def _eps_from_paths(paths, width, height):
    lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {width} {height}",
        "%%Creator: 3FA AUTO TRACE",
        "%%EndComments",
        "0 0 0 setrgbcolor",
        "newpath",
    ]
    for contour in paths:
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        x, y = pts[0]
        lines.append(f"{x:.2f} {height-y:.2f} moveto")
        for x, y in pts[1:]:
            lines.append(f"{x:.2f} {height-y:.2f} lineto")
        lines.append("closepath")
    lines += ["fill", "showpage", "%%EOF"]
    return "\n".join(lines)

def trace_image(data: bytes):
    try:
        image = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Fail imej tidak sah: {exc}")

    rgba = np.array(image)
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 12]
    contours.sort(key=cv2.contourArea, reverse=True)
    contours = contours[:500]

    height, width = gray.shape
    paths = [_svg_path_from_contour(c) for c in contours]
    paths = [p for p in paths if p]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<g fill="black" fill-rule="evenodd">{"".join(f'<path d="{p}"/>' for p in paths)}</g>
</svg>"""

    eps = _eps_from_paths(contours, width, height)
    quality = min(99, max(60, int(70 + min(len(paths), 150) / 3)))

    return {
        "width": width,
        "height": height,
        "contours": len(paths),
        "quality": quality,
        "svg": svg,
        "eps_base64": base64.b64encode(eps.encode()).decode(),
    }
