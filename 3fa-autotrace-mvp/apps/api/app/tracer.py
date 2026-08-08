import base64
import io
import cv2
import numpy as np
from PIL import Image


# ============================================================
# 3FA AUTO TRACE V2
# Smooth Bézier vector tracing
# ============================================================

def angle_between(a, b, c):
    """
    Return turning angle at point b.
    """
    v1 = a.astype(float) - b.astype(float)
    v2 = c.astype(float) - b.astype(float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 180.0

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def bezier_path(points, corner_angle=55.0, tension=0.32):
    """
    Convert a closed polygon into a smooth Bézier path.

    Sharp corners remain mostly straight.
    Curved sections receive smooth cubic Bézier handles.
    """

    pts = np.asarray(points, dtype=float)

    if len(pts) < 3:
        return ""

    # Remove duplicate neighbouring points
    clean = [pts[0]]

    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > 1.0:
            clean.append(p)

    pts = np.asarray(clean)

    if len(pts) < 3:
        return ""

    n = len(pts)

    commands = []

    # Start
    p0 = pts[0]
    commands.append(f"M {p0[0]:.2f} {p0[1]:.2f}")

    for i in range(n):
        prev = pts[(i - 1) % n]
        curr = pts[i]
        nxt = pts[(i + 1) % n]

        # Detect sharp corner
        angle = angle_between(prev, curr, nxt)

        p_prev = prev
        p_curr = curr
        p_next = nxt

        d1 = np.linalg.norm(p_curr - p_prev)
        d2 = np.linalg.norm(p_next - p_curr)

        if d1 < 0.001 or d2 < 0.001:
            continue

        # Sharp corner:
        # use straight line instead of aggressively rounding it.
        if angle < corner_angle:
            commands.append(
                f"L {p_curr[0]:.2f} {p_curr[1]:.2f}"
            )
            continue

        # Smooth Bézier handles
        incoming = p_curr - (p_curr - p_prev) * tension
        outgoing = p_curr + (p_next - p_curr) * tension

        # Previous segment control point
        prev_control = p_curr - (p_next - p_prev) * tension

        # Next segment control point
        next_control = p_curr + (p_next - p_prev) * tension

        if i == 0:
            continue

        commands.append(
            f"C "
            f"{prev_control[0]:.2f} {prev_control[1]:.2f}, "
            f"{next_control[0]:.2f} {next_control[1]:.2f}, "
            f"{p_curr[0]:.2f} {p_curr[1]:.2f}"
        )

    commands.append("Z")

    return " ".join(commands)


def eps_path(points, height, corner_angle=55.0, tension=0.32):
    """
    Convert Bézier SVG-style geometry into EPS commands.
    """

    pts = np.asarray(points, dtype=float)

    if len(pts) < 3:
        return ""

    clean = [pts[0]]

    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > 1.0:
            clean.append(p)

    pts = np.asarray(clean)

    if len(pts) < 3:
        return ""

    n = len(pts)

    p0 = pts[0]

    commands = [
        f"{p0[0]:.2f} {height - p0[1]:.2f} moveto"
    ]

    for i in range(1, n):
        prev = pts[(i - 1) % n]
        curr = pts[i]
        nxt = pts[(i + 1) % n]

        angle = angle_between(prev, curr, nxt)

        if angle < corner_angle:
            commands.append(
                f"{curr[0]:.2f} {height - curr[1]:.2f} lineto"
            )
            continue

        control1 = curr - (nxt - prev) * tension
        control2 = curr + (nxt - prev) * tension

        commands.append(
            f"{control1[0]:.2f} {height - control1[1]:.2f} "
            f"{control2[0]:.2f} {height - control2[1]:.2f} "
            f"{curr[0]:.2f} {height - curr[1]:.2f} curveto"
        )

    commands.append("closepath")

    return "\n".join(commands)


def trace_image(data: bytes):

    try:
        image = Image.open(
            io.BytesIO(data)
        ).convert("RGBA")

    except Exception as exc:
        raise ValueError(
            f"Fail imej tidak sah: {exc}"
        )

    rgba = np.array(image)

    rgb = cv2.cvtColor(
        rgba,
        cv2.COLOR_RGBA2RGB
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # ========================================================
    # 1. UPSCALE
    # ========================================================

    scale = 2

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # ========================================================
    # 2. CLEAN IMAGE
    # ========================================================

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ========================================================
    # 3. AUTO THRESHOLD
    # ========================================================

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ========================================================
    # 4. MORPHOLOGICAL CLEANUP
    # ========================================================

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # ========================================================
    # 5. FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # Ignore tiny noise
    contours = [
        c for c in contours
        if cv2.contourArea(c) >= 25
    ]

    contours.sort(
        key=cv2.contourArea,
        reverse=True
    )

    # Safety limit
    contours = contours[:300]

    # ========================================================
    # 6. REDUCE NODES
    # ========================================================

    smooth_contours = []

    for contour in contours:

perimeter = cv2.arcLength(
    contour,
    True
)

epsilon = max(
    0.5,
    perimeter * 0.001
)
        )

        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(simplified) >= 3:
            smooth_contours.append(
                simplified
            )

    height, width = gray.shape

    # ========================================================
    # 7. SVG
    # ========================================================

    svg_paths = []

    for contour in smooth_contours:

        points = contour.reshape(
            -1,
            2
        )

        path = bezier_path(
            points,
            corner_angle=55.0,
            tension=0.28
        )

        if path:
            svg_paths.append(
                f'<path d="{path}"/>'
            )

    svg = f'''<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">

<g
fill="black"
fill-rule="evenodd"
stroke="none">

{"".join(svg_paths)}

</g>
</svg>'''

    # ========================================================
    # 8. EPS
    # ========================================================

    eps_lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {width} {height}",
        "%%Creator: 3FA AUTO TRACE V2",
        "%%Title: Smooth Bézier Vector",
        "%%EndComments",
        "0 0 0 setrgbcolor",
    ]

    for contour in smooth_contours:

        points = contour.reshape(
            -1,
            2
        )

        path = eps_path(
            points,
            height,
            corner_angle=55.0,
            tension=0.28
        )

        if path:
            eps_lines.append(path)

    eps_lines.extend([
        "fill",
        "showpage",
        "%%EOF"
    ])

    eps = "\n".join(
        eps_lines
    )

    # ========================================================
    # 9. QUALITY SCORE
    # ========================================================

    total_paths = len(svg_paths)

    quality = min(
        98,
        max(
            80,
            int(
                90 +
                min(total_paths, 80) / 10
            )
        )
    )

    return {
        "width": width,
        "height": height,
        "contours": total_paths,
        "quality": quality,
        "svg": svg,
        "eps_base64": base64.b64encode(
            eps.encode()
        ).decode(),
    }
