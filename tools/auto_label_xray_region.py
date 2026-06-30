from pathlib import Path
import argparse
import cv2
import numpy as np

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def sorted_image_paths(input_dir: Path):
    image_paths = [
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(
        image_paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem.lower()
    )


def clip_box(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), img_w))
    y2 = max(y1 + 1, min(int(round(y2)), img_h))
    return x1, y1, x2, y2


def expand_box(
    x1, y1, x2, y2, img_w, img_h, margin_x=0.06, margin_top=0.01, margin_bottom=0.12
):
    box_w = x2 - x1
    box_h = y2 - y1
    return clip_box(
        x1 - box_w * margin_x,
        y1 - box_h * margin_top,
        x2 + box_w * margin_x,
        y2 + box_h * margin_bottom,
        img_w,
        img_h,
    )


def to_yolo_format(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def manual_adjust_box(
    x1,
    y1,
    x2,
    y2,
    img_w,
    img_h,
    move_x=0.0,
    move_y=0.0,
    grow_left=0.0,
    grow_right=0.0,
    grow_top=0.0,
    grow_bottom=0.0,
):
    """
    Ajuste manual final da caixa.

    move_x / move_y:
        deslocam a caixa em percentagem da dimensão total da imagem.
        move_x positivo = direita; negativo = esquerda.
        move_y positivo = baixo; negativo = cima.

    grow_left/right/top/bottom:
        aumentam a caixa em percentagem da própria largura/altura da caixa.
        Ex.: grow_right=0.25 acrescenta 25% da largura atual só à direita.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    x1 = x1 - box_w * grow_left
    x2 = x2 + box_w * grow_right
    y1 = y1 - box_h * grow_top
    y2 = y2 + box_h * grow_bottom

    dx = img_w * move_x
    dy = img_h * move_y

    x1 = x1 + dx
    x2 = x2 + dx
    y1 = y1 + dy
    y2 = y2 + dy

    return clip_box(x1, y1, x2, y2, img_w, img_h)


def get_largest_component_bbox(mask, min_area=0):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    best_box = None
    best_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_box = (int(x), int(y), int(x + w), int(y + h), int(area))

    return best_box


def contiguous_runs(values):
    runs = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def largest_run(values, min_len=1):
    runs = [run for run in contiguous_runs(values) if run[1] - run[0] >= min_len]
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def morph_1d(values, kernel_size):
    kernel_size = max(3, int(kernel_size))
    line = (values.astype(np.uint8) * 255)[None, :]
    kernel = np.ones((1, kernel_size), np.uint8)
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel, iterations=2)
    line = cv2.morphologyEx(line, cv2.MORPH_OPEN, kernel, iterations=1)
    return line[0] > 0


def remove_colored_annotations(bgr):
    """
    Remove linhas/texto coloridos de previews antigos.
    Nas imagens originais RX isto quase não altera nada, porque normalmente são cinzentas.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = ((saturation > 45) & (value > 45)).astype(np.uint8) * 255

    if np.count_nonzero(color_mask) == 0:
        return bgr, color_mask

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    cleaned = cv2.inpaint(bgr, color_mask, 3, cv2.INPAINT_TELEA)
    return cleaned, color_mask


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def expand_panel_box(box, img_w, img_h, margin=0.005):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return clip_box(
        x1 - w * margin,
        y1 - h * margin,
        x2 + w * margin,
        y2 + h * margin,
        img_w,
        img_h,
    )


def detect_panel_by_projection(gray, color_mask, black_threshold):
    """
    Deteta a placa RX por projeções horizontais/verticais.
    Isto evita que texto lateral ou ruído preto façam a caixa virar 1.000 x 1.000.
    """
    img_h, img_w = gray.shape[:2]
    color_free = color_mask == 0
    thresholds = sorted(set([black_threshold, 10, 14, 18, 22, 26, 30, 35, 40]))
    candidates = []

    for threshold in thresholds:
        mask = ((gray > threshold) & color_free).astype(np.uint8)
        col_fraction = mask.mean(axis=0)
        row_fraction = mask.mean(axis=1)

        col_cut = max(0.16, min(0.55, float(np.percentile(col_fraction, 90)) * 0.35))
        row_cut = max(0.16, min(0.55, float(np.percentile(row_fraction, 90)) * 0.35))

        valid_cols = morph_1d(col_fraction > col_cut, max(5, img_w * 0.012))
        valid_rows = morph_1d(row_fraction > row_cut, max(5, img_h * 0.012))

        x_run = largest_run(valid_cols, min_len=int(img_w * 0.18))
        y_run = largest_run(valid_rows, min_len=int(img_h * 0.18))

        if x_run is None or y_run is None:
            continue

        x1, x2 = x_run
        y1, y2 = y_run
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < img_w * img_h * 0.08:
            continue

        full_penalty = 0.0
        if x1 <= 2 and x2 >= img_w - 2:
            full_penalty += 0.35
        if y1 <= 2 and y2 >= img_h - 2:
            full_penalty += 0.20

        # Preferir uma placa grande, mas penalizar a imagem inteira.
        score = (area / (img_w * img_h)) - full_penalty + threshold * 0.001
        candidates.append((score, x1, y1, x2, y2))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, x1, y1, x2, y2 = candidates[0]
    return expand_panel_box((x1, y1, x2, y2), img_w, img_h, margin=0.005)


def detect_panel_by_components(gray, color_mask, black_threshold):
    img_h, img_w = gray.shape[:2]
    mask = ((gray > black_threshold) & (color_mask == 0)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    component = get_largest_component_bbox(mask, min_area=int(img_w * img_h * 0.05))
    if component is None:
        return None

    x1, y1, x2, y2, _ = component
    return expand_panel_box((x1, y1, x2, y2), img_w, img_h, margin=0.005)


def detect_xray_panel(gray, color_mask, black_threshold=12):
    """
    Combina dois métodos:
    1) componentes conectadas;
    2) projeções verticais/horizontais.

    Se um método apanhar a imagem inteira por erro, o outro tende a salvar o resultado.
    """
    img_h, img_w = gray.shape[:2]
    projection_box = detect_panel_by_projection(gray, color_mask, black_threshold)
    component_box = detect_panel_by_components(gray, color_mask, black_threshold)

    if projection_box is None and component_box is None:
        return 0, 0, img_w, img_h
    if projection_box is None:
        return component_box
    if component_box is None:
        return projection_box

    component_full = (component_box[2] - component_box[0]) > img_w * 0.90 and (
        component_box[3] - component_box[1]
    ) > img_h * 0.90

    projection_sensible = (
        box_area(projection_box) > img_w * img_h * 0.16
        and (projection_box[2] - projection_box[0]) > img_w * 0.22
        and (projection_box[3] - projection_box[1]) > img_h * 0.35
    )

    if component_full and projection_sensible:
        return projection_box

    # Se a projeção for plausível e não for demasiado pequena face ao componente, usa-a.
    if (
        projection_sensible
        and box_area(projection_box) > box_area(component_box) * 0.35
    ):
        return projection_box

    return component_box


def estimate_background(panel_gray, valid_mask):
    h, w = panel_gray.shape[:2]
    border = max(12, int(min(h, w) * 0.10))
    blur = cv2.GaussianBlur(panel_gray, (5, 5), 0)

    ys, xs = np.where(valid_mask > 0)
    if len(xs) < 100:
        values = blur.reshape(-1)
        return float(np.percentile(values, 55))

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    strips = [
        blur[y_min : min(y_max, y_min + border), x_min:x_max].reshape(-1),
        blur[max(y_min, y_max - border) : y_max, x_min:x_max].reshape(-1),
        blur[y_min:y_max, x_min : min(x_max, x_min + border)].reshape(-1),
        blur[y_min:y_max, max(x_min, x_max - border) : x_max].reshape(-1),
    ]

    values = np.concatenate([s for s in strips if s.size > 0])
    values = values[values > 4]
    if len(values) < 100:
        values = blur[valid_mask > 0]

    return float(np.percentile(values, 55))


def erase_inner_panel_borders(mask):
    h, w = mask.shape[:2]
    bx = max(2, int(w * 0.01))
    by = max(2, int(h * 0.01))
    cleaned = mask.copy()
    cleaned[:by, :] = 0
    cleaned[:, :bx] = 0
    cleaned[:, w - bx :] = 0
    # Não apagamos a base, porque o punho pode tocar no limite inferior.
    return cleaned


def keep_non_tiny_components(mask, min_ratio=0.00025):
    h, w = mask.shape[:2]
    min_area = int(h * w * min_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if bw < w * 0.012 or bh < h * 0.012:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 6.0 or aspect < 0.04:
            continue
        cleaned[labels == i] = 255

    return cleaned


def remove_bad_edge_components(mask):
    """
    Remove bordos da placa, marcas e texto junto às extremidades.
    Mantém componentes grandes que possam corresponder à mão/punho.
    """
    h, w = mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        touches_top = y <= 2
        touches_left = x <= 2
        touches_right = x + bw >= w - 3
        touches_bottom = y + bh >= h - 3
        touches_edge = touches_top or touches_left or touches_right or touches_bottom

        big_anatomy_like = (
            area > h * w * 0.025
            and bw > w * 0.12
            and bh > h * 0.25
            and 0.10 < cx / w < 0.90
        )

        thin_border_like = (bw > w * 0.85 and bh < h * 0.05) or (
            bh > h * 0.85 and bw < w * 0.05
        )

        if thin_border_like:
            continue
        if touches_edge and not big_anatomy_like:
            continue

        cleaned[labels == i] = 255

    return cleaned


def select_anatomy_seed(mask):
    """
    Escolhe a componente anatómica antes do fecho morfológico pesado.
    Esta é a alteração mais importante: texto/ruído deixam de ser unidos à mão.
    """
    h, w = mask.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    candidates = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]

        if area < h * w * 0.002:
            continue
        if bw < w * 0.08 or bh < h * 0.12:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.12 or aspect > 2.50:
            continue

        center_x_score = 1.0 - abs(cx / w - 0.50) * 2.0
        center_y_score = 1.0 - abs(cy / h - 0.56) * 1.5
        area_score = min(area / (h * w * 0.18), 1.0)

        edge_penalty = 0.0
        if x <= 2 or x + bw >= w - 3 or y <= 2:
            edge_penalty += 1.0
        if y < h * 0.08 and bh < h * 0.35:
            edge_penalty += 0.8

        score = (
            2.0 * area_score
            + 0.7 * center_x_score
            + 0.3 * center_y_score
            - edge_penalty
        )
        candidates.append((score, i))

    if not candidates:
        return mask

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_label = candidates[0][1]

    seed = np.zeros_like(mask)
    seed[labels == best_label] = 255
    return seed


def fallback_tight_box(panel_gray, valid_mask):
    """
    Último recurso: cria uma caixa ainda assim apertada, nunca a imagem inteira.
    """
    h, w = panel_gray.shape[:2]
    blur = cv2.GaussianBlur(panel_gray, (5, 5), 0)
    values = blur[valid_mask > 0]

    if len(values) < 100:
        return int(w * 0.20), int(h * 0.05), int(w * 0.80), int(h * 0.98)

    threshold = max(np.percentile(values, 65), np.percentile(values, 50) + 5)
    mask = ((blur > threshold) & (valid_mask > 0)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    component = get_largest_component_bbox(mask, min_area=int(h * w * 0.005))
    if component is None:
        return int(w * 0.20), int(h * 0.05), int(w * 0.80), int(h * 0.98)

    x1, y1, x2, y2, _ = component
    return x1, y1, x2, y2


def detect_hand_wrist_inside_panel(
    gray, panel_box, x_margin=0.06, top_margin=0.012, bottom_margin=0.12, debug=False
):
    img_h, img_w = gray.shape[:2]
    px1, py1, px2, py2 = panel_box
    panel = gray[py1:py2, px1:px2].copy()
    ph, pw = panel.shape[:2]

    if ph < 30 or pw < 30:
        return panel_box, {}

    blur = cv2.GaussianBlur(panel, (5, 5), 0)

    # Máscara válida: ignora margens laterais/topo, mas permite o punho tocar em baixo.
    margin_x = max(3, int(pw * 0.035))
    margin_top = max(3, int(ph * 0.025))
    margin_bottom = max(3, int(ph * 0.005))

    valid_mask = np.zeros((ph, pw), np.uint8)
    valid_mask[margin_top : ph - margin_bottom, margin_x : pw - margin_x] = 255

    bg_value = estimate_background(blur, valid_mask)
    valid_values = blur[valid_mask > 0]
    p50, p58, p60, p70, p75, p85, p95 = np.percentile(
        valid_values, [50, 58, 60, 70, 75, 85, 95]
    )

    dynamic_offset = max(5.0, 0.055 * (p95 - bg_value))
    threshold = max(bg_value + dynamic_offset, p58)
    threshold = min(threshold, p75)

    intensity_mask = ((blur > threshold) & (valid_mask > 0)).astype(np.uint8) * 255

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blur)
    equalized_values = equalized[valid_mask > 0]
    otsu_threshold, _ = cv2.threshold(
        equalized_values.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    percentile_threshold = np.percentile(equalized_values, 60)
    contrast_mask = (
        ((equalized > otsu_threshold) | (equalized > percentile_threshold))
        & (valid_mask > 0)
    ).astype(np.uint8) * 255

    edges = cv2.Canny(equalized, 25, 85)
    edges = cv2.bitwise_and(edges, valid_mask)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    combined = cv2.bitwise_or(intensity_mask, contrast_mask)
    combined = cv2.bitwise_or(combined, edges)
    combined = erase_inner_panel_borders(combined)
    combined = keep_non_tiny_components(combined, min_ratio=0.00025)
    combined = remove_bad_edge_components(combined)

    seed = select_anatomy_seed(combined)

    # Fecho morfológico só depois de isolar a mão/punho.
    kernel_close_size = max(11, int(round(min(pw, ph) * 0.030)))
    if kernel_close_size % 2 == 0:
        kernel_close_size += 1

    kernel_dilate_size = max(7, int(round(min(pw, ph) * 0.020)))
    if kernel_dilate_size % 2 == 0:
        kernel_dilate_size += 1

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_close_size, kernel_close_size)
    )
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_dilate_size, kernel_dilate_size)
    )

    final_mask = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=2)
    final_mask = cv2.morphologyEx(
        final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )

    component = get_largest_component_bbox(final_mask, min_area=int(ph * pw * 0.004))
    seed_component = get_largest_component_bbox(seed, min_area=int(ph * pw * 0.002))

    if component is None:
        x1, y1, x2, y2 = fallback_tight_box(panel, valid_mask)
    else:
        x1, y1, x2, y2, _ = component

    if seed_component is not None:
        sx1, sy1, sx2, sy2, _ = seed_component
        seed_h = sy2 - sy1

        # O topo vem da seed limpa, porque a dilatação pode puxar a caixa para cima por causa de ruído.
        y1 = max(0, sy1 - max(5, int(seed_h * 0.010)))
        x1 = min(x1, sx1)
        x2 = max(x2, sx2)
        y2 = max(y2, sy2)

    x1, y1, x2, y2 = expand_box(
        x1,
        y1,
        x2,
        y2,
        img_w=pw,
        img_h=ph,
        margin_x=x_margin,
        margin_top=top_margin,
        margin_bottom=bottom_margin,
    )

    if y2 > ph * 0.90:
        y2 = min(ph, int(y2 + (y2 - y1) * 0.03))

    gx1, gy1, gx2, gy2 = clip_box(px1 + x1, py1 + y1, px1 + x2, py1 + y2, img_w, img_h)

    # Sanity check: se voltou a ficar quase igual à placa inteira, usa a seed limpa como caixa base.
    panel_w = px2 - px1
    panel_h = py2 - py1
    final_w = gx2 - gx1
    final_h = gy2 - gy1

    if (
        seed_component is not None
        and final_w > panel_w * 0.98
        and final_h > panel_h * 0.98
    ):
        sx1, sy1, sx2, sy2, _ = seed_component
        tx1, ty1, tx2, ty2 = expand_box(
            sx1,
            sy1,
            sx2,
            sy2,
            img_w=pw,
            img_h=ph,
            margin_x=max(x_margin, 0.08),
            margin_top=top_margin,
            margin_bottom=bottom_margin,
        )
        gx1, gy1, gx2, gy2 = clip_box(
            px1 + tx1, py1 + ty1, px1 + tx2, py1 + ty2, img_w, img_h
        )

    debug_data = {}
    if debug:
        debug_data = {
            "combined": combined,
            "seed": seed,
            "final_mask": final_mask,
            "background": bg_value,
            "threshold": threshold,
        }

    return (gx1, gy1, gx2, gy2), debug_data


def save_debug_images(debug_dir, image_path, debug_data):
    if not debug_data:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    for name in ["combined", "seed", "final_mask"]:
        if name in debug_data and debug_data[name] is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_{name}.png"), debug_data[name])


def process_image(
    image_path,
    labels_dir,
    preview_dir,
    debug_dir,
    class_id=0,
    black_threshold=12,
    x_margin=0.06,
    top_margin=0.012,
    bottom_margin=0.12,
    move_x=0.0,
    move_y=0.0,
    grow_left=0.0,
    grow_right=0.0,
    grow_top=0.0,
    grow_bottom=0.0,
    save_debug=False,
    draw_panel=False,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERRO] Não consegui abrir: {image_path}")
        return False

    img_h, img_w = image.shape[:2]

    cleaned_image, color_mask = remove_colored_annotations(image)
    gray_clean = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    panel_box = detect_xray_panel(
        gray_original,
        color_mask=color_mask,
        black_threshold=black_threshold,
    )

    hand_box, debug_data = detect_hand_wrist_inside_panel(
        gray_clean,
        panel_box,
        x_margin=x_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        debug=save_debug,
    )

    x1, y1, x2, y2 = hand_box

    # Ajuste manual final, útil para afinar casos específicos sem alterar o algoritmo.
    x1, y1, x2, y2 = manual_adjust_box(
        x1,
        y1,
        x2,
        y2,
        img_w=img_w,
        img_h=img_h,
        move_x=move_x,
        move_y=move_y,
        grow_left=grow_left,
        grow_right=grow_right,
        grow_top=grow_top,
        grow_bottom=grow_bottom,
    )

    x_center, y_center, width, height = to_yolo_format(x1, y1, x2, y2, img_w, img_h)

    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        )

    preview = image.copy()

    if draw_panel:
        px1, py1, px2, py2 = panel_box
        cv2.rectangle(preview, (px1, py1), (px2, py2), (130, 130, 130), 2)

    # Caixa final em azul: esta é a caixa que vai para o YOLO.
    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)

    text = f"YOLO: {class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
    text_x = max(5, x1)
    text_y = max(25, y1 - 8 if y1 > 35 else y1 + 25)
    cv2.putText(
        preview,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    preview_path = preview_dir / image_path.name
    cv2.imwrite(str(preview_path), preview)

    if save_debug:
        save_debug_images(debug_dir, image_path, debug_data)

    print(
        f"[OK] {image_path.name} -> {label_path.name} | "
        f"YOLO {class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label para RX mão/punho em formato YOLO."
    )

    parser.add_argument(
        "--input", default="data/raw", help="Pasta com imagens originais"
    )
    parser.add_argument(
        "--labels", default="data/auto_labels", help="Pasta onde guardar .txt YOLO"
    )
    parser.add_argument(
        "--preview", default="data/auto_preview", help="Pasta onde guardar previews"
    )
    parser.add_argument(
        "--debug",
        default="data/auto_debug_masks",
        help="Pasta onde guardar máscaras de debug",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Classe YOLO")
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=12,
        help="Limiar base para separar preto exterior da radiografia",
    )

    parser.add_argument(
        "--x-margin",
        type=float,
        default=0.06,
        help="Margem lateral automática da caixa final",
    )
    parser.add_argument(
        "--top-margin",
        type=float,
        default=0.012,
        help="Margem superior automática da caixa final",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=0.12,
        help="Margem inferior automática da caixa final, útil para incluir punho",
    )

    # Ajustes manuais finais:
    # move-x positivo move para a direita; negativo move para a esquerda.
    # move-y positivo move para baixo; negativo move para cima.
    # grow-* aumenta apenas um lado da caixa.
    parser.add_argument(
        "--move-x",
        type=float,
        default=0.0,
        help="Move a caixa no eixo X em fração da largura total da imagem. Ex.: 0.05 direita, -0.05 esquerda",
    )
    parser.add_argument(
        "--move-y",
        type=float,
        default=0.0,
        help="Move a caixa no eixo Y em fração da altura total da imagem. Ex.: 0.03 baixo, -0.03 cima",
    )
    parser.add_argument(
        "--grow-left",
        type=float,
        default=0.0,
        help="Aumenta a caixa para a esquerda em fração da largura atual da caixa",
    )
    parser.add_argument(
        "--grow-right",
        type=float,
        default=0.0,
        help="Aumenta a caixa para a direita em fração da largura atual da caixa",
    )
    parser.add_argument(
        "--grow-top",
        type=float,
        default=0.0,
        help="Aumenta a caixa para cima em fração da altura atual da caixa",
    )
    parser.add_argument(
        "--grow-bottom",
        type=float,
        default=0.0,
        help="Aumenta a caixa para baixo em fração da altura atual da caixa",
    )

    parser.add_argument(
        "--save-debug", action="store_true", help="Guarda máscaras intermédias"
    )
    parser.add_argument(
        "--draw-panel",
        action="store_true",
        help="Desenha também a placa RX em cinzento no preview",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    labels_dir = Path(args.labels)
    preview_dir = Path(args.preview)
    debug_dir = Path(args.debug)

    if not input_dir.exists():
        raise FileNotFoundError(f"A pasta de entrada não existe: {input_dir}")

    image_paths = sorted_image_paths(input_dir)
    print(f"Encontradas {len(image_paths)} imagens em {input_dir}")

    if not image_paths:
        print("Nenhuma imagem encontrada.")
        return

    ok_count = 0
    for image_path in image_paths:
        success = process_image(
            image_path=image_path,
            labels_dir=labels_dir,
            preview_dir=preview_dir,
            debug_dir=debug_dir,
            class_id=args.class_id,
            black_threshold=args.black_threshold,
            x_margin=args.x_margin,
            top_margin=args.top_margin,
            bottom_margin=args.bottom_margin,
            move_x=args.move_x,
            move_y=args.move_y,
            grow_left=args.grow_left,
            grow_right=args.grow_right,
            grow_top=args.grow_top,
            grow_bottom=args.grow_bottom,
            save_debug=args.save_debug,
            draw_panel=args.draw_panel,
        )
        if success:
            ok_count += 1

    print("\nProcesso concluído.")
    print(f"Imagens processadas com sucesso: {ok_count}/{len(image_paths)}")
    print(f"Labels guardadas em: {labels_dir}")
    print(f"Previews guardadas em: {preview_dir}")
    if args.save_debug:
        print(f"Máscaras de debug guardadas em: {debug_dir}")


if __name__ == "__main__":
    main()
