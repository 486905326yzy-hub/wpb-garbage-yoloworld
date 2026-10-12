import os

import yaml


class PromptVocabulary:
    """Validated prompt list and canonical subtype mapping."""

    def __init__(self, config_path):
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(config_path)))
        if not os.path.isfile(expanded):
            raise ValueError("prompt config does not exist: %s" % expanded)
        with open(expanded, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        classes = config.get("classes")
        if not isinstance(classes, dict) or not classes:
            raise ValueError("prompt config must contain a non-empty 'classes' mapping")

        self.config_path = expanded
        self.output_class = str(config.get("output_class", "garbage")).strip() or "garbage"
        self.output_class_id = int(config.get("output_class_id", 0))
        self.prompts = []
        self.subtype_by_prompt = {}
        self.subtype_thresholds = {}

        seen = set()
        for subtype, settings in classes.items():
            subtype = str(subtype).strip()
            if not subtype:
                raise ValueError("garbage subtype names cannot be empty")
            if isinstance(settings, list):
                prompts = settings
                threshold = 0.0
            elif isinstance(settings, dict):
                prompts = settings.get("prompts", [])
                threshold = float(settings.get("min_confidence", 0.0))
            else:
                raise ValueError("settings for %s must be a mapping or list" % subtype)
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("subtype %s must contain at least one prompt" % subtype)
            self.subtype_thresholds[subtype] = threshold
            for prompt in prompts:
                prompt = str(prompt).strip()
                normalized = prompt.casefold()
                if not prompt:
                    raise ValueError("empty prompt in subtype %s" % subtype)
                if normalized in seen:
                    raise ValueError("duplicate prompt: %s" % prompt)
                seen.add(normalized)
                self.prompts.append(prompt)
                self.subtype_by_prompt[normalized] = subtype

    def candidate(self, box, score, prompt):
        prompt = str(prompt)
        subtype = self.subtype_by_prompt.get(prompt.casefold(), "unknown")
        if float(score) < self.subtype_thresholds.get(subtype, 0.0):
            return None
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        return {
            "class_name": self.output_class,
            "class_id": self.output_class_id,
            "subtype": subtype,
            "prompt": prompt,
            "score": float(score),
            "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        }


def intersection_over_union(left, right):
    ax1, ay1, ax2, ay2 = left["bbox"]
    bx1, by1, bx2, by2 = right["bbox"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def canonical_nms(candidates, iou_threshold=0.45, max_detections=50):
    """Collapse duplicate aliases because every configured object is garbage."""
    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept = []
    while remaining and len(kept) < max(1, int(max_detections)):
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            candidate
            for candidate in remaining
            if intersection_over_union(best, candidate) < float(iou_threshold)
        ]
    return kept


def draw_detections(image, detections):
    import cv2

    colors = {
        "chips_can": (0, 80, 255),
        "cookie_box": (40, 180, 255),
        "iced_tea_bottle": (0, 210, 255),
        "cola_bottle": (30, 30, 220),
        "shampoo_bottle": (255, 120, 20),
        "apple": (50, 50, 240),
        "banana": (0, 230, 230),
        "paper_ball": (230, 230, 230),
    }
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = colors.get(detection["subtype"], (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = "garbage/%s %.2f" % (detection["subtype"], detection["score"])
        text_y = max(18, y1 - 6)
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return image
