#!/usr/bin/env python3
"""Offline single-image check using only already-local model assets."""

import argparse
import json
import os

import cv2
import torch
from garbage_yoloworld.core import PromptVocabulary, canonical_nms, draw_detections
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--canonical-nms-iou", type=float, default=0.45)
    return parser.parse_args()


def require_file(path, name):
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(expanded):
        raise SystemExit("%s not found: %s" % (name, expanded))
    return expanded


def main():
    args = parse_args()
    image_path = require_file(args.image, "image")
    model_path = require_file(args.model, "model")
    prompt_path = require_file(args.prompts, "prompt config")
    output_path = os.path.abspath(os.path.expanduser(args.output))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("OpenCV could not decode image: %s" % image_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    vocabulary = PromptVocabulary(prompt_path)
    model = YOLOWorld(model_path)
    model.set_classes(vocabulary.prompts)
    with torch.inference_mode():
        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=100,
            verbose=False,
        )[0]

    candidates = []
    names = getattr(result, "names", None) or getattr(model, "names", {})
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, class_ids):
            prompt = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else str(class_id)
            candidate = vocabulary.candidate(box, score, prompt)
            if candidate is not None:
                candidates.append(candidate)
    detections = canonical_nms(candidates, args.canonical_nms_iou, 100)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, draw_detections(image.copy(), detections)):
        raise SystemExit("failed to write output: %s" % output_path)
    print(json.dumps({"output": output_path, "detections": detections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
