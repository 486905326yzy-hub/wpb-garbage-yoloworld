#!/usr/bin/env python3
import json
import os
import threading
import time
import warnings

import cv2
import numpy as np
import rospy
import torch
from garbage_yoloworld.core import (
    PromptVocabulary,
    canonical_nms,
    draw_detections,
    intersection_over_union,
)
from garbage_yoloworld.msg import GarbageDetection, GarbageDetectionArray
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated.*",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)


def resolve_local_file(path, description):
    expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(expanded):
        raise ValueError("%s does not exist: %s" % (description, expanded))
    return expanded


class GarbageYoloWorldNode:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.model_path = resolve_local_file(rospy.get_param("~model_path"), "model_path")
        prompt_path = resolve_local_file(rospy.get_param("~prompt_config"), "prompt_config")
        self.vocabulary = PromptVocabulary(prompt_path)
        self.device = str(rospy.get_param("~device", "cuda:0"))
        self.require_gpu = bool(rospy.get_param("~require_gpu", True))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.conf = float(rospy.get_param("~conf", 0.10))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.canonical_nms_iou = float(rospy.get_param("~canonical_nms_iou", 0.45))
        self.max_det = int(rospy.get_param("~max_det", 50))
        self.confirmation_hits = max(1, int(rospy.get_param("~confirmation_hits", 2)))
        self.confirmation_iou = float(rospy.get_param("~confirmation_iou", 0.30))
        self.confirmation_max_age = float(rospy.get_param("~confirmation_max_age", 0.75))
        self.process_every_n = max(1, int(rospy.get_param("~process_every_n", 3)))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.warmup_enabled = bool(rospy.get_param("~warmup", True))
        self.log_interval = float(rospy.get_param("~log_interval", 5.0))

        self.frame_count = 0
        self.success_count = 0
        self.error_count = 0
        self.last_log_time = 0.0
        self.paused = False
        self.pending_image = None
        self.tracks = []
        self.condition = threading.Condition()
        self.stop_event = threading.Event()

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            message = "CUDA requested but torch.cuda.is_available() is false"
            if self.require_gpu:
                raise RuntimeError(message)
            rospy.logwarn("%s; falling back to CPU", message)
            self.device = "cpu"
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        from ultralytics import YOLOWorld

        rospy.loginfo("Loading existing YOLO-World model (no download): %s", self.model_path)
        self.model = YOLOWorld(self.model_path)
        self.model.set_classes(self.vocabulary.prompts)
        rospy.loginfo(
            "Configured %d open-vocabulary prompts from %s",
            len(self.vocabulary.prompts),
            self.vocabulary.config_path,
        )
        self.warmup()

        self.det_pub = rospy.Publisher("~detections", GarbageDetectionArray, queue_size=1)
        self.details_pub = rospy.Publisher("~details", String, queue_size=1)
        self.has_garbage_pub = rospy.Publisher("~has_garbage", Bool, queue_size=1)
        self.debug_pub = (
            rospy.Publisher("~debug_image", Image, queue_size=1)
            if self.publish_debug
            else None
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.pause_sub = rospy.Subscriber("~pause", Bool, self.pause_callback, queue_size=1)
        self.worker = threading.Thread(target=self.inference_loop, name="garbage-yoloworld", daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "Independent garbage detector ready: image=%s device=%s process_every_n=%d",
            self.image_topic,
            self.device,
            self.process_every_n,
        )

    def warmup(self):
        if not self.warmup_enabled:
            return
        image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        start = time.time()
        with torch.inference_mode():
            self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )
        rospy.loginfo("Garbage YOLO-World warm-up complete in %.2fs", time.time() - start)

    def pause_callback(self, message):
        with self.condition:
            self.paused = bool(message.data)
            if self.paused:
                self.pending_image = None
            self.condition.notify_all()
        rospy.loginfo("Garbage inference %s", "paused" if self.paused else "resumed")

    def image_callback(self, message):
        with self.condition:
            if self.paused:
                return
            self.frame_count += 1
            if self.frame_count % self.process_every_n:
                return
            self.pending_image = (message, time.time())
            self.condition.notify()

    def inference_loop(self):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.condition:
                while self.pending_image is None and not self.stop_event.is_set() and not rospy.is_shutdown():
                    self.condition.wait(timeout=0.5)
                if self.stop_event.is_set() or rospy.is_shutdown():
                    return
                pending = self.pending_image
                self.pending_image = None
                if self.paused:
                    continue
            try:
                self.process_image(*pending)
            except Exception as exc:
                self.error_count += 1
                rospy.logerr_throttle(2.0, "Garbage YOLO-World recovered from error: %s", exc)

    def process_image(self, message, received_at):
        image = self.image_to_bgr(message)
        start = time.time()
        with torch.inference_mode():
            result = self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )[0]

        candidates = []
        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, scores, class_ids):
                prompt = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else str(class_id)
                candidate = self.vocabulary.candidate(box, score, prompt)
                if candidate is None:
                    continue
                self.clip_bbox(candidate, image.shape[1], image.shape[0])
                candidates.append(candidate)
        raw_detections = canonical_nms(candidates, self.canonical_nms_iou, self.max_det)
        detections = self.confirm_detections(raw_detections, time.time())

        output = GarbageDetectionArray()
        output.header = message.header
        output.image_width = image.shape[1]
        output.image_height = image.shape[0]
        for item in detections:
            detection = GarbageDetection()
            detection.class_name = item["class_name"]
            detection.class_id = item["class_id"]
            detection.subtype = item["subtype"]
            detection.prompt = item["prompt"]
            detection.score = item["score"]
            detection.xmin, detection.ymin, detection.xmax, detection.ymax = item["bbox"]
            detection.center_x = int((detection.xmin + detection.xmax) / 2)
            detection.center_y = int((detection.ymin + detection.ymax) / 2)
            output.detections.append(detection)
        self.det_pub.publish(output)
        self.has_garbage_pub.publish(Bool(data=bool(detections)))

        details = {
            "stamp": {"secs": message.header.stamp.secs, "nsecs": message.header.stamp.nsecs},
            "frame_id": message.header.frame_id,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "detections": detections,
        }
        self.details_pub.publish(String(data=json.dumps(details, ensure_ascii=False, separators=(",", ":"))))
        if self.debug_pub is not None:
            debug = draw_detections(image.copy(), detections)
            self.debug_pub.publish(self.bgr_to_image(debug, message))

        now = time.time()
        self.success_count += 1
        if now - self.last_log_time >= self.log_interval:
            rospy.loginfo(
                "Garbage detections=%d inference=%.1fms input_age=%.2fs successes=%d errors=%d",
                len(detections),
                (now - start) * 1000.0,
                now - received_at,
                self.success_count,
                self.error_count,
            )
            self.last_log_time = now

    def confirm_detections(self, detections, now):
        active = [
            track
            for track in self.tracks
            if now - track["last_seen"] <= self.confirmation_max_age
        ]
        used = set()
        updated = []
        confirmed = []
        for detection in detections:
            best_index = None
            best_iou = self.confirmation_iou
            for index, track in enumerate(active):
                if index in used:
                    continue
                overlap = intersection_over_union(detection, track["detection"])
                if overlap >= best_iou:
                    best_index = index
                    best_iou = overlap
            hits = 1
            if best_index is not None:
                used.add(best_index)
                hits = active[best_index]["hits"] + 1
            track = {
                "detection": detection,
                "hits": hits,
                "last_seen": now,
            }
            updated.append(track)
            if hits >= self.confirmation_hits:
                published = dict(detection)
                published["track_hits"] = hits
                confirmed.append(published)
        updated.extend(track for index, track in enumerate(active) if index not in used)
        self.tracks = updated
        return confirmed

    @staticmethod
    def clip_bbox(candidate, width, height):
        x1, y1, x2, y2 = candidate["bbox"]
        candidate["bbox"] = [
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(0, min(x2, width - 1)),
            max(0, min(y2, height - 1)),
        ]

    @staticmethod
    def image_to_bgr(message):
        encoding = message.encoding.lower()
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
        if channels is None:
            raise ValueError("unsupported image encoding: %s" % message.encoding)
        row_bytes = message.width * channels
        if message.step < row_bytes:
            raise ValueError("invalid image step")
        raw = np.frombuffer(message.data, dtype=np.uint8)
        needed = message.height * message.step
        if raw.size < needed:
            raise ValueError("truncated image data")
        image = raw[:needed].reshape(message.height, message.step)[:, :row_bytes]
        if channels == 1:
            return cv2.cvtColor(image.reshape(message.height, message.width), cv2.COLOR_GRAY2BGR)
        image = image.reshape(message.height, message.width, channels)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image.copy()

    @staticmethod
    def bgr_to_image(image, source):
        output = Image()
        output.header = source.header
        output.height, output.width = image.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = image.tobytes()
        return output

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.pending_image = None
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)


def main():
    rospy.init_node("garbage_yoloworld")
    node = GarbageYoloWorldNode()
    rospy.spin()
    node.close()


if __name__ == "__main__":
    main()
