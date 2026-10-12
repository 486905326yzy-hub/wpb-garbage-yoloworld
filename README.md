# ROS YOLO-World Garbage Detector

An independent ROS Noetic experiment that uses YOLO-World open-vocabulary
prompts to detect the eight garbage types used in a 2026 service-robot task.
It was separated from the robot's owner-search application so it can be tested,
tuned, or stopped without starting navigation, voice, or manipulation nodes.

中文简介：这是一个独立的 ROS Noetic 垃圾识别节点。它用 YOLO-World
运行时文本提示识别赛项物品，不修改或启动机器人原有业务程序。

## What it detects

The default vocabulary in `config/garbage_prompts.yaml` covers:

- potato-chips can/tube
- cookie box/package
- iced-tea bottle
- cola/soda bottle
- shampoo bottle
- apple
- banana
- crumpled paper ball

Several English prompt aliases are assigned to each subtype. All accepted
detections are published with the canonical class `garbage`; the winning
subtype and prompt are preserved in the structured and JSON outputs.

This approach does not learn the white background of a product sheet: the
checkpoint receives text prompts at runtime. That improves portability, but it
does not guarantee performance on every floor. Real camera images from the
competition environment are still required for threshold and prompt tuning.

## Design

```text
existing ROS camera -> latest-frame worker -> YOLO-World + text prompts
                    -> alias NMS -> two-frame confirmation
                    -> detections / JSON / boolean / debug image
```

- Uses the newest pending frame instead of building an inference backlog.
- Applies per-subtype confidence thresholds.
- Collapses overlapping aliases with a second NMS pass.
- Requires two spatially consistent hits by default to suppress one-frame noise.
- Does not start a camera, robot base, navigation stack, or other task nodes.

## Requirements

- Ubuntu 20.04 and ROS Noetic (the tested platform)
- Python 3 with `numpy`, `PyYAML`, and OpenCV
- PyTorch
- an Ultralytics release that provides `ultralytics.YOLOWorld`
- a compatible YOLO-World `.pt` checkpoint stored locally

Model weights are intentionally not included in this repository. This avoids
redistributing a large third-party checkpoint and prevents an existing local
model from being overwritten.

## Build

Clone this repository as a catkin package:

```bash
cd ~/catkin_ws/src
git clone https://github.com/486905326yzy-hub/wpb-garbage-yoloworld.git garbage_yoloworld
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Install Python dependencies in the same interpreter used by ROS:

```bash
python3 -m pip install numpy PyYAML opencv-python torch ultralytics
```

Choose PyTorch/CUDA packages that match the host NVIDIA driver; the generic
command above is only illustrative.

## Run

Start the camera driver separately, then pass an absolute model path:

```bash
roslaunch garbage_yoloworld garbage_yoloworld.launch \
  model_path:=/absolute/path/to/yolov8s-world.pt \
  image_topic:=/kinect2/qhd/image_color_rect \
  device:=cuda:0
```

Visualize the annotated stream:

```bash
rqt_image_view /garbage_detection/debug_image
```

Pause or resume inference without stopping the node:

```bash
rostopic pub -1 /garbage_detection/pause std_msgs/Bool 'data: true'
rostopic pub -1 /garbage_detection/pause std_msgs/Bool 'data: false'
```

## Outputs

- `/garbage_detection` — `garbage_yoloworld/GarbageDetectionArray`
- `/garbage_detection/details` — `std_msgs/String` containing JSON
- `/garbage_detection/has_garbage` — `std_msgs/Bool`
- `/garbage_detection/debug_image` — annotated `sensor_msgs/Image`
- `/garbage_detection/pause` — optional `std_msgs/Bool` input

Each structured detection contains `class_name`, `class_id`, `subtype`,
`prompt`, score, bounding box, and image-space center.

## Offline image test

```bash
rosrun garbage_yoloworld garbage_yoloworld_image.py \
  --image /path/to/test-scene.jpg \
  --model /absolute/path/to/yolov8s-world.pt \
  --prompts "$(rospack find garbage_yoloworld)/config/garbage_prompts.yaml" \
  --output /tmp/garbage-debug.jpg
```

The command prints JSON detections and writes an annotated image.

## Tuning and current validation

Edit `config/garbage_prompts.yaml` to add aliases or adjust
