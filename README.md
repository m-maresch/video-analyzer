# video-analyzer

`video-analyzer` is a Python library for sampling frames from videos and
classifying them with an image-classification model. A C++
engine decodes and preprocesses videos concurrently, then
feeds sampled frames through batched model inference.

## Installation

Install Python, PyTorch, torchvision, CMake 3.18 or newer, a C++20 compiler,
OpenCV, and LibTorch. Make sure CMake can find Torch and OpenCV.

Clone the repository and install the package:

```
git clone https://github.com/m-maresch/video-analyzer
cd video-analyzer
pip install .
```

The `pip install .` command takes care of building the C++ engine and installs the Python library.

## Usage

Create or provide a model, then pass its path to `VideoAnalyzer`:

```python
from video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer(
    model_path="mobilenet_v3_small.pt",
    batch_size=16,
    target_size=(224, 224),
)
results = analyzer.analyze(
    ["video1.mp4", "video2.mp4"],
    sample_interval_secs=1.0,
)
```

`analyze` returns a dictionary keyed by absolute video paths. Each value is a
list of sampled-frame results containing the timestamp, frame index, and
predicted class index.

## Processing Pipeline

The C++ engine starts a producer thread for each input video. These threads
decode, sample, and preprocess frames concurrently, then enqueue them in a
bounded thread-safe queue. A consumer thread collects queued frames into
batches and runs model inference. This overlaps video decoding and
preprocessing across videos with batched inference.

## Example

`analyze_video.py` demonstrates the library with a pretrained MobileNetV3-Small
model. Run it with one or more video paths:

```
python analyze_video.py ./video1.mp4 ./video2.mp4
```

The example exports the pretrained model to
`mobilenet_v3_small.pt` if it does not exist. The first export may download
weights from torchvision. The script prints the timestamp, frame number, and ImageNet class label for each sampled frame.

### Performance Tuning

Use `--sample-interval` to change the time between sampled frames:

| Command | Behavior |
| :--- | :--- |
| `python analyze_video.py video.mp4` | Sample approximately one frame per second (default). |
| `python analyze_video.py --sample-interval 2.5 video.mp4` | Sample approximately one frame every 2.5 seconds. |

Larger intervals process fewer frames and reduce analysis time.

## License

This project is licensed under the MIT License. See the `LICENSE.txt` file for details. Third-party library notices are documented in `THIRD-PARTY-NOTICES.txt`.

## Acknowledgments

This project was developed with the assistance of Google Gemini 3 and GitHub Copilot.

## Dependencies

Thanks to everyone contributing to any of the following projects:

- OpenCV
- PyTorch and torchvision
- pybind11
- Scikit-build-core
