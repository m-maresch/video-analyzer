import argparse
import os

import torch
import torchvision.models as models

from video_analyzer import VideoAnalyzer


def export_model(path="mobilenet_v3_small.pt"):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    class_names = weights.meta["categories"]

    if not os.path.exists(path):
        print("Exporting MobileNetV3-Small...")
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        example_input = torch.randn(1, 3, 224, 224)
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, path)

    return path, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Video analytics engine")

    parser.add_argument(
        "video_paths",
        nargs="+",
        help="Path(s) to the input video file(s) to analyze.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Interval in seconds between sampled audio/video frames.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_file, class_names = export_model()

    analyzer = VideoAnalyzer(
        model_path=model_file,
        batch_size=16,
        target_size=(224, 224),
        use_cuda=torch.cuda.is_available(),
    )

    videos = args.video_paths
    print(f"Analyzing {len(videos)} video(s)...")

    results = analyzer.analyze(videos, sample_interval_secs=args.sample_interval)

    for video_path, frames in results.items():
        print(f"\nResults for: {video_path}")
        for entry in frames:
            print(
                f"  Time: {entry['timestamp']:.1f}s | Frame: {entry['frame_index']} | Class: {class_names[entry['predicted_class']]}"
            )


if __name__ == "__main__":
    main()
