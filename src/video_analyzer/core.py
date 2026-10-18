import os
from typing import List, Dict, Union

try:
    import _video_analyzer_engine

    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class VideoAnalyzer:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        target_size: tuple = (224, 224),
        use_cuda: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.batch_size = batch_size
        self.target_width, self.target_height = target_size
        self.use_cuda = use_cuda

        if not ENGINE_AVAILABLE:
            raise RuntimeError(
                "C++ extension '_video_analyzer_engine' is not compiled or installed."
            )

    def analyze(
        self, video_paths: Union[str, List[str]], sample_interval_secs: float = 1.0
    ) -> Dict[str, List[Dict]]:
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        resolved_paths = []
        for path in video_paths:
            if os.path.exists(path):
                resolved_paths.append(os.path.abspath(path))
            else:
                raise FileNotFoundError(f"Video not found at {path}")

        if not resolved_paths:
            raise ValueError("No valid video paths provided.")

        analysis_results = _video_analyzer_engine.run_pipeline(
            video_paths=resolved_paths,
            model_path=self.model_path,
            sample_interval_secs=sample_interval_secs,
            batch_size=self.batch_size,
            target_width=self.target_width,
            target_height=self.target_height,
            use_cuda=self.use_cuda,
        )

        grouped_results = {path: [] for path in resolved_paths}
        for result in analysis_results:
            if result.video_path in grouped_results:
                grouped_results[result.video_path].append(
                    {
                        "timestamp": result.timestamp,
                        "frame_index": result.frame_index,
                        "predicted_class": result.predicted_class,
                    }
                )

        return grouped_results
