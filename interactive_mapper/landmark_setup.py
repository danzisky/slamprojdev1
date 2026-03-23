"""Interactive landmark placement for the interactive_mapper package."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2

if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    SLAM_ROVER_DIR = PACKAGE_DIR.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from interactive_mapper.map_manager import MapManager  # type: ignore
else:
    from .map_manager import MapManager


class LandmarkSetupSession:
    """Small interactive UI for placing and saving map landmarks."""

    def __init__(
        self,
        map_file: str | Path,
        landmarks_file: Optional[str | Path] = None,
        window_name: str = "Landmark Setup",
        resolution: Optional[float] = None
    ) -> None:
        self.map_manager = MapManager(map_file)
        self.window_name = window_name
        self.landmarks_file = Path(landmarks_file) if landmarks_file is not None else self._default_landmark_file()
        self.landmarks: list[tuple[int, int]] = []
        self.resolution = resolution
        self._window_initialized = False

        if self.landmarks_file.exists():
            self.load_landmarks(self.landmarks_file)

    def _default_landmark_file(self) -> Path:
        return self.map_manager.map_file.with_suffix(".landmarks.json")

    def load_landmarks(self, file_path: str | Path) -> list[tuple[int, int]]:
        """Load landmarks from JSON file if it exists."""
        path = Path(file_path)
        if not path.exists():
            self.landmarks = []
            return self.landmarks

        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload.get("landmarks", [])
        self.landmarks = [(int(item[0]), int(item[1])) for item in points]
        return self.landmarks

    def save_landmarks(self, file_path: Optional[str | Path] = None) -> Path:
        """Save landmarks to JSON in the same format used by localization scripts."""
        path = Path(file_path) if file_path is not None else self.landmarks_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "resolution": self.resolution,
            "map_size": [self.map_manager.grid_width, self.map_manager.grid_height],
            "landmarks": [[int(x), int(y)] for x, y in self.landmarks],
            "count": len(self.landmarks),
            "map_file": str(self.map_manager.map_file),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {len(self.landmarks)} landmarks to: {path}")
        return path

    def add_landmark(self, x: int, y: int) -> None:
        point = (int(x), int(y))
        if point not in self.landmarks:
            self.landmarks.append(point)
            print(f"Added landmark at {point}")

    def remove_last_landmark(self) -> None:
        if not self.landmarks:
            return
        removed = self.landmarks.pop()
        print(f"Removed landmark at {removed}")

    def _ensure_window(self) -> None:
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, self.map_manager.grid_width, self.map_manager.grid_height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._window_initialized = True

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if 0 <= x < self.map_manager.grid_width and 0 <= y < self.map_manager.grid_height:
            self.add_landmark(x, y)

    def draw(self) -> None:
        viz = cv2.cvtColor(self.map_manager.occupancy_grid, cv2.COLOR_GRAY2BGR)

        for index, (x, y) in enumerate(self.landmarks, start=1):
            cv2.circle(viz, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(viz, (x, y), 10, (0, 0, 180), 2)
            cv2.putText(viz, str(index), (x + 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.putText(viz, f"Landmarks: {len(self.landmarks)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(viz, "Left click: add", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz, "u: undo last", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz, "s: save", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(viz, "q/esc: quit", (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, viz)

    def run(self, wait_ms: int = 50) -> list[tuple[int, int]]:
        """Run interactive landmark setup loop."""
        self._ensure_window()

        print("=" * 50)
        print("LANDMARK SETUP MODE")
        print("=" * 50)
        print("Left click: add landmark")
        print("u: undo last landmark")
        print("s: save landmarks")
        print("q or esc: quit")
        print("=" * 50)
        print(f"Map: {self.map_manager.map_file}")
        print(f"Output JSON: {self.landmarks_file}")

        while True:
            self.draw()
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save_landmarks()
            if key == ord("u"):
                self.remove_last_landmark()

        if self._window_initialized:
            cv2.destroyWindow(self.window_name)
            self._window_initialized = False

        return list(self.landmarks)


def run_landmark_setup_session(
    map_file: str | Path,
    landmarks_file: Optional[str | Path] = None,
    window_name: str = "Landmark Setup",
    resolution: Optional[float] = None
) -> list[tuple[int, int]]:
    """Convenience API used by interactive_mapper.main."""
    session = LandmarkSetupSession(
        map_file=map_file,
        landmarks_file=landmarks_file,
        window_name=window_name,
        resolution=resolution
    )
    return session.run()


if __name__ == "__main__":
    from interactive_mapper.config import LANDMARKS_FILE, LANDMARK_SETUP_WINDOW_NAME, MAP_FILE, RESOLUTION_M_PER_PX # type: ignore

    run_landmark_setup_session(
        map_file=MAP_FILE,
        landmarks_file=LANDMARKS_FILE,
        window_name=LANDMARK_SETUP_WINDOW_NAME,
        resolution=RESOLUTION_M_PER_PX
    )