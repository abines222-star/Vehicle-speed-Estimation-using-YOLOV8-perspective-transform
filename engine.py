"""Background detection, tracking, speed estimation, and output recording."""
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from speed import SpeedEstimator, video_source


class TrafficEngine:
    CSV_FIELDS = ["id", "class", "max_speed", "avg_speed", "time_of_max_speed", "direction"]

    def __init__(self):
        self.lock = threading.RLock()
        self.jpeg = None
        self.vehicles = {}
        self.events = []
        self.fastest_id = None
        self.running = False
        self.error = None
        self.video_time = 0.0
        self.processing_fps = 0.0
        self.cached_result = False
        self.thread = None
        self._last_csv_write = 0.0
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def start(self, force_process=False):
        if self.thread and self.thread.is_alive():
            return False
        if not force_process and self._load_saved_result():
            return False
        self.running = True
        self.thread = threading.Thread(target=self.run, name="traffic-processor", daemon=True)
        self.thread.start()
        return True

    def _input_signature(self):
        source = video_source(config.VIDEO_PATH)
        if isinstance(source, int):
            return None
        path = Path(source)
        if not path.is_file():
            return None
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}

    def _load_saved_result(self):
        signature = self._input_signature()
        if signature is None or not config.OUTPUT_VIDEO.is_file() or not config.VEHICLES_LOG.is_file():
            return False

        source_capture = cv2.VideoCapture(signature["path"])
        result_capture = cv2.VideoCapture(str(config.OUTPUT_VIDEO))
        source_frames = source_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        result_frames = result_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        source_capture.release()
        result_fps = result_capture.get(cv2.CAP_PROP_FPS)
        result_capture.release()
        if source_frames < 1 or result_fps <= 0 or abs(source_frames - result_frames) > 1:
            return False

        metadata = None
        if config.RESULT_METADATA.is_file():
            try:
                metadata = json.loads(config.RESULT_METADATA.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if metadata.get("input") != signature:
                return False

        if metadata:
            records = metadata.get("vehicles", [])
            self.events = metadata.get("events", [])
            self.video_time = float(metadata.get("video_time", result_frames / result_fps))
        else:
            records = []
            with config.VEHICLES_LOG.open("r", newline="", encoding="utf-8") as input_file:
                for row in csv.DictReader(input_file):
                    average = float(row["avg_speed"] or 0)
                    records.append({
                        "id": int(row["id"]),
                        "class": row["class"],
                        "current_speed": None,
                        "max_speed": float(row["max_speed"] or 0),
                        "avg_speed": average,
                        "speed_total": average,
                        "speed_samples": int(average > 0),
                        "time_of_max_speed": row["time_of_max_speed"] or "--:--:--",
                        "video_time_of_max_speed": 0.0,
                        "first_seen_time": "--:--:--",
                        "last_seen_time": row["time_of_max_speed"] or "--:--:--",
                        "last_video_time": 0.0,
                        "spot": None,
                        "direction": row["direction"] or "UNKNOWN",
                        "_overspeed_active": False,
                    })
            self.events = []
            self.video_time = result_frames / result_fps

        self.vehicles = {int(record["id"]): record for record in records}
        self.fastest_id = max(
            self.vehicles,
            key=lambda track_id: self.vehicles[track_id]["max_speed"],
            default=None,
        )
        self.running = False
        self.cached_result = True
        if metadata is None:
            self._write_result_metadata()
        return True

    def run(self):
        capture = None
        writer = None
        completed = False
        try:
            model_path = config.MODEL_PATH
            if not model_path.is_absolute():
                model_path = config.PROJECT_ROOT / model_path
            model = YOLO(str(model_path))
            capture = cv2.VideoCapture(video_source(config.VIDEO_PATH))
            if not capture.isOpened():
                raise RuntimeError("Could not open video input: {}".format(config.VIDEO_PATH))

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            estimator = SpeedEstimator(fps, (width, height))
            writer = cv2.VideoWriter(
                str(config.OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                writer.release()
                writer = None
                raise RuntimeError("Could not create output video: {}".format(config.OUTPUT_VIDEO))

            frame_no = 0
            last_boxes = []
            started_at = time.monotonic()
            while True:
                frame_started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_no += 1
                detect_this_frame = (frame_no - 1) % max(1, config.DETECTION_STRIDE) == 0
                if detect_this_frame:
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml", classes=list(config.CLASSES),
                        conf=config.CONF, imgsz=config.INFERENCE_SIZE, verbose=False,
                    )[0]
                    with self.lock:
                        for record in self.vehicles.values():
                            record["current_speed"] = None
                    boxes_to_draw = []
                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                            track_id = int(track_id)
                            class_id = int(class_id)
                            if class_id not in config.CLASSES:
                                continue
                            x1, y1, x2, y2 = map(int, box)
                            contact_x, contact_y = (x1 + x2) // 2, y2
                            current_speed = estimator.update(track_id, frame_no, contact_x, contact_y)
                            road_spot = estimator.project(contact_x, contact_y)
                            wall_now = datetime.now().strftime("%H:%M:%S")
                            video_time = frame_no / fps
                            with self.lock:
                                record = self.vehicles.get(track_id)
                                if record is None:
                                    record = {
                                        "id": track_id,
                                        "class": config.CLASSES[class_id],
                                        "current_speed": None,
                                        "max_speed": 0.0,
                                        "avg_speed": 0.0,
                                        "speed_total": 0.0,
                                        "speed_samples": 0,
                                        "time_of_max_speed": "--:--:--",
                                        "video_time_of_max_speed": 0.0,
                                        "first_seen_time": wall_now,
                                        "last_seen_time": wall_now,
                                        "last_video_time": round(video_time, 2),
                                        "spot": None,
                                        "direction": "UNKNOWN",
                                        "_overspeed_active": False,
                                    }
                                    self.vehicles[track_id] = record
                                old_direction = record["direction"]
                                record["class"] = config.CLASSES[class_id]
                                record["last_seen_time"] = wall_now
                                record["last_video_time"] = round(video_time, 2)
                                record["spot"] = {
                                    "pixel_x": contact_x,
                                    "pixel_y": contact_y,
                                    **(road_spot or {}),
                                }
                                record["direction"] = estimator.direction(track_id)
                                if record["first_seen_time"] == wall_now and record["last_video_time"] == round(video_time, 2):
                                    if not any(event["id"] == track_id and event["event"] == "detected" for event in self.events):
                                        self._add_event(record, "detected", wall_now, video_time)
                                if record["direction"] in ("UP", "DOWN") and old_direction not in ("UNKNOWN", record["direction"]):
                                    self._add_event(record, "direction_change", wall_now, video_time)
                                if current_speed is not None:
                                    record["current_speed"] = round(current_speed, 1)
                                    record["speed_total"] += current_speed
                                    record["speed_samples"] += 1
                                    record["avg_speed"] = round(
                                        record["speed_total"] / record["speed_samples"], 1
                                    )
                                    if current_speed > record["max_speed"]:
                                        record["max_speed"] = round(current_speed, 1)
                                        record["time_of_max_speed"] = wall_now
                                        record["video_time_of_max_speed"] = round(video_time, 2)
                                    is_overspeed = current_speed > config.SPEED_LIMIT
                                    if is_overspeed and not record["_overspeed_active"]:
                                        self._add_event(record, "overspeed", wall_now, video_time)
                                    record["_overspeed_active"] = is_overspeed
                                else:
                                    record["_overspeed_active"] = False
                            boxes_to_draw.append((track_id, config.CLASSES[class_id], (x1, y1, x2, y2)))
                    last_boxes = boxes_to_draw
                else:
                    boxes_to_draw = last_boxes

                with self.lock:
                    if detect_this_frame:
                        self.fastest_id = max(
                            self.vehicles,
                            key=lambda item: self.vehicles[item]["max_speed"],
                            default=None,
                        )
                    fastest = dict(self.vehicles[self.fastest_id]) if self.fastest_id is not None else None
                    self.video_time = frame_no / fps
                    self.processing_fps = round(frame_no / max(time.monotonic() - started_at, 0.001), 1)

                for track_id, class_name, (x1, y1, x2, y2) in boxes_to_draw:
                    with self.lock:
                        record = dict(self.vehicles[track_id])
                        fastest_id = self.fastest_id
                    color = (0, 0, 255) if track_id == fastest_id else (40, 210, 120)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    speed_text = "--.-" if record["current_speed"] is None else "{:.1f}".format(record["current_speed"])
                    label = "{} #{} {} km/h".format(class_name, track_id, speed_text)
                    cv2.putText(frame, label, (x1, max(y1 - 9, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2, cv2.LINE_AA)

                self._draw_hud(frame, fastest)
                writer.write(frame)
                jpeg_ok, encoded = cv2.imencode(".jpg", frame)
                if jpeg_ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                if time.monotonic() - self._last_csv_write >= 30:
                    self.write_csv()
                delay = (1.0 / fps) - (time.monotonic() - frame_started)
                if delay > 0:
                    time.sleep(delay)
            self.video_time = frame_no / fps
            completed = True
        except Exception as error:
            with self.lock:
                self.error = str(error)
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            self.write_csv()
            if completed and self.error is None:
                self._write_result_metadata()
            with self.lock:
                self.running = False

    def _visible_vehicle_rows(self, rows):
        return [row for row in rows if row.get("class") != "bike"]

    def _visible_events(self, events):
        return [event for event in events if event.get("event") != "overspeed"]

    def _write_result_metadata(self):
        signature = self._input_signature()
        if signature is None:
            return
        with self.lock:
            vehicles = self._visible_vehicle_rows([dict(record) for record in self.vehicles.values()])
            events = self._visible_events([dict(event) for event in self.events])
            video_time = self.video_time
        metadata = {"input": signature, "vehicles": vehicles, "events": events, "video_time": video_time}
        temporary_path = config.RESULT_METADATA.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(metadata), encoding="utf-8")
        temporary_path.replace(config.RESULT_METADATA)

    def cached_video_stream(self):
        while True:
            capture = cv2.VideoCapture(str(config.OUTPUT_VIDEO))
            if not capture.isOpened():
                capture.release()
                return
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        return
                    encoded_ok, encoded = cv2.imencode(".jpg", frame)
                    if encoded_ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                    time.sleep(1.0 / fps)
            finally:
                capture.release()

    def _draw_hud(self, frame, fastest):
        overlay = frame.copy()
        cv2.rectangle(overlay, (14, 14), (410, 126), (13, 22, 30), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        if fastest:
            video_seconds = int(fastest["video_time_of_max_speed"])
            video_clock = "{:02d}:{:02d}".format(video_seconds // 60, video_seconds % 60)
            first_line = "FASTEST: {} ID {}".format(fastest["class"].upper(), fastest["id"])
            second_line = "SPEED: {:.1f} km/h".format(fastest["max_speed"])
            third_line = "TIME: {}   VIDEO: {}".format(fastest["time_of_max_speed"], video_clock)
        else:
            first_line, second_line = "FASTEST: --", "SPEED: --.- km/h"
            third_line = "TIME: --:--:--   VIDEO: --:--"
        lines = [
            (first_line, 44, 0.64, (110, 235, 190)),
            (second_line, 76, 0.68, (245, 248, 248)),
            (third_line, 105, 0.48, (195, 209, 213)),
        ]
        for text, y, scale, color in lines:
            cv2.putText(frame, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _add_event(self, record, event_type, wall_time, video_time):
        spot = record.get("spot") or {}
        self.events.insert(0, {
            "event": event_type,
            "id": int(record["id"]),
            "class": record["class"],
            "direction": record["direction"],
            "speed": record["current_speed"],
            "wall_time": wall_time,
            "video_time": round(video_time, 2),
            "spot": dict(spot),
        })
        del self.events[100:]

    def write_csv(self):
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with self.lock:
            rows = self._visible_vehicle_rows([dict(record) for record in self.vehicles.values()])
        rows.sort(key=lambda item: item["max_speed"], reverse=True)
        temporary_path = config.VEHICLES_LOG.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "id": row["id"], "class": row["class"], "max_speed": row["max_speed"],
                    "avg_speed": row["avg_speed"], "time_of_max_speed": row["time_of_max_speed"],
                    "direction": row["direction"],
                })
        temporary_path.replace(config.VEHICLES_LOG)
        self._last_csv_write = time.monotonic()

    def vehicles_snapshot(self):
        with self.lock:
            rows = self._visible_vehicle_rows([dict(record) for record in self.vehicles.values()])
            fastest_id = None
            if self.fastest_id is not None:
                fastest_record = self.vehicles.get(self.fastest_id)
                if fastest_record is not None and fastest_record.get("class") != "bike":
                    fastest_id = self.fastest_id
            if fastest_id is None:
                visible_ids = [track_id for track_id, record in self.vehicles.items() if record.get("class") != "bike"]
                fastest_id = max(visible_ids, key=lambda item: self.vehicles[item]["max_speed"], default=None)
            fastest = dict(self.vehicles[fastest_id]) if fastest_id is not None else None
            events = self._visible_events([dict(event) for event in self.events])
            running, error, video_time = self.running, self.error, self.video_time
            processing_fps = self.processing_fps
        rows.sort(key=lambda item: item["max_speed"], reverse=True)
        valid_averages = [row["avg_speed"] for row in rows if row["speed_samples"]]
        summary = {
            "total_vehicles": len(rows),
            "average_speed": round(sum(valid_averages) / len(valid_averages), 1) if valid_averages else 0.0,
        }
        directional_fastest = {
            direction: next((dict(row) for row in rows if row["direction"] == direction), None)
            for direction in ("UP", "DOWN")
        }
        for row in rows:
            row.pop("speed_total", None)
            row.pop("speed_samples", None)
            row.pop("_overspeed_active", None)
        return {"vehicles": rows, "fastest": fastest, "summary": summary,
            "running": running, "error": error, "video_time": video_time,
            "speed_limit": config.SPEED_LIMIT, "directional_fastest": directional_fastest,
            "events": events, "active_vehicles": sum(row["current_speed"] is not None for row in rows),
            "processing_fps": processing_fps, "detection_stride": config.DETECTION_STRIDE,
            "cached_result": self.cached_result}

    def fastest_snapshot(self):
        with self.lock:
            visible_ids = [track_id for track_id, record in self.vehicles.items() if record.get("class") != "bike"]
            if not visible_ids:
                return None
            fastest_id = max(visible_ids, key=lambda item: self.vehicles[item]["max_speed"], default=None)
            if fastest_id is None:
                return None
            return dict(self.vehicles[fastest_id])
