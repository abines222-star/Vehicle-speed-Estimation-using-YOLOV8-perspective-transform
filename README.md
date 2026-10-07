# Traffic Speed Tracker

Live vehicle detection and ByteTrack tracking with perspective-based speed estimates, a fastest-vehicle HUD, a Flask dashboard, CSV results, and an annotated MP4 recording.

## Install and run

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The default input is `DATA/INPUTS/sample3.mp4`; set `VIDEO_PATH = 0` in `config.py` to use a webcam. YOLOv8n weights download automatically the first time the detector starts.

The first run processes the input and saves the annotated video and vehicle results. Later runs reuse matching completed results and open the dashboard with the saved video playing; they do not run detection again. Use `python app.py --process` to explicitly process the source again, or `python app.py --no-browser` to keep the browser closed.

For CPU throughput, inference defaults to 416 pixels and samples every third frame (`INFERENCE_SIZE` and `DETECTION_STRIDE` in `config.py`). Speed uses original video frame numbers, while the annotated output still writes every frame. Lowering the inference size or increasing the stride can improve throughput further, but may reduce detection detail or track continuity.

## Calibration

Run `python calibrate.py` and click four corners of a road rectangle in this order: top-left, top-right, bottom-right, bottom-left. Copy the printed points into `SOURCE_POINTS` in `config.py`, and measure that rectangle's real width and length for `REAL_WIDTH_M` and `REAL_LENGTH_M`. The bundled sample has a full-frame starter rectangle sized 7 m by 20 m so the project runs immediately; replace it with measured road-only points and dimensions for meaningful speed accuracy.

Speed estimation maps each vehicle's bottom-center road contact point from image pixels to meters using a homography computed from the four calibration corners. Distance between track samples divided by elapsed video time gives meters per second, converted to km/h and smoothed. Perspective and scale errors in the selected points or real-world dimensions directly affect every speed estimate, so calibration accuracy determines speed accuracy.

## Outputs

- `DATA/OUTPUTS/vehicles_log.csv`: per-vehicle final statistics, rewritten every 30 seconds and when processing ends.
- `DATA/OUTPUTS/output.mp4`: annotated video with bounding boxes and the fastest-vehicle HUD.

The dashboard keeps the final leaderboard available after the video ends. Ultralytics YOLOv8 is licensed under AGPL-3.0.
