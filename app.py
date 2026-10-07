"""Flask dashboard, cached-result viewer, and MJPEG stream."""
import argparse
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template

import config
from engine import TrafficEngine

app = Flask(__name__, template_folder=str(config.PROJECT_ROOT / "templates"))
engine = TrafficEngine()

@app.route("/")
def index():
    return render_template("index.html")

def mjpeg():
    if engine.cached_result:
        yield from engine.cached_video_stream()
        return
    while True:
        with engine.lock:
            frame = engine.jpeg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/vehicles")
def vehicles():
    return jsonify(engine.vehicles_snapshot())


@app.route("/api/fastest")
def fastest():
    return jsonify(engine.fastest_snapshot())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Speed Tracker dashboard")
    parser.add_argument("--process", action="store_true", help="reprocess the configured source video")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    arguments = parser.parse_args()
    engine.start(force_process=arguments.process)
    if not arguments.no_browser:
        threading.Timer(1.0, webbrowser.open_new_tab, ["http://127.0.0.1:{}".format(config.PORT)]).start()
    app.run(host="0.0.0.0", port=config.PORT, threaded=True, use_reloader=False)
