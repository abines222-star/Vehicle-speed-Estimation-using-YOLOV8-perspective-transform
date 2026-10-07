import unittest

from engine import TrafficEngine


class DashboardFilterTests(unittest.TestCase):
    def test_bikes_and_overspeed_are_filtered_from_snapshot(self):
        engine = TrafficEngine()
        engine.vehicles = {
            1: {
                "id": 1,
                "class": "car",
                "current_speed": 48.0,
                "max_speed": 48.0,
                "avg_speed": 48.0,
                "speed_total": 48.0,
                "speed_samples": 1,
                "time_of_max_speed": "12:00:00",
                "video_time_of_max_speed": 10.0,
                "first_seen_time": "12:00:00",
                "last_seen_time": "12:00:00",
                "last_video_time": 10.0,
                "spot": None,
                "direction": "UP",
                "_overspeed_active": False,
            },
            2: {
                "id": 2,
                "class": "bike",
                "current_speed": 31.0,
                "max_speed": 31.0,
                "avg_speed": 31.0,
                "speed_total": 31.0,
                "speed_samples": 1,
                "time_of_max_speed": "12:00:10",
                "video_time_of_max_speed": 20.0,
                "first_seen_time": "12:00:10",
                "last_seen_time": "12:00:10",
                "last_video_time": 20.0,
                "spot": None,
                "direction": "DOWN",
                "_overspeed_active": False,
            },
        }
        engine.events = [
            {
                "event": "overspeed",
                "id": 1,
                "class": "car",
                "direction": "UP",
                "speed": 80.0,
                "wall_time": "12:00:15",
                "video_time": 25.0,
                "spot": {},
            },
            {
                "event": "detected",
                "id": 3,
                "class": "car",
                "direction": "DOWN",
                "speed": 40.0,
                "wall_time": "12:00:20",
                "video_time": 30.0,
                "spot": {},
            },
        ]
        engine.fastest_id = 1

        snapshot = engine.vehicles_snapshot()

        self.assertEqual([vehicle["class"] for vehicle in snapshot["vehicles"]], ["car"])
        self.assertEqual(snapshot["summary"]["bike_count"], 0)
        self.assertEqual(snapshot["summary"]["overspeed_count"], 0)
        self.assertFalse(any(event["event"] == "overspeed" for event in snapshot["events"]))


if __name__ == "__main__":
    unittest.main()
