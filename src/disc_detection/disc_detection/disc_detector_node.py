#!/usr/bin/env python3
"""
disc_detector_node.py
---------------------
Detects a white disc on the ground using FastSAM (color only).
Depth-based 3D projection is stubbed out and ready for a future PR.

Publishes:
  /disc/mask          (sensor_msgs/Image)      — binary mask of the disc
  /disc/viz           (sensor_msgs/Image)      — RGB frame with overlay
  /disc/centroid_2d   (geometry_msgs/Point)    — pixel centroid (u, v, 0)
  /disc/detected      (std_msgs/Bool)          — detection flag for nav

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from ultralytics import FastSAM


# ── Colour filter for "white" in HSV space ────────────────────────────────────
WHITE_HSV_LOW  = np.array([0,   0, 180], dtype=np.uint8)
WHITE_HSV_HIGH = np.array([180, 45, 255], dtype=np.uint8)

# ── Shape filter ──────────────────────────────────────────────────────────────
MIN_CIRCULARITY  = 0.65   # 1.0 = perfect circle
MIN_MASK_AREA_PX = 500    # ignore tiny detections
MAX_MASK_AREA_PX = 80_000 # ignore detections that are basically the whole frame


class DiscDetectorNode(Node):

    def __init__(self):
        super().__init__("disc_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("model",       "FastSAM-s.pt")
        self.declare_parameter("imgsz",        640)
        self.declare_parameter("conf",         0.4)
        self.declare_parameter("iou",          0.9)
        self.declare_parameter("device",       "cpu")   # or "cuda:0"
        self.declare_parameter("fps_limit",    10.0)    # max inference Hz
        self.declare_parameter("input_topic",  "/camera/color/image_raw")

        model_path = self.get_parameter("model").value
        self.imgsz  = self.get_parameter("imgsz").value
        self.conf   = self.get_parameter("conf").value
        self.iou    = self.get_parameter("iou").value
        self.device = self.get_parameter("device").value
        fps_limit   = self.get_parameter("fps_limit").value
        input_topic = self.get_parameter("input_topic").value

        self._min_dt = 1.0 / fps_limit          # seconds between inferences
        self._last_inference_time = 0.0

        # ── Model ─────────────────────────────────────────────────────────────
        self.get_logger().info(f"Loading FastSAM from: {model_path}")
        self.model = FastSAM(model_path)
        self.get_logger().info("FastSAM ready.")

        # ── Bridge ────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── QoS — sensor best-effort matches RealSense defaults ───────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_color = self.create_subscription(
            Image, input_topic, self._color_cb, sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_mask       = self.create_publisher(Image, "/disc/mask",        10)
        self.pub_viz        = self.create_publisher(Image, "/disc/viz",         10)
        self.pub_centroid   = self.create_publisher(Point, "/disc/centroid_2d", 10)
        self.pub_detected   = self.create_publisher(Bool,  "/disc/detected",    10)

        self.get_logger().info(
            f"Subscribed to {input_topic} | Inference cap: {fps_limit:.1f} Hz"
        )

    # ── Callback ──────────────────────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._last_inference_time) < self._min_dt:
            return                          # throttle to fps_limit
        self._last_inference_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        mask, centroid = self._detect(frame)

        detected = centroid is not None

        # ── Publish detected flag ─────────────────────────────────────────────
        self.pub_detected.publish(Bool(data=detected))

        # ── Publish mask ──────────────────────────────────────────────────────
        mask_img = (mask * 255).astype(np.uint8) if detected else \
                   np.zeros(frame.shape[:2], dtype=np.uint8)
        self.pub_mask.publish(
            self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
        )

        # ── Publish viz ───────────────────────────────────────────────────────
        viz = self._draw_overlay(frame.copy(), mask_img, centroid)
        self.pub_viz.publish(
            self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
        )

        # ── Publish centroid ──────────────────────────────────────────────────
        if detected:
            u, v = centroid
            self.pub_centroid.publish(Point(x=float(u), y=float(v), z=0.0))
            self.get_logger().debug(f"Disc centroid → ({u:.0f}, {v:.0f})")
        else:
            self.get_logger().debug("Disc not detected.")

    # ── Detection logic ───────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray):
        """
        Run FastSAM → filter masks by (1) white colour and (2) circularity.
        Returns (best_mask, centroid) or (None, None).
        """
        results = self.model(
            frame,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return None, None

        result = results[0]
        if result.masks is None:
            return None, None

        best_mask      = None
        best_score     = -1.0
        best_centroid  = None

        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask_global = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)

        for raw_mask in result.masks.data:
            # Convert tensor → uint8 numpy mask at original resolution
            mask_np = raw_mask.cpu().numpy().astype(np.uint8)
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

            area = int(mask_np.sum())
            if area < MIN_MASK_AREA_PX or area > MAX_MASK_AREA_PX:
                continue

            # ── White colour check ────────────────────────────────────────────
            white_pixels  = int(cv2.bitwise_and(white_mask_global, white_mask_global,
                                                mask=mask_np).sum() / 255)
            white_ratio   = white_pixels / area
            if white_ratio < 0.55:              # at least 55 % of mask is white
                continue

            # ── Circularity check ─────────────────────────────────────────────
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt          = max(contours, key=cv2.contourArea)
            perimeter    = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity  = 4 * np.pi * cv2.contourArea(cnt) / (perimeter ** 2)
            if circularity < MIN_CIRCULARITY:
                continue

            # ── Combined score (circularity weighted higher) ──────────────────
            score = 0.6 * circularity + 0.4 * white_ratio
            if score > best_score:
                best_score    = score
                best_mask     = mask_np
                M             = cv2.moments(mask_np)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_centroid = (cx, cy)

        return best_mask, best_centroid

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, mask_img, centroid):
        if mask_img.any():
            overlay        = frame.copy()
            green_layer    = np.zeros_like(frame)
            green_layer[:, :, 1] = mask_img   # green channel
            cv2.addWeighted(green_layer, 0.4, overlay, 1.0, 0, overlay)
            frame = overlay

        if centroid is not None:
            u, v = centroid
            cv2.circle(frame, (u, v), 8,  (0, 255, 0), -1)
            cv2.circle(frame, (u, v), 12, (255, 255, 255), 2)
            cv2.putText(frame, f"disc ({u},{v})", (u + 15, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No disc detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DiscDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()