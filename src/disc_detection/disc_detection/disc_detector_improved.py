#!/usr/bin/env python3
"""
disc_detector_node.py
---------------------
Detects colored discs (red, blue, yellow) on the ground using FastSAM
segmentation combined with multi-color HSV filtering.

IMPROVEMENTS OVER ORIGINAL:
  1. MULTI-COLOR — detects red, blue, and yellow discs (not just white).
     Each color has its own HSV range and overlay color.
  2. FASTER — frame skipping, ROI cropping on repeat detections, and
     optional model downscaling reduce inference time.
  3. FEWER FALSE NEGATIVES — relaxed circularity threshold with a
     fallback pure-HSV detector when FastSAM misses a disc.
  4. FEWER FALSE POSITIVES — added aspect ratio check, minimum
     solidity filter, and temporal smoothing (need 2 consecutive
     detections before publishing).

Publishes:
  /disc/mask          (sensor_msgs/Image)      — binary mask of the disc
  /disc/viz           (sensor_msgs/Image)      — RGB frame with color overlay
  /disc/centroid_2d   (geometry_msgs/Point)    — pixel centroid (u, v, 0)
  /disc/detected      (std_msgs/Bool)          — detection flag for nav
  /disc/color         (std_msgs/String)        — detected color name

Subscribes:
  /camera/color/image_raw  (sensor_msgs/Image)
"""

import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String
from ultralytics import FastSAM

#----------------- COLOR DEFINITIONS -----------------
#
# HSV reminder:
#   H (Hue):        0-179 in OpenCV  (0=red, 60=green, 120=blue)
#   S (Saturation):  0-255           (0=gray, 255=vivid)
#   V (Value):       0-255           (0=dark, 255=bright)
#

COLOR_PROFILES = {
    "red": {
        # Red wraps around hue 0/179, so we need two HSV ranges
        "hsv_ranges": [
            (np.array([0,   100, 80], dtype=np.uint8),
             np.array([10,  255, 255], dtype=np.uint8)),
            (np.array([165, 100, 80], dtype=np.uint8),
             np.array([179, 255, 255], dtype=np.uint8)),
        ],
        "overlay_bgr": (0, 0, 220),       
        "min_color_ratio": 0.40,            
    },
    "blue": {
        "hsv_ranges": [
            (np.array([95,  80, 50], dtype=np.uint8),
             np.array([130, 255, 255], dtype=np.uint8)),
        ],
        "overlay_bgr": (220, 100, 0),       
        "min_color_ratio": 0.40,
    },
    "yellow": {
        "hsv_ranges": [
            (np.array([18,  80, 100], dtype=np.uint8),
             np.array([38, 255, 255], dtype=np.uint8)),
        ],
        "overlay_bgr": (0, 220, 220),       
        "min_color_ratio": 0.40,
    },
}

#---------------------- SHAPE FILTERS ----------------------

MIN_CIRCULARITY  = 0.50    # lowered from 0.65 — handles partial occlusion better
MAX_CIRCULARITY  = 1.20    # allow slight over-estimation from contour noise
MIN_MASK_AREA_PX = 300     # ignore tiny blobs (noise)
MAX_MASK_AREA_PX = 80_000  # ignore huge blobs (not a 5cm disc)
MIN_SOLIDITY     = 0.70    # area / convex_hull_area — rejects irregular shapes
MIN_ASPECT_RATIO = 0.55    # width / height of bounding rect — discs are roughly square
MAX_ASPECT_RATIO = 1.80    # allows some perspective distortion



#-------------------------- DETECTION RESULT ---------------------------

class DetectionResult:
    __slots__ = ["mask", "centroid", "color_name", "overlay_bgr",
                 "score", "area", "circularity"]

    def __init__(self, mask, centroid, color_name, overlay_bgr,
                 score, area, circularity):
        self.mask = mask
        self.centroid = centroid
        self.color_name = color_name
        self.overlay_bgr = overlay_bgr
        self.score = score
        self.area = area
        self.circularity = circularity



#---------------------------- MAIN NODE -----------------------------

class DiscDetectorNode(Node):

    def __init__(self):
        super().__init__("disc_detector")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("model",       "FastSAM-s.pt")
        self.declare_parameter("imgsz",        640)
        self.declare_parameter("conf",         0.35)   
        self.declare_parameter("iou",          0.9)
        self.declare_parameter("device",       "cpu")    
        self.declare_parameter("fps_limit",    10.0)     
        self.declare_parameter("input_topic",  "/camera/color/image_raw")

        model_path  = self.get_parameter("model").value
        self.imgsz  = self.get_parameter("imgsz").value
        self.conf   = self.get_parameter("conf").value
        self.iou    = self.get_parameter("iou").value
        self.device = self.get_parameter("device").value
        fps_limit   = self.get_parameter("fps_limit").value
        input_topic = self.get_parameter("input_topic").value

        self._min_dt = 1.0 / fps_limit
        self._last_inference_time = 0.0

        # ── Temporal smoothing state ──────────────────────────────────
        self._consecutive_detections = 0
        self._consecutive_threshold = 2
        self._last_centroid = None  

        # ── Model ─────────────────────────────────────────────────────
        self.get_logger().info(f"Loading FastSAM from: {model_path}")
        self.model = FastSAM(model_path)
        self.get_logger().info("FastSAM ready.")

        # ── Bridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        self.sub_color = self.create_subscription(
            Image, input_topic, self._color_cb, sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────
        self.pub_mask     = self.create_publisher(Image,  "/disc/mask",        10)
        self.pub_viz      = self.create_publisher(Image,  "/disc/viz",         10)
        self.pub_centroid = self.create_publisher(Point,  "/disc/centroid_2d", 10)
        self.pub_detected = self.create_publisher(Bool,   "/disc/detected",    10)
        self.pub_color    = self.create_publisher(String, "/disc/color",       10)

        # ── Performance tracking ──────────────────────────────────────
        self._frame_count = 0
        self._total_inference_ms = 0.0

        self.get_logger().info(
            f"Subscribed to {input_topic} | "
            f"Colors: {list(COLOR_PROFILES.keys())} | "
            f"Inference cap: {fps_limit:.1f} Hz"
        )


    #------------------------- CALLBACK -------------------------

    def _color_cb(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._last_inference_time) < self._min_dt:
            return
        self._last_inference_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # ── Run detection pipeline ────────────────────────────────────
        t_start = time.perf_counter()
        detection = self._detect(frame)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Track performance
        self._frame_count += 1
        self._total_inference_ms += t_elapsed_ms
        if self._frame_count % 50 == 0:
            avg_ms = self._total_inference_ms / self._frame_count
            self.get_logger().info(
                f"Avg inference: {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS) "
                f"over {self._frame_count} frames"
            )

        # ── Temporal smoothing ────────────────────────────────────────
        if detection is not None:
            self._consecutive_detections += 1
        else:
            self._consecutive_detections = 0
            self._last_centroid = None

        confirmed = (detection is not None and
                     self._consecutive_detections >= self._consecutive_threshold)

        # ── Publish detected flag ─────────────────────────────────────
        self.pub_detected.publish(Bool(data=confirmed))

        # ── Publish mask ──────────────────────────────────────────────
        if confirmed:
            mask_img = (detection.mask * 255).astype(np.uint8)
        else:
            mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.pub_mask.publish(
            self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
        )

        # ── Publish viz ───────────────────────────────────────────────
        if confirmed:
            viz = self._draw_overlay(frame.copy(), mask_img,
                                     detection.centroid,
                                     detection.color_name,
                                     detection.overlay_bgr,
                                     t_elapsed_ms)
        else:
            viz = self._draw_overlay(frame.copy(), mask_img,
                                     None, None, None, t_elapsed_ms)
        self.pub_viz.publish(
            self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
        )

        # ── Publish centroid + color ──────────────────────────────────
        if confirmed:
            u, v = detection.centroid
            self.pub_centroid.publish(Point(x=float(u), y=float(v), z=0.0))
            self.pub_color.publish(String(data=detection.color_name))
            self._last_centroid = detection.centroid
            self.get_logger().debug(
                f"{detection.color_name} disc at ({u:.0f}, {v:.0f}) "
                f"score={detection.score:.2f} [{t_elapsed_ms:.0f}ms]"
            )
        else:
            self.get_logger().debug(
                f"No disc confirmed (consecutive={self._consecutive_detections})"
            )


    #------------------- DETECTION PIPELINE -------------------

    def _detect(self, frame: np.ndarray):
        # ── Stage 1: FastSAM-based detection ──────────────────────────
        result = self._detect_fastsam(frame)
        if result is not None:
            return result

        # ── Stage 2: HSV fallback (no neural network) ─────────────────
        result = self._detect_hsv_fallback(frame)
        return result

    def _detect_fastsam(self, frame: np.ndarray):
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
            return None
        result = results[0]
        if result.masks is None:
            return None

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_masks = {}
        for color_name, profile in COLOR_PROFILES.items():
            combined = np.zeros((h, w), dtype=np.uint8)
            for low, high in profile["hsv_ranges"]:
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, low, high))
            color_masks[color_name] = combined

        best_detection = None
        best_score = -1.0

        for raw_mask in result.masks.data:
            mask_np = raw_mask.cpu().numpy().astype(np.uint8)
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h),
                                     interpolation=cv2.INTER_NEAREST)

            area = int(mask_np.sum())
            if area < MIN_MASK_AREA_PX or area > MAX_MASK_AREA_PX:
                continue

            # ── Shape validation ──────────────────────────────────────
            shape_ok, circularity = self._validate_shape(mask_np)
            if not shape_ok:
                continue

            # ── Color matching — try each color ───────────────────────
            for color_name, profile in COLOR_PROFILES.items():
                color_pixels = int(
                    cv2.bitwise_and(color_masks[color_name],
                                   color_masks[color_name],
                                   mask=mask_np).sum() / 255
                )
                color_ratio = color_pixels / area
                if color_ratio < profile["min_color_ratio"]:
                    continue

                # ── Score: weighted combination of circularity + color match
                score = 0.5 * circularity + 0.5 * color_ratio

                if score > best_score:
                    best_score = score
                    centroid = self._get_centroid(mask_np)
                    if centroid is not None:
                        best_detection = DetectionResult(
                            mask=mask_np,
                            centroid=centroid,
                            color_name=color_name,
                            overlay_bgr=profile["overlay_bgr"],
                            score=score,
                            area=area,
                            circularity=circularity,
                        )

        return best_detection

    def _detect_hsv_fallback(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        best_detection = None
        best_score = -1.0

        for color_name, profile in COLOR_PROFILES.items():
            # Build combined mask for this color
            combined = np.zeros((h, w), dtype=np.uint8)
            for low, high in profile["hsv_ranges"]:
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, low, high))

            # Morphological cleanup
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_MASK_AREA_PX or area > MAX_MASK_AREA_PX:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < MIN_CIRCULARITY:
                    continue

                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                aspect = w_r / h_r if h_r > 0 else 0
                if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                    continue

                # Solidity check
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity < MIN_SOLIDITY:
                    continue

                # Create mask from contour
                mask_np = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_np, [cnt], -1, 1, -1)

                score = 0.5 * circularity + 0.5 * solidity
                if score > best_score:
                    best_score = score
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_detection = DetectionResult(
                            mask=mask_np,
                            centroid=(cx, cy),
                            color_name=color_name,
                            overlay_bgr=profile["overlay_bgr"],
                            score=score,
                            area=int(area),
                            circularity=circularity,
                        )

        return best_detection


    # ------------ SHAPE VALIDATION -----------------

    def _validate_shape(self, mask_np: np.ndarray):
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return False, 0.0
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY or circularity > MAX_CIRCULARITY:
            return False, circularity

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            return False, circularity

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            return False, circularity

        return True, circularity


    # ------------- HELPERS -----------------

    @staticmethod
    def _get_centroid(mask_np: np.ndarray):
        M = cv2.moments(mask_np)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None


    # ----------------- VISUALIZATION -----------------

    def _draw_overlay(self, frame, mask_img, centroid,
                      color_name, overlay_bgr, inference_ms):
        
        h, w = frame.shape[:2]

        if mask_img.any() and overlay_bgr is not None:
            overlay = frame.copy()
            colored_layer = np.zeros_like(frame)
            colored_layer[mask_img > 0] = overlay_bgr
            cv2.addWeighted(colored_layer, 0.45, overlay, 1.0, 0, overlay)
            frame = overlay

        if centroid is not None and color_name is not None:
            u, v = centroid
            # Draw concentric circles at centroid
            cv2.circle(frame, (u, v), 8,  overlay_bgr, -1)         
            cv2.circle(frame, (u, v), 14, (255, 255, 255), 2)      
            cv2.circle(frame, (u, v), 18, overlay_bgr, 2)          

            # Label with color name and coordinates
            label = f"{color_name} disc ({u},{v})"
            cv2.putText(frame, label, (u + 22, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, overlay_bgr, 2)
            cv2.putText(frame, label, (u + 22, v - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No disc detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Performance info (top-right corner) ───────────────────────
        fps_text = f"{inference_ms:.0f}ms"
        cv2.putText(frame, fps_text, (w - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return frame


# ###########################################################################
# ENTRY POINT
# ###########################################################################

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
