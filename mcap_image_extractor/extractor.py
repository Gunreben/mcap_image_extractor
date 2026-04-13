"""
Core extraction logic for MCAP Image Extractor.

Handles reading MCAP bags, extracting images at configurable intervals,
collecting GPS/IMU/odometry metadata, extracting pointclouds, and
generating metadata CSV files suitable for CVAT labeling workflows.
"""

import bisect
import csv
import logging
import math
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import quote

import cv2
import numpy as np
import yaml

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic type classification
# ---------------------------------------------------------------------------

IMAGE_TYPES = {
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/CompressedImage',
}

GPS_TYPES = {
    'novatel_oem7_msgs/msg/BESTPOS',
    'sensor_msgs/msg/NavSatFix',
}

IMU_TYPES = {
    'sensor_msgs/msg/Imu',
}

POINTCLOUD_TYPES = {
    'sensor_msgs/msg/PointCloud2',
}

ODOM_TYPES = {
    'nav_msgs/msg/Odometry',
}


def categorize_topic(msg_type: str) -> str:
    """Classify a ROS message type into a human-readable category."""
    if msg_type in IMAGE_TYPES:
        return 'image'
    if msg_type in GPS_TYPES:
        return 'gps'
    if msg_type in IMU_TYPES:
        return 'imu'
    if msg_type in POINTCLOUD_TYPES:
        return 'pointcloud'
    if msg_type in ODOM_TYPES:
        return 'odom'
    return 'other'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TopicInfo:
    """Information about a single topic in an MCAP bag."""
    name: str
    type: str
    message_count: int
    category: str


@dataclass
class CameraIntrinsics:
    """Parsed camera intrinsics from a calibration YAML file."""
    camera_name: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray        # 3x3
    distortion_model: str            # 'equidistant', 'rational_polynomial', …
    distortion_coefficients: np.ndarray
    source_file: str = ''            # path to the YAML file


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""
    bag_paths: List[str]
    output_dir: str
    image_topics: List[str]
    gps_topics: List[str] = field(default_factory=list)
    imu_topics: List[str] = field(default_factory=list)
    pointcloud_topics: List[str] = field(default_factory=list)
    odom_topics: List[str] = field(default_factory=list)
    export_mode: str = 'interval'   # 'interval' or 'random_count'
    frame_interval: float = 1.0      # seconds between extracted frames
    random_sample_count: int = 50    # images to export per topic in random mode
    image_format: str = 'png'        # 'png' or 'jpg'
    jpeg_quality: int = 95           # 1-100, only used for JPEG
    generate_metadata_csv: bool = True
    extract_pointclouds: bool = False
    rectify: bool = False
    # topic → CameraIntrinsics mapping (only used when rectify=True)
    calibration_map: Dict[str, 'CameraIntrinsics'] = field(default_factory=dict)


@dataclass
class ExtractionStats:
    """Statistics from an extraction run."""
    total_messages: int = 0
    images_extracted: int = 0
    images_skipped: int = 0
    pointclouds_extracted: int = 0
    gps_messages: int = 0
    imu_messages: int = 0
    bags_processed: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bag inspection
# ---------------------------------------------------------------------------

@dataclass
class BagSummary:
    """Summary information for an inspected MCAP bag."""
    topics: List[TopicInfo]
    duration: float  # bag duration in seconds (0.0 if unknown)


def inspect_bag(bag_path: str) -> BagSummary:
    """
    Read an MCAP file and return topic info plus the bag duration.

    Args:
        bag_path: Path to the .mcap file.

    Returns:
        A BagSummary with sorted topics and duration in seconds.
    """
    topics: List[TopicInfo] = []
    duration = 0.0

    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if not summary:
            return BagSummary(topics=topics, duration=duration)

        # Compute bag duration from statistics
        if summary.statistics:
            start_ns = summary.statistics.message_start_time
            end_ns = summary.statistics.message_end_time
            if end_ns > start_ns:
                duration = (end_ns - start_ns) * 1e-9

        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            msg_type = schema.name if schema else 'unknown'

            msg_count = 0
            if summary.statistics:
                msg_count = summary.statistics.channel_message_counts.get(
                    channel_id, 0
                )

            topics.append(TopicInfo(
                name=channel.topic,
                type=msg_type,
                message_count=msg_count,
                category=categorize_topic(msg_type),
            ))

    return BagSummary(
        topics=sorted(topics, key=lambda t: t.name),
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_topic_name(topic: str) -> str:
    """Convert a ROS topic name into a valid directory name."""
    return topic.strip('/').replace('/', '_')


def stamp_to_sec(stamp) -> float:
    """Convert a ROS2 Time stamp to seconds since epoch."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_to_ns(stamp) -> int:
    """Convert a ROS2 Time stamp to integer nanoseconds since epoch."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_ns_to_sec(stamp_ns: int) -> float:
    """Convert integer nanoseconds since epoch to seconds."""
    return float(stamp_ns) * 1e-9


def format_timestamp_ns(stamp_ns: int) -> str:
    """Format nanoseconds since epoch as ``sec.nanosec``."""
    sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
    return f"{sec}.{nanosec:09d}"


def encode_topic_for_filename(topic: str) -> str:
    """Percent-encode a topic name so the exact topic is preserved in filenames."""
    return quote(topic, safe='')


def build_image_filename(topic: str, stamp_ns: int, ext: str) -> str:
    """Build a reversible filename containing the exact topic and timestamp."""
    return f"{encode_topic_for_filename(topic)}__{format_timestamp_ns(stamp_ns)}.{ext}"


def find_nearest(sorted_data: List[Dict], target_time: float,
                 max_dt: float = 2.0) -> Optional[Dict]:
    """Find the data entry closest in time using binary search.

    Args:
        sorted_data: List of dicts, each containing a 'timestamp' key,
                     sorted by timestamp ascending.
        target_time: Target timestamp in seconds.
        max_dt: Maximum allowable time difference in seconds.

    Returns:
        The nearest dict entry, or None if nothing is within *max_dt*.
    """
    if not sorted_data:
        return None

    times = [d['timestamp'] for d in sorted_data]
    idx = bisect.bisect_left(times, target_time)

    best = None
    best_dt = float('inf')
    for i in (idx - 1, idx):
        if 0 <= i < len(sorted_data):
            dt = abs(sorted_data[i]['timestamp'] - target_time)
            if dt < best_dt:
                best_dt = dt
                best = sorted_data[i]

    return best if best_dt <= max_dt else None


# ---------------------------------------------------------------------------
# Image decoding and saving
# ---------------------------------------------------------------------------

def decode_image(msg, msg_type: str) -> Optional[np.ndarray]:
    """Decode a ROS image message to a BGR numpy array.

    Supports sensor_msgs/msg/CompressedImage and sensor_msgs/msg/Image
    with common encodings.
    """
    try:
        if msg_type == 'sensor_msgs/msg/CompressedImage':
            data = msg.data
            if isinstance(data, (list, tuple)):
                data = bytes(data)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        if msg_type == 'sensor_msgs/msg/Image':
            data = msg.data
            if isinstance(data, (list, tuple)):
                data = bytes(data)
            encoding = msg.encoding.lower()
            h, w = msg.height, msg.width

            _ENC_MAP = {
                'bgr8':       lambda d, h, w: np.frombuffer(d, np.uint8).reshape(h, w, 3),
                'rgb8':       lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w, 3),
                                  cv2.COLOR_RGB2BGR),
                'bgra8':      lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w, 4),
                                  cv2.COLOR_BGRA2BGR),
                'rgba8':      lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w, 4),
                                  cv2.COLOR_RGBA2BGR),
                'mono8':      lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w),
                                  cv2.COLOR_GRAY2BGR),
                'mono16':     lambda d, h, w: cv2.cvtColor(
                                  (np.frombuffer(d, np.uint16).reshape(h, w) // 256
                                   ).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR),
                '16uc1':      lambda d, h, w: cv2.cvtColor(
                                  (np.frombuffer(d, np.uint16).reshape(h, w) // 256
                                   ).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR),
                'bayer_rggb8': lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w),
                                  cv2.COLOR_BayerRG2BGR),
                'bayer_bggr8': lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w),
                                  cv2.COLOR_BayerBG2BGR),
                'bayer_gbrg8': lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w),
                                  cv2.COLOR_BayerGB2BGR),
                'bayer_grbg8': lambda d, h, w: cv2.cvtColor(
                                  np.frombuffer(d, np.uint8).reshape(h, w),
                                  cv2.COLOR_BayerGR2BGR),
            }

            decoder_fn = _ENC_MAP.get(encoding)
            if decoder_fn is None:
                logger.warning("Unsupported image encoding: %s", encoding)
                return None
            return decoder_fn(data, h, w)

    except Exception as e:
        logger.warning("Failed to decode image (%s): %s", msg_type, e)

    return None


def save_image(img: np.ndarray, path: str, fmt: str,
               jpeg_quality: int = 95) -> bool:
    """Save a BGR numpy image to disk as PNG or JPEG."""
    try:
        if fmt in ('jpg', 'jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        elif fmt == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # fast compression
        else:
            params = []
        cv2.imwrite(path, img, params)
        return True
    except Exception as e:
        logger.error("Failed to save image %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Camera calibration loading and image rectification
# ---------------------------------------------------------------------------

def load_intrinsics_yaml(yaml_path: str) -> Optional[CameraIntrinsics]:
    """Parse a ``*.intrinsics.yaml`` calibration file.

    Expected YAML keys: ``camera_name``, ``image_width``, ``image_height``,
    ``camera_matrix`` (with ``data``), ``distortion_model``,
    ``distortion_coefficients`` (with ``data``).
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if data is None:
            return None

        cam_matrix = np.array(
            data['camera_matrix']['data'], dtype=np.float64,
        ).reshape(3, 3)

        dist_coeffs = np.array(
            data['distortion_coefficients']['data'], dtype=np.float64,
        )

        return CameraIntrinsics(
            camera_name=data.get('camera_name', ''),
            image_width=int(data.get('image_width', 0)),
            image_height=int(data.get('image_height', 0)),
            camera_matrix=cam_matrix,
            distortion_model=data.get('distortion_model', ''),
            distortion_coefficients=dist_coeffs,
            source_file=yaml_path,
        )
    except Exception as e:
        logger.warning("Failed to load intrinsics from %s: %s", yaml_path, e)
        return None


def scan_calibration_dir(
    dir_path: str,
) -> Dict[str, CameraIntrinsics]:
    """Scan a directory for ``*.intrinsics.yaml`` files.

    Returns:
        Dict mapping ``camera_name`` → :class:`CameraIntrinsics`.
    """
    result: Dict[str, CameraIntrinsics] = {}
    cal_dir = Path(dir_path)
    if not cal_dir.is_dir():
        return result

    for p in sorted(cal_dir.glob('*.intrinsics.yaml')):
        intr = load_intrinsics_yaml(str(p))
        if intr and intr.camera_name:
            result[intr.camera_name] = intr

    return result


def auto_map_topics_to_calibrations(
    image_topics: List[str],
    calibrations: Dict[str, CameraIntrinsics],
) -> Dict[str, str]:
    """Try to automatically match image topics to camera calibration names.

    Heuristic:
      1. For ``/camera/{label}/…`` topics build ``camera_{label}`` and look
         for an exact ``camera_name`` match.
      2. Fall back to checking whether any ``camera_name`` appears as a
         substring of the topic (handles ZED-style names, etc.).

    Returns:
        Dict mapping **topic** → **camera_name** for every topic that could
        be matched.  Topics without a match are omitted.
    """
    mapping: Dict[str, str] = {}
    cam_names = list(calibrations.keys())

    for topic in image_topics:
        parts = [p for p in topic.split('/') if p]

        # Strategy 1: /camera/{label}/… → camera_{label}
        matched = False
        if len(parts) >= 2 and parts[0] == 'camera':
            candidate = f"camera_{parts[1]}"
            if candidate in calibrations:
                mapping[topic] = candidate
                matched = True

        # Strategy 2: substring match of camera_name in the full topic
        if not matched:
            for cname in cam_names:
                # Build tokens from camera_name for flexible matching
                # e.g. "zed_left_camera" → check if topic contains "zed_left"
                # or the full name
                name_tokens = cname.replace('camera_', '').replace('_camera', '')
                if cname in topic.replace('/', '_') or name_tokens in topic.replace('/', '_'):
                    mapping[topic] = cname
                    break

    return mapping


def rectify_image(
    img: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Undistort / rectify an image using the given camera intrinsics.

    Supports ``equidistant`` / ``fisheye`` and ``rational_polynomial``
    distortion models.  Unknown models return the image unchanged.
    """
    K = intrinsics.camera_matrix
    D = intrinsics.distortion_coefficients
    model = intrinsics.distortion_model.lower()

    try:
        if model in ('equidistant', 'fisheye', 'kannala_brandt'):
            # Use fisheye undistort with reduced FOV to avoid black edges
            Knew = K.copy()
            Knew[0, 0] *= 0.5
            Knew[1, 1] *= 0.5
            return cv2.fisheye.undistortImage(
                img, K=K, D=D[:4], Knew=Knew,
            )

        if model in ('rational_polynomial', 'plumb_bob'):
            if np.allclose(D, 0):
                return img  # already undistorted
            h, w = img.shape[:2]
            new_K, _roi = cv2.getOptimalNewCameraMatrix(
                K, D, (w, h), 1, (w, h),
            )
            return cv2.undistort(img, K, D, None, new_K)

    except Exception as e:
        logger.warning("Rectification failed (%s): %s", model, e)

    return img


# ---------------------------------------------------------------------------
# GPS / Odom extraction helpers
# ---------------------------------------------------------------------------

def extract_gps_data(msg, msg_type: str) -> Optional[Dict]:
    """Extract lat/lon/alt from a GPS message."""
    try:
        if msg_type == 'novatel_oem7_msgs/msg/BESTPOS':
            return {
                'lat': float(msg.lat),
                'lon': float(msg.lon),
                'alt': float(msg.hgt),
                'timestamp': stamp_to_sec(msg.header.stamp),
            }
        if msg_type == 'sensor_msgs/msg/NavSatFix':
            return {
                'lat': float(msg.latitude),
                'lon': float(msg.longitude),
                'alt': float(msg.altitude),
                'timestamp': stamp_to_sec(msg.header.stamp),
            }
    except Exception as e:
        logger.warning("Failed to extract GPS data: %s", e)
    return None


def extract_odom_data(msg) -> Optional[Dict]:
    """Extract heading (yaw) and speed from an Odometry message."""
    try:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.sqrt(vx ** 2 + vy ** 2)

        q = msg.pose.pose.orientation
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        heading = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        return {
            'speed': speed,
            'heading': heading,
            'timestamp': stamp_to_sec(msg.header.stamp),
        }
    except Exception as e:
        logger.warning("Failed to extract odom data: %s", e)
    return None


# ---------------------------------------------------------------------------
# Pointcloud PLY export
# ---------------------------------------------------------------------------

def save_pointcloud_ply(msg, path: str) -> bool:
    """Save a PointCloud2 message as a binary little-endian PLY file."""
    try:
        data = msg.data
        if isinstance(data, (list, tuple)):
            data = bytes(data)

        point_step = msg.point_step
        num_points = len(data) // point_step
        if num_points == 0:
            return False

        # Parse field offsets
        fields: Dict[str, int] = {}
        for f in msg.fields:
            name = f.name if hasattr(f, 'name') else f['name']
            offset = f.offset if hasattr(f, 'offset') else f['offset']
            fields[name] = offset

        if not all(k in fields for k in ('x', 'y', 'z')):
            return False

        x_off, y_off, z_off = fields['x'], fields['y'], fields['z']

        # Extract XYZ
        raw = np.frombuffer(data, dtype=np.uint8).reshape(num_points, point_step)
        x = np.frombuffer(raw[:, x_off:x_off + 4].tobytes(), dtype=np.float32)
        y = np.frombuffer(raw[:, y_off:y_off + 4].tobytes(), dtype=np.float32)
        z = np.frombuffer(raw[:, z_off:z_off + 4].tobytes(), dtype=np.float32)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[valid], y[valid], z[valid]

        # Optional colour
        has_color = 'rgb' in fields or 'rgba' in fields
        r = g = b = None
        if has_color:
            c_off = fields.get('rgba', fields.get('rgb'))
            rgba_raw = np.frombuffer(
                raw[:, c_off:c_off + 4].tobytes(), dtype=np.uint8
            ).reshape(-1, 4)[valid]
            r, g, b = rgba_raw[:, 0], rgba_raw[:, 1], rgba_raw[:, 2]

        n_valid = int(len(x))

        # Write binary little-endian PLY
        with open(path, 'wb') as f:
            header = "ply\nformat binary_little_endian 1.0\n"
            header += f"element vertex {n_valid}\n"
            header += "property float x\nproperty float y\nproperty float z\n"
            if has_color:
                header += ("property uchar red\nproperty uchar green\n"
                           "property uchar blue\n")
            header += "end_header\n"
            f.write(header.encode('ascii'))

            if has_color:
                for i in range(n_valid):
                    f.write(struct.pack('<fff', x[i], y[i], z[i]))
                    f.write(struct.pack('BBB', r[i], g[i], b[i]))
            else:
                for i in range(n_valid):
                    f.write(struct.pack('<fff', x[i], y[i], z[i]))

        return True

    except Exception as e:
        logger.error("Failed to save pointcloud %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class McapImageExtractor:
    """
    Extract images, GPS/IMU metadata, and pointclouds from MCAP bag files.

    Designed for data preparation workflows targeting CVAT labeling:
    - Configurable frame interval to avoid near-duplicate frames.
    - Native resolution output as PNG or high-quality JPEG.
    - Automatic GPS/odometry synchronisation per image frame.
    - Combined metadata CSV for downstream tooling.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.stats = ExtractionStats()
        self._cancelled = False
        self._progress_cb: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Register a ``callback(fraction, message)`` for progress updates."""
        self._progress_cb = callback

    def cancel(self):
        """Request graceful cancellation of the current extraction."""
        self._cancelled = True

    # ----- internal helpers -----

    def _progress(self, fraction: float, message: str):
        if self._progress_cb:
            self._progress_cb(fraction, message)

    # ----- public API -----

    def extract(self) -> ExtractionStats:
        """Run the full extraction pipeline across all configured bags."""
        self.stats = ExtractionStats()
        self._cancelled = False

        output_dir = Path(self.config.output_dir).expanduser().resolve()
        images_dir = output_dir / 'images'

        # Create per-topic output directories for images
        for topic in self.config.image_topics:
            (images_dir / sanitize_topic_name(topic)).mkdir(
                parents=True, exist_ok=True
            )

        # Create pointcloud directories if needed
        if self.config.extract_pointclouds and self.config.pointcloud_topics:
            pc_base = output_dir / 'pointclouds'
            for topic in self.config.pointcloud_topics:
                (pc_base / sanitize_topic_name(topic)).mkdir(
                    parents=True, exist_ok=True
                )

        # Accumulated metadata across all bags
        all_metadata: List[Dict] = []
        all_gps: List[Dict] = []
        all_odom: List[Dict] = []

        # Frame counters persist across bags to avoid filename collisions
        frame_counters: Dict[str, int] = {}
        random_sampling_plans: Dict[str, Dict[str, Set[int]]] = {}

        if self.config.export_mode == 'random_count' and self.config.image_topics:
            self._progress(0.0, "Planning random image samples...")
            random_sampling_plans = self._plan_random_sampling()
            if not random_sampling_plans:
                self._progress(0.0, "WARNING: random sampling plan is empty — "
                               "no images will be exported.")

        total_bags = len(self.config.bag_paths)
        for bag_idx, bag_path in enumerate(self.config.bag_paths):
            if self._cancelled:
                break
            try:
                self._process_bag(
                    bag_path, bag_idx, total_bags,
                    images_dir, output_dir,
                    all_metadata, all_gps, all_odom,
                    frame_counters,
                    random_sampling_plans.get(bag_path, {}),
                )
                self.stats.bags_processed += 1
            except Exception as e:
                error = f"Error processing {bag_path}: {e}"
                logger.error(error)
                self.stats.errors.append(error)

        # Write CSVs
        if self.config.generate_metadata_csv and all_metadata:
            self._write_metadata_csv(output_dir, all_metadata,
                                     all_gps, all_odom)
        if all_gps:
            self._write_gps_csv(output_dir, all_gps)

        self._progress(1.0, "Extraction complete!")
        return self.stats

    # ----- per-bag processing -----

    def _process_bag(
        self, bag_path: str, bag_idx: int, total_bags: int,
        images_dir: Path, output_dir: Path,
        all_metadata: List[Dict],
        all_gps: List[Dict],
        all_odom: List[Dict],
        frame_counters: Dict[str, int],
        random_sampling_plan: Dict[str, Set[int]],
    ):
        bag_name = Path(bag_path).stem
        self._progress(bag_idx / total_bags,
                       f"Opening {bag_name}...")

        # Count total messages for progress reporting
        total_messages = 0
        with open(bag_path, 'rb') as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary and summary.statistics:
                total_messages = summary.statistics.message_count

        # Build set of topics we actually want to read
        topics_to_read: Set[str] = set(self.config.image_topics)
        topics_to_read.update(self.config.gps_topics)
        topics_to_read.update(self.config.imu_topics)
        topics_to_read.update(self.config.odom_topics)
        if self.config.extract_pointclouds:
            topics_to_read.update(self.config.pointcloud_topics)

        # Frame interval resets per bag (each bag is an independent recording)
        last_extract_time: Dict[str, float] = {}
        image_message_indices: Dict[str, int] = {}

        msg_count = 0
        with open(bag_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, decoded in reader.iter_decoded_messages():
                if self._cancelled:
                    return

                msg_count += 1
                self.stats.total_messages += 1
                topic = channel.topic
                msg_type = schema.name

                # Progress update every 500 messages
                if msg_count % 500 == 0:
                    frac = msg_count / max(total_messages, 1)
                    overall = (bag_idx + frac) / total_bags
                    self._progress(
                        overall,
                        f"[{bag_name}] {msg_count}/{total_messages} messages"
                    )

                if topic not in topics_to_read:
                    continue

                # --- Images ---
                if topic in self.config.image_topics:
                    image_message_indices.setdefault(topic, 0)
                    self._handle_image(
                        decoded, msg_type, topic, bag_name,
                        images_dir, output_dir,
                        last_extract_time,
                        frame_counters, all_metadata,
                        random_sampling_plan,
                        image_message_indices[topic],
                    )
                    image_message_indices[topic] += 1

                # --- GPS ---
                elif topic in self.config.gps_topics:
                    gps = extract_gps_data(decoded, msg_type)
                    if gps:
                        gps['bag_file'] = bag_name
                        all_gps.append(gps)
                        self.stats.gps_messages += 1

                # --- Odometry ---
                elif topic in self.config.odom_topics:
                    odom = extract_odom_data(decoded)
                    if odom:
                        odom['bag_file'] = bag_name
                        all_odom.append(odom)

                # --- IMU ---
                elif topic in self.config.imu_topics:
                    self.stats.imu_messages += 1

                # --- Pointclouds ---
                elif (topic in self.config.pointcloud_topics
                      and self.config.extract_pointclouds):
                    self._handle_pointcloud(
                        decoded, topic, output_dir,
                        last_extract_time, frame_counters,
                    )

    # ----- message handlers -----

    def _handle_image(
        self, msg, msg_type: str, topic: str, bag_name: str,
        images_dir: Path, output_dir: Path,
        last_extract_time: Dict[str, float],
        frame_counters: Dict[str, int],
        all_metadata: List[Dict],
        random_sampling_plan: Dict[str, Set[int]],
        image_message_index: int,
    ):
        stamp_ns = stamp_to_ns(msg.header.stamp)
        timestamp = stamp_ns_to_sec(stamp_ns)

        if self.config.export_mode == 'interval':
            if topic in last_extract_time:
                if timestamp - last_extract_time[topic] < self.config.frame_interval:
                    self.stats.images_skipped += 1
                    return
        elif image_message_index not in random_sampling_plan.get(topic, set()):
            self.stats.images_skipped += 1
            return

        img = decode_image(msg, msg_type)
        if img is None:
            self.stats.images_skipped += 1
            return

        # Optional rectification
        if self.config.rectify and topic in self.config.calibration_map:
            img = rectify_image(img, self.config.calibration_map[topic])

        frame_counters.setdefault(topic, 0)
        frame_counters[topic] += 1
        frame_num = frame_counters[topic]

        ext = self.config.image_format
        topic_dir = images_dir / sanitize_topic_name(topic)
        filename = build_image_filename(topic, stamp_ns, ext)
        filepath = topic_dir / filename

        if save_image(img, str(filepath), ext, self.config.jpeg_quality):
            self.stats.images_extracted += 1
            last_extract_time[topic] = timestamp
            all_metadata.append({
                'frame_id': frame_num,
                'filename': str(filepath.relative_to(output_dir)),
                'bag_file': bag_name,
                'topic': topic,
                'timestamp': timestamp,
                'timestamp_ns': stamp_ns,
                'width': img.shape[1],
                'height': img.shape[0],
            })
        else:
            self.stats.errors.append(f"Failed to save {filepath}")

    def _plan_random_sampling(self) -> Dict[str, Dict[str, Set[int]]]:
        """Plan per-bag, per-topic image indices for random sampling."""
        bag_topics: Dict[str, Dict[str, List[int]]] = {}
        bag_capacities: Dict[str, int] = {}

        total_bags = len(self.config.bag_paths)
        for i, bag_path in enumerate(self.config.bag_paths):
            bag_name = Path(bag_path).stem
            self._progress(
                0.0,
                f"Collecting timestamps from {bag_name} "
                f"({i + 1}/{total_bags})...",
            )
            topic_timestamps = self._collect_image_timestamps(bag_path)
            bag_topics[bag_path] = topic_timestamps

            for topic, ts_list in topic_timestamps.items():
                self._progress(
                    0.0,
                    f"  {topic}: {len(ts_list):,} decoded messages",
                )

            bag_capacities[bag_path] = self._random_sampling_capacity(
                topic_timestamps
            )
            self._progress(
                0.0,
                f"  {bag_name} sampling capacity: "
                f"{bag_capacities[bag_path]:,}",
            )

        total_capacity = sum(bag_capacities.values())
        if total_capacity <= 0:
            warning = (
                "Random sampling could not be planned: no bag contains "
                "decodable messages for all selected image topics."
            )
            self._progress(0.0, f"ERROR: {warning}")
            self.stats.errors.append(warning)
            return {}

        requested = min(self.config.random_sample_count, total_capacity)
        if requested < self.config.random_sample_count:
            self._progress(
                0.0,
                f"Note: requested {self.config.random_sample_count:,} samples "
                f"per topic but only {requested:,} are available.",
            )

        allocations = self._allocate_random_counts(requested, bag_capacities)
        plans: Dict[str, Dict[str, Set[int]]] = {}
        for bag_path, topic_timestamps in bag_topics.items():
            count = allocations.get(bag_path, 0)
            if count <= 0:
                continue
            plans[bag_path] = self._build_bag_random_plan(topic_timestamps, count)

        total_planned = sum(
            len(indices)
            for plan in plans.values()
            for indices in plan.values()
        )
        self._progress(
            0.0,
            f"Random plan: {requested:,} samples/topic, "
            f"{total_planned:,} total image slots across "
            f"{len(plans)} bag(s)",
        )
        return plans

    def _collect_image_timestamps(self, bag_path: str) -> Dict[str, List[int]]:
        """Collect per-topic message log_times for the selected image topics.

        Uses the MCAP message envelope ``log_time`` (always present) instead
        of decoding each message, so this cannot silently fail.
        """
        image_topics = list(self.config.image_topics)
        timestamps: Dict[str, List[int]] = {
            topic: [] for topic in image_topics
        }

        with open(bag_path, 'rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, _decoded in (
                reader.iter_decoded_messages(topics=image_topics)
            ):
                if channel.topic in timestamps:
                    timestamps[channel.topic].append(message.log_time)

        return timestamps

    def _random_sampling_capacity(
        self, topic_timestamps: Dict[str, List[int]]
    ) -> int:
        """Return the max shared random samples available for one bag."""
        if not self.config.image_topics:
            return 0
        per_topic_counts = []
        for topic in self.config.image_topics:
            count = len(topic_timestamps.get(topic, []))
            if count <= 0:
                return 0
            per_topic_counts.append(count)
        return min(per_topic_counts)

    def _allocate_random_counts(
        self, requested: int, bag_capacities: Dict[str, int]
    ) -> Dict[str, int]:
        """Distribute requested random samples across bags by capacity."""
        positive = {
            bag_path: capacity
            for bag_path, capacity in bag_capacities.items()
            if capacity > 0
        }
        total_capacity = sum(positive.values())
        if requested <= 0 or total_capacity <= 0:
            return {}

        allocations = {bag_path: 0 for bag_path in positive}
        remainders = []
        assigned = 0

        for bag_path, capacity in positive.items():
            exact = requested * capacity / total_capacity
            base = min(capacity, int(math.floor(exact)))
            allocations[bag_path] = base
            assigned += base
            remainders.append((exact - base, bag_path))

        remaining = requested - assigned
        for _fraction, bag_path in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if allocations[bag_path] >= positive[bag_path]:
                continue
            allocations[bag_path] += 1
            remaining -= 1

        if remaining > 0:
            for bag_path in sorted(positive):
                while remaining > 0 and allocations[bag_path] < positive[bag_path]:
                    allocations[bag_path] += 1
                    remaining -= 1

        return allocations

    def _build_bag_random_plan(
        self, topic_timestamps: Dict[str, List[int]], count: int
    ) -> Dict[str, Set[int]]:
        """Pick shared timestamps, then match each topic to the nearest frame."""
        if count <= 0:
            return {}

        anchor_topic = min(
            self.config.image_topics,
            key=lambda topic: len(topic_timestamps.get(topic, [])),
        )
        anchor_timestamps = topic_timestamps.get(anchor_topic, [])
        anchor_indices = sorted(random.sample(range(len(anchor_timestamps)), count))
        target_timestamps = [anchor_timestamps[idx] for idx in anchor_indices]

        plan: Dict[str, Set[int]] = {anchor_topic: set(anchor_indices)}
        for topic in self.config.image_topics:
            if topic == anchor_topic:
                continue
            selected = self._select_nearest_unique_indices(
                topic_timestamps.get(topic, []),
                target_timestamps,
            )
            plan[topic] = set(selected)

        return plan

    def _select_nearest_unique_indices(
        self, timestamps: List[int], target_timestamps: List[int]
    ) -> List[int]:
        """Greedily assign each target to the nearest unused timestamp."""
        used: Set[int] = set()
        selected: List[int] = []

        for target in target_timestamps:
            idx = self._find_nearest_unused_index(timestamps, target, used)
            if idx is None:
                break
            used.add(idx)
            selected.append(idx)

        return selected

    def _find_nearest_unused_index(
        self, timestamps: List[int], target: int, used: Set[int]
    ) -> Optional[int]:
        """Find the nearest timestamp index that has not been assigned yet."""
        if not timestamps or len(used) >= len(timestamps):
            return None

        idx = bisect.bisect_left(timestamps, target)
        left = idx - 1
        right = idx

        while left >= 0 or right < len(timestamps):
            candidates = []
            if left >= 0 and left not in used:
                candidates.append(left)
            if right < len(timestamps) and right not in used:
                candidates.append(right)
            if candidates:
                return min(candidates, key=lambda i: abs(timestamps[i] - target))
            left -= 1
            right += 1

        return None

    def _handle_pointcloud(
        self, msg, topic: str, output_dir: Path,
        last_extract_time: Dict[str, float],
        frame_counters: Dict[str, int],
    ):
        timestamp = stamp_to_sec(msg.header.stamp)
        key = f"pc_{topic}"

        if key in last_extract_time:
            if timestamp - last_extract_time[key] < self.config.frame_interval:
                return

        frame_counters.setdefault(key, 0)
        frame_counters[key] += 1

        pc_dir = output_dir / 'pointclouds' / sanitize_topic_name(topic)
        ply_path = pc_dir / f"{frame_counters[key]:07d}.ply"

        if save_pointcloud_ply(msg, str(ply_path)):
            self.stats.pointclouds_extracted += 1
            last_extract_time[key] = timestamp

    # ----- CSV writers -----

    def _write_metadata_csv(
        self, output_dir: Path,
        metadata: List[Dict],
        gps_data: List[Dict],
        odom_data: List[Dict],
    ):
        """Write a combined metadata CSV with GPS/odom matched to each frame."""
        csv_path = output_dir / 'metadata.csv'

        gps_sorted = sorted(gps_data, key=lambda d: d['timestamp'])
        odom_sorted = sorted(odom_data, key=lambda d: d['timestamp'])

        fieldnames = [
            'frame_id', 'filename', 'bag_file', 'topic', 'timestamp',
            'timestamp_ns',
            'width', 'height',
            'gps_lat', 'gps_lon', 'gps_alt',
            'heading_deg', 'speed_mps',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in metadata:
                row: Dict[str, Any] = {
                    'frame_id': entry['frame_id'],
                    'filename': entry['filename'],
                    'bag_file': entry['bag_file'],
                    'topic': entry['topic'],
                    'timestamp': format_timestamp_ns(entry['timestamp_ns']),
                    'timestamp_ns': entry['timestamp_ns'],
                    'width': entry['width'],
                    'height': entry['height'],
                    'gps_lat': '',
                    'gps_lon': '',
                    'gps_alt': '',
                    'heading_deg': '',
                    'speed_mps': '',
                }

                gps = find_nearest(gps_sorted, entry['timestamp'], max_dt=2.0)
                if gps:
                    row['gps_lat'] = f"{gps['lat']:.10f}"
                    row['gps_lon'] = f"{gps['lon']:.10f}"
                    row['gps_alt'] = f"{gps['alt']:.3f}"

                odom = find_nearest(odom_sorted, entry['timestamp'], max_dt=1.0)
                if odom:
                    row['heading_deg'] = f"{odom['heading']:.3f}"
                    row['speed_mps'] = f"{odom['speed']:.3f}"

                writer.writerow(row)

        logger.info("Metadata CSV written to %s", csv_path)

    def _write_gps_csv(self, output_dir: Path, gps_data: List[Dict]):
        """Write a standalone GPS trajectory CSV."""
        csv_path = output_dir / 'gps_trajectory.csv'

        fieldnames = ['timestamp', 'lat', 'lon', 'alt', 'bag_file']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gps in sorted(gps_data, key=lambda d: d['timestamp']):
                writer.writerow({
                    'timestamp': f"{gps['timestamp']:.9f}",
                    'lat': f"{gps['lat']:.10f}",
                    'lon': f"{gps['lon']:.10f}",
                    'alt': f"{gps['alt']:.3f}",
                    'bag_file': gps.get('bag_file', ''),
                })

        logger.info("GPS trajectory CSV written to %s", csv_path)
