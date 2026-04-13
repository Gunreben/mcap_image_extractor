# MCAP Image Extractor

Extract image frames from ROS 2 MCAP bag files for **CVAT labeling** and data preparation workflows.

## Features

- **GUI application** — add multiple MCAP bags, browse and select topics, configure extraction
- **Smart frame extraction** — configurable time interval (e.g. 1 frame / second) to avoid near-duplicate frames
- **Random matched sampling** — export exactly `x` images per selected image topic using shared random timestamps matched as closely as possible
- **Native resolution** — images are saved at original sensor resolution (e.g. ZED 2i)
- **Format choice** — PNG (lossless) or high-quality JPEG with adjustable quality
- **Traceable filenames** — image names include the exact topic and exact ROS timestamp
- **GPS/GNSS synchronisation** — matches NovAtel BESTPOS or NavSatFix to each extracted frame
- **Odometry sync** — extracts heading and speed from `nav_msgs/Odometry` topics
- **Metadata CSV** — `filename, timestamp, gps_lat, gps_lon, gps_alt, heading, speed` per frame
- **Stereo pointcloud export** — saves PointCloud2 topics as binary PLY files
- **GPS trajectory CSV** — standalone trajectory file for mapping tools

## Supported Message Types

| Category    | ROS 2 Types                                               |
|-------------|-----------------------------------------------------------|
| Image       | `sensor_msgs/msg/Image`, `sensor_msgs/msg/CompressedImage`|
| GPS         | `novatel_oem7_msgs/msg/BESTPOS`, `sensor_msgs/msg/NavSatFix` |
| IMU         | `sensor_msgs/msg/Imu`                                     |
| Pointcloud  | `sensor_msgs/msg/PointCloud2`                             |
| Odometry    | `nav_msgs/msg/Odometry`                                   |

## Installation

### Option A: pip install (standalone, no ROS 2 needed)

```bash
cd mcap_image_extractor
pip install -r requirements.txt
pip install -e .
```

### Option B: colcon build (ROS 2 workspace)

```bash
cd ~/ros2_ws
colcon build --packages-select mcap_image_extractor
source install/setup.bash
```

## Usage

### Launch the GUI

```bash
# After pip install
mcap_image_extractor

# Or run directly
python -m mcap_image_extractor.gui
```

### GUI Workflow

1. **Add Bags** — click "Add Bags" to select one or more `.mcap` files
2. **Select Topics** — image topics are auto-selected; add GPS/Odom topics for metadata
3. **Configure** — choose interval sampling or random matched sampling, then set output format and output directory
4. **Extract** — click "Extract" and monitor progress in the log

### Quick-Select Buttons

- **Select All Images** — check all image topics at once
- **Select All GPS** — include GPS for metadata CSV
- **Select All Odom** — include odometry for heading/speed

## Output Structure

```
output_dir/
├── images/
│   ├── zed_zed_node_left_raw_image_raw_color_compressed/
│   │   ├── %2Fzed%2Fzed_node%2Fleft_raw%2Fimage_raw_color%2Fcompressed__1713182400.123456789.png
│   │   ├── %2Fzed%2Fzed_node%2Fleft_raw%2Fimage_raw_color%2Fcompressed__1713182401.456789123.png
│   │   └── ...
│   └── camera_rear_left_image_raw_compressed/
│       └── ...
├── pointclouds/           (if enabled)
│   └── zed_zed_node_point_cloud_cloud_registered/
│       ├── 0000001.ply
│       └── ...
├── metadata.csv           (if GPS/Odom topics selected)
└── gps_trajectory.csv     (if GPS topics selected)
```

### metadata.csv Columns

| Column       | Description                                    |
|-------------|------------------------------------------------|
| frame_id    | Sequential frame number per topic              |
| filename    | Relative path to the extracted image           |
| bag_file    | Source MCAP bag file name                      |
| topic       | ROS topic the image was extracted from         |
| timestamp   | Image timestamp (seconds, nanosecond precision)|
| timestamp_ns| Exact image timestamp in nanoseconds           |
| width       | Image width in pixels                          |
| height      | Image height in pixels                         |
| gps_lat     | Nearest GPS latitude (if available)            |
| gps_lon     | Nearest GPS longitude (if available)           |
| gps_alt     | Nearest GPS altitude (if available)            |
| heading_deg | Vehicle heading in degrees (from odometry)     |
| speed_mps   | Vehicle speed in m/s (from odometry)           |

## CVAT Integration

The extracted images are ready for direct upload to CVAT:

1. Create a CVAT task
2. Upload the contents of the per-topic image folder
3. Use `metadata.csv` for geo-referenced downstream processing

## Dependencies

- `mcap` + `mcap-ros2-support` — reading MCAP files (no ROS 2 installation required)
- `opencv-python` — image decoding and encoding
- `numpy` — array operations
- `PySide6` + `qt-material` — modern dark-themed GUI
