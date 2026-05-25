# ─────────────────────────────────────────────────────────────────
# Base: ROS 2 Humble on Ubuntu 22.04
# ─────────────────────────────────────────────────────────────────
FROM osrf/ros:humble-desktop

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ─────────────────────────────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # ROS tools
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-cv-bridge \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-image-transport \
    ros-humble-rqt-image-view \
    ros-humble-rosbag2-storage-mcap \
    # Python
    python3-pip \
    python3-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────
# Python dependencies (FastSAM + vision)
# ─────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    "ultralytics>=8.2.0" \
    "opencv-python>=4.8.0" \
    "numpy>=1.24.0,<2.0.0"

# ─────────────────────────────────────────────────────────────────
# Pre-download FastSAM model weights so first run is instant
# ─────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir matplotlib --upgrade && \
    python3 -c "from ultralytics import FastSAM; FastSAM('FastSAM-s.pt')"

# ─────────────────────────────────────────────────────────────────
# ROS 2 workspace skeleton
# The actual source code is mounted in at runtime (see compose),
# but we create the folder structure here so colcon is happy.
# ─────────────────────────────────────────────────────────────────
RUN mkdir -p /ros2_ws/src

WORKDIR /ros2_ws

# ─────────────────────────────────────────────────────────────────
# rosdep bootstrap (run once at build time)
# ─────────────────────────────────────────────────────────────────
RUN rosdep init || true && rosdep update

# ─────────────────────────────────────────────────────────────────
# Source ROS 2 automatically in every shell
# ─────────────────────────────────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc

# ─────────────────────────────────────────────────────────────────
# Mount points (populated at runtime via docker-compose volumes)
#   /ros2_ws/src   ← your ROS 2 packages / codebase
#   /bags          ← your rosbag files
# ─────────────────────────────────────────────────────────────────
VOLUME ["/ros2_ws/src", "/bags"]

# ─────────────────────────────────────────────────────────────────
# Entrypoint: build workspace if src has packages, then open shell
# ─────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]