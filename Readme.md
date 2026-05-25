# FRE Task 4 — Docker Environment

## Folder layout

```
fre_task_4/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── README.md
├── bags/               ← drop your .bag files here
│   └── my_recording/
└── src/                ← your ROS 2 packages go here
    └── disc_detection/
        ├── disc_detection/
        │   ├── __init__.py
        │   └── disc_detector_node.py
        ├── launch/
        ├── resource/
        ├── package.xml
        └── setup.py
```

## First time setup

```bash
# Allow Docker to open GUI windows on your desktop
xhost +local:docker

# Build the image (takes a few minutes, downloads weights)
docker compose build

# Start the container
docker compose run --rm ros2
```

## Daily workflow (3 terminals)

Open 3 terminals on your VM. In each one, exec into the container:

```bash
# Terminal 1 — play your bag
docker compose run --rm ros2
ros2 bag play /bags/my_recording

# Terminal 2 — run the detector
docker exec -it fre_task4 bash
ros2 launch disc_detection disc_detection.launch.py

# Terminal 3 — inspect topics / run RViz
docker exec -it fre_task4 bash
ros2 topic echo /disc/detected
# or
rviz2
```



## Rebuild after code changes

Because src/ is mounted, code changes are live inside the container.
Just rebuild the workspace:

```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```


There is also a fix_bag_metadata that helps make the ros bag done with jazzy to work with humble.