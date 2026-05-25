from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model",       default_value="FastSAM-s.pt"),
        DeclareLaunchArgument("device",      default_value="cpu"),
        DeclareLaunchArgument("fps_limit",   default_value="10.0"),
        DeclareLaunchArgument("input_topic", default_value="/camera/color/image_raw"),
        Node(
            package="disc_detection",
            executable="disc_detector",
            name="disc_detector",
            output="screen",
            parameters=[{
                "model":       LaunchConfiguration("model"),
                "device":      LaunchConfiguration("device"),
                "fps_limit":   LaunchConfiguration("fps_limit"),
                "input_topic": LaunchConfiguration("input_topic"),
                "imgsz":       640,
                "conf":        0.4,
                "iou":         0.9,
            }],
        ),
    ])
