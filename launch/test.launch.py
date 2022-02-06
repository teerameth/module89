from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='module89',
            executable='chessboard_encoder_fake.py',
            name='chessboard_encoder_fake'
        ),
        Node(
            package='module89',
            executable='camera_fake.py',
            name='camera_fake',
        ),
        Node(
            package='module89',
            executable='chessboard_detector_fake.py',
            name='chessboard_detector_fake'
        ),
        Node(
            package='module89',
            executable='chessboard_locator.py',
            name='chessboard_locator'
        ),
        Node(
            package='module89',
            executable='chessboard_tracker.py',
            name='chessboard_tracker'
        )
    ])