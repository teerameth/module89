from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='module89',
            executable='camera.py',
            name='camera0',
            parameters=[{
                'name': 0,
                'id': 0,
            }],
        ),

        Node(
            package='module89',
            executable='camera.py',
            name='camera1',
            parameters=[{
                'name': 1,
                'id': 2,
            }],
        ),
        Node(
            package='module89',
            executable='chessboard_tracker.py',
            name='chessboard_tracker'
        ),
        Node(
            package='module89',
            executable='chessboard_classifier.py',
            name='chessboard_classifier'
        )
    ])