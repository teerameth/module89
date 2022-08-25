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
            executable='chessboard_estimator.py',
            name='chessboard_estimator',
        ),
        Node(
            package='module89',
            executable='pseudo_state.py',
            name='FEN_tracker',
        ),
        Node(
            package='module89',
            executable='GameController.py',
            name='GameController',
        ),
    ])