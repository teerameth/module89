from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
MODEL_FILE_NAME = 'chessboard.onnx'

def generate_launch_description():
    dope_encoder_node = ComposableNode(
            name='dope_encoder',
            package='isaac_ros_dnn_encoders',
            plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
            parameters=[{
                'network_image_width': 640,
                'network_image_height': 480,
                'network_image_encoding': 'rgb8',
                'network_normalization_type': 'positive_negative'
            }],
            remappings=[('encoded_tensor', 'tensor_pub'), ('image', 'dope/input')])

    dope_inference_node = ComposableNode(
        name='dope_inference',
        package='isaac_ros_tensor_rt',
        plugin='isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models', MODEL_FILE_NAME),
            'engine_file_path': '/tmp/trt_engine.plan',
            'input_tensor_names': ['input'],
            'input_binding_names': ['input'],
            'output_tensor_names': ['output'],
            'output_binding_names': ['output'],
            'verbose': False
        }])
    container = ComposableNodeContainer(
        name='dope_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[dope_encoder_node, dope_inference_node],
        output='screen',
    )
    return LaunchDescription([
        container,
        Node(
            package='module89',
            executable='chessboard_encoder_fake.py',
            name='chessboard_encoder_fake'
        ),
        Node(
            package='module89',
            executable='chessboard_locator_dope.py',
            name='chessboard_locator_dope',
            remappings=[('pose', 'dope/output')]
        ),
        Node(
            package='module89',
            executable='camera.py',
            name='camera0',
            parameters=[{
                'id': 0,
            }],
            remappings=[('camera0/image', 'dope/input')]
        ),
        # Node(
        #     package='module89',
        #     executable='camera.py',
        #     name='camera1',
        #     parameters=[{
        #         'id': 1,
        #     }],
        # ),
        # Node(
        #     package='module89',
        #     executable='chessboard_detector_fake.py',
        #     name='chessboard_detector_fake'
        # ),
        # Node(
        #     package='module89',
        #     executable='chessboard_locator_dope.py',
        #     name='chessboard_locator'
        # ),
        # Node(
        #     package='module89',
        #     executable='chessboard_tracker.py',
        #     name='chessboard_tracker'
        # ),


    ])