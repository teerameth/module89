from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os
MODEL_FILE_NAME = 'chessboard.onnx'

def generate_launch_description():
    dope_encoder_node0 = ComposableNode(
            name='dope_encoder0',
            package='isaac_ros_dnn_encoders',
            plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
            parameters=[{
                'network_image_width': 640,
                'network_image_height': 480,
                'network_image_encoding': 'rgb8',
                'network_normalization_type': 'positive_negative'
            }],
            remappings=[('encoded_tensor', 'dope0/tensor_pub'), ('image', 'camera0/image')])

    dope_inference_node0 = ComposableNode(
        name='dope_inference0',
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
        }],
            remappings=[('tensor_pub', 'dope0/tensor_pub'), ('/tensor_sub', 'dope0/tensor_sub')]
        )

    dope_encoder_node1 = ComposableNode(
        name='dope_encoder1',
        package='isaac_ros_dnn_encoders',
        plugin='isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': 640,
            'network_image_height': 480,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative'
        }],
        remappings=[('encoded_tensor', 'dope1/tensor_pub'), ('image', 'camera1/image')])

    dope_inference_node1 = ComposableNode(
        name='dope_inference1',
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
        }],
        remappings=[('tensor_pub', 'dope1/tensor_pub'), ('/tensor_sub', 'dope1/tensor_sub')]
    )


    dope_decoder_node0 = Node(
        name='chessboard_locator_dope0',
        package='module89',
        executable='chessboard_locator_dope.py',
        remappings=[('/tensor_sub', 'dope0/tensor_sub'), ('pose', 'dope0/output')]
    )

    dope_decoder_node1 = Node(
        name='chessboard_locator_dope1',
        package='module89',
        executable='chessboard_locator_dope.py',
        remappings=[('/tensor_sub', 'dope1/tensor_sub'), ('pose', 'dope1/output')]
    )
    container0 = ComposableNodeContainer(
        name='dope_container0',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[dope_encoder_node0, dope_inference_node0],
        output='screen',
    )
    container1 = ComposableNodeContainer(
        name='dope_container1',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[dope_encoder_node1, dope_inference_node1],
        output='screen',
    )
    return LaunchDescription([
        container0,
        container1,
        dope_decoder_node0,
        dope_decoder_node1,
        Node(
            package='module89',
            executable='chessboard_tracker.py',
            name='chessboard_tracker'
        ),
        Node(
            package='module89',
            executable='chessboard_classifier.py',
            name='chessboard_classifier'
        ),
        Node(
            package='module89',
            executable='camera.py',
            name='camera0',
            parameters=[{
                'id': 0,
            }],
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


    ])