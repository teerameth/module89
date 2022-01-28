cmake_minimum_required(VERSION 3.5)
project(module89)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)

## msg file for service-client ##
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/FindChessboardPose.srv"
)

# Include Cpp "include" directory
install(DIRECTORY
  launch
  config
  urdf
  meshes
  models
  rviz
  param
  DESTINATION share/${PROJECT_NAME}/
)

# Create Cpp executable
# add_executable(hello_world src/hello_world.cpp)
# target_include_directories(hello_world PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)

# add_executable(example_velocity examples/example_velocity.cpp)
# ament_target_dependencies(example_velocity
#   rclcpp
#   std_msgs
# )


# Install Cpp executables
# install(TARGETS hello_world example_velocity
#   DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  #scripts/teleop_twist_keyboard.py
  #scripts/publisher_member_function.py
  #scripts/teleop_omni_joystick.py
  #scripts/teleop_twist_joystick.py
  #scripts/teleop_planar_joystick.py
  #scripts/aruco_service.py
  #scripts/commander.py
  #scripts/robot_navigator.py
  #scripts/get_image.py
  scripts/camera.py
  scripts/camera_fake.py
  scripts/chessboard_tracker.py
  scripts/chessboard_detector.py
  scripts/chessboard_dope.py
  scripts/chessboard_encoder_fake.py
  scripts/geometry.py
  scripts/transform.py
  DESTINATION lib/${PROJECT_NAME}
)

  
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
