cmake_minimum_required(VERSION 3.5)
project(module89)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
ament_auto_find_build_dependencies()

## msg file for service-client ##
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/ChessboardImgPose.msg
  # "srv/ChessboardDetection.srv"
  # "srv/ChessboardPose.srv"
  # "srv/AddThreeInts.srv"
  DEPENDENCIES sensor_msgs geometry_msgs
)

install(DIRECTORY
  launch
  config
  models
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
  scripts/camera.py
  scripts/camera_fake.py
  scripts/chessboard_tracker.py
  scripts/chessboard_detector.py
  scripts/chessboard_locator_dope.py
  scripts/chessboard_classifier.py
  scripts/chessboard_encoder_fake.py
  scripts/geometry.py
  scripts/transform.py
  scripts/example_client.py
  scripts/example_service.py
  scripts/encoder_dynamixel.py
  scripts/dataset_gatherer.py
  DESTINATION lib/${PROJECT_NAME}
)

  
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()

# ament_package()
ament_auto_package(INSTALL_TO_SHARE)