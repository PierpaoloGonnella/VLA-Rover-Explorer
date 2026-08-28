cmake_minimum_required(VERSION 3.8)
project(rover_explorer_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs objdetect)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(rclcpp REQUIRED)
find_package(ros2_aruco_interfaces QUIET)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RoverPose.msg"
  "msg/LegalActions.msg"
  "msg/PolicyDecision.msg"
  "msg/VlmAdvisory.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(rover_explorer
  PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rover_explorer"
)

add_library(rover_deterministic_logic STATIC src/deterministic_logic.cpp)
target_include_directories(rover_deterministic_logic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

function(add_rover_cpp_node target)
  add_executable(${target} src/${target}.cpp ${ARGN})
  target_include_directories(${target} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${target}
    rover_deterministic_logic
    rclcpp::rclcpp
    diagnostic_msgs::diagnostic_msgs__rosidl_typesupport_cpp
    geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
    nav_msgs::nav_msgs__rosidl_typesupport_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    std_msgs::std_msgs__rosidl_typesupport_cpp
    "${cpp_typesupport_target}"
  )
  install(TARGETS ${target} RUNTIME DESTINATION lib/${PROJECT_NAME})
endfunction()

add_rover_cpp_node(motion_node)
add_rover_cpp_node(guard_node)
add_rover_cpp_node(coverage_node)
add_rover_cpp_node(obstacle_grid_node)
add_rover_cpp_node(localizer_node src/localization.cpp)
target_include_directories(localizer_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(localizer_node ${OpenCV_LIBS})
if(ros2_aruco_interfaces_FOUND)
  rosidl_get_typesupport_target(
    aruco_cpp_typesupport_target ros2_aruco_interfaces "rosidl_typesupport_cpp")
  target_compile_definitions(localizer_node PRIVATE ROVER_HAS_ROS2_ARUCO=1)
  target_link_libraries(localizer_node "${aruco_cpp_typesupport_target}")
endif()

# Offline parity/performance probe. It has no ROS graph presence and is never
# launched by either runtime configuration.
add_executable(localizer_probe test/localizer_probe.cpp src/localization.cpp)
target_include_directories(localizer_probe PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(localizer_probe
  ${OpenCV_LIBS}
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)
install(TARGETS localizer_probe RUNTIME DESTINATION lib/${PROJECT_NAME})

# Integration-heavy Python nodes remain Python. The migrated implementations
# are also installed under explicit *_python names for parity checks/fallback;
# launch files start only the native names, preventing competing publishers.
set(ROVER_PYTHON_NODES
  camera_node
  localizer_node_python
  ble_bridge_node
  policy_vlm_node
  policy_classic_node
  logger_node
  simulator_node
  motion_node_python
  guard_node_python
  coverage_node_python
  obstacle_grid_node_python
)

if(WIN32)
  foreach(PYTHON_NODE_EXECUTABLE IN LISTS ROVER_PYTHON_NODES)
    set(PYTHON_NODE_MODULE "${PYTHON_NODE_EXECUTABLE}")
    if(PYTHON_NODE_EXECUTABLE MATCHES "_python$")
      string(REGEX REPLACE "_python$" "" PYTHON_NODE_MODULE "${PYTHON_NODE_EXECUTABLE}")
    endif()
    configure_file(
      scripts/node_launcher.cpp.in
      "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_NODE_EXECUTABLE}_launcher.cpp"
      @ONLY
    )
    add_executable(${PYTHON_NODE_EXECUTABLE}_launcher
      "${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_NODE_EXECUTABLE}_launcher.cpp"
    )
    target_link_libraries(${PYTHON_NODE_EXECUTABLE}_launcher PRIVATE Python3::Python)
    set_target_properties(
      ${PYTHON_NODE_EXECUTABLE}_launcher PROPERTIES OUTPUT_NAME ${PYTHON_NODE_EXECUTABLE})
    install(
      TARGETS ${PYTHON_NODE_EXECUTABLE}_launcher RUNTIME DESTINATION lib/${PROJECT_NAME})
  endforeach()
else()
  foreach(PYTHON_NODE_EXECUTABLE IN LISTS ROVER_PYTHON_NODES)
    install(PROGRAMS scripts/${PYTHON_NODE_EXECUTABLE} DESTINATION lib/${PROJECT_NAME})
  endforeach()
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp")
  ament_add_gtest(test_deterministic_logic test/test_deterministic_logic.cpp)
  if(TARGET test_deterministic_logic)
    target_link_libraries(test_deterministic_logic rover_deterministic_logic)
    target_include_directories(test_deterministic_logic PRIVATE include)
  endif()
  ament_add_gtest(test_localization test/test_localization.cpp src/localization.cpp)
  if(TARGET test_localization)
    target_include_directories(test_localization PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_localization
      ${OpenCV_LIBS}
      sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    )
  endif()
  ament_add_pytest_test(test_legacy ${CMAKE_CURRENT_SOURCE_DIR}/../tests TIMEOUT 240
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
    ENV "TEMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp" "TMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp"
  )
  ament_add_pytest_test(test_safety test/test_safety.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
    ENV "TEMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp" "TMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp"
  )
  set(ROS_TEST_LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/ros_logs")
  file(MAKE_DIRECTORY "${ROS_TEST_LOG_DIR}")
  add_launch_test(test/test_motion_launch.py TARGET test_motion_launch TIMEOUT 90)
  add_launch_test(test/test_guard_launch.py TARGET test_guard_launch TIMEOUT 60)
  add_launch_test(test/test_sim_launch.py TARGET test_sim_launch TIMEOUT 90)
  add_launch_test(
    test/test_localizer_parity_launch.py TARGET test_localizer_parity_launch TIMEOUT 120)
  add_launch_test(
    test/test_localizer_safety_launch.py TARGET test_localizer_safety_launch TIMEOUT 90)
  set_tests_properties(
    test_motion_launch test_guard_launch test_sim_launch
    test_localizer_parity_launch test_localizer_safety_launch
    PROPERTIES ENVIRONMENT
      "ROS_LOG_DIR=${ROS_TEST_LOG_DIR};FASTDDS_BUILTIN_TRANSPORTS=UDPv4"
  )
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_export_include_directories(include)
ament_package()
