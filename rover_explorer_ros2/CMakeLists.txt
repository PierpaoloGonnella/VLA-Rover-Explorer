cmake_minimum_required(VERSION 3.8)
project(rover_explorer_ros2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(rosidl_default_generators REQUIRED)
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

set(ROVER_NODE_SCRIPTS
  scripts/camera_node
  scripts/localizer_node
  scripts/ble_bridge_node
  scripts/guard_node
  scripts/coverage_node
  scripts/obstacle_grid_node
  scripts/policy_vlm_node
  scripts/policy_classic_node
  scripts/motion_node
  scripts/logger_node
  scripts/simulator_node
)

if(WIN32)
  foreach(NODE_SCRIPT IN LISTS ROVER_NODE_SCRIPTS)
    get_filename_component(NODE_MODULE "${NODE_SCRIPT}" NAME)
    configure_file(
      scripts/node_launcher.cpp.in
      "${CMAKE_CURRENT_BINARY_DIR}/${NODE_MODULE}_launcher.cpp"
      @ONLY
    )
    add_executable(${NODE_MODULE}_launcher
      "${CMAKE_CURRENT_BINARY_DIR}/${NODE_MODULE}_launcher.cpp"
    )
    target_link_libraries(${NODE_MODULE}_launcher PRIVATE Python3::Python)
    set_target_properties(${NODE_MODULE}_launcher PROPERTIES OUTPUT_NAME ${NODE_MODULE})
    install(TARGETS ${NODE_MODULE}_launcher RUNTIME DESTINATION lib/${PROJECT_NAME})
  endforeach()
else()
  install(PROGRAMS
    ${ROVER_NODE_SCRIPTS}
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp")
  ament_add_pytest_test(test_legacy ${CMAKE_CURRENT_SOURCE_DIR}/../tests TIMEOUT 240
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
    ENV "TEMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp" "TMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp"
  )
  ament_add_pytest_test(test_safety test/test_safety.py
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/.."
    ENV "TEMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp" "TMP=${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp"
  )
  add_launch_test(test/test_motion_launch.py TIMEOUT 60)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
