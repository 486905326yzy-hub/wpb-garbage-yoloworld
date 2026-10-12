cmake_minimum_required(VERSION 3.0.2)
project(garbage_yoloworld)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  GarbageDetection.msg
  GarbageDetectionArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/garbage_yoloworld_image.py
  scripts/garbage_yoloworld_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
