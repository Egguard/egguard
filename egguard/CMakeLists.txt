cmake_minimum_required(VERSION 3.5)

# find dependencies
project(egguard)

find_package(ament_cmake REQUIRED)
find_package(egguard_custom_interfaces REQUIRED) 
find_package(egguard_mode_manager REQUIRED) 
find_package(egguard_nav2_system REQUIRED)
find_package(egguard_world REQUIRED) 
find_package(async_web_server_cpp REQUIRED) 
find_package(web_video_server REQUIRED) 
find_package(egguard_computer_vision REQUIRED)
ament_package()