cmake_minimum_required(VERSION3.5)

# find dependencies
project(egguard)

find_package(ament_cmake REQUIRED)
find_package(egguard_custom_interfaces REQUIRED) 
find_package(egguard_mode_manager REQUIRED) 
ament_package()