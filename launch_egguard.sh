#!/bin/bash

# EggGuard Project Launch Script
# This script automates the process of launching the EggGuard ROS 2 project

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a package is installed
check_package() {
    if ! dpkg -l | grep -q $1; then
        echo -e "${YELLOW}Installing $1...${NC}"
        sudo apt install -y $1
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install $1. Exiting.${NC}"
            exit 1
        fi
    fi
}

# Function to open a new terminal and execute a command
run_in_new_terminal() {
    title=$1
    command=$2
    
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$title" -- bash -c "cd $(pwd); source ./install/setup.bash; echo -e '${GREEN}Running: $command${NC}'; $command; echo 'Press Enter to close this terminal...'; read"
    elif command -v xterm &> /dev/null; then
        xterm -title "$title" -e "cd $(pwd); source ./install/setup.bash; echo -e '${GREEN}Running: $command${NC}'; $command; echo 'Press Enter to close this terminal...'; read"
    else
        echo -e "${RED}No suitable terminal emulator found (gnome-terminal or xterm)${NC}"
        echo -e "${YELLOW}Installing xterm...${NC}"
        sudo apt install -y xterm
        xterm -title "$title" -e "cd $(pwd); source ./install/setup.bash; echo -e '${GREEN}Running: $command${NC}'; $command; echo 'Press Enter to close this terminal...'; read"
    fi
}

# Check if we're in the right directory
if [[ "$(basename $(pwd))" != "egguard" || ! -f "./install/setup.bash" ]]; then
    echo -e "${RED}Error: This script must be run from the root of your EggGuard workspace (/egguard)${NC}"
    exit 1
fi

# Check and install dependencies
echo -e "${GREEN}Checking and installing dependencies...${NC}"
check_package "curl"

# Update the system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update

# Build the project
echo -e "${GREEN}Building the project...${NC}"
colcon build --parallel-workers $(nproc) --packages-skip egguard
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed. Please fix the errors and try again.${NC}"
    exit 1
fi

# Source the setup script
source ./install/setup.bash

# Set Gazebo model path
export GAZEBO_MODEL_PATH=$HOME/egguard/install/egguard_world/share/egguard_world/models/model_editor_models:$HOME/egguard/install/egguard_world/share/egguard_world/models
echo -e "${GREEN}Set GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}${NC}"

# Wait between launching components
wait_time=6

# Launch components in separate terminals
echo -e "${GREEN}Launching EggGuard components in separate terminals...${NC}"

run_in_new_terminal "EggGuard World" "ros2 launch egguard_world egguard_world.launch.py"
sleep $wait_time

run_in_new_terminal "Provide Map" "ros2 launch egguard_nav2_system provide_map.launch.py use_sim_time:=False"
sleep $wait_time

run_in_new_terminal "Load Map" "ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap \"{map_url: $HOME/egguard/egguard_nav2_system/config/my_map.yaml}\""
sleep $wait_time

run_in_new_terminal "Mode Manager" "ros2 launch egguard_mode_manager initial_mode_publisher.launch.py"
sleep $wait_time

run_in_new_terminal "Autonomous Controller Launch" "ros2 launch egguard_nav2_system autonomous_controller.launch.py"
sleep 20

run_in_new_terminal "Autonomous Controller Run" "ros2 run egguard_nav2_system autonomous_controller"
sleep $wait_time

run_in_new_terminal "Rosbridge Server" "ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
sleep 2

run_in_new_terminal "Manual Controller" "ros2 launch egguard_nav2_system manual_controller.launch.py"
sleep 2

run_in_new_terminal "Web Video Server" "ros2 run web_video_server web_video_server"
sleep 2

run_in_new_terminal "OpenCV Egg Detection" "ros2 launch egguard_computer_vision egguard_vision.launch.py"
sleep 2

# Open browser with camera stream
if command -v xdg-open &> /dev/null; then
    echo -e "${GREEN}Opening camera stream in browser...${NC}"
    xdg-open http://0.0.0.0:8080/stream?topic=/camera/image_raw
elif command -v firefox &> /dev/null; then
    firefox http://0.0.0.0:8080/stream?topic=/image
elif command -v google-chrome &> /dev/null; then
    google-chrome http://0.0.0.0:8080/stream?topic=/camera/image_raw
else
    echo -e "${YELLOW}Please open this URL in your browser:${NC}"
    echo -e "${GREEN}http://0.0.0.0:8080/stream?topic=/camera/image_raw${NC}"
fi

echo -e "${GREEN}All EggGuard components have been started!${NC}"
echo -e "${YELLOW}Note: To use any new terminal manually, remember to run:${NC}"
echo -e "${GREEN}source $(pwd)/install/setup.bash${NC}"