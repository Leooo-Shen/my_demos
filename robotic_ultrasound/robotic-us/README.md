# robotic-us
Practical course Computational Surgineingeering at TUM

# Important notes when using the robot
## General setups
1. run `cd ~/catkin_ws/`, `source devel/setup.bash`, and `roscore` in one terminal
2. run `rosparam set /iiwa/toolName convex` in another terminal
3. Start ROSSmartServo application on the IIWA robot

## Safety
1. Run roscore first, then start the SmartServo mode.
2. In the right-bottom side, click the icon and set the robot moving speed below 30%.
3. Have lots of `sleep` in scripts. 
4. Have the loop that keep comparing current pose with target pose. Only if the distance is very small can we publish next pose.
5. Always put your hands on the big red button to stop the robot immediately in case of any danger.
6. When finishing experiments, always stop the SmartServo program, then stop the roscore.

## Use EpiphanGrabber in ImFusion
1. `cd ~/imfusion/src/EpiphanGrabber_yuan`
2. cmake and make the project
3. in `build`, copy the .so file
4. put it to `catkin_ws/devel/lib`
5. `source ~/catkin_ws/devel/setup.bash`
6. `ImFusionSuite`