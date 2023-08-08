# Initial Point Detection

The goal is to use intel realsense camera to retrieve depth and color image, perform handeye calibration for setting the relation between the robotic coordinates and the camera coordinates, then detect the target phantom and set up the object coordinates, in order to detect the starting points.

# Dependencies

- ROS noetic


# Setup ROS workspace

```bash
# navigate to your workspace
cd intialPointDetection & catkin_init_workspace
cd .. & catkin_make
```

Then add the ros packages under `src` folder and compile.



