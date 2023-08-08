#!/bin/bash

docker run -it --net=host osrf/ros:noetic-desktop-full roscore -p 33341
