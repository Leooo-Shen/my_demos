from RobotControl import RobotControl
from Pipeline import Pipeline
import rospy
import logging
import utils

if __name__ == "__main__":
    print("Starting pipeline...")
    robot = RobotControl()
    rospy.init_node("robot", anonymous=True)
    pipeline = Pipeline(robot)
    pipeline.run()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        logging.error("ROSInterruptException")

    # utils.visualize_log(robot.logfile)