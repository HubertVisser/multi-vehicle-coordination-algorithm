import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Quaternion
import tf.transformations as tf

def create_rectangle_marker(marker_id, pose, width, height):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "rectangle"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose = pose
    marker.scale.x = width
    marker.scale.y = height
    marker.scale.z = 0.1  # Thickness of the rectangle
    marker.color.a = 1.0  # Alpha
    marker.color.r = 1.0  # Red
    marker.color.g = 0.0  # Green
    marker.color.b = 0.0  # Blue
    return marker

def pose_to_quaternion(yaw):
    """Convert a yaw angle (in radians) to a quaternion."""
    return Quaternion(*tf.quaternion_from_euler(0, 0, yaw))

def create_pose(x, y, yaw):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0
    pose.orientation = pose_to_quaternion(yaw)
    return pose

if __name__ == "__main__":
    rospy.init_node("rectangle_marker_publisher")
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    marker_id = 0

    while not rospy.is_shutdown():
        # Example pose
        x = 1.0
        y = 2.0
        yaw = 0.5  # radians

        pose = create_pose(x, y, yaw)
        marker = create_rectangle_marker(marker_id, pose, width=1.0, height=0.5)
        marker_pub.publish(marker)

        marker_id += 1
        rate.sleep()