"""
Provides QoS configuration for the '/manual_nav' topic.
"""
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

def get_manual_nav_qos_profile() -> QoSProfile:
    """
    Returns a QoSProfile for the '/manual_nav' topic.

    Returns:
        QoSProfile: The QoS profile for '/manual_nav' topic.
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE
    )