"""
Provides QoS configuration for the '/mode' topic.
"""
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, HistoryPolicy

def get_common_qos_profile() -> QoSProfile:
    """
    Returns a QoSProfile for the '/mode' topic.

    Ensures the latest mode is available for new subscribers by keeping only the last message.

    Returns:
        QoSProfile: The QoS profile for '/mode' topic.
    """
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,  # Keep only the last message
        depth=1,  # Store just the most recent message
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL  # Ensure new subscribers get the latest message
    )
