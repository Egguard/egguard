class Mode:
    """
    Class containing constants for the possible operational modes of the robot.
    """
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"  
    EMERGENCY = "emergency"
    
    @classmethod
    def is_valid_mode(cls, mode_str):
        """
        Check if the provided string represents a valid robot mode.
        
        Parameters:
        -----------
        mode_str : str
            String to check
            
        Returns:
        --------
        bool
            True if the string matches a valid mode, False otherwise
        """
        return mode_str in [cls.MANUAL, cls.AUTONOMOUS, cls.EMERGENCY]