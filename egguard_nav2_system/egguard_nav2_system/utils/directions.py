class Direction:
    """
    Class containing constants for the possible directions of manual nav.
    """
    FORWARD = "forward"
    RIGHT = "right"  
    LEFT = "left"
    
    @classmethod
    def is_valid_mode(cls, direction):
        """
        Check if the provided string represents a valid direction
        
        Parameters:
        -----------
        mode_str : str
            String to check
            
        Returns:
        --------
        bool
            True if the string matches a valid direction, False otherwise
        """
        return direction in [cls.FORWARD, cls.RIGHT, cls.LEFT]