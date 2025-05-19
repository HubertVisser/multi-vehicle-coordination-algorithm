import numpy as np

class SlackTracker:
    def __init__(self, settings):
        """
        Initialize the SlackTracker to store max lower and upper slack values per prediction step.
        """
        self._N = settings["N"]  # Number of prediction steps
        self._max_sl_pred = np.zeros(self._N)  # Array to store max lower slack values
        self._max_su_pred = np.zeros(self._N)  # Array to store max upper slack values

        self._max_sl = []
        self._max_su = []

    def update(self, solver):
        """
        Update the max lower and upper slack values for each prediction step.
        Args:
            solver: The solver object to retrieve slack values from.
        """
        for k in range(self._N):
            sl = solver.get(k, 'sl')  # Get lower slack for step k
            su = solver.get(k, 'su')  # Get upper slack for step k
            self._max_sl_pred[k] = np.max(sl)  # Store the max lower slack
            self._max_su_pred[k] = np.max(su)  # Store the max upper slack
        
        self.add_max_sl(np.max(self._max_sl_pred))  # Add the max lower slack to the list
        self.add_max_su(np.max(self._max_su_pred))  # Add the max lower slack to the list

    def get_max_sl(self):
        """
        Get the max lower slack values.
        Returns:
            np.ndarray: Array of max lower slack values.
        """
        return self._max_sl

    def get_max_su(self):
        """
        Get the max upper slack values.
        Returns:
            np.ndarray: Array of max upper slack values.
        """
        return self._max_su

    def add_max_sl(self, sl):
        """
        Add a max lower slack value to the list.
        Args:
            sl: The max lower slack value to add.
        """
        self._max_sl.append(sl)
    
    def add_max_su(self, su):
        """
        Add a max upper slack value to the list.
        Args:
            su: The max upper slack value to add.
        """
        self._max_su.append(su)

    def print(self):
        """
        Print the max lower and upper slack values for debugging.
        """
        print("Max Lower Slack (sl):", self._max_sl)
        print("Max Upper Slack (su):", self._max_su)