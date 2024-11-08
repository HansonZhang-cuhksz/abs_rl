# ABS_RL
An adaptive bitrate streaming algorithm implementation based on reinforcement learning.

# studentcode_123090823.py
This program provided an interface for model deployment.

# dqn_model.pth
This is the reinforcement learning model file.

# req.txt
This implementation uses 3rd-party pytorch library and its dependencies. The conda environment is exported in this file.

# grader.py
The original grader code will run into timing issue because the waiting time is not enough for subprocess to import pytorch library.

This modified version of grader only changes the sleeping time at line 49 to 2 seconds. Everything else remain unchanged.

**Please do not use this code other than grading assignment.**