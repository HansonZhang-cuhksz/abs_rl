# ABS_RL
An adaptive bitrate streaming algorithm implementation based on reinforcement learning.

# Content
## studentcode_123090823.py
This program provided an interface for model deployment.

## dqn_model.pth
This is the reinforcement learning model file.

## req.txt
This implementation uses 3rd-party pytorch library and its dependencies. The conda environment is exported in this file.

## grader.py
The original grader code will run into timing issue because the waiting time is not enough for subprocess to import pytorch library.

This modified version of grader only changes the sleeping time at line 49 to 2 seconds. Everything else remain unchanged.

## torch_train.py
This is the program for training the model. Grading does not dependent on this program.

# Grading
1. Install conda environment with reference to req.txt.
2. Replace grader program with grader.py in this directory.
3. Put studentcode_123090823.py and dqn_model.pth in the same directory.
4. Run grader.py.

**Please do not use this code other than grading assignment.**