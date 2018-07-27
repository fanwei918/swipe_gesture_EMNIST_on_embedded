# swipe_gesture_EMNIST_on_embedded
swipe gesture recognition on embedded system with touch screen


Data_precessing.cpp

This function is the C rewrite of the data processing part in the python code. takes the trace coordinates, crop it, build 28*28 mat and fill in the blanks, get the flattened 28*28 = 784 1d-vec , represents the hand writting, which is the input of NN.cpp
