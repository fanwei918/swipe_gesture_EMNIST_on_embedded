# swipe_gesture_EMNIST_on_embedded
swipe gesture recognition on embedded system with touch screen


Data_precessing.cpp

This function is the C rewrite of the data processing part in the python code. takes the trace coordinates, crop it, build 28*28 mat and fill in the blanks, get the flattened 28*28 = 784 1d-vec , represents the hand writting, which is the input of NN.cpp


FNN_MINST_and_some_letter.cpp

C version of the 2 layer FNN in the python code. input scaled by *256, (1->256),  weights scaled by 2**18 , and bias scaled same as input (*256)
in the end of 1st layer, scaled back by >>18; in 2nd layer, didn;t bother to get the softmax output ,just compare which one is biggest.
