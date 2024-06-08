# -*- coding: utf-8 -*-
"""
Created on Wed May 17 02:53:19 2023

@author: Jose
"""
import cv2 # OpenCV to access camera modelues and convert colores
import mediapipe as mp # media pipe to detect hands 
import numpy as np # numpy to do calculations
import tensorflow as tf # to load trained model


labels_dict_one_hand = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', # dictionary to show name of the letter we are detecting (one hand)
               5: 'G', 6: 'H', 7: 'I', 8: 'J',
               9: 'K', 10: 'L', 11: 'M', 12: 'N',
               13: 'O', 14: 'P',
               15: 'R', 16: 'S', 17: 'T', 18: 'U',
               19: 'V', 20: 'W', 21: 'Y',
               22: 'Z'}

labels_dict_two_hand = {0: 'F', 1: 'Enie', 2: 'Q', 3: 'X'} # dictionary to show name of the letter we are detecting (two hands)

model_1 = tf.keras.models.load_model("model_redes_one_hand.h5") # load one hand model (it is not necessary to train NN, you can try RFC, SVC, etc)

model_2 = tf.keras.models.load_model("model_redes_two_hand.h5") # load two hands model (it is not necessary to train NN, you can try RFC, SVC, etc)

cap = cv2.VideoCapture(0) # access camera module

mp_hands = mp.solutions.hands # this is the hand detection algorithm 
mp_drawing = mp.solutions.drawing_utils # to draw the hand solutions
mp_drawing_styles = mp.solutions.drawing_styles # to edit drowing style

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2) # recognizing hands on image with 0.2 of confidence


while True: # to keep program active

    data_aux = [] # to save coordinates
    x_ = [] # to save 'X' coordinates
    y_ = [] # to save 'Y' coordinates

    ret, frame = cap.read() # to extract frame from camera

    H, W, _ = frame.shape # extract height and width from frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # change frame color from BGR to RGB

    results = hands.process(frame_rgb) # process hands
    if results.multi_hand_landmarks: # if we detect a hand then
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # draw points
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(), # styles
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks: # for every landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x # extract 'X' coordinate
                y = hand_landmarks.landmark[i].y # extract 'Y' coordinate

                x_.append(x) # save 'X' coordinate
                y_.append(y) # save 'Y' corodinate

            for i in range(len(hand_landmarks.landmark)): # for every landmark 
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_)) # same reference lvl to all data
                data_aux.append(y - min(y_)) # same reference lvl to all data

        x1 = int(min(x_) * W) - 10 # this is to show results above hand
        y1 = int(min(y_) * H) - 10 # this is to show results above hand

        x2 = int(max(x_) * W) - 10 # this is to show results above hand
        y2 = int(max(y_) * H) - 10 # this is to show results above hand
        
        if len(data_aux) == 42: # if we detect only one hand then
            
            data_shape = np.reshape(np.array(data_aux), (-1, 42)) # reshape the data
    
            prediction = model_1.predict(data_shape) # predict with one-hand model
    
            predicted_character = labels_dict_one_hand[prediction.argmax()] # extract probability of prediction
            
            predicted_proba = np.max(prediction) # probability of that letter
            
            print(predicted_character, predicted_proba) # print results
            
            show_proba = str(np.round(predicted_proba, 2)) # probability to show
            
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # create a rectangle enclosing the hand
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA) # show letter detected
            cv2.putText(frame, show_proba, (x1 + 70, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA) # show probability 
            
        elif len(data_aux) == 84: # if we detect two hands
            
            data_shape = np.reshape(np.array(data_aux), (-1, 84)) # reshape the data
            prediction = model_2.predict(data_shape) # predict with two-hands model
    
            predicted_character = labels_dict_two_hand[prediction.argmax()] # extract probability of prediction

            print(predicted_character, predicted_proba) # probability of that letter
            
            show_proba = str(np.round(predicted_proba, 2)) # probability to show
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # create a rectangle enclosing the hands 
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA) # show letter detected
            cv2.putText(frame, show_proba, (x1 +70, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA) # show probability       


    cv2.imshow('frame', frame) # show frame 
    cv2.waitKey(1) # update every milisecond

cap.release() # close the program 
cv2.destroyAllWindows()