# -*- coding: utf-8 -*-
"""
Created on Wed May 17 02:30:51 2023

@author: Jose
"""

import os # to extract and save data on folders
import pickle # to save the data 

import mediapipe as mp # to recognize an specific item in a photo
import cv2 # to do some data clieaning


mp_hands = mp.solutions.hands # this is the hand detection algorithm 
mp_drawing = mp.solutions.drawing_utils # to draw the hand solutions
mp_drawing_styles = mp.solutions.drawing_styles # to edit drowing style

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2) # recognizing hands on image with 0.2 of confidence

DATA_DIR = 'data' # where we save the images collected for one-hand sign

data = [] # to save coordinates
labels = [] # to save labels


for dir_ in os.listdir(DATA_DIR): # iterate over every folder and image
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # get the image
        data_aux = [] # save 'X' and 'Y' as a coordinate

        x_ = [] # to extract 'X' coordinates
        y_ = [] # to extract 'Y' coordinates
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) # read the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the colors from BGR to RGB

        results = hands.process(img_rgb) # identify hands on images
        if results.multi_hand_landmarks: # if we detect something then
            for hand_landmarks in results.multi_hand_landmarks: # for every landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x # extract 'X' coordinates
                    y = hand_landmarks.landmark[i].y # extract 'Y' coordinates

                    x_.append(x) # append them to the list
                    y_.append(y) # append them to the list

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x # extract 'X' coordinate
                    y = hand_landmarks.landmark[i].y # extract 'Y' coordinate
                    data_aux.append(x - min(x_)) # this is to have same reference lvl in data
                    data_aux.append(y - min(y_)) # this is to have same reference lvl in data

            data.append(data_aux) # save coordinates as a vector
            labels.append(dir_) # add label to data

f = open('data.pickle', 'wb') # write data as binary data
pickle.dump({'data': data, 'labels': labels}, f) # include the data
f.close() # close file