# -*- coding: utf-8 -*-
"""
Created on Thu May 18 02:53:22 2023

@author: Jose
"""

import os # import to access folders and files

import cv2 # open CV to access camera module



DATA_DIR = 'data_dos_manos' # path where data is going to be saved
if not os.path.exists(DATA_DIR): # if the folder do not exist then this creates the folder
    os.makedirs(DATA_DIR) # creates folder

dataset_size = 400 # amount of data to collect for every letter (400 images for two-hand sign)


labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', # only capture F, Q, Ñ and X (Ñ is hard to save as file so we choosed Enie instead)
               5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N',
               14: 'Enie', 15: 'O', 16: 'P', 17: 'Q',
               18: 'R', 19: 'S', 20: 'T', 21: 'U',
               22: 'V', 23: 'W', 24: 'X', 25: 'Y',
               26: 'Z'}

cap = cv2.VideoCapture(0) # this access to the camera module desire (take care if you have multiple cameras available for your device)
for j in range(17, 18): # you can change this to cover all letters, however I found out it is easier to record one letter at a time 
    if not os.path.exists(os.path.join(DATA_DIR, labels_dict.get(j))): # it is gonna create a unique carpet for this letter (so images are saved with their label)
        os.makedirs(os.path.join(DATA_DIR, labels_dict.get(j))) # creates the folder 

    print('Recolectando data de la clase {}'.format(labels_dict.get(j))) # this to check wich letter are we recording

    done = False
    while True:
        ret, frame = cap.read() # we are gonna gave this little interface
        cv2.putText(frame, 'Presiona "G" para grabar', (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA) # just to set set the camera in the right position and lighting (press 'G' to record)
        cv2.imshow('frame', frame) # this is to show the frame 
        if cv2.waitKey(25) == ord('g'):
            break

    counter = 0 # set the counter
    while counter < dataset_size: 
        ret, frame = cap.read() # we capture the image
        cv2.imshow('frame', frame) 
        cv2.waitKey(25) # we wait 25 ms
        cv2.imwrite(os.path.join(DATA_DIR, labels_dict.get(j), '{}.jpg'.format(counter)), frame) # we write the image as RecordedLetterName_iterarion.jpg

        counter += 1

cap.release() # then we close the window
cv2.destroyAllWindows()