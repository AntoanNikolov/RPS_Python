from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import random
import time
import os
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./Tensorflow/keras_model.h5", compile=False)

# Load the labels
class_names = open("./Tensorflow/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)


def capture_inp():
    global model, class_names, camera, cv2, np

    result = "" 
    start_time = time.time() #we are using the time function to make the camera run for 4 seconds so the user can reposition their hand if necessary
    while True:

        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)
        cv2.waitKey(1)  # Add delay to update window

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        result = class_name[2:].strip() #.strip fixed an error somewhere somehow by removing empty string (trims whitespace)
        
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        os.system('cls') #removes clutter from terminal
        print("The camera is registering ",result)
        if elapsed_time > 4:
            break #exit while true

    os.system('cls') #removes "the camera is registering..." from end result in the terminal
    return result

#function for ai to choose what to play
def ai_choice():
    possible_choices = ["Rock", "Paper", "Scissors"]
    rand_num = random.randint(0,2)
    choice = possible_choices[rand_num]
    #the ai chooses randomly between the 3 elements in the array
    return choice


#function taht decides who won
def decide_winner(ai_choice_var, user_choice):
    #I am using a dictionary because it is tidier than using a bunch of if statements
    win_map = {
        "Rock":"Scissors",
        "Paper":"Rock",
        "Scissors":"Paper"
    }
    if ai_choice_var == user_choice:
        return "It's a draw!"
    user_wins_against = win_map.get(user_choice) #The Keras Model lable that has been chosen by the camera is assigned to a key in the dictionary. This way we know what the user wins against.
    if ai_choice_var == user_wins_against: #if the code has chosen an option that the user wins against, the user wins.
        return "You win!"
    else:
        return "You lose!"



def main_game():
    print("Get your hand ready!")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Capturing...")
    user_choice = capture_inp()
    ai_choice_var = ai_choice()
    
    time.sleep(1)
    print("You have chosen",user_choice)
    print("The AI has chosen:",ai_choice_var)
    print(decide_winner(ai_choice_var, user_choice))


main_game()
camera.release()
cv2.destroyAllWindows()