import cv2
 
# our image
img_file = 'car image.png'
 
 # our pre-trained car classifier
car_classifier_file = 'car_detector.xml'
 
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale(needed for haar cascade)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to black and white 

# create our car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_and_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
# car2 = cars[1]
# (x, y, w, h) = car2
    cv2.rectangle(img, (x, y),(x+w, y+h),(0, 0, 255), 2)

# print out number of cars
# print(cars)
 
# Display the image with faces spotted
cv2.imshow("My Image", img) ###

# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

print("Code Completed")