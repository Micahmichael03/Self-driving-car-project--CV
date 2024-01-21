import cv2

# our image
img_file = 'car image.png'
# our video
video = cv2.VideoCapture('india.mp4')

# our pre-trained car classifier
car_classifier_file = 'car_detector.xml'

# create our car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)

# Run forever until car stops or something. Or crashes
while True:
    # Read the current frame
    (read_successful, frame)= video.read()

    # safe coding
    if read_successful:
        # convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    else:
        break
    
    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y),(x+w, y+h),(0, 0, 255), 2)
    
    # Display the image with faces spotted
    cv2.imshow("My Image", frame) ###

    # Dont autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)
    
    # stop if Q key is pressed
    if key==81 or key==113:
        break

print("Code Completed")