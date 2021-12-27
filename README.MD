# Live Background Remover in OpenCV Python
Getting started

We first need to create a Npy file that detects a specific color in order to subtract that color from our main video and replace it with our image
```
$ python3 trackbar.py
```
Select your desired colors using Trackbars and hit 'S' to save
```
$ python3 test.py 
```
Now check to see if the npy file is detecting your desired color
```
$ python3 main.py
```
This command will start the main program

 you can replace bg.jpg with your desired image type
```
video = cv2.VideoCapture(0)
image = cv2.imread("bg.jpg")
```
change ```cv2.imread("bg.jpg")``` path with your desired image path

Feel Free To Contribute

