# Live Background Remover in OpenCV Python

how to start 
```bash
$ python3 trackbar.py
```

create your desired penval.npy file using Trackbars and hit 'S' to save 

```bash
$ python3 test.py 
```
now check that the npy file is detecting your desired color or not

```bash
$ python3 main.py
```
to start the main program

replace bg.jpg with your desired image type

```python
video = cv2.VideoCapture(0)
image = cv2.imread("bg.jpg")
```
change ```cv2.imread("bg.jpg")``` path with your desired image path

Feel Free To  Contribute