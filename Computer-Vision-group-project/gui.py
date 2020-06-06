import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
apps = []

# Открыть файл(работа с 1 изображением)

def addApp():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File",
                                          filetypes=(("JPG", "*.jpg"), ("PNG", "*.png"), ("JPEG", "*.jpeg")))
    if filename != "":
        apps.append(filename)
        label1 = tk.Label(frame, text=apps, bg="gray")
        label1.pack()
        detect_img_object(filename)
        drawEdges(filename)


def drawEdges(image):
    img = cv2.imread(image)
    cv2.imshow("Your Photo", img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img, 0, 200)
    cv2.imshow("Edged photo", img)

def detect_img_object(image):
    img = cv2.imread(image)
    edged = cv2.Canny(img, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("Output", img)

    idx = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cv2.imshow(str(idx) + '.png', new_img)
    cv2.waitKey(0)


# Открыть файл (работа с 2 изображениями)

def addApp2():
    filename1 = filedialog.askopenfilename(initialdir="/", title="Select #1 File",
                                          filetypes=(("JPG", "*.jpg"), ("PNG", "*.png"), ("JPEG", "*.jpeg")))
    filename2 = filedialog.askopenfilename(initialdir="/", title="Select #2 File",
                                           filetypes=(("JPG", "*.jpg"), ("PNG", "*.png"), ("JPEG", "*.jpeg")))
    if filename1 != "":
        apps.append(filename1)
        apps.append(filename2)
        label1 = tk.Label(frame, text=apps, bg="gray")
        label1.pack()
        draw_features_orb(filename1,filename2) #Sau
        draw_features_sift(filename1,filename2) #Sul


def draw_features_orb(image1,image2):
    import cv2 as cv
    img1 = cv.imread(image1, cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(image2, cv2.IMREAD_GRAYSCALE)  # trainImage
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ORB", img3)

def draw_features_sift(image1,image2):
    import cv2 as cv
    img1 = cv.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv.imread(image2, cv2.IMREAD_GRAYSCALE)
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Sift", img3)

# Синий экран
canvas = tk.Canvas(root, height=300, width=300, bg="blue")
canvas.pack()
# Белый экран
frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
# Label
label = tk.Label(frame, text="Choose your photo", fg="black")
label.pack()
# Button

openFile = tk.Button(root, text="Open File", fg='white', bg="gray", command=addApp)
openFile.place(relx=0.4, rely=0.6)
l1 = tk.Label(root, text="1 photo:", fg="black")
l1.place(relx=0.1, rely=0.6)

# Second Button
openFiles = tk.Button(root, text="Features", fg='white', bg="gray", command=addApp2)
openFiles.place(relx=0.4, rely=0.8)
label2 = tk.Label(root, text="2 photoes:", fg="black")
label2.place(relx=0.1, rely=0.8)

root.mainloop()

#pip install opencv-python==3.4.2.16
#pip install opencv-contrib-python==3.4.2.16