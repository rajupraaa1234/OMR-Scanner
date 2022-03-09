from importlib.resources import path
from traceback import print_tb
import cv2
from cv2 import THRESH_BINARY_INV
import numpy as np
import Utils
#####################################################
path = "1.jpg"
widhImg =500
heightImg = 500
question = 5
choice = 5
ans = [1,0,2,0,4]
#####################################################

# PREPROCESSING
img=cv2.imread(path)
img = cv2.resize(img,(widhImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,40,90)

#Finding All Contours
countours , hirerarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,countours,-1,(0,255,0),5)


#find Rectangle
rectCon = Utils.rectContous(countours)
biggestCont = Utils.getCornorPoints(rectCon[0])

if biggestCont.size != 0:
    cv2.drawContours(imgBiggestContours,biggestCont,-1,(0,255,0),10)
    biggestCont = Utils.reorder(biggestCont)
    pt1 = np.float32(biggestCont) #prepare for Wrap
    pts2 = np.float32([[0, 0],[widhImg, 0], [0, heightImg],[widhImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pt1,pts2) #Get Transformation Matrix
    imgWrapColored = cv2.warpPerspective(img,matrix,(widhImg,heightImg))
    cv2.imshow("question",imgWrapColored)

    #Apply Thershold (for detect which option has been marked )
    imgWrapGray = cv2.cvtColor(imgWrapColored,cv2.COLOR_BGR2GRAY)
    imgThersh = cv2.threshold(imgWrapGray,150,255,THRESH_BINARY_INV)[1]

    # getting all boxes merked and un marked both in linear list
    boxes = Utils.splitBox(imgThersh)
    #countNonZero => The function returns the number of non-zero elements in src (like Pixel value)
    # print(cv2,cv2.countNonZero(boxes[1]).countNonZero(boxes[2]))  # for marked options always have maximum pixel value than unmarked options

    #Getting No Zero Pixel value of each box
    myPixelVal = np.zeros((question,choice)) # creatting 2D array of same of total question
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC +=1
        if(countC == choice):
            countR +=1
            countC = 0
    print(myPixelVal) 

    #Now I am finding all the marked column nnumber w.r.t Question and appended in myIndex arr
    myIndex = []
    for x in range(0,question):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    print(myIndex)

    # now finding Score     
    score = 0
    for i in range(0,question):
        if ans[i] == myIndex[i]:
            score +=1

    print("Final Score is : ",score)        

   
    imgBlack = np.zeros_like(img) 
    imageArray = ([img,imgGray,imgBlur,imgCanny],[imgContours,imgBiggestContours,imgWrapColored,imgThersh] )
    imgStacked = Utils.stackImages(imageArray,0.5)
    cv2.imshow("imgStacked",imgStacked)
    cv2.waitKey(0)



