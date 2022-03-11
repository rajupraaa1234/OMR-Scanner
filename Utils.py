from pickle import FALSE, TRUE
from tkinter.messagebox import QUESTION
from traceback import print_tb
import cv2
from cv2 import THRESH_BINARY_INV
import numpy as np

############################
question = 10
choice = 4
path = "full_omr.png"
widhImg = 1100
heightImg = 1100
#############################

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver


def rectContous(contours):
    rectCon=[]
    count =0
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)    # giving me 4 points of coordinate
            if len(approx)==4:
                rectCon.append(i)
                count +=1
    print("count" , count)            
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
    return rectCon 
        
# for find 4 point        
def getCornorPoints(cont):
    peri = cv2.arcLength(cont,True)
    approx = cv2.approxPolyDP(cont,0.02*peri,True)
    return approx


# found all the 4 points of largest Contours 
#
# [[230  5]
# [231 638]
# [432 637]
# [424  12]]

# [ 235  869 1069  436] got the array of sum (x+y) => (230 + 5) => 235 where minimum sum is denoted the origin 
#
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]      # origin
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]      # Max height and width
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]  

    return myPointsNew

# Sp
def splitBox(img):
    count = 0
    rows = np.vsplit(img,question)   
    boxes = []
    for r in rows:
        cols = np.hsplit(r,choice+1)
        flag = FALSE
        for box in cols:
            if flag == FALSE : 
                flag = TRUE
                continue
            boxes.append(box)
            if(count<=4):
                #cv2.imshow("Splite",box)
                count +=1

    return boxes


def ExtractInnerImage():
    img=cv2.imread(path)
    img = cv2.resize(img,(widhImg,heightImg))
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,40,80)

    countours , hirerarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours,countours,-1,(0,255,0),4)
    rectCon = rectContous(countours)
    biggestCont = getCornorPoints(rectCon[0])

    if biggestCont.size != 0:
        cv2.drawContours(imgBiggestContours,biggestCont,-1,(0,255,0),10)
        biggestCont = reorder(biggestCont)
        pt1 = np.float32(biggestCont) #prepare for Wrap
        pts2 = np.float32([[0, 0],[widhImg, 0], [0, heightImg],[widhImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pt1,pts2) #Get Transformation Matrix
        imgWrapColored = cv2.warpPerspective(img,matrix,(widhImg,heightImg))
        #cv2.imshow("question from method",imgWrapColored)
        return imgWrapColored


def getAttemtedQuestionList(biggestCont,imgBiggestContours,img,ans):
    if biggestCont.size != 0:
        cv2.drawContours(imgBiggestContours,biggestCont,-1,(0,255,0),10)
        biggestCont = reorder(biggestCont)
        pt1 = np.float32(biggestCont) #prepare for Wrap
        pts2 = np.float32([[0, 0],[widhImg, 0], [0, heightImg],[widhImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pt1,pts2) #Get Transformation Matrix
        imgWrapColored = cv2.warpPerspective(img,matrix,(widhImg,heightImg))
        # cv2.imshow("question",imgWrapColored)
        # cv2.waitKey(0)
        
        #Apply Thershold (for detect which option has been marked )
        imgWrapGray = cv2.cvtColor(imgWrapColored,cv2.COLOR_BGR2GRAY)
        imgThersh = cv2.threshold(imgWrapGray,90,225,THRESH_BINARY_INV)[1]

        # getting all boxes merked and un marked both in linear list
        boxes = splitBox(imgThersh)
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

        print("Score for current box : ",score)  

        return myIndex      

   
        # imgBlack = np.zeros_like(img) 
        # imageArray = ([img,imgGray,imgBlur,imgCanny],[imgContours,imgBiggestContours,imgWrapColored,imgThersh] )
        # cv2.imshow("imgContours",imgContours)
        # cv2.imshow("imgBiggestContours",imgBiggestContours)
        # cv2.imshow("imgThersh",imgThersh)
        # imgStacked = stackImages(imageArray,0.5)
        # cv2.imshow("imgStacked",imgStacked)
        # cv2.waitKey(0)





