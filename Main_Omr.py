from asyncio.windows_events import NULL
from importlib.resources import path
from traceback import print_tb
import cv2
from cv2 import THRESH_BINARY_INV
import numpy as np
import Utils
import json
#####################################################
path = "5.JPG"
widhImg = 1100
heightImg = 1100
question = 10
choice = 4
#####################################################
# Declare the Answer for 1 to 10 Question 
ans = [[0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3],
       [0,2,3,3,1,0,2,3,1,3] 
      ]
#####################################################

# PREPROCESSING
#img=Utils.ExtractInnerImage()
img=cv2.imread(path)
img = cv2.resize(img,(widhImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,30,70)    # fining all edges by using Canny method

#Finding All Contours
countours , hirerarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,countours,-1,(0,255,0),4)


#find Rectangle
rectCon = Utils.rectContous(countours)
#print("size--->",rectCon)
#biggestCont = Utils.getCornorPoints(rectCon[5])



AllMCQBoxes = [4,3,7,8,9,6,5]
count = 0
FinalGrade = 0
for i in AllMCQBoxes:
    biggestCont = Utils.getCornorPoints(rectCon[i])
    CurrMcqBox = Utils.getAttemtedQuestionList(biggestCont,imgBiggestContours,img,ans[count])
    for j in range(0,question):
        if ans[count][j] == CurrMcqBox[j]:
            FinalGrade +=1
    count +=1


def getResult():
    JsonRes = {
        "Result": FinalGrade,
    }
    return json.dumps(JsonRes)
   
print("Final Grade",getResult())        




