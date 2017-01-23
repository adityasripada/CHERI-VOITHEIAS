import numpy as np
import cv2
import time

#To load the camera from specified port (mostly 0 or 1. Rarely 2)
camport=0
cap = cv2.VideoCapture(camport) 	#640,480

while(True):
    #To read a frame from the camera
    ret, frame = cap.read()
    frame1 = frame
    if ret==True:
		
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		faces = cv2.CascadeClassifier('haarcascade_eye.xml')
		detected = faces.detectMultiScale(frame, 1.3, 5)
	
		
		
		pupilFrame = frame
		pupilO = frame
		windowClose = np.ones((8,8),np.uint8)
		windowOpen = np.ones((7,7),np.uint8)
		windowDilate = np.ones((15,15),np.uint8)
		windowErode = np.ones((5,5),np.uint8)
		

		for (x,y,w,h) in detected:
			#cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255),1)
			cv2.rectangle(frame1, (x,y), ((x+w),(y+h)), (0,200,255),2)	
			pupilFrame = cv2.equalizeHist(frame[int(y+(h*.25)):int((y+h)), int(x):int((x+w))])
			pupilO = pupilFrame
			#cv2.imshow('Kaa',pupilO)
			ret, pupilFrame = cv2.threshold(pupilFrame,50,255,cv2.THRESH_BINARY)		
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
			#pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_DILATE, windowDilate)
			pupilFrame = cv2.dilate(pupilFrame,windowDilate,iterations = 1)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
			pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)			
			#cv2.imshow('After morphology',pupilFrame)
			
			threshold = cv2.inRange(pupilFrame,150,255)		
			contours, hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			#cv2.imshow('Thresh',threshold) 
			
			
			if len(contours) >= 2:
				
				maxArea = 0
				MAindex = 0			
				distanceX = []		
				currentIndex = 0 
				for cnt in contours:
					area = cv2.contourArea(cnt)
					center = cv2.moments(cnt)
					if center['m00']==0:
						continue
					cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
					distanceX.append(cx)	
					if area > maxArea:
						maxArea = area
						MAindex = currentIndex
					currentIndex = currentIndex + 1
		
				del contours[MAindex]		#remove the picture frame contour
				del distanceX[MAindex]
			
			

			if len(contours) >= 2:		#alterations for right eye
				if distanceX !=[]:
					edgeOfEye = distanceX.index(min(distanceX))
				else:
					edgeOfEye = distanceX.index(max(distanceX))	
				del contours[edgeOfEye]
				del distanceX[edgeOfEye]

			if len(contours) >= 1:		#get largest blob
				maxArea = 0
				for cnt in contours:
					area = cv2.contourArea(cnt)
					if area > maxArea:
						maxArea = area
						largeBlob = cnt
					
			if len(largeBlob) > 0:	
				center = cv2.moments(largeBlob)
				cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
				cv2.circle(pupilO,(cx,cy),5,255,-1)
				#print cx,cy
				#cv2.drawContours(frame1,[largeBlob],-1,(0,0,255),2)
				cv2.circle(frame1,(int(x+cx+10),int(y+cy)),2,[0,0,255],-1)
				
		#ret, ulti_thresh = cv2.threshold(cv2.absdiff(pupilO,pupilFrame),70,255,cv2.THRESH_BINARY_INV)
		
		
		#cv2.imshow('feed',frame1)
		cv2.imshow('frame',pupilO)
		#cv2.imshow('frame2',pupilFrame)
		if cv2.waitKey(25) & 0xFF == 27:
			break
    else:
	print "Cam not Found in port "+str(camport)	
	break
cap.release()
cv2.destroyAllWindows()
