# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:42:25 2022

@author: rupik
"""
import numpy as np
import cv2
import os
import imutils

#eucl dist
def distance(v1,v2):
    return np.sqrt((v1-v2)**2).sum()

# user defined funn for knn

# UDF of Knearest Neighbours (take sample train, test, no of neighbours)
def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
        # label 
		iy = train[i, -1]
		# Compute the distance from test point
        # call distance function to calculate distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
        
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1]) 
	return output[0][index]

# collect test samples from video
vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:/Users/rupik/Documents/Edu labs/haarcascade_frontalface_default.xml")

## .npy image path 
dataset_path = "C:/Users/rupik/Documents/Edu labs/faces/"

face_data = [] 
labels = []
class_id = 0
names = {}


# Dataset prepration for all input samples 
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
        # Adding named ID from fx samples [-4] :
		names[class_id] = fx[:-4]
        # dataitems collected in previous step 
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)
        # np.ones: create an array of shape of data_item
		target = class_id * np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

# concatinating face data
face_dataset = np.concatenate(face_data, axis=0)
# concatinating face data [0]
#                         [1]
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret, frame = vid.read()
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 5
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

        # pass trainset data by flatteing it just CNN 
        # flatten array train sample data & train knn algo
		out = knn(trainset, face_section.flatten())

		# Draw rectangle in the original image
		cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    # Faces is the Frame name 
    # frame is video data (Unstructured Samples)
	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()