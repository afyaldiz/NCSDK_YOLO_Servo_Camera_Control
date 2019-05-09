# yedekkod

# USAGE

# python ncsvize1.py --graph graphs/mobilenetgraph --display 1

# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.5 --display 1



# import the necessary packages

from mvnc import mvncapi as mvnc

from imutils.video import VideoStream

from imutils.video import FPS

import argparse

import numpy as np

import time

import cv2

import wiringpi

from Tkinter import * 

import threading



a=150

my_secim=0

global secim

secim=0

global secimm

secimm=0

delay_period = 0.05

my_selection = 0

def selection():  

	global my_selection

	my_selection = int(radio.get())

	selection = "Takip Eilecek Nesne ", my_selection

	#print("sel func: ", my_selection)

	label.config(text = selection) 

def secim():  

	global secim

	global secimm

	secimm=secimm+1





def silme():

	global my_selection

	global secim

	global secimm

	R1.deselect()

	R2.deselect()

	R3.deselect()

	R4.deselect()

	R5.deselect()

	R6.deselect()

	R7.deselect()

	R8.deselect()

	R9.deselect()

	R10.deselect()

	R11.deselect()

	R12.deselect()

	R13.deselect()

	R14.deselect()

	R15.deselect()

	R16.deselect()

	R17.deselect()

	R18.deselect()

	R19.deselect()

	R20.deselect()

	R21.deselect()

	my_selection=0

	secim=0

	secimm=0



top = Tk() 

top.geometry("250x500")  

radio = IntVar() 

lbl = Label(text = "Takip Edilecek Nesneyi Seciniz:")  

lbl.pack()

B1=Button(top, text="Cikis", command=top.quit).pack()

B2=Button(top, text ="Secimi Sifirla", command = silme).pack()

B3=Button(top, text ="Digeri", command = secim).pack()



R1 = Radiobutton(top, text="background", variable=radio, value=1, command=selection)  

R1.pack(anchor=W)



R2 = Radiobutton(top, text="aeroplane", variable=radio, value=2,  

command=selection)  

R2.pack(anchor=W)



R3 = Radiobutton(top, text="bicycle", variable=radio, value=3,  

command=selection)  

R3.pack(anchor=W)



R4 = Radiobutton(top, text="bird", variable=radio, value=4,command=selection)  

R4.pack(anchor=W)



R5 = Radiobutton(top, text="boat", variable=radio, value=5,  

command=selection)  

R5.pack(anchor=W)



R6 = Radiobutton(top, text="boottle", variable=radio, value=6,  

command=selection)  

R6.pack(anchor=W)



R7 = Radiobutton(top, text="bus", variable=radio, value=7,  

command=selection)  

R7.pack(anchor=W)



R8 = Radiobutton(top, text="car", variable=radio, value=8,  

command=selection)  

R8.pack(anchor=W) 



R9 = Radiobutton(top, text="cat", variable=radio, value=9,  

command=selection)  

R9.pack(anchor=W) 



R10 = Radiobutton(top, text="chair", variable=radio, value=10,  

command=selection)  

R10.pack(anchor=W) 



R11 = Radiobutton(top, text="cow", variable=radio, value=11,  

command=selection)  

R11.pack(anchor=W)



R12 = Radiobutton(top, text="diningtable", variable=radio, value=12,  

command=selection)  

R12.pack(anchor=W)



R13 = Radiobutton(top, text="dog", variable=radio, value=13,  

command=selection)  

R13.pack(anchor=W)



R14 = Radiobutton(top, text="horse", variable=radio, value=14,  

command=selection)  

R14.pack(anchor=W)



R15 = Radiobutton(top, text="motorbike", variable=radio, value=15,  

command=selection)  

R15.pack(anchor=W)



R16 = Radiobutton(top, text="person", variable=radio, value=16,  

command=selection)  

R16.pack(anchor=W)



R17 = Radiobutton(top, text="pottedplant", variable=radio, value=17,  

command=selection)  

R17.pack(anchor=W)



R18 = Radiobutton(top, text="sheep", variable=radio, value=18,  

command=selection)  

R18.pack(anchor=W)



R19 = Radiobutton(top, text="sofa", variable=radio, value=19,  

command=selection)  

R19.pack(anchor=W)



R20 = Radiobutton(top, text="train", variable=radio, value=20,  

command=selection)  

R20.pack(anchor=W)



R21 = Radiobutton(top, text="tvmonitor", variable=radio, value=21,  

command=selection)  

R21.pack(anchor=W)





liste=("background", "aeroplane", "bicycle", "bird",

	"boat", "bottle", "bus", "car", "cat", "chair", "cow",

	"diningtable", "dog", "horse", "motorbike", "person",

	"pottedplant", "sheep", "sofa", "train", "tvmonitor")



	

wiringpi.wiringPiSetupGpio()



# set #18 to be a PWM output

wiringpi.pinMode(18, wiringpi.GPIO.PWM_OUTPUT)



# set the PWM mode to milliseconds stype

wiringpi.pwmSetMode(wiringpi.GPIO.PWM_MODE_MS)



# divide down clock

wiringpi.pwmSetClock(192)

wiringpi.pwmSetRange(2000)



wiringpi.pwmWrite(18,a)





# initialize the list of class labels our network was trained to

# detect, then generate a set of bounding box colors for each class

CLASSES = ("background", "aeroplane", "bicycle", "bird",

	"boat", "bottle", "bus", "car", "cat", "chair", "cow",

	"diningtable", "dog", "horse", "motorbike", "person",

	"pottedplant", "sheep", "sofa", "train", "tvmonitor")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



# frame dimensions should be sqaure

PREPROCESS_DIMS = (300, 300)

DISPLAY_DIMS = (600, 600)



# calculate the multiplier needed to scale the bounding boxes

DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]



def preprocess_image(input_image):

	# preprocess the image

	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)

	preprocessed = preprocessed - 127.5

	preprocessed = preprocessed * 0.007843

	preprocessed = preprocessed.astype(np.float16)



	# return the image to the calling function

	return preprocessed

	

def predict(image, graph):

	# preprocess the image

	image = preprocess_image(image)



	# send the image to the NCS and run a forward pass to grab the

	# network predictions

	graph.LoadTensor(image, None)

	(output, _) = graph.GetResult()



	# grab the number of valid object predictions from the output,

	# then initialize the list of predictions

	num_valid_boxes = output[0]

	predictions = []



	# loop over results

	for box_index in range(num_valid_boxes):

		# calculate the base index into our array so we can extract

		# bounding box information

		base_index = 7 + box_index * 7



		# boxes with non-finite (inf, nan, etc) numbers must be ignored

		if (not np.isfinite(output[base_index]) or

			not np.isfinite(output[base_index + 1]) or

			not np.isfinite(output[base_index + 2]) or

			not np.isfinite(output[base_index + 3]) or

			not np.isfinite(output[base_index + 4]) or

			not np.isfinite(output[base_index + 5]) or

			not np.isfinite(output[base_index + 6])):

			continue



		# extract the image width and height and clip the boxes to the

		# image size in case network returns boxes outside of the image

		# boundaries

		(h, w) = image.shape[:2]

		x1 = max(0, int(output[base_index + 3] * w))

		y1 = max(0, int(output[base_index + 4] * h))

		x2 = min(w,	int(output[base_index + 5] * w))

		y2 = min(h,	int(output[base_index + 6] * h))



		# grab the prediction class label, confidence (i.e., probability),

		# and bounding box (x, y)-coordinates

		pred_class = int(output[base_index + 1])

		pred_conf = output[base_index + 2]

		pred_boxpts = ((x1, y1), (x2, y2))



		# create prediciton tuple and append the prediction to the

		# predictions list

		prediction = (pred_class, pred_conf, pred_boxpts)

		predictions.append(prediction)



	# return the list of predictions to the calling function

	return predictions



# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()

ap.add_argument("-g", "--graph", required=True,

	help="path to input graph file")

ap.add_argument("-c", "--confidence", default=.5,

	help="confidence threshold")

ap.add_argument("-d", "--display", type=int, default=0,

	help="switch to display image on screen")

args = vars(ap.parse_args())



# grab a list of all NCS devices plugged in to USB

print("[BILGI] NCS cihazi araniyor...")

devices = mvnc.EnumerateDevices()



# if no devices found, exit the script

if len(devices) == 0:

	print("[BILGI] NCS cihazi bulunmadi")

	quit()



# use the first device since this is a simple test script

# (you'll want to modify this is using multiple NCS devices)

print("[BILGI] cihaz {} bulundu. device0 de kullanilabilir. "

	"cihaz aciliyor...".format(len(devices)))

device = mvnc.Device(devices[0])

device.OpenDevice()



# open the CNN graph file

print("[BILGI] Graph dosyasi Raspberry Pi 3te yUkleniyor...")

with open(args["graph"], mode="rb") as f:

	graph_in_memory = f.read()



# load the graph into the NCS

print("[BILGI] Graph Dosyasi NCSye gidiyor...")

graph = device.AllocateGraph(graph_in_memory)



# open a pointer to the video stream thread and allow the buffer to

# start to fill, then start the FPS counter

print("[BILGI] FPS Sayici Basliyor")

vs = VideoStream(0).start()

time.sleep(1)

fps = FPS().start()





# loop over frames from the video file stream

def video_stream():

	while True:

		try:

		

			global a

			global my_secim

			global secim

			global secimm

			# grab the frame from the threaded video stream

			# make a copy of the frame and resize it for display/video purposes

			frame = vs.read()

			image_for_result = frame.copy()

			image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

			#nesneler=np.zeros(10)

			#nesneler1=np.zeros(10)

			#nesneler_conf=np.zeros(10)

			# use the NCS to acquire predictions

			predictions = predict(frame, graph)

			sinif=[]

			b=0

			# loop over our predictions

			for (i, pred) in enumerate(predictions):

				# extract prediction data for readability

				(pred_class, pred_conf, pred_boxpts) = pred

				

				#kare.append(pred_boxpts[1][0]+pred_boxpts[0][0])

#				sirali_kare.append(pred_boxpts[1][0]-pred_boxpts[0][0])

#				sirali_kare.sort(reverse=True)

				sinif.append(pred_class)



					



				# filter out weak detections by ensuring the `confidence`

				# is greater than the minimum confidence

				if pred_conf > args["confidence"]:

					# print prediction to terminal



					print("[BILGI] Nesne={}, Tahmin Orani={},".format(CLASSES[pred_class], pred_conf))



					# check if we should show the prediction data

					# on the frame

					if args["display"] > 0:

						#print(my_selection)

						# build a label consisting of the predicted class and

						# associated probability

						i=i+1

						try:

							if i>1:

								for j in range(0,i-1,1):

									if sinif[j]==sinif[j+1]:

										b=b+1

										#print str(b) + "tane ayni sinif var"

										label = "{}: {:.2f}%".format(CLASSES[pred_class]+ str(b),pred_conf * 100)

								

									elif sinif[j]!=sinif[j+1]:

										b=0

									else:

										break

							else:

								b=0

						except:

							break

						

						label = "{}: {:.2f}%".format(CLASSES[pred_class]+ str(b),

								pred_conf * 100)

					

						

						

						# extract information from the prediction boxpoints

						(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])

						ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)

						ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)

						(startX, startY) = (ptA[0], ptA[1])

						y = startY - 15 if startY - 15 > 15 else startY + 15

						# display the rectangle and label text

						cv2.rectangle(image_for_result, ptA, ptB,

							COLORS[pred_class], 2)

						cv2.circle(image_for_result,(300,300), 5,(255,0,0),-1)

						cv2.putText(image_for_result, label, (startX, y),

						cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)



						

						if not CLASSES[pred_class]+ str(b)==liste[my_selection-1]+str(secimm):

							continue

						nesnenin_ortasi=(ptA[0]+ptB[0])/2

						x3=600-nesnenin_ortasi

						#print nesnenin_ortasi

						if x3<150:

							print ("NESNE SOLDA")

							#print a

							#50 ile 250 arasi

							a=a-8

							wiringpi.pwmWrite(18,a)

					#		time.sleep(delay_period)

							if a<=55:

								a=65

						elif x3>450:

							print ("NESNE SAGDA")

							#print a

							a=a+8

							wiringpi.pwmWrite(18,a)

					#		time.sleep(delay_period)					

							if a>=240:

								a=235

						elif x3>150 and x3<450:

							print("NESNE ORTADA")

							#print a

							wiringpi.pwmWrite(18,a)

	 #						time.sleep(delay_period)

				

			 #			if CLASSES[pred_class]==liste[my_selection-1]:

					#		break

		

						i=0

		

				

			# check if we should display the frame on the screen

			# with prediction data (you can achieve faster FPS if you

			# do not output to the screen)

			if args["display"] > 0:

				# display the frame to the screen

				cv2.imshow("Output", image_for_result)

				key = cv2.waitKey(1) & 0xFF



				# if the `q` key was pressed, break from the loop

				if key == ord("q"):

					break



			# update the FPS counter

			fps.update()

	

		# if "ctrl+c" is pressed in the terminal, break from the loop

		except KeyboardInterrupt:

			break



		# if there's a problem reading a frame, break gracefully

		except AttributeError:

			break





th= threading.Thread(target=video_stream) #initialise the thread

th.setDaemon(True)

th.start() #start the thread

label = Label(top)  

label.pack()  

top.mainloop()

# stop the FPS counter timer

fps.stop()





# destroy all windows if we are displaying them

if args["display"] > 0:

	cv2.destroyAllWindows()

# stop the video stream

vs.stop()



# clean up the graph and device

graph.DeallocateGraph()

device.CloseDevice()





# display FPS information

print("[BILGI] Gecen zaman: {:.2f}".format(fps.elapsed()))

print("[BILGI] Yaklasik FPS: {:.2f}".format(fps.fps()))





