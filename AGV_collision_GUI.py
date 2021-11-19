import cv2
import numpy as np
import time
import datetime
import glob
from tkinter import *
from tkinter import ttk, messagebox
import RPi.GPIO as GPIO
import configparser

def generate_ini(cfg_name = r'/home/auo/AGV/collision_cfg.ini'):
    msg_result = messagebox.askokcancel("Notification", "是否覆蓋現有設定檔？")
    if msg_result == True:
        with open(cfg_name, 'w') as cfgfile:
            cfg = configparser.ConfigParser()
            sectione_name = ['Collision']
            for i in sectione_name:
                cfg.add_section(i)
                if i == 'Collision':
                    cfg.set(i, 'lane_left_top_x', str(lane_left_top.get()))
                    cfg.set(i, 'warn_left_bottom_x', str(lane_left_bottom.get()))
                    cfg.set(i, 'lane_right_top_x', str(lane_right_top.get()))
                    cfg.set(i, 'warn_right_bottom_x', str(lane_right_bottom.get()))
                    cfg.set(i, 'warn_top_y', str(warn_line.get()))
                    cfg.set(i, 'stop_top_y', str(stop_line.get()))
            cfg.write(cfgfile)
            cfgfile.close()
            #lane_cfg.destroy()
            #cv2.destroyAllWindows()

def exit_btn():
    msg_result = messagebox.askokcancel("Notification", "確定離開？")
    if msg_result == True:
        lane_cfg.destroy()
        cv2.destroyAllWindows()

#讀取設定檔
config = configparser.ConfigParser()
config.read("/home/auo/AGV/collision_cfg.ini")
lane_left_top_x = config["Collision"]["lane_left_top_x"]
warn_left_bottom_x = config["Collision"]["warn_left_bottom_x"]
lane_right_top_x = config["Collision"]["lane_right_top_x"]
warn_right_bottom_x = config["Collision"]["warn_right_bottom_x"]
warn_top_y = config["Collision"]["warn_top_y"]
stop_top_y = config["Collision"]["stop_top_y"] 


sec = 5

warn_frame_max = 5
OK_frame_max = 10

RelayA = [21, 20] #[21, 20, 26]
CH1 = 21
CH2 = 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(RelayA, GPIO.OUT, initial=GPIO.HIGH)
time.sleep(1)		

lane_cfg = Tk()

net = cv2.dnn_DetectionModel('/home/auo/AGV/cfg/custom-yolov4-tiny_14.cfg', '/home/auo/AGV/model/custom-yolov4-tiny_14_69.weights')
net.setInputSize(416, 416)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

#net = cv2.dnn.readNet('custom-yolov4-detector_last.weights', 'custom-yolov4-detector.cfg')
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#model = cv2.dnn_DetectionModel(net)
#model.setInputParams(size = (512, 512), scale = 1.0/255)

vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = vc.read()

img_width = frame.shape[1]
img_heigth = frame.shape[0]
lane_left_top = Scale(lane_cfg, label='左車道線頂部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_left_bottom = Scale(lane_cfg, label='左車道線底部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_right_top = Scale(lane_cfg, label='右車道線頂部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_right_bottom = Scale(lane_cfg, label='右車道線底部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
warn_line = Scale(lane_cfg, label='減速範圍', length=100, from_=int(img_heigth/2), to=img_heigth - 1)
stop_line = Scale(lane_cfg, label='停止範圍', length=100, from_=int(img_heigth/2), to=img_heigth - 1)
chk_btn = Button(lane_cfg, bg='orange', text='更新設定檔', command=generate_ini)
exit_btn = Button(lane_cfg, bg='orange', text='離開程式', command=exit_btn)
'''
lane_left_top.set(int(img_width/3)) 
lane_left_bottom.set(int(img_width/4))
lane_right_top.set(int(img_width/3*2))
lane_right_bottom.set(int(img_width/4*3))
warn_line.set(int(img_heigth/4*3))
stop_line.set(int(img_heigth/5*4))
'''
lane_left_top.set(int(lane_left_top_x))
lane_left_bottom.set(int(warn_left_bottom_x))
lane_right_top.set(int(lane_right_top_x))
lane_right_bottom.set(int(warn_right_bottom_x))
warn_line.set(int(warn_top_y))
stop_line.set(int(stop_top_y))

chk_btn.config(height='2')
exit_btn.config(height='2')

lane_left_top.pack()
lane_left_bottom.pack()
lane_right_top.pack()
lane_right_bottom.pack()
warn_line.pack()
stop_line.pack()
chk_btn.pack(fill='x')
exit_btn.pack(fill='x')

size = (img_width, img_heigth)
#out = cv2.VideoWriter('output/G0210936.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 3, (480,360))

#video to frame & frame to video
fps = vc.get(cv2.CAP_PROP_FPS)
print(fps)
warn_frame_count = 0 #物體出現在警戒區內的次數
warn_bool = False #檢查是否有物體出現在警戒區內
stop = False
slow = False
tStart = time.time()
while(1):
	ret, frame = vc.read()
	if time.time() - tStart > sec:
		
		now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
		cv2.imwrite("/home/auo/AGV/1111/" + now + ".jpg", frame)
		tStart = time.time()
	if frame is not None:
	#if idx % 20 == 0:
		#if idx % 100 == 0:
		#	print("processing " + str(idx) + " images")
		#start = time.time()
		classes, confidences, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.4) #0.9s
		lane_cfg.update()
		#車道左線頂 x
		lane_left_top_x = lane_left_top.get()
		#車道右線頂 x
		lane_right_top_x = lane_right_top.get()
		#警戒區底線左 x (車道左線底 x)
		warn_left_bottom_x = lane_left_bottom.get()
		#警戒區底線右 x (車道右線底 x)
		warn_right_bottom_x = lane_right_bottom.get()
		#警戒區頂線 y
		warn_top_y = warn_line.get()
		#警戒區停止線 y
		warn_stop_y = stop_line.get()
		#警戒區底線 y
		warn_bottom_y = img_heigth - 1
		#先給警戒線上黑色，避免找錯警戒角點
		cv2.line(frame, (0, warn_top_y), (img_width - 1, warn_top_y), (0,0,0), 5)
		#畫綠車道線
		cv2.line(frame, (lane_left_top_x, int(img_heigth/5)), (warn_left_bottom_x, img_heigth - 1), (0,255,0), 5)
		cv2.line(frame, (lane_right_top_x, int(img_heigth/5)), (warn_right_bottom_x, img_heigth - 1), (0,255,0), 5)
		#找警戒線與車道的交叉點
		warn_line_where = np.where(frame[warn_top_y,:,1] == 255)
		#警戒區頂線左 x
		warn_left_top_x = warn_line_where[0][0]
		#警戒區頂線右 x
		warn_right_top_x = warn_line_where[0][len(warn_line_where[0]) - 1]
		#產生警戒區域遮罩
		warn_mask = np.zeros((img_heigth, img_width, 3), dtype=np.uint8)
		#roi_corners = np.array([[(580,815), (1305,815), (1458,1079), (432,1079)]], dtype=np.int32)
		roi_corners = np.array([[(warn_left_top_x, warn_top_y), (warn_right_top_x, warn_top_y), (warn_right_bottom_x, warn_bottom_y), (warn_left_bottom_x, warn_bottom_y)]], dtype=np.int32)
		channel_count = 3
		ignore_mask_color = (255,)*channel_count
		cv2.fillPoly(warn_mask, roi_corners, ignore_mask_color)
		stop = False
		slow = False
		#綠線
		#cv2.line(frame, (lane_left_top_x, 282), (warn_left_bottom_x, 1079), (0,255,0), 5)
		#cv2.line(frame, (lane_right_top_x, 282), (warn_right_bottom_x, 1079), (0,255,0), 5)
		#黃警戒線
		cv2.line(frame, (0, warn_top_y), (img_width - 1, warn_top_y), (0,255,255), 5)
		#紅警戒線
		cv2.line(frame, (0, warn_stop_y), (img_width - 1, warn_stop_y), (0,0,255), 5)
		#end = time.time()
		#print(end - start)
		warn_bool = False
		warn_sum = 0
		if len(boxes) > 0:
			for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
				pstring = str(int(100 * confidence)) + "%"
				x_left, y_top, width, height = box
		#		x_center = int(x_left + (width/2)) + 205
		#		y_center = int(y_top + (height/2)) + 90
		#		pyautogui.moveTo(x_center, y_center, duration=0.25)
		#		time.sleep(0.5)
		#		pyautogui.click()
		#		time.sleep(0.5)
				boundingBox = [
					(x_left, y_top), #左上頂點
					(x_left, y_top + height), #左下頂點
					(x_left + width, y_top + height), #右下頂點
					(x_left + width, y_top) #右上頂點
				]
				textCoord = (x_left, y_top - 10)
				if classId == 0:
					rectColor = (0, 0, 255)
					pstring = "cones " + pstring
				elif classId == 1:
					rectColor = (0, 255, 0)
					pstring = "cart " + pstring
				elif classId == 2:
					rectColor = (255, 0, 255)
					pstring = "pp_box " + pstring
				elif classId == 3:
					rectColor = (0, 255, 255)
					pstring = "person " + pstring
				elif classId == 4:
					rectColor = (255, 255, 0)
					pstring = "spacer " + pstring
				elif classId == 5:
					rectColor = (128, 128, 255)
					pstring = "stacker " + pstring
				elif classId == 6:
					rectColor = (128, 255, 0)
					pstring = "trolley " + pstring
				elif classId == 7:
					rectColor = (255, 128, 0)
					pstring = "board " + pstring
				elif classId == 8:
					rectColor = (255, 255, 128)
					pstring = "glass_set " + pstring
				elif classId == 9:
					rectColor = (255, 128, 128)
					pstring = "forklift " + pstring
				elif classId == 10:
					rectColor = (255, 0, 128)
					pstring = "chair " + pstring
				elif classId == 11:
					rectColor = (128, 0, 255)
					pstring = "footboard " + pstring
				elif classId == 12:
					rectColor = (128, 255, 255)
					pstring = "fire extinguisher " + pstring
				elif classId == 13:
					continue
				#if int(100 * confidence) > 30:
				# 在影像中標出Box邊界和類別、信心度
				cv2.rectangle(frame, boundingBox[0], boundingBox[2], rectColor, 2)
				cv2.putText(frame, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
				#判斷角點是否在警戒區域內
				mask_obj_bottom = warn_mask[y_top + height - 1, x_left:x_left + width - 1] 
				warn_sum = np.sum(mask_obj_bottom == 255) / 3.0 #obj底線共有多少個點在警戒區內
				#if 255 in warn_mask[y_top + height - 1, x_left:x_left + width - 1]:
				if warn_sum / len(mask_obj_bottom) > 0.3:
					warn_bool = True
					warn_frame_count = warn_frame_count + 1
					if warn_frame_count > warn_frame_max:
						#判斷角點落在黃區 or 紅區
						if (y_top + height) <= warn_stop_y:
							OK_frame_max = 0
							slow = True
						elif (y_top + height) > warn_stop_y:
							OK_frame_max = 0
							stop = True
			if not warn_bool:
				warn_frame_count = 0
				OK_frame_max = OK_frame_max + 1
			if stop:
				cv2.putText(frame, "Stop", (int(img_width/3),int(img_heigth/5)), cv2.FONT_HERSHEY_DUPLEX, 5, (0,0,255), 8)
				GPIO.output(CH1, GPIO.LOW)
				GPIO.output(CH2, GPIO.HIGH)
			elif slow:
				cv2.putText(frame, "Slow Down", (int(img_width/3),int(img_heigth/5)), cv2.FONT_HERSHEY_DUPLEX, 5, (0,128,255), 8)
				GPIO.output(CH1, GPIO.HIGH)
				GPIO.output(CH2, GPIO.LOW)
		else:
			OK_frame_max = OK_frame_max + 1
		if OK_frame_max >= 10:
                        slow = False
                        stop = False
			GPIO.output(CH1, GPIO.HIGH)
			GPIO.output(CH2, GPIO.HIGH)
		#frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
		#out.write(frame)
		#warn_mask = cv2.resize(warn_mask, (640, 480), interpolation=cv2.INTER_AREA)
		frame_new = np.hstack((warn_mask,frame))
		cv2.imshow("frame", frame_new)
		#cv2.imshow("mask", warn_mask)
		cv2.waitKey(1)
	else:
		continue
out.release()
vc.release()  
