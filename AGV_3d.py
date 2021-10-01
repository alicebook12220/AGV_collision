import time
import glob
import configparser
import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
import RPi.GPIO as GPIO

import pyrealsense2 as rs
import numpy as np
import cv2

import datetime
time.sleep(10)
def generate_ini(cfg_name = r'/home/auo/realsense/collision_cfg.ini'):
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
                    cfg.set(i, 'slow_dis', str(slow_value.get()))
                    cfg.set(i, 'stop_dis', str(stop_value.get()))
                    cfg.set(i, 'slow_obj_pixel', str(slow_obj.get()))
                    cfg.set(i, 'stop_obj_pixel', str(stop_obj.get()))
            cfg.write(cfgfile)
            cfgfile.close()
            #lane_cfg.destroy()
            #cv2.destroyAllWindows()

def exit_btn():
    msg_result = messagebox.askokcancel("Notification", "確定離開？")
    if msg_result == True:
        lane_cfg.destroy()
        cv2.destroyAllWindows()
        out.release()
#錄影
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/home/auo/realsense/video/" + now + ".mp4", fourcc, 15.0, (1280, 480))
#讀取設定檔
config = configparser.ConfigParser()
config.read("/home/auo/realsense/collision_cfg.ini")
lane_left_top_x = config["Collision"]["lane_left_top_x"]
warn_left_bottom_x = config["Collision"]["warn_left_bottom_x"]
lane_right_top_x = config["Collision"]["lane_right_top_x"]
warn_right_bottom_x = config["Collision"]["warn_right_bottom_x"]
warn_top_y = config["Collision"]["warn_top_y"]
slow_dis = config["Collision"]["slow_dis"] 
stop_dis = config["Collision"]["stop_dis"] 
slow_obj_pixel = config["Collision"]["slow_obj_pixel"]
stop_obj_pixel = config["Collision"]["stop_obj_pixel"]
#可容許警告的frame數量
warn_frame_max = 15
#繼電器腳位
RelayA = [21, 20] #[21, 20, 26]
CH1 = 21
CH2 = 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(RelayA, GPIO.OUT, initial=GPIO.HIGH)
time.sleep(1)		

#tk GUI介面
lane_cfg = Tk()
lane_cfg.title("避障參數設定")

#----------取得realsense 3d相機影像------------------
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
# Convert images to numpy arrays
color_image = np.asanyarray(color_frame.get_data())
#----------取得realsense 3d相機影像------------------

img_width = color_image.shape[1]
img_heigth = color_image.shape[0]
lane_left_top = Scale(lane_cfg, label='左車道線頂部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_left_bottom = Scale(lane_cfg, label='左車道線底部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_right_top = Scale(lane_cfg, label='右車道線頂部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
lane_right_bottom = Scale(lane_cfg, label='右車道線底部 x 座標', length=300, from_=0, to=img_width - 1, orient=HORIZONTAL)
slow_value = Scale(lane_cfg, label='減速距離（毫米）', length=300, from_=0, to=3000, orient=HORIZONTAL)
stop_value = Scale(lane_cfg, label='停止距離（毫米）', length=300, from_=0, to=2000, orient=HORIZONTAL)
slow_obj = Scale(lane_cfg, label='減速物體大小（像素數量）', length=300, from_=0, to=100000, orient=HORIZONTAL)
stop_obj = Scale(lane_cfg, label='停止物體大小（像素數量）', length=300, from_=0, to=100000, orient=HORIZONTAL)
warn_line = Scale(lane_cfg, label='警戒範圍', length=100, from_=0, to=img_heigth - 1)

chk_btn = Button(lane_cfg, bg='orange', text='更新設定檔', command=generate_ini)
exit_btn = Button(lane_cfg, bg='orange', text='離開程式', command=exit_btn)
statusbar = Label(lane_cfg, text="Text", bd=2, relief=SUNKEN, anchor=W)

#車道線及警戒線初始值
lane_left_top.set(int(lane_left_top_x))
lane_left_bottom.set(int(warn_left_bottom_x))
lane_right_top.set(int(lane_right_top_x))
lane_right_bottom.set(int(warn_right_bottom_x))
slow_value.set(int(slow_dis))
stop_value.set(int(stop_dis))
slow_obj.set(int(slow_obj_pixel))
stop_obj.set(int(stop_obj_pixel))
warn_line.set(int(warn_top_y)) 

chk_btn.config(height='2')
exit_btn.config(height='2')

statusbar.configure(text="Text Updated")

lane_left_top.pack()
lane_left_bottom.pack()
lane_right_top.pack()
lane_right_bottom.pack()
slow_value.pack()
stop_value.pack()
slow_obj.pack()
stop_obj.pack()
warn_line.pack()
chk_btn.pack(fill='x')
exit_btn.pack(fill='x')
statusbar.pack(side='bottom', fill='x')

warn_frame_count = 0 #物體出現在警戒區內的次數
video_status = True
start = time.time()
try:
    while True:
        if time.time() - start > 180:
            out.release()
            video_status = True
        if video_status:
            #錄影
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(now + ".mp4", fourcc, 30.0, (1280, 480))
            video_status = False
        lane_cfg.update()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

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
        #警戒區底線 y
        warn_bottom_y = img_heigth - 1
        #先給警戒線上黑色，避免找錯警戒角點
        cv2.line(color_image, (0, warn_top_y), (img_width - 1, warn_top_y), (0,0,0), 5)
        #畫綠車道線
        cv2.line(color_image, (lane_left_top_x, 0), (warn_left_bottom_x, img_heigth - 1), (0,255,0), 5)
        cv2.line(color_image, (lane_right_top_x, 0), (warn_right_bottom_x, img_heigth - 1), (0,255,0), 5)
        #找警戒線與車道的交叉點
        warn_line_where = np.where(color_image[warn_top_y,:,1] == 255)
        #警戒區頂線左 x
        warn_left_top_x = warn_line_where[0][0]
        #警戒區頂線右 x
        warn_right_top_x = warn_line_where[0][len(warn_line_where[0]) - 1]
        #產生警戒區域遮罩
        warn_mask = np.zeros((img_heigth, img_width, 1), dtype=np.uint8)
        roi_corners = np.array([[(warn_left_top_x, warn_top_y), (warn_right_top_x, warn_top_y), (warn_right_bottom_x, warn_bottom_y), (warn_left_bottom_x, warn_bottom_y)]], dtype=np.int32)
        channel_count = 1
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(warn_mask, roi_corners, ignore_mask_color)
        warn_mask = np.squeeze(warn_mask) #(480,640,1) to (480,640)
        stop = False
        slow = False
        #黃警戒線
        cv2.line(color_image, (0, warn_top_y), (img_width - 1, warn_top_y), (0,255,255), 5)

        warn_bool = False
        warn_sum = 0

        #判斷是否有物體在警戒區域內
        #減速距離
        slow_dis_val = slow_value.get()
        #停止距離
        stop_dis_val = stop_value.get()
        #物體大小
        slow_obj_val = slow_obj.get()
        stop_obj_val = stop_obj.get()
        #計算近距離pixel數量
        depth_warn_mask = depth_image[warn_mask == 255]
        depth_warn_mask[depth_warn_mask == 0] = 50000
        slow_depth = depth_warn_mask[depth_warn_mask < slow_dis_val]
        stop_depth = depth_warn_mask[depth_warn_mask < stop_dis_val]
        statusbar_text = "slow area:" + str(len(slow_depth)) + " stop area:" + str(len(stop_depth))
        statusbar.configure(text = statusbar_text)
        
        if len(stop_depth) > stop_obj_val:
            cv2.putText(color_image, "Stop", (int(img_width/4),int(img_heigth/4)), cv2.FONT_HERSHEY_DUPLEX, 5, (0,0,255), 6)
            warn_frame_count = warn_frame_count + 1
            GPIO.output(CH1, GPIO.LOW)
            GPIO.output(CH2, GPIO.HIGH)
        elif len(slow_depth) > slow_obj_val:
            cv2.putText(color_image, "Slow", (int(img_width/4),int(img_heigth/4)), cv2.FONT_HERSHEY_DUPLEX, 5, (0,128,255), 6)
            warn_frame_count = warn_frame_count + 1
            GPIO.output(CH1, GPIO.HIGH)
            GPIO.output(CH2, GPIO.LOW)
        else:
            warn_frame_count = 0
            GPIO.output(CH1, GPIO.HIGH)
            GPIO.output(CH2, GPIO.HIGH)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        out.write(images)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
    out.release()
