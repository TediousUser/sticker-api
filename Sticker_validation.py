from ultralytics import YOLO
import os
import random
import shutil
from pathlib import Path
import torch
import cv2
import glob
import numpy as np

class StickerValidation():

    def __init__(self):
        # Load the YOLOv8 model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO('best_100.pt').to(self.device)  

    def StickerDetection(self,image_path):

        frame = cv2.imread(image_path)
        results = self.model(frame)
        annotated_frame = results[0].plot(labels=True,conf=False)
        for result in results:
            boxes = result.boxes
            class_ids = boxes.cls.tolist()  # List of class IDs
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]

        return class_names
    
    def get_largest_rectangle_contour(self,contours, min_area=1000):

        largest_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                if area > max_area:
                    max_area = area
                    largest_contour = box
        return largest_contour

    def get_skew_angle_from_box(self,box):
        # Use box points to compute angle
        pt1, pt2 = box[0], box[1]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def check_sticker_alignment(self,image_path, angle_threshold=2):
        img = cv2.imread(image_path)
        if img is None:
            print("Image not found.")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box = self.get_largest_rectangle_contour(contours)

        if box is None:
            print("No significant contour found.")
            return

        angle = self.get_skew_angle_from_box(box)
        # print(angle)
        aligned = abs(angle) <= angle_threshold

        return aligned
    
    def check_sticker_centering_yolo(self,image_path, conf_threshold=0.3, max_offset=20):
        
        StickerFlag = True
        # Read image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        battery_center = (w // 2, h // 2)

        # Inference
        results = self.model(img)[0]
        boxes = results.boxes

        # Filter by confidence
        boxes = [box for box in boxes if box.conf.item() > conf_threshold]

        # print(boxes)
        if not boxes:
            StickerFlag = False
            return StickerFlag

        # Assume first detection is the sticker (or filter based on class if multi-class model)
        box = boxes[0].xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = box
        sticker_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Distance from battery center
        dx = abs(sticker_center[0] - battery_center[0])
        dy = abs(sticker_center[1] - battery_center[1])

        centered = dx <= max_offset and dy <= max_offset

        status = "Centered" if centered else "Not Centered"
    
        if StickerFlag != False :
            if status == "Centered":
                return True
            else :
                return False
            

    def CheckSticker(self,img_path):

        angle_alignment = self.check_sticker_alignment(img_path)
        sticker_center = self.check_sticker_centering_yolo(img_path)

        if not (angle_alignment and sticker_center):

            message= "Not Aligned"

        else:
            message = "Aligned"


        return message
    
    def StickerStatus(self,image_path):
        detection = self.StickerDetection(image_path)

        if len(detection) != 0:

            det_list = detection[0].split(" ")

            if "Top" not in det_list:

                message = self.CheckSticker(image_path)

                if message == "Not Aligned":
                    return "NOK" + "," + message
                
                elif message == "Aligned":
                    return "OK" 

            else :
                message = detection[0]
                return "OK" + "," + "Sticker present"

        else :
            message = "NOK" + "," + "Sticker absent"
            return message
    

    
    

