import cv2
from ultralytics import YOLO
import numpy as np
from LaneDetection import * # importing lane detection task
import time

############### Object Detection + Distance Estimation + Data Pipepine #################

model = YOLO("best.pt")

focal_length2 = ((103*280)/138) # for rider
focal_length1 = ((43*2000)/175) # for car

def vehicle_ahead_in_region(region, centerxy):
    region_left = np.array([(0, 500), (558, 425), (0, 638), (290, 638)], np.int32)
    region_center = np.array([(559, 425), (559, 425), (291, 638), (785, 638)], np.int32)
    region_right = np.array([(560, 425), (1138, 400), (786, 638), (1138, 638)], np.int32)

    regions = {'left': region_left, 'center': region_center, 'right': region_right}
    target_region = regions[region]


    if cv2.pointPolygonTest(target_region, centerxy, False) >= 0:
        return True
    return False


def process_boxes(image, focal_length1, focal_length2, left_lane_type, mid_lane_type, right_lane_type):

    results = model.predict(image, line_width=10, device=0)
    boxes = results[ 0 ].boxes

    color = [(255,150,0),(0,69,255),(0,165, 255),(128,0,128),(0,69,255)]

    min_bicycle_distance = 6
    min_vehicle_distance = 18

    car_on_left = False
    car_in_front = False
    car_on_right = False
    bicycle_left = False
    bicycle_center = False
    bicycle_right = False

    num_f = 90
    f_c = 0

    for box in boxes:
        cords = box.xyxy[ 0 ].tolist()
        cls = box.cls[ 0 ].item()
        conf = box.conf[ 0 ].item()


        if cls in [0,2]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_obj = x2 - x1


            if cls == 0:
                W = 175
                focal_length = focal_length1
            elif cls == 2:
                W = 138
                focal_length = focal_length2

            d_obj = ((W*focal_length)/w_obj)

            x1, y1, x2, y2 = map(round, [ x1, y1, x2, y2 ])
            # print('cls:', cls)
            # print('FL:', focal_length)
            # print('w_px:', w_obj)
            # print('W:', W)
            d_obj = int(round(d_obj / 100))
           # print('dist:', d_obj)

            # print('-----------^')
            if d_obj < 21:

                label01 = "%.fm" % (d_obj)
                label = "%s:%.fm" % (results[0].names[cls], d_obj)

                cv2.rectangle(image, (x1, y1), (x2, y2), color[int(cls)], 2)
                cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(cls)], 2)

                if cls == 0:  # Car
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center_xy = (center_x, center_y)

                    if vehicle_ahead_in_region('left', center_xy):
                        car_on_left = True
                    elif vehicle_ahead_in_region('center', center_xy):
                        car_in_front = True
                    elif vehicle_ahead_in_region('right', center_xy):
                        car_on_right = True

                    if d_obj < min_vehicle_distance:
                        min_vehicle_distance = d_obj

                elif cls == 2:  # Bicycle
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center_xy = (center_x, center_y)
                    center_L = (x2, y2)
                    center_R = (x1, y1)

                    if vehicle_ahead_in_region('left', center_xy):
                        bicycle_left = True
                        #closest_bicycle_position = 'left'
                    elif vehicle_ahead_in_region('center', center_xy):
                        bicycle_center = True
                       # closest_bicycle_position = 'center'
                    elif vehicle_ahead_in_region('right', center_xy):
                        bicycle_right = True
                       # closest_bicycle_position = 'right'

                    if d_obj < min_bicycle_distance:
                        min_bicycle_distance = d_obj

            # label = "%s:%.fm : %.f%%" % (results[0].names[cls], d_obj, conf)
            # label01 = "%.fm" % (d_obj)
            #
            # cv2.rectangle(image, (x1, y1), (x2, y2), color[int(cls)], 1)
            # cv2.putText(image, label01, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[int(cls)], 1)

    # To Display Result/ Output on Frame.
    def text_display(text, x,y):
        f_center = (image.shape[1] // 2, 30)
        f_color = (0, 0, 0)  # black color
        return cv2.putText(image, text, (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, f_color, 2)

    if bicycle_center and min_bicycle_distance < 6 and left_lane_type == 'Solid' and mid_lane_type == 'Solid':
         print("Solid Lane. Slow Down and Keep Follow")
         text = ("Solid Lane. Slow Down and Keep Follow")
         text_display(text, 220, 25)
    elif bicycle_center and min_bicycle_distance < 6 :
        print("Slow Down! Bicycle detected within 5 meter in center lane.")
        text = ("Slow Down! Bicycle detected within 5 meter in center lane.")
        text_display(text, 220, 25)
        if car_on_left and car_on_right and car_in_front and min_vehicle_distance < 16:
            print("Cars detected on both sides. Cannot overtake the bicycle. Slow Down and Follow It")
            text = ("Cars detected on both sides. Cannot overtake the bicycle. Slow Down and Follow It")
            text_display(text, 220, 85)
        elif not car_on_left and not car_in_front and left_lane_type != 'Solid':
            print("No car detected on the left lane. Safe to overtake from the left.")
            text = ("No Car Ahead in Near. Safe to Overtake from the left.")
            text_display(text, 220, 85)
        elif not car_on_right and not car_in_front and mid_lane_type != 'Solid':
            print("No car detected on the right lane. Safe to overtake from the right.")
            text = ("No Car Ahead in Near. Safe to overtake from the right.")
            text_display(text, 220, 85)
    elif bicycle_left and min_bicycle_distance < 6:
        print("Bicycle detected within 5 meters in the left lane.")
        text = ("Bicycle detected within 5 meters in the left lane.")
        text_display(text, 200, 30)
        if car_in_front and car_on_right and min_vehicle_distance < 15:
            print("Cars Ahead in the Right and Center Lanes. Cannot Overtake.")
            text = ("Cars Ahead in the Right and Center Lanes.")
            text_display(text, 200, 45)
        elif car_in_front and not car_on_right and min_vehicle_distance < 15:
            print("Car Ahead in the Right Lane. Follow Center Lane Only.")
            text = ("Car Ahead. Cannot Overtake. Follow Center Lane Only.")
            text_display(text, 200, 45)
        elif not car_in_front and not car_on_right and mid_lane_type != 'Solid':
            print("No Car Ahead. Safely Can Keep Driving or Overtaking")
            text = ("No Car Ahead. Safely Can Keep Driving or Overtaking")
            text_display(text, 200, 45)
    elif bicycle_right and min_bicycle_distance < 6:
        print("Bicycle detected within 5 meters in the right lane.")
        text = ("Bicycle detected within 5 meters in the right lane.")
        text_display(text, 200, 30)
        if car_on_left and car_in_front and min_vehicle_distance < 15:
            print("Cars detected in the left and center lanes. Cannot overtake the bicycle.")
            text = ("Cars detected in the left and center lanes. Cannot overtake the bicycle.")
            text_display(text, 200, 45)
        elif car_on_left and not car_in_front and min_vehicle_distance < 15:
            print("Cars detected in the Left lane. Follow Center Lane Only.")
            text = ("Cars detected in the Left lane. Follow Center Lane Only.")
            text_display(text, 200, 45)
        elif not car_in_front and not (car_on_left and min_vehicle_distance < 14) and left_lane_type != 'Solid':
            print("No Car Ahead. Safely Can Keep Driving or Overtaking")
            text = ("No Car Ahead. Safely Can Keep Driving or Overtaking.")
            if f_c < 200:
                text_display(text, 200, 45)
                f_c += 1
    else:
        print("Can Keep Driving")
        text = ("Can Keep Driving")
        text_display(text, 200, 25)

    return image

# img = cv2.imread('refimg4bike02.png')
# image = cv2.resize(img, (1136, 639))
#
# outimg = process_boxes(image, focal_length1, focal_length2)
# cv2.imwrite(r'C:\Users\abm_0\OneDrive\Desktop\RefEstImg01.jpg', outimg)

video_path = "./images/F_01.mp4"
##cv2.resize(img, (1136,639))
cap = cv2.VideoCapture(video_path)

#----------OUT VID----------
#output_path = "C:\\Users\\abm_0\\OneDrive\\Desktop\\outimgs\\09.mp4"
#frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#codec = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter(output_path, codec, 30, (1136, 639))


while cap.isOpened():

    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (1136, 639))

        start_time = time.perf_counter()

        #results = model.predict(frame, show_labels=True)


        out_cv, swindow, left_lane_type, mid_lane_type, right_lane_type = detect_lanes(frame)

        out_yolo = process_boxes(frame, focal_length1, focal_length2, left_lane_type, mid_lane_type, right_lane_type)

        #results = score_frame(out_img)
        #frame = plot_boxes(results, out_img)

        #out_yolo = process_boxes(frame, focal_length1, focal_length2)

        merged_out = cv2.addWeighted(out_yolo,1,out_cv,0.4,0)

        end_time = time.perf_counter()
        fpss = 1 / round(end_time - start_time, 3)
        cv2.putText(frame, f'FPS: {int(fpss)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        #annotated_frame = results[0].plot()


        cv2.imshow("YOLOv8 Inference", merged_out)
        #out.write(merged_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
#out.release()

# Closes all the frames
cv2.destroyAllWindows()