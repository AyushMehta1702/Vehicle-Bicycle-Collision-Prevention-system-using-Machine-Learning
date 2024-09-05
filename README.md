
# Bi-Cycle Detection using Machine Learning for Situation Awareness

## Motivation

With the growing global emphasis on healthy lifestyles and environmental protection, there has been a significant rise in the adoption of eco-friendly transportation, such as bicycles. However, the safety of cyclists sharing the road with vehicles is a pressing concern. The increasing number of accidents involving cyclists and vehicles highlights the need for intelligent systems that can aid in accident prevention and ensure safer bicycle-vehicle interactions.

## Problem Statement

Cyclist safety remains a critical issue due to the severity of accidents when they share roads with vehicles. The main factors contributing to these accidents include inadequate visibility, improper vehicle maneuvers, and lack of cyclist awareness by drivers. This research addresses the urgent need to develop an intelligent system that can prevent accidents by detecting bicycles, analyzing vehicle-bicycle interaction scenarios, and aiding in decision-making to enhance road safety.

## Objective

The objective of this research is to develop a comprehensive Bicycle Detection System using Machine Learning and Computer Vision techniques. The system integrates various components:

- **Lane Detection:** Identifying lane boundaries and categorizing them.
- **Object Detection:** Detecting bicycles, vehicles, and other relevant objects.
- **Distance Estimation:** Calculating the distance between detected objects and the vehicle.
- **Situation Analysis:** Understanding the vehicle-bicycle interaction scenarios.
- **Decision Making:** Providing recommendations based on the detected situations, such as whether to stay behind, overtake, or continue driving.

## Methodology

### 1. Process Flow Overview
- **Input Frame:** Video frames are fed into the system.
- **Lane Detection:** Detects lanes using image processing techniques such as color thresholding, perspective transformation, and sliding window methods.
- **Object Detection:** YOLOv8 model is trained for detecting objects like pedestrians, cyclists, cars, trucks, and buses.
- **Distance Estimation:** Calculates the distance between the detected objects and the vehicle using bounding box coordinates.
- **Situation Analysis and Decision Making:** Analyzes the detected lanes and objects to decide the appropriate vehicle action.

### 2. Lane Detection
- Image enhancement techniques are applied, including resizing, brightness, and contrast adjustments.
- Perspective transformation and sliding window methods are used to detect multiple lanes.
- Lanes are classified into three types: **Solid** (Yellow), **Bicycle Dotted** (Red), and **Vehicle Dotted** (Blue).

### 3. Object Detection
- The YOLOv8 model is trained using the BDD10K dataset, focusing on five key classes: Pedestrian, Rider(cyclist), Car, Truck, and Bus.
- Parameters like image size, epochs, and batch size are optimized for the best performance.

### 4. Distance Estimation
- The distance between the vehicle and detected objects is estimated using the focal length and pixel width of the objects in the bounding box.

### 5. Situation Analysis and Decision Making
- The system categorizes the detected objects and lanes into regions and decides whether the vehicle should **slow down, stay behind, or overtake**.

## Scenario Creation with CarMaker

To create realistic vehicle-bicycle interaction scenarios, I used **CarMaker**, a simulation software widely used for testing driver assistance systems and autonomous driving technologies. The specific scenarios tested in this project were designed to replicate common and critical interactions between vehicles and bicycles, such as:

- **Right Hook**
- **Left Cross**
- **Sideswipe**
- **Overtaking on a Shared Lane**
- **Solid Lane with Traffic**

After creating these scenarios in CarMaker, the simulations were exported as video files. These videos were then used as test inputs for the detection system developed in this project.


## Implementation

### Lane Detection
- Various techniques are implemented to enhance image quality and accurately detect lanes, even under challenging conditions like bright sunlight.

### Object Detection
- The model is trained and validated using a subset of the BDD10K dataset, with results showing reasonable accuracy in detecting the five targeted classes.

### Distance Estimation
- The model calculates the distance from the vehicle to detected objects using a pre-calibrated focal length.

### Integration
- The integration of **lane detection, object detection, and distance estimation** provides a comprehensive situational analysis, guiding the vehicle's decisions.

## Results and Evaluation

### Lane Detection
- The system performed well under standard conditions, but accuracy dropped in adverse weather or bright sunlight.
- Videos of different driving scenarios show the system's performance under various conditions.

### Object Detection
- The model achieved good precision and recall across different classes, with the best results for detecting cars.

### Integration
- Videos demonstrating the integrated system's performance in real-time scenarios are included. The system successfully guides the vehicle based on the detected lanes and objects.

## Conclusion

This research successfully developed a system that integrates lane detection, object detection, distance estimation, and situation analysis. The system enhances decision-making in real-time, promoting safer interactions between bicycles and vehicles on the road. The results demonstrate the system's potential to contribute to intelligent transportation systems, with future advancements expected in image enhancement, model optimization, and the use of multi-camera setups for more accurate detection and analysis.

## Future Work

- Improving image enhancement techniques for better lane detection in challenging lighting conditions.
- Implementing lane coordination-based guidance systems.
- Optimizing YOLOv8 weights using Sparsification techniques.
- Exploring stereo camera setups for more precise distance estimation.
- Expanding to multi-camera setups for a comprehensive vehicle vision system.

## References

[1] European Cyclists' Federation, ‘The Benefits Of Cycling’, 2018. [Link](https://ecf.com/policy-areas/cycling-economy/economic-benefits)  
[2] ‘Death on the roads’, World Health Organization, Live Statistics Counter of Deaths on the road. [Link](https://extranet.who.int/roadsafety/death-on-the-roads/#ticker/cyclists)  
[3] International Transport Forum: Road Safety Report 2021 Germany. [Link](https://www.itf-oecd.org/sites/default/files/germany-road-safety.pdf)  
[4] Schreck, Benjamin. "Cycling and designing for cyclists in Germany: Road safety, Guidelines and Research." Transactions on Transport Sciences 8.1 (2017): 44-57.  
[5] Bikeway Study Part Two: Your Speed, Your Choice July 24, 2020. [Link](https://cyclingsavvy.org/2020/07/motorist-caused-bike-crashes/)  
[6] Relja Novović, "Cycling in traffic: Typical risky situations," Bike Gremlin, 2023. [Link](https://bike.bikegremlin.com/1855/typical-risky-traffic-situations-cyclists/#b)  
[7] Urban Bikeway Design Guide, Nacto. [Link](https://nacto.org/publication/urban-bikeway-design-guide/intersection-treatments/through-bike-lanes/)  
[8] Computer Vision Zone, "Distance Estimation Figure". [Link](https://www.computervision.zone/lessons/code-and-files-13/)

## Demo Videos

- **Right Hook (Safe distance > 7m)**  
  [![Watch the video](Rhook01_test_01.mp4)
- **Overtake (Safe distance > 7m)**
- **Solid Lane – Traffic (Safe distance > 7m)**

[Attach the relevant videos demonstrating different scenarios]

