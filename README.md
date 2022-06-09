# yolov4-custom-functions
## Getting Started
### Conda (Recommended)



# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```


## Downloading Official Pre-trained Weights
YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we will use the pre-trained weights.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights



## YOLOv4 Using Tensorflow (tf, .pb model)
To implement YOLOv4 using TensorFlow, first we convert the .weights into the corresponding TensorFlow model files and then run the model.
```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 



# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.avi


If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` and .weights file in above commands.

<strong>Note:</strong> You can also run the detector on multiple images at once by changing the --images flag like such ``--images "./data/images/kite.jpg, ./data/images/dog.jpg"``

### Result Image(s) (Regular TensorFlow)
You can find the outputted image(s) showing the detections saved within the 'detections' folder.
#### Pre-trained YOLOv4 Model Example
<p align="center"><img src="data/helpers/result.png" width="640"\></p>

### Result Video
Video saves wherever you point --output flag to. If you don't set the flag then your video will not be saved with detections on it.
<p align="center"><img src="data/helpers/demo.gif"\></p>

## YOLOv4-Tiny using TensorFlow
The following commands will allow you to run yolov4-tiny model.
```
# yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

# Run yolov4-tiny tensorflow model
python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --images ./data/images/kite.jpg --tiny
```
<a name="custom"/>

## Custom YOLOv4 Using TensorFlow
The following commands will allow you to run your custom yolov4 model. (video and webcam commands work as well)
```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg
```

#### Custom YOLOv4 Model Example (see video link above to train this model)
<p align="center"><img src="data/helpers/custom_result.png" width="640"\></p>

## Custom Functions and Flags
Here is how to use all the currently supported custom functions and flags that I have created.

<a name="counting"/>

### Counting Objects (total objects or per class)
I have created a custom function within the file [core/functions.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/functions.py) that can be used to count and keep track of the number of objects detected at a given moment within each image or video. It can be used to count total objects found or can count number of objects detected per class.

#### Count Total Objects
To count total objects all that is needed is to add the custom flag "--count" to your detect.py or detect_video.py command.
```
# Run yolov4 model while counting total objects detected
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count
```
Running the above command will count the total number of objects detected and output it to your command prompt or shell as well as on the saved detection as so:
<p align="center"><img src="data/helpers/total_count.png" width="640"\></p>

#### Count Objects Per Class
To count the number of objects for each individual class of your object detector you need to add the custom flag "--count" as well as change one line in the detect.py or detect_video.py script. By default the count_objects function has a parameter called <strong>by_class</strong> that is set to False. If you change this parameter to <strong>True</strong> it will count per class instead.

To count per class make detect.py or detect_video.py look like this:
<p align="center"><img src="data/helpers/by_class_config.PNG" width="640"\></p>

Then run the same command as above:
```
# Run yolov4 model while counting objects per class
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count
```
Running the above command will count the number of objects detected per class and output it to your command prompt or shell as well as on the saved detection as so:
<p align="center"><img src="data/helpers/perclass_count.png" width="640"\></p>

<strong>Note:</strong> You can add the --count flag to detect_video.py commands as well!

<a name="info"/>

### Print Detailed Info About Each Detection (class, confidence, bounding box coordinates)
I have created a custom flag called <strong>INFO</strong> that can be added to any detect.py or detect_video.py commands in order to print detailed information about each detection made by the object detector. To print the detailed information to your command prompt just add the flag `--info` to any of your commands. The information on each detection includes the class, confidence in the detection and the bounding box coordinates of the detection in xmin, ymin, xmax, ymax format.

If you want to edit what information gets printed you can edit the <strong>draw_bbox</strong> function found within the [core/utils.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/utils.py) file. The line that prints the information looks as follows:
<p align="center"><img src="data/helpers/info_details.PNG" height="50"\></p>

Example of info flag added to command:
```
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --info
```
Resulting output within your shell or terminal:
<p align="center"><img src="data/helpers/info_output.PNG" height="100"\></p>

<strong>Note:</strong> You can add the --info flag to detect_video.py commands as well!

<a name="crop"/>

### Crop Detections and Save Them as New Images
I have created a custom function within the file [core/functions.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/functions.py) that can be applied to any detect.py or detect_video.py commands in order to crop the YOLOv4 detections and save them each as their own new image. To crop detections all you need to do is add the `--crop` flag to any command. The resulting cropped images will be saved within the <strong>detections/crop/</strong> folder.
  
 Example of crop flag added to command:
```
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --crop
```
 Here is an example of one of the resulting cropped detections from the above command.
 <p align="center"><img src="data/helpers/crop_example.png" height="250"\></p>
 
<a name="license"/>




python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)


detect_video.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/video.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.25)
  --count: count objects within video
    (default: False)
  --dont_show: dont show video output
    (default: False)
  --info: print info on detections
    (default: False)
  --crop: crop detections and save as new images
    (default: False)
```

### References  

   Huge shoutout goes to hunglc007 for creating the backbone of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * https://github.com/theAIGuysCode
#   a p p l e _ o r a n g e _ d e t e c t i o n _ y o l o v 4 - c o u n t  
 