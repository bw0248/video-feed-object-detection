# Object detection overlay for video feeds

## TODO

* [ ] get google meet video calls to work with firefox
* make tiny/yolo v3 configurable
* update readme
* small dia of processing pipeline
* add license
* upload

Just a small proof of concept for a real-time object detection overlay in video/webcam feeds. 
Due to V4l2 dependency only working on linux currently.

* Object detection is based on [YOLOv3](https://pjreddie.com/darknet/yolo/)

## Dependencies

* [Opencv (preferably with GPU support)](https://github.com/opencv/opencv)
* [V4l2loopback](https://github.com/umlaeute/v4l2loopback)

## Tested with

* mpv
* google meet via chrome
* ffplay/ffmpeg
* cheese

## Build and run

*Note: If the virtual cam has not yet already been created you will need sudo rights to install the kernel module*
```
$ git clone https://github.com/bw0248/video-feed-object-detection.git
$ cd video-feed-object-detection/
$ ./run
```

## Usage

Now you see a small gui with your actual webcam feed, use the virtual webcam to have additional object detection overlay.

Examples (need to have ffplay, cheese, mpv)

* ffplay /dev/video9
* cheese -d vfod-cam
* mpv av://video4linux:/dev/video9
* select the virtual cam in video calls

