#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#define VIDEO_OUT "/dev/video9"

//const std::string MODEL_CONFIG = "./darknet/cfg/yolov3.cfg";
//const std::string MODEL_WEIGHTS =  "./darknet/yolov3.weights";
const std::string MODEL_CONFIG = "./darknet/cfg/yolov3-tiny.cfg";
const std::string MODEL_WEIGHTS =  "./darknet/yolov3-tiny.weights";

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

int main() {
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    
    std::vector<std::string> classes;

    std::string classesFile = "coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) { classes.push_back(line); }
    
    std::cout << "config: " << MODEL_CONFIG << std::endl;
    std::cout << "weights: " << MODEL_WEIGHTS << std::endl;

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cam(2);
    if(!cam.isOpened()) { exit(1); }
    cam.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    std::cout << "cam is opened" << std::endl;

    // open and configure output (virtual webcam)
    int output = open(VIDEO_OUT, O_RDWR);
    if (output < 0) {
        std::cerr << "ERROR: could not open output device\n" <<  strerror(errno);
        return -2;
    }

    struct v4l2_format vid_format;
    memset(&vid_format, 0, sizeof(vid_format));
    vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(output, VIDIOC_G_FMT, &vid_format) < 0) {
        std::cerr << "ERROR: unable to get video format!\n" << strerror(errno);
        return -1;
    }

    // configure desired video format on device                                 
    size_t framesize = FRAME_WIDTH * FRAME_HEIGHT * 3;                          
    vid_format.fmt.pix.width = FRAME_WIDTH;                                     
    vid_format.fmt.pix.height = FRAME_HEIGHT;                                   
    vid_format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;                        
    vid_format.fmt.pix.sizeimage = framesize;                                   
    vid_format.fmt.pix.field = V4L2_FIELD_NONE;                                 

    if (ioctl(output, VIDIOC_S_FMT, &vid_format) < 0) {                         
        std::cerr << "ERROR: unable to set video format!\n" << strerror(errno); 
        return -1;                                                              
    }  

    
    auto frameStart = std::chrono::high_resolution_clock::now();
    while (1) {
        cv::Mat frame;
        cam >> frame;
        //std::cout << "dims: (" << width << "|" << height << ")" << std::endl; 

        static cv::Mat blob;
        cv::dnn::blobFromImage(
                frame, 
                blob, 
                1/255.0,
                cv::Size(FRAME_WIDTH, FRAME_HEIGHT),
                cv::Scalar(0, 0, 0),
                true,
                false
        );

        // propagate through network
        net.setInput(blob);
        std::vector<cv::Mat> outVector;
        net.forward(outVector, net.getUnconnectedOutLayersNames());

        //// annotate bbs
        //int numRowsDetection = outVector.rows;  // number of detected objects
        //int objInfo = outVector.cols;   // [x, y, w, h, class 1, class 2, ...]

        //for (int i = 0; i < numRowsDetection; i++) {
        //    Mat scores = outVector.row(
        //}

        auto frameEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frameDuration = frameEnd - frameStart;
        frameStart = frameEnd;
        int fps = (1/(frameDuration.count()));

        cv::putText(
                frame, 
                std::to_string(fps),
                cv::Point(30, 30),
                cv::FONT_HERSHEY_PLAIN,
                1,
                CV_RGB(100, 255, 0),
                1,
                cv::LINE_AA
        );
        cv::imshow("webcam", frame);


        size_t written = write(output, frame.data, framesize);
        if (written < 0) {
            std::cerr << "ERROR: could not write to output device!\n";
            close(output);
            break;
        }        


        if(cv::waitKey(10) == 27) { break; }
    }

    cam.release();
    std::cout << "cam released" << std::endl;
    close(output);

    return 0;
}



