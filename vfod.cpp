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
#include <algorithm>

const std::string MODEL_CONFIG = "../models/yolov3.cfg";
const std::string MODEL_WEIGHTS =  "../models/yolov3.weights";
const std::string MODEL_CONFIG_TINY = "../models/yolov3-tiny.cfg";
const std::string MODEL_WEIGHTS_TINY =  "../models/yolov3-tiny.weights";

const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};

const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

char* getCmdOption(char** start, char** end, const std::string& option) {
    char** itr = std::find(start, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool findCmdOption(char** start, char** end, const std::string& option) {
    return std::find(start, end, option) != end;
}

int main(int argc, char** argv) {
    const std::string modelConfig = MODEL_CONFIG;
    const std::string modelWeights = MODEL_WEIGHTS;
    if (findCmdOption(argv, argv + argc, "--tiny")) {
        std::cout << "using tiny model for detection" << std::endl;
        const std::string modelConfig = MODEL_CONFIG_TINY;
        const std::string modelWeights = MODEL_WEIGHTS_TINY;
    }

    bool gui = false;
    if (findCmdOption(argv, argv + argc, "--gui")) {
        gui = true;
    }

    bool fps = false;
    if (findCmdOption(argv, argv + argc, "--fps")) {
        fps = true;
    }

    std::string webcamDev = "/dev/video0";
    char* webcamDevOption = getCmdOption(argv, argv + argc, "--dev");
    if (webcamDevOption) {
        webcamDev = webcamDevOption;
        std::cout << "using " << webcamDev << " for capturing" << std::endl;
    }

    if (!findCmdOption(argv, argv + argc, "--virtual-dev")) {
        std::cout << "need to supply virtual device, exiting" << std::endl;
        return 1;
    }
    char* virtualCamDev = (getCmdOption(argv, argv + argc, "--virtual-dev"));
    std::cout << "using " << virtualCamDev << " as virtual cam" << std::endl;

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;

    std::vector<std::string> classes;

    std::string classesFile = "../models/coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) { classes.push_back(line); }

    //std::cout << "read " << classes.size() << " labels" << std::endl;
    //std::cout << "config: " << modelConfig << std::endl;
    //std::cout << "weights: " << modelWeights << std::endl;

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);

    // try to use gpu ,fall back to cpu if gpu can't be accessed
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Opencv videocapture - default color format is 24bit BGR
    //cv::VideoCapture cam(webcamDev);
    cv::VideoCapture cam(webcamDev);
    if(!cam.isOpened()) {
        std::cout << "could not open webcam device " << webcamDev << " exiting" << std::endl;
        exit(1);
    }
    cam.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cam.set(cv::CAP_PROP_CONVERT_RGB, 1);

    // open and configure output (virtual webcam)
    int output = open(virtualCamDev, O_RDWR);
    if (output < 0) {
        std::cerr << "ERROR: could not open output device - " <<  strerror(errno) << std::endl;
        return -2;
    }

    struct v4l2_format vid_format;
    memset(&vid_format, 0, sizeof(vid_format));
    vid_format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(output, VIDIOC_G_FMT, &vid_format) < 0) {
        std::cerr << "ERROR: unable to get video format -" << strerror(errno) << std::endl;
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
        std::cerr << "ERROR: unable to set video format - " << strerror(errno) << std::endl;
        return -1;
    }

    auto outputNames = net.getUnconnectedOutLayersNames();
    auto frameStart = std::chrono::high_resolution_clock::now();
    cv::Mat frame;
    cv::Mat blob;
    std::vector<cv::Mat> detections;

    while (1) {
        cam >> frame;
        if (gui) {
            cv::imshow("cam", frame);
        }

        cv::dnn::blobFromImage(
                frame,
                blob,
                0.00392,
                cv::Size(FRAME_WIDTH, FRAME_HEIGHT),
                cv::Scalar(0, 0, 0),
                true,
                false,
                CV_32F
                );

        net.setInput(blob);
        net.forward(detections, outputNames);

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        for (auto& output: detections) {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++) {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width/2, y - height/2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++) {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD) {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++) {
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        }

        for (int c= 0; c < NUM_CLASSES; c++) {
            for (size_t i = 0; i < indices[c].size(); ++i) {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                label_ss << classes[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                cv::rectangle(
                        frame,
                        cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10),
                        cv::Point(rect.x + label_bg_sz.width, rect.y),
                        color,
                        cv::FILLED
                        );
                cv::putText(
                        frame,
                        label.c_str(),
                        cv::Point(rect.x, rect.y - baseline - 5),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        cv::Scalar(0, 0, 0)
                        );
            }
        }

        // annotate fps
        if (fps) {
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
        }

        // convert frame to RGB and write to virtual cam
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        size_t written = write(output, frame.data, framesize);
        if (written < 0) {
            std::cerr << "ERROR: could not write to output device!\n";
            close(output);
            break;
        }

        if(cv::waitKey(1) == 27) { break; }
    }

    cam.release();
    std::cout << "cam released" << std::endl;
    close(output);

    return 0;
}

