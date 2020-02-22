#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

template <typename T>
class ring_buffer {
    std::vector<T> dataBuffer;
    int size;
    int index;
public:
    ring_buffer(int size_dataBuffer)
    {
        size = size_dataBuffer;
        index =0;
    }
    void push_back(T data)
    {
        dataBuffer.insert(data, index % size);
        index++;
    }
};
#endif /* dataStructures_h */
