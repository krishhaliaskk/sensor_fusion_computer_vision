#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    cv::imshow("input", img);
    cv::waitKey();

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32F);
    int blocksize = 2;
    int aperture = 3;
	
	cv::Mat dst_norm, dst_norm_scaled;
    double k=0.04;
    int minresponse = 100;
    double t1 = cv::getTickCount();
    cv::cornerHarris(img, dst, blocksize, aperture, k);
    double t2 = cv::getTickCount();
    cout<< "HarrisCorner : "<< (t2 - t1)/cv::getTickFrequency()<< endl;

    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    cv::imshow("harris", dst_norm_scaled);
    cv::waitKey(0);
    double maxoverlap =0.0;
    vector<cv::KeyPoint> keypoints;
    for(int i=0; i< dst_norm_scaled.rows; i++)
    {
        for(int j=0;j < dst_norm_scaled.cols; j++)
        {
            int response;
            response = (int)dst_norm.at<float>(i,j);
            if(response > minresponse)
            {
                cv::KeyPoint kp;
                kp.pt = cv::Point2f(j, i);
                kp.response = response;
                kp.size = 2 * aperture;

                bool boverlap = false;
                for(auto it = keypoints.begin(); it !=keypoints.end(); ++it)
                {
                    double kpoverlap = cv::KeyPoint::overlap(kp, *it);
                    if(kpoverlap > maxoverlap)
                    {
                        boverlap = true;
                        if(kp.response > (*it).response)
                        {
                            (*it) = kp;
                            break;
                        }
                    }
                    //printf("loop\n");
                }
                if(!boverlap)
                {
                    keypoints.push_back(kp);
                }
            }
        }
    }
    string windowName;
    windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
}

void cornernessGFTT()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    cv::imshow("input", img);
    cv::waitKey();

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32F);
    int blocksize = 6;
    double maxoverlap =0.0;
    double mindistance = (1 - maxoverlap) * blocksize;
    int maxcorners = img.rows * img.cols / std::max(1.0, mindistance);
    double qualitylevel = 0.01;
    bool useharris = false;
    int aperture = 3;
    double k=0.04;
    vector<cv::KeyPoint> kptShiTomasi;
    vector<cv::Point2f> corners;
    cv::Mat dst_norm, dst_norm_scaled;

    int minresponse = 100;
    double t1 = cv::getTickCount();
    cv::goodFeaturesToTrack(img, corners, maxcorners, qualitylevel, mindistance, cv::Mat(), blocksize, useharris, k);
    double t2 = cv::getTickCount();
    cout<< "goodFeaturesToTrack : " << (t2 - t1)/cv::getTickFrequency()<< endl;
    for(auto it = corners.begin(); it != corners.end(); it++)
    {
        cv::KeyPoint newkp;
        newkp.pt = cv::Point2f((*it).x, (*it).y);
        newkp.size = blocksize;
        kptShiTomasi.push_back(newkp);
    }

    string windowName;
    windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);

    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    vector<cv::KeyPoint> kp2d;
    t1 = cv::getTickCount();
    detector->detect(img, kp2d, cv::Mat());
    t2 = cv::getTickCount();
    cout<< "FAST : " << (t2 - t1)/cv::getTickFrequency()<< endl;
    cv::Mat img_vis = img.clone();
    cv::drawKeypoints(img, kp2d, img_vis);
    cv::imshow("FAST", img_vis);
    cv::waitKey(0);
}

int main()
{
    cornernessGFTT();
    cornernessHarris();
}
