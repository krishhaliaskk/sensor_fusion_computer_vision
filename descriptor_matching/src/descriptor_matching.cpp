#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "structIO.hpp"

using namespace std;
// 2 images to compare
// kpts 1 and 2 from each images
// kpt descriptions 1 and 2 from each images

// output matches (kpt point pairs that match)
// to find the match we have two steps
// 1) a metric for comparison of matching vectors (descriptorType hamming or L2)
// 2) choosing the pairs of kpt desc for comparisons (matcherType BF FLANN)

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType)
{

    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // dmatch algo choices BF, FLANN
    if (matcherType.compare("MAT_BF") == 0)
    {        
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching";
    }

    // NN KNN
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // TODO : implement k-nearest-neighbor matching
        vector<vector<cv::DMatch>> matches_knn;
        matcher->knnMatch(descSource, descRef, matches_knn, 2);
        float dist_threshold = 0.8f;
        int good_matches = 0;
        for (int i=0; i< matches_knn.size(); i++)
        {
            float dist_ratio = matches_knn[i][0].distance / matches_knn[i][1].distance;
            if( dist_ratio < dist_threshold)
            {
                matches.push_back(matches_knn[i][0]);
                good_matches++;
            }
        }
        // TODO : filter matches using descriptor distance ratio test
        float discarded_mtch = 1.0 - (float)good_matches/(float)matches_knn.size();
        std::cout << "discarded_matches" << discarded_mtch;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    string windowName = "Matching keypoints between two camera images (best 50)";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0);
}

int main()
{
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");    
    vector<cv::KeyPoint> kptsSource, kptsRef;
    cv::Mat descSource, descRef;

    // SIFT

    readKeypoints("../dat/C35A5_KptsSource_SIFT.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_SIFT.dat", kptsRef);
    readDescriptors("../dat/C35A5_DescSource_SIFT.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_SIFT.dat", descRef);
/*
    // BRISK LARGE
    readKeypoints("../dat/C35A5_KptsSource_BRISK_large.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_BRISK_large.dat", kptsRef);
    readDescriptors("../dat/C35A5_DescSource_BRISK_large.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_BRISK_large.dat", descRef);

    // BRISK SMALL
    readKeypoints("../dat/C35A5_KptsSource_BRISK_small.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_BRISK_small.dat", kptsRef);
    readDescriptors("../dat/C35A5_DescSource_BRISK_small.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_BRISK_small.dat", descRef);
*/
    vector<cv::DMatch> matches;
    string matcherType = "MAT_FLANN";
    string descriptorType = "DES_BINARY"; 
    string selectorType = "SEL_KNN";
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType);
}
