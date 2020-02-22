#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void gradientSobel()
{
    // TODO: Based on the image gradients in both x and y, compute an image 
    // which contains the gradient magnitude according to the equation at the 
    // beginning of this section for every pixel position. Also, apply different 
    // levels of Gaussian blurring before applying the Sobel operator and compare the results.
    cv::Mat img = cv::imread("../images/img1gray.png");
    cv::imshow("input", img);
    cv::waitKey(0);

    cv::Mat img_grey;
    cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);
    cv::imshow("grey", img_grey);
    cv::waitKey(0);

    // gaussian 5X5
    float gauss_data[25] = {1,4,2,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1};
    for(int i=0;i < 25; i++)
    {
        gauss_data[i] /= 273;
    }
    cv::Mat gauss_kernel = cv::Mat(5, 5, CV_32F, gauss_data);
    cv::Mat result;
    cv::filter2D(img_grey, result, -1, gauss_kernel, cv::Point(-1, 1), 0, cv::BORDER_DEFAULT);
    cv::imshow("output", result);
    cv::waitKey(0);

    float sobel_data_x[9] ={-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobel_data_y[9] ={-1, -2, -1, 0, 0, 0, 1, 2, 1};

    cv::Mat sobel_x = cv::Mat(3, 3, CV_32F, sobel_data_x);
    cv::Mat sobel_y = cv::Mat(3, 3, CV_32F, sobel_data_y);

    cv::Mat grad_x, grad_y;

    cv::filter2D(result, grad_x, -1, sobel_x);
    cv::filter2D(result, grad_y, -1, sobel_y);

    cv::imshow("grad_x", grad_x);
    cv::waitKey(0);
    cv::imshow("grad_y", grad_y);
    cv::waitKey(0);

    cv::Mat grad_mag = result.clone();

    for(int i=0;i<grad_mag.rows; i++)
    {
        for(int j=0; j< grad_mag.cols; j++)
        {
            double x2= pow(grad_x.at<unsigned char>(i,j), 2);
            double y2= pow(grad_y.at<unsigned char>(i, j), 2);
            grad_mag.at<unsigned char>(i, j) = sqrt(x2 + y2);
        }
    }

    cv::imshow("grad_mag", grad_mag);
    cv::waitKey(0);

}

int main()
{
    gradientSobel();
}
