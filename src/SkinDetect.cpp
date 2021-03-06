#include "SkinDetect.h"
#include "config.h"

#include <stdio.h>
#include <time.h>
#include <iostream>

using namespace std;

SkinDetector::SkinDetector(int w, int h) 
{
    width = w;
    height = h;
    index = 0;

    dst_img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgHSV = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3); //HSV image
    imgHue = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1); //color
    imgSat = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1); //saturation
    imgGra = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1); //gray
}

IplImage* SkinDetector::detect(const IplImage* img, int val)
{
    assert(img->width == width && img->height == height);

    clock_t start = clock();

    cvCvtColor(img, imgHSV, CV_BGR2HSV);
    cvSplit(imgHSV, imgHue, imgSat, imgGra, 0);

    unsigned char* pDst = (unsigned char*)dst_img->imageData;
    unsigned char* pHue = (unsigned char*)imgHue->imageData;
    unsigned char* pGra = (unsigned char*)imgGra->imageData;
    for(int h = 0; h < dst_img->height; h ++)
    {
        for(int w = 0; w < dst_img->width; w++)
        {
            if (pHue[w] >= SKIN_HUE_LOWER_BOUND && pHue[w] <= SKIN_HUE_UPPER_BOUND &&
                    pGra[w] >= SKIN_GRAY_LOWER_BOUND && pGra[w] <= SKIN_GRAY_UPPER_BOUND)
            {
                pDst[w] = val;
            }
            else
            {
                pDst[w] = 0;
            }
        }
        pDst += dst_img->widthStep;
        pHue += imgHue->widthStep;
        pGra += imgGra->widthStep;
    }
    
    return dst_img;
}
