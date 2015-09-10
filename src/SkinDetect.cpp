#include "SkinDetect.h"
#include "config.h"

#include <stdio.h>
#include <time.h>
#include <iostream>

using namespace std;

SkinDetector::SkinDetector() 
{
    dst_img = 0;
    
    imgHSV = 0;
    imgHue = 0;
    imgSat = 0;
    imgGra = 0;
}

IplImage* SkinDetector::detect(const IplImage* img)
{
    clock_t start = clock();

    if(dst_img && (dst_img->width != img->width && dst_img->height != img->height))
    {
        cvReleaseImage(&dst_img);
        dst_img = 0;
    }
    if(imgHSV && (imgHSV->width != img->width && imgHSV->height != img->height))
    {
        cvReleaseImage(&imgHSV);
        imgHSV = 0;
    }
    if(imgHue && (imgHue->width != img->width && imgHue->height != img->height))
    {
        cvReleaseImage(&imgHue);
        imgHue = 0;
    }
    if(imgSat && (imgSat->width != img->width && imgSat->height != img->height))
    {
        cvReleaseImage(&imgSat);
        imgSat = 0;
    }
    if(imgGra && (imgGra->width != img->width && imgGra->height != img->height))
    {
        cvReleaseImage(&imgGra);
        imgGra = 0;
    }


    if(!dst_img)
    {
        if(param["debug"] == "yes")
            cout << "SkinDetector:create dst_img" << endl;
        dst_img = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    }
    if(!imgHSV)
    {
        if(param["debug"] == "yes")
            cout << "SkinDetector:create imgHSV" << endl;
        imgHSV = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 3); //HSV image
    }
    if(!imgHue)
    {
        if(param["debug"] == "yes")
            cout << "SkinDetector:create imgHue" << endl;
        imgHue = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1); //color
    }
    if(!imgSat)
    {
        if(param["debug"] == "yes")
            cout << "SkinDetector:create imgSat" << endl;
        imgSat = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1); //saturation
    }
    if(!imgGra)
    {
        if(param["debug"] == "yes")
            cout << "SkinDetector:create imgGra" << endl;
        imgGra = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1); //gray
    }

    cout << "SkinDetector: elapse " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;

    cvCvtColor(img, imgHSV, CV_BGR2HSV);
    cout << "SkinDetector: cvtColor, elapse " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;
    cvSplit(imgHSV, imgHue, imgSat, imgGra, 0);
    cout << "SkinDetector: splited, elapse " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;

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
                pDst[w] = 255;
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
    
    /*cvSaveImage("dst.jpg", dst_img);
    cvSaveImage("hue.jpg", imgHue);
    cvSaveImage("gra.jpg", imgGra)*/;
    return dst_img;
}
