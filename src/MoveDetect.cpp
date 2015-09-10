#include "MoveDetect.h"
#include "tool.h"
#include "config.h"

#include <iostream>
using namespace std;

MoveDetector::MoveDetector()
{
    if(param["debug"] == "yes")
        cout << "MoveDetector construct" << endl;
    imgLast = 0;
    imgTmp = 0;
    imgMask = 0;
    frame_index = 0;
}

/* Detect movement using frame difference.
 * The input image is 1 channel gray image.
 * The result is a mask image represents the region of movements.
 * */
IplImage* MoveDetector::detect(const IplImage* img)
{
    if(imgLast && (imgLast->width != img->width || imgLast->height != img->height))
    {
        if(param["debug"] == "yes")
            cout << "imgLast size mismatch" << endl;
        cvReleaseImage(&imgLast);
        imgLast = 0;
        frame_index = 0;
    }
    if(imgTmp && (imgTmp->width != img->width || imgTmp->height != img->height))
    {
        if(param["debug"] == "yes")
            cout << "imgTmp size mismatch" << endl;
        cvReleaseImage(&imgTmp);
        imgTmp = 0;
        frame_index = 0;
    }
    if(imgMask && (imgMask->width != img->width || imgMask->height != img->height))
    {
        if(param["debug"] == "yes")
            cout << "imgMask size mismatch" << endl;
        cvReleaseImage(&imgMask);
        imgMask = 0;
        frame_index = 0;
    }

    if(!imgLast)
        imgLast = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    if(!imgTmp)
        imgTmp = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    if(!imgMask)
        imgMask = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    
    cout << "move detector inited" << endl;

    if(param["debug"] == "yes")
        cout << "MoveDetector frame index:" << frame_index << endl;
    
    if(0 == frame_index)
    {
        cout << "first frame" << endl;
        
        setImageVal(imgMask, 255);
        cvCopy(img, imgLast);
        frame_index ++;
        return imgMask;
    }

    cvAbsDiff(imgLast, img, imgTmp);
    cvThreshold(imgTmp, imgMask, 10, 255, CV_THRESH_BINARY);
    
    cvCopy(img, imgLast);

    frame_index++;

    return imgMask;
}
