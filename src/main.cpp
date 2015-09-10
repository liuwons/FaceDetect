#include <iostream>
#include <string.h>
#include <map>
#include <stdio.h>
using namespace std;

#include "cv_header.h"
#include "tool.h"
#include "SkinDetect.h"
#include "MoveDetect.h"
#include "FaceDetect.h"
#include "config.h"

int main(int argc, char** argv)
{
    if(!parse_param(argc, argv, param))
    {
        cerr << "ERROR: Parse arguments failed!" << endl;
        return -1;
    }

    if(param["debug"] == "yes")
    {
        if(param["mode"] == MODE_VIDEO)
            cout << "detect mode is video mode" << endl;
        else
            cout << "detect mode is image mode" << endl;

        cout << "input file name:" << param["file"] << endl;
    }

    if(param["mode"] == MODE_VIDEO)
    {
        CvCapture* capture = cvCaptureFromFile(param["file"].data());
        if(!capture)
        {
            cerr << "Error: open video failed" << endl;
            return 1;
        }

        int fr_count = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
        cout << "frame count: " << fr_count << endl;

        int fr_start = 0;
        int fr_stop = 10;
        int fr_index = 0;

        FaceDetector*  fd = 0;

        IplImage* frame = 0;
        while(frame = cvQueryFrame(capture))
        {
            if(!fd)
            {
                fd = new FaceDetector(frame->width, frame->height, atoi(param["fps"].data()), atoi(param["fbuf"].data()));
            }
            IplImage* imgSmooth = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
            cvSmooth(frame, imgSmooth, CV_GAUSSIAN, 3, 0, 0);
            
            if(param["debug"] == "yes")
            {
                char fname[256];
                sprintf(fname, "%s/smooth%d.jpg", param["log"].data(), fr_index);
                cout << "save " << string(fname) << endl;
                cvSaveImage(fname, imgSmooth);
            }

            vector<CvRect> rects = fd->getCandidateRect(imgSmooth);
            fd->detect(imgSmooth, cvRect(0, 0, 0, 0));
            
            fr_index ++;

            cvReleaseImage(&imgSmooth);
        }

        cvReleaseCapture(&capture);

    }
    else
    {
        IplImage* img = cvLoadImage(param["file"].data(), CV_LOAD_IMAGE_COLOR);
        
        IplImage* imgSmooth = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 3);
        cvSmooth(img, imgSmooth, CV_GAUSSIAN, 3, 0, 0);
        
        /*IplImage* imgGray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
        cvCvtColor(imgSmooth, imgGray, CV_BGR2GRAY);*/

        if(param["debug"] == "yes")
        {
            cout << "save smooth.jpg" << endl;
            cvSaveImage("smooth.jpg", imgSmooth);
            /*cout << "save gray.bmp" << endl;
            cvSaveImage("gray.bmp", imgGray);*/
        }

        /*FaceDetector fd;
        vector<CvRect> rects = fd.getCandidateRect(imgSmooth);*/

    }

    return 0;
}
