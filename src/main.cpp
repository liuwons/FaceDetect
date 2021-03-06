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
		CvCapture* capture = NULL;
        if(param["file"] == "camera")
        {
            capture = cvCreateCameraCapture(0);
        }
        else
        {
            capture = cvCaptureFromFile(param["file"].data());
        }
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

        cvNamedWindow("FaceDetect", CV_WINDOW_AUTOSIZE);

        IplImage* frame = 0;
        while(frame = cvQueryFrame(capture))
        {
            if(!fd)
            {
                fd = new FaceDetector(frame->width, frame->height, atoi(param["fps"].data()), atoi(param["fbuf"].data()));
            }
            IplImage* imgSmooth = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
            cvSmooth(frame, imgSmooth, CV_GAUSSIAN, 3, 0, 0);
            
            vector<Rect> faces = fd->detectAll(imgSmooth);
            for(int k = 0; k < faces.size(); k ++)
            {
                Rect rect = faces[k];
                cvRectangle(imgSmooth, 
                        cvPoint(rect.x, rect.y), 
                        cvPoint(rect.x + rect.width, rect.y + rect.height),
                        cvScalar(0, 0, 255));
            }
            cvShowImage("FaceDetect", imgSmooth);
			cvWaitKey(1);
            
            fr_index ++;

            if(param["debug"] == "yes")
            {
                char fname[256];
                sprintf(fname, "%s/smooth%d.jpg", param["log"].data(), fr_index);
                cout << "save " << string(fname) << endl;
                cvSaveImage(fname, imgSmooth);
            }


            cvReleaseImage(&imgSmooth);

        }

        cvReleaseCapture(&capture);

    }
    else
    {

        IplImage* img = cvLoadImage(param["file"].data(), CV_LOAD_IMAGE_COLOR);
        
        IplImage* imgSmooth = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 3);
        cvSmooth(img, imgSmooth, CV_GAUSSIAN, 3, 0, 0);
		
		FaceDetector* fd = new FaceDetector(img->width, img->height, atoi(param["fps"].data()), atoi(param["fbuf"].data()));
        
		vector<Rect> faces = fd->detectAll(imgSmooth);
		for (int k = 0; k < faces.size(); k++)
		{
			Rect rect = faces[k];
			cvRectangle(imgSmooth,
				cvPoint(rect.x, rect.y),
				cvPoint(rect.x + rect.width, rect.y + rect.height),
				cvScalar(0, 0, 255));
		}

		if (param["debug"] == "yes")
		{
			char fname[256];
			sprintf(fname, "%s/smooth%d.jpg", param["log"].data(), 5648);
			cout << "save " << string(fname) << endl;
			cvSaveImage(fname, imgSmooth);
		}

    }

    return 0;
}
