#include "FaceDetect.h"
#include "config.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <stdio.h>
#include <time.h>
using namespace cv;

using namespace std;

const string FaceDetector::cascade_path = "haarcascade_frontalface_alt.xml";
const int FaceDetector::MIN_SIZE = 10;

FaceDetector::FaceDetector()
{
    imgMask = 0;
    imgGray = 0;
    index = 0;

    cascade_loaded = false;

    if(cascade.load(cascade_path))
    {
        cascade_loaded = true;
    }
    else
    {
        cerr << "load cascade failed" << endl;
    }
}

vector<Rect> FaceDetector::detect(const IplImage* img, CvRect rect)
{
    clock_t start = clock();

    vector<Rect> faces;
    
    cascade.detectMultiScale(Mat(img), faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

    cout << "FaceDetector::detect, elapse: " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;


    cout << "face count:" << faces.size() << endl;

    return faces;
}

vector<CvRect> FaceDetector::getCandidateRect(const IplImage* img)
{
    clock_t start, finish;

    start = clock();

    if(imgMask && (imgMask->width != img->width || imgMask->height != img->height))
    {
        cvReleaseImage(&imgMask);
        imgMask = 0;
        index = 0;
    }
    
    if(imgGray && (imgGray->width != img->width || imgGray->height != img->height))
    {
        cvReleaseImage(&imgGray);
        imgGray = 0;
        index = 0;
    }

    if(!imgMask)
        imgMask = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
    if(!imgGray)
        imgGray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

    if(param["debug"] == "yes")
        cout << "convert color" << endl;
    cvCvtColor(img, imgGray, CV_BGR2GRAY);

    cout << "elapse:" << (double)(clock()-start)/CLOCKS_PER_SEC << endl;

    if(param["debug"] == "yes")
        cout << "move detect" << endl;
    IplImage* imgMaskMove = md.detect(imgGray);
    
    cout << "elapse:" << (double)(clock()-start)/CLOCKS_PER_SEC << endl;
    
    if(param["debug"] == "yes")
        cout << "skin detect" << endl;
    IplImage* imgMaskSkin = sd.detect(img);

    cout << "elapse:" << (double)(clock()-start)/CLOCKS_PER_SEC << endl;

    cvOr(imgMaskMove, imgMaskSkin, imgMask);

    vector<CvRect> rects = regionAnalyze(imgMask, MIN_SIZE);
    
    finish = clock();
    double dura = difftime(finish, start);
    cout << "FaceDetector::getCandidateRect, time elapse:" << dura/CLOCKS_PER_SEC << endl;
    
    if(param["debug"] == "yes")
    {
        char fname[256];
        
        sprintf(fname, "mask_move%d.bmp", index);
        cout << "save mask_move.bmp" << endl;
        cvSaveImage(fname, imgMaskMove);
        
        sprintf(fname, "mask_skin%d.bmp", index);
        cout << "save mask_skin.bmp" << endl;
        cvSaveImage(fname, imgMaskSkin);
        
        sprintf(fname, "mask%d.bmp", index);
        cout << "save mask.bmp" << endl;
        cvSaveImage(fname, imgMask);
    }


    index ++;


    return rects;
}

vector<CvRect> regionAnalyze(IplImage* imgMask, int min_size)
{
    vector<CvRect> res;

    unsigned char* p = (unsigned char*)imgMask->imageData;
    for(int i = 0; i < imgMask->height; i ++)
    {
        int left = 0, right = 0;
        while(right < imgMask->width)
        {
            while(p[left] == 0 && left < imgMask->width)
                left ++;
            right = left;
            while(p[right] && right < imgMask->width)
                right ++;
            if(right - left > min_size)
            {
                for(int j = left; j < right; j ++)
                    p[j] = 1;
            }
            left = right;
        }
        p += imgMask->widthStep;
    }

    for(int i = 0; i < imgMask->width; i ++)
    {
        p = (unsigned char*)imgMask->imageData + i;
        int top = 0, bottom = 0;
        while(bottom < imgMask->height)
        {
            while(p[top*imgMask->widthStep] == 0 && top < imgMask->height)
                top++;
            bottom = top;
            while(p[bottom*imgMask->widthStep] && bottom < imgMask->height)
                bottom++;
            if(bottom - top > min_size)
            {
                for(int j = top; j < bottom; j ++)
                {
                    p[j*imgMask->widthStep] = 1;
                }
            }
            top = bottom;
        }
    }
    
    p = (unsigned char*)imgMask->imageData;
    for(int i = 0; i < imgMask->height; i ++)
    {
        for(int j = 0; j < imgMask->width; j ++)
        {
            if(p[j] == 1)
                p[j] = 255;
            else if(p[j] == 255)
                p[j] = 0;
        }
        p += imgMask->widthStep;
    }

    return res;
}
