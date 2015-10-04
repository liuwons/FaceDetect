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

const string FaceDetector::DEFAULT_CASCADE_PATH = "cascade.xml";
const int FaceDetector::MIN_SIZE = 10;

FaceDetector::FaceDetector(int w, int h, int fp, int fb, const char* cp)
{
    width = w;
    height = h;
    fps = fp;
    imgBufLen = fb;

    imgBuf = new IplImage*[imgBufLen];
    for(int i = 0; i < imgBufLen; i ++)
    {
        imgBuf[i] = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    }

    imgMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgGray = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgBack = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

    md = new BackgroundDiffMoveDetector(width, height, fp, fb);
    sd = new SkinDetector(width, height);
    
    index = 0;

    if(cp)
        cascade_path = cp;
    else
        cascade_path = DEFAULT_CASCADE_PATH;
	cout << "cascade path:" << cascade_path << endl;
	bool loadr = cascade.load(cascade_path);
	cout << "load cascade:" << loadr << endl;
    assert(loadr);

    sd = new SkinDetector(w, h);
    md = new BackgroundDiffMoveDetector(w, h, fp, fb);

    ii = new IntImage(w, h);
}

vector<Rect> FaceDetector::detectAll(const IplImage* img)
{
    IplImage* mask = getMask(img);
    return detect(img, mask);
}

vector<Rect> FaceDetector::detect(const IplImage* img, const IplImage* mask)
{
    clock_t start = clock();

    vector<Rect> faces;
   
    /*if(param["debug"] == "yes")
    {
        cout << "calc integImage" << endl;
    }*/

    //IntImage* ii = new IntImage(mask);
    ii->calcIntg(mask);
	cout << "got integral image" << endl;

    /*if(param["debug"] == "yes")
    {
        cout << "detectMultiScale" << endl;
    }*/

    cascade.detectMultiScale(Mat(img), ii, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(18, 18));

    clock_t finish = clock();

    if(param["debug"] == "yes")
    {
        cout << "FaceDetector::detect cost time: " << difftime(finish, start)/CLOCKS_PER_SEC << " secs" << endl;
        cout << "face count:" << faces.size() << endl;
    }

    return faces;
}

IplImage* FaceDetector::getMask(const IplImage* img)
{
    assert(img->width == width && img->height == height);

    clock_t start, finish;
    if(param["debug"] == "yes")
    {
        start = clock();
    }

    cvCvtColor(img, imgGray, CV_BGR2GRAY);

    const IplImage* imgMaskMove = md->detect(imgGray, 1);
	cout << "got move mask" << endl;
    
    const IplImage* imgMaskSkin = sd->detect(img, 1);
	cout << "got skin mask" << endl;

    cvAnd(imgMaskMove, imgMaskSkin, imgMask);
    //cvOr(imgMaskMove, imgMaskSkin, imgMask);

    //vector<CvRect> rects = regionAnalyze(imgMask, MIN_SIZE);
    
    /*if(param["debug"] == "yes")
    {
        finish = clock();
        double dura = difftime(finish, start);
        cout << "FaceDetector::getCandidateRect cost time:" << dura/CLOCKS_PER_SEC << " secs" << endl;
    }*/

    /*if(param["debug"] == "yes")
    {
        char fname[256];
        
        sprintf(fname, "%s/mask_move%d.bmp", param["log"].data(), index);
        cout << "save mask_move.bmp" << endl;
        cvSaveImage(fname, imgMaskMove);
        
        sprintf(fname, "%s/mask_skin%d.bmp", param["log"].data(), index);
        cout << "save mask_skin.bmp" << endl;
        cvSaveImage(fname, imgMaskSkin);
        
        sprintf(fname, "%s/mask%d.bmp", param["log"].data(), index);
        cout << "save mask.bmp" << endl;
        cvSaveImage(fname, imgMask);
    }*/

    index ++;

    return imgMask;
}

/*
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
}*/
