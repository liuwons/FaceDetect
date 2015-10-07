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

	cout << "cascade path to load:" << cascade_path << endl;
    bool loaded = cascade.load(cascade_path);
	assert(loaded);

    sd = new SkinDetector(w, h);
    md = new BackgroundDiffMoveDetector(w, h, fp, fb);

    ii = new IntImage(w, h);
}

vector<Rect> FaceDetector::detectAll(const IplImage* img, CvRect* searched)
{
    IplImage* mask = getMask(img);
	CvRect ori = getRegion(mask, 10);
	cout << "region:" << ori.x << " " << ori.y << " " << ori.width << " " << ori.height << endl;
	if (searched)
	{
		*searched = ori;
	}
    return detect(img, mask, ori);
}

vector<Rect> FaceDetector::detect(const IplImage* img, const IplImage* mask, CvRect ori)
{
    clock_t start = clock();

    vector<Rect> faces;
   
    ii->calcIntg(mask);

    cascade.detectMultiScale(ori, Mat(img), ii, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(18, 18));

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
    
    const IplImage* imgMaskSkin = sd->detect(img, 1);

    cvAnd(imgMaskMove, imgMaskSkin, imgMask);

	if (param["debug"] == "yes")
		cout << "get mask cost time:" << difftime(clock(), start) / CLOCKS_PER_SEC << endl;

    index ++;

    return imgMask;
}

CvRect FaceDetector::getRegion(const IplImage* mask, int th)
{
	CvRect result;
	
	clock_t st = clock();
	IplImage* img = (IplImage*)cvClone(mask);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	
	vector<CvRect> regions;
	int xmax = 0, xmin = width, ymax = 0, ymin = height;
	for (; contour != 0; contour = contour->h_next)
	{
		CvRect rec = cvBoundingRect(contour, 0);
		if (rec.width < th || rec.height < th)
			continue;

		if (rec.x < xmin)
			xmin = rec.x;
		if (rec.y < ymin)
			ymin = rec.y;
		if (rec.x + rec.width > xmax)
			xmax = rec.x + rec.width;
		if (rec.y + rec.height > ymax)
			ymax = rec.y + rec.height;

	}

	if (xmax > xmin)
	{
		result = cvRect(xmin, ymin, xmax - xmin, ymax - ymin);
	}
	else
	{
		result = cvRect(0, 0, width, height);
	}

	cout << "contour ana time:" << difftime(clock(), st) / CLOCKS_PER_SEC << endl;

	return result;
}