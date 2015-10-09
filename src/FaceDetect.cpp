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
    imgCont = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

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

FaceDetector::~FaceDetector()
{
	for (int i = 0; i < imgBufLen; i++)
	{
		cvReleaseImage(&imgBuf[i]);
	}
	delete[] imgBuf;

	cvReleaseImage(&imgMask);
	cvReleaseImage(&imgGray);
	cvReleaseImage(&imgBack);
	cvReleaseImage(&imgCont);

	delete md;
	delete sd;
	delete ii;
}

vector<Rect> FaceDetector::detectAll(IplImage* img, CvRect* searched)
{
    IplImage* mask = getMask(img);
	CvRect ori;

	if (opt)
	{
		ori = analyze(mask, 10, imgGray);
	}
	else
	{
		ori = cvRect(0, 0, mask->width, mask->height);
	}

	//cout << "region:" << ori.x << " " << ori.y << " " << ori.width << " " << ori.height << endl;
	if (searched)
	{
		*searched = ori;
	}
    vector<Rect> results = detect(imgGray, mask, ori);
	last_detected = results;

	return results;
}

vector<Rect> FaceDetector::detect(IplImage* img, IplImage* mask, CvRect ori)
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

IplImage* FaceDetector::getMask(IplImage* img)
{
    assert(img->width == width && img->height == height);

    clock_t start, finish;
    if(param["debug"] == "yes")
    {
        start = clock();
    }

    cvCvtColor(img, imgGray, CV_BGR2GRAY);

    IplImage* imgMaskMove = md->detect(imgGray, 1);

	// Set rectangles of the last detected faces as ROI.
	for (vector<Rect>::iterator iter = last_detected.begin(); iter < last_detected.end(); iter++)
	{
		unsigned char* p = (unsigned char*)imgMaskMove->imageData + iter->y * imgMaskMove->widthStep;
		int y1 = iter->y + iter->height;
		int x1 = iter->x + iter->width;
		for (int y = iter->y; y < y1; y++)
		{
			for (int x = iter->x; x < x1; x++)
			{
				p[x] = 1;
			}
			p += imgMaskMove->widthStep;
		}
	}
    
    const IplImage* imgMaskSkin = sd->detect(img, 1);

    cvAnd(imgMaskMove, imgMaskSkin, imgMask);

	if (param["debug"] == "yes")
		cout << "get mask cost time:" << difftime(clock(), start) / CLOCKS_PER_SEC << endl;

    index ++;

    return imgMask;
}

CvRect FaceDetector::analyze(IplImage* mask, int th, IplImage* src)
{
	clock_t st = clock();
	
	CvRect result;
	cvCopy(mask, imgCont);
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	cvFindContours(imgCont, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	cvZero(imgCont);
	
	vector<CvRect> regions;
	int xmax = 0, xmin = width, ymax = 0, ymin = height;
	int contour_index = 0;
	for (; contour != 0; contour = contour->h_next)
	{
		CvRect rec = cvBoundingRect(contour, 0);
		if (rec.width < th || rec.height < th || cvContourArea(contour) < th*th/2)
			continue;

		contour_index++;

		// Fill all the contour regions on the temp mask.
		cvDrawContours(imgCont, contour, cvScalar(contour_index, contour_index, contour_index), cvScalar(0, 0, 0), CV_FILLED, CV_FILLED);

		if (rec.x < xmin)
			xmin = rec.x;
		if (rec.y < ymin)
			ymin = rec.y;
		if (rec.x + rec.width > xmax)
			xmax = rec.x + rec.width;
		if (rec.y + rec.height > ymax)
			ymax = rec.y + rec.height;

	}
	cvClearMemStorage(storage);

	//assert(contour_index < 256);

	char fname2[256];
	sprintf(fname2, "log/gray_before%d.bmp", index);
	cvSaveImage(fname2, src);

	unsigned char* psrc;
	unsigned char* pcon;
	hist.clear();
	psrc = (unsigned char*)src->imageData;
	pcon = (unsigned char*)imgCont->imageData;
	for (int h = 0; h < src->height; h++)
	{
		for (int w = 0; w < src->width; w++)
		{
			if (pcon[w] != 0)
			{
				hist.pix_count++;
				hist.bin[psrc[w]] ++;
			}
		}
		psrc += src->widthStep;
		pcon += imgCont->widthStep;
	}
	hist.normalize();
	psrc = (unsigned char*)src->imageData;
	pcon = (unsigned char*)imgCont->imageData;
	for (int h = 0; h < src->height; h++)
	{
		for (int w = 0; w < src->width; w++)
		{
			int id = pcon[w];
			if (id != 0)
			{
				psrc[w] = hist.map[psrc[w]];
			}
		}
		psrc += src->widthStep;
		pcon += imgCont->widthStep;
	}

	// Get the mininal rectangle that contains all the contours.
	if (xmax > xmin)
	{
		result = cvRect(xmin, ymin, xmax - xmin, ymax - ymin);
	}
	else
	{
		result = cvRect(0, 0, width, height);
	}

	cout << "contour ana time:" << difftime(clock(), st) / CLOCKS_PER_SEC << endl;

	char fname1[256];
	sprintf(fname1, "log/gray_after%d.bmp", index);
	cvSaveImage(fname1, src);


	char fname[256];
	sprintf(fname, "log/cont%d_%d.bmp", index, contour_index);
	cvSaveImage(fname, imgCont);

	return result;
}
