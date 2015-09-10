#include "MoveDetect.h"
#include "tool.h"
#include "config.h"

#include <iostream>
#include <stdio.h>
using namespace std;

FrameDiffMoveDetector::FrameDiffMoveDetector(int w, int h)
{
    width = w;
    height = h;

    imgLast = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgTmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

    index = 0;
}

/* Detect movement using frame difference.
 * The input image is 1 channel gray image.
 * The result is a mask image represents the region of movements.
 * */
const IplImage* FrameDiffMoveDetector::detect(const IplImage* img)
{
    assert(img->width == width && img->height == height);

    if(0 == index)
    {
        setImageVal(imgMask, 255);
    }
    else
    {
        cvAbsDiff(imgLast, img, imgTmp);
        cvThreshold(imgTmp, imgMask, 10, 255, CV_THRESH_BINARY);
    }

    cvCopy(img, imgLast);

    index++;

    return imgMask;
}



BackgroundDiffMoveDetector::BackgroundDiffMoveDetector(int w, int h, int fp, int fb, BackgroundMethod m)
{
    if(param["debug"] == "yes")
    {
        cout << "BackgroundDiffMoveDetector: fps[" << fp << "], buflen[" << fb << "]" << endl;
    }

    method = m;
    width = w;
    height = h;
    fps = fp;
    imgBufLen = fb;
    index = 0;

    imgBack = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    imgTmp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

    imgBuf = new IplImage*[imgBufLen];
    for(int i = 0; i < imgBufLen; i ++)
    {
        imgBuf[i] = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    }

    cbuf = 0;
    ibuf = 0;
    bp = 0;

    if(method == METHOD_AVERAGE)
    {
        cbuf = new int[width*height];
    }
    else
    {
        ibuf = new unsigned char[width*height*imgBufLen];
        bp = new unsigned char*[imgBufLen];
    }

}

int compare_uchar(const void* a, const void* b)
{
    unsigned char m = *((unsigned char*)a);
    unsigned char n = *((unsigned char*)b);

    return m > n ? 1 : -1;
}

const IplImage* BackgroundDiffMoveDetector::detect(const IplImage* img)
{
    assert(img->width == width && img->height == height);

    if(method == METHOD_AVERAGE)
    {
        if(index % fps == 0)
        {
            int cur = index / fps;
            if(cur == imgBufLen)
            {
                memset(cbuf, 0, width*height*sizeof(int));

                for(int k = 0; k < imgBufLen; k ++)
                {
                    unsigned char* p = (unsigned char*)imgBuf[k]->imageData;
                    int* pbuf = cbuf;
                    for(int i = 0; i < height; i ++)
                    {
                        for(int j = 0; j < width; j ++)
                        {
                            pbuf[j] += p[j];
                        }
                        p += imgBuf[k]->widthStep;
                        pbuf += width;
                    }
                }

                unsigned char* p = (unsigned char*)imgBack->imageData;
                int* pbuf = cbuf;
                for(int i = 0; i < height; i ++)
                {
                    for(int j = 0; j < width; j ++)
                    {
                        p[j] = pbuf[j]/imgBufLen;
                    }

                    pbuf += width;
                    p += imgBack->widthStep;
                }
                int bcur = index/fps/imgBufLen;
                char fn[256];
                sprintf(fn, "%s/background%d.bmp", param["log"].data(), bcur);
                cvSaveImage(fn, imgBack);
            }
            else
            {
                cvCopy(img, imgBuf[cur]);
            }
        }

        index++;

        cvAbsDiff(imgBack, img, imgTmp);
        cvThreshold(imgTmp, imgMask, 10, 255, CV_THRESH_BINARY);

        return imgMask;
    }
    else
    {
        if(index % fps == 0)
        {
            int cur = index / fps;
            if(cur % imgBufLen == 0)
            {
                cout << "calc new background" << endl;
                unsigned char* pm = (unsigned char*)imgBack->imageData;
                for(int k = 0; k < imgBufLen; k ++)
                    bp[k] = (unsigned char*)imgBuf[k]->imageData;

                for(int i = 0; i < height; i ++)
                {
                    for(int j = 0; j < width; j ++)
                    {
                        for(int n = 0; n < imgBufLen; n ++)
                            ibuf[n] = bp[n][j];
                        qsort(ibuf, imgBufLen, 1, compare_uchar);
                        pm[j] = ibuf[imgBufLen/2];
                    }

                    pm += imgBack->widthStep;
                    for(int k = 0; k < imgBufLen; k ++)
                        bp[k] += imgBuf[k]->widthStep;
                }
                int bcur = index/fps/imgBufLen;
                char fn[256];
                sprintf(fn, "%s/background%d.bmp", param["log"].data(), bcur);
                cvSaveImage(fn, imgBack);
            }
            else
            {
                cvCopy(img, imgBuf[cur%imgBufLen]);
            }
        }

        index++;

        cvAbsDiff(imgBack, img, imgTmp);
        cvThreshold(imgTmp, imgMask, 10, 255, CV_THRESH_BINARY);

        return imgMask;
    }

}
