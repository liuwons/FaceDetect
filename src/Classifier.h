#include "cv_header.h"

#ifndef CLASSIFIER_H
#define CLASSIFIER_h

#include <iostream>
using namespace std;

class IntImage
{
    public:
        IntImage(const IplImage* img)
        {
            assert(img->nChannels == 1 && img->width >= 0 && img->height >= 0);

            width = img->width;
            height = img->height;

            data = new int[width*height];

            calcIntg(img);
        }

        IntImage(int w, int h)
        {
            width = w;
            height = h;

            data = new int[w*h];
        }

        void calcIntg(const IplImage* img)
        {
            assert(img->width == width && img->height == height);

            int sum;
            unsigned char* p;
            
            sum = 0;
            p = (unsigned char*)img->imageData;
            for(int i = 0; i < width; i ++)
            {
                sum += p[i];
                data[i] = sum;
            }

            sum = 0;
            int k = 0;
            p = (unsigned char*)img->imageData;
            for(int i = 0; i < height; i ++)
            {
                sum += p[0];
                data[k] = sum;
                k += width;
                p += img->widthStep;
            }

            p = (unsigned char*)img->imageData;
            int *ip = data;
            for(int i = 1; i < height; i ++)
            {
                ip += width;
                p += img->widthStep;
                for(int j = 1; j < width; j ++)
                {
                    ip[j] = *(ip+j-1) + *(ip+j-width) - *(ip+j-width-1) + p[j];
                }
            }

        }

        ~IntImage()
        {
            delete[] data;
        }

        int width;
        int height;
        int* data;
};


class MaskCascadeClassifier : public CascadeClassifier
{
    public:
        CV_WRAP virtual void detectMultiScale(CvRect ori, const Mat& image,
                                   const IntImage* ii,
                                   CV_OUT vector<Rect>& objects,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

        CV_WRAP virtual void detectMultiScale(CvRect ori, const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size(),
                                   bool outputRejectLevels=false );

    protected:
        virtual bool detectSingleScale(CvRect oRect, CvRect cRect, const Mat& image, int stripStart, int stripEnd, Size processingRectSize,
                int stripSize, int yStep, double factor, vector<Rect>& candidates,
                vector<int>& rejectLevels, vector<double>& levelWeights, bool outputRejectLevels=false);

        friend class MaskCascadeClassifierInvoker;

        const IntImage* intImg;

};

#endif
