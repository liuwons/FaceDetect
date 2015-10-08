#ifndef MOVEDETECT_H
#define MOVEDETECT_H

#include "cv_header.h"

class MoveDetector
{
    public:
        virtual IplImage* detect(const IplImage* img, int val) = 0;
};

class FrameDiffMoveDetector : public MoveDetector
{
    private:
        IplImage* imgLast;
        IplImage* imgTmp;
        IplImage* imgMask;
        int index;
        int width;
        int height;
    public:
        FrameDiffMoveDetector(int w, int h);

        /* Detect movement using frame difference.
         * The input image is 1 channel gray image.
         * The result is a mask image represents the region of movements.
         * */
        IplImage* detect(const IplImage* img, int val = 1);
};

enum BackgroundMethod
{
    METHOD_AVERAGE = 0,
    METHOD_MEDIUM = 1
};

class BackgroundDiffMoveDetector : public MoveDetector
{
    private:
        BackgroundMethod method;

        int index;
        int width;
        int height;

        int fps;
        int imgBufLen;

        IplImage** imgBuf;
        IplImage* imgBack;
        IplImage* imgMask;
        IplImage* imgTmp;

        int* cbuf;
        unsigned char* ibuf;
        unsigned char** bp;

    public:
        BackgroundDiffMoveDetector(int w, int h, int fp, int fb, BackgroundMethod m = METHOD_MEDIUM);
        IplImage* detect(const IplImage* img, int val = 1);
};

#endif
