#ifndef MOVEDETECT_H
#define MOVEDETECT_H

#include "cv_header.h"

class MoveDetector
{
    private:
        IplImage* imgLast;
        IplImage* imgTmp;
        IplImage* imgMask;
        int frame_index;
    public:
        MoveDetector();

        /* Detect movement using frame difference.
         * The input image is 1 channel gray image.
         * The result is a mask image represents the region of movements.
         * */
        IplImage* detect(const IplImage* img);

};

#endif
