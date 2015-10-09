#ifndef SKINDETECT_H
#define SKINDETECT_H

#include "cv_header.h"

class SkinDetector
{
    private:
        int width;
        int height;
        int index;

        IplImage* dst_img;
            
        IplImage* imgHSV; //HSV image
        IplImage* imgHue; //color
        IplImage* imgSat; //saturation
        IplImage* imgGra; //gray
    public:
        const static unsigned char SKIN_GRAY_UPPER_BOUND = 250;
        const static unsigned char SKIN_GRAY_LOWER_BOUND = 15;
        const static unsigned char SKIN_HUE_LOWER_BOUND = 3;
        const static unsigned char SKIN_HUE_UPPER_BOUND = 33;
        SkinDetector(int w, int h);
		~SkinDetector();
        IplImage* detect(const IplImage* img, int val = 1);
};

#endif
