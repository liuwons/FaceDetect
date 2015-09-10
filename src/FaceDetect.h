#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <vector>

#include "MoveDetect.h"
#include "SkinDetect.h"

using std::vector;

class FaceDetector
{
    private:
        CascadeClassifier cascade;
        bool cascade_loaded;

        MoveDetector md;
        SkinDetector sd;
        
        IplImage* imgMask;
        IplImage* imgGray;
        
        int index;
    public:
        const static string cascade_path;// = "haarcascade_frontalface_alt.xml"; 
        const static int MIN_SIZE;

        FaceDetector();
        vector<Rect> detect(const IplImage* img, CvRect rect);
        vector<CvRect> getCandidateRect(const IplImage* img);
};

vector<CvRect> regionAnalyze(IplImage* imgMask, int min_size);

#endif
