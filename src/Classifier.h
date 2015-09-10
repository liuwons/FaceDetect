#include "cv_header.h"

#ifndef CLASSIFIER_H
#define CLASSIFIER_h
class MaskCascadeClassifier : public CascadeClassifier
{
    public:
        CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size() );

        CV_WRAP virtual void detectMultiScale( const Mat& image,
                                   CV_OUT vector<Rect>& objects,
                                   vector<int>& rejectLevels,
                                   vector<double>& levelWeights,
                                   double scaleFactor=1.1,
                                   int minNeighbors=3, int flags=0,
                                   Size minSize=Size(),
                                   Size maxSize=Size(),
                                   bool outputRejectLevels=false );

};

#endif
