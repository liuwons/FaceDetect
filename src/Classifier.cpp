#include "Classifier.h"

#include <iostream>
using namespace std;

#include "config.h"

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };


class MaskCascadeClassifierInvoker : public ParallelLoopBody
{
public:
    MaskCascadeClassifierInvoker( MaskCascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor,
        vector<Rect>& _vec, vector<int>& _levels, vector<double>& _weights, bool outputLevels, const Mat& _mask, Mutex* _mtx)
    {
        classifier = &_cc;
        processingRectSize = _sz1;
        stripSize = _stripSize;
        yStep = _yStep;
        scalingFactor = _factor;
        rectangles = &_vec;
        rejectLevels = outputLevels ? &_levels : 0;
        levelWeights = outputLevels ? &_weights : 0;
        mask = _mask;
        mtx = _mtx;
    }

    void operator()(const Range& range) const
    {
        Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();

        Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));

        const IntImage* intImg = classifier->intImg;
        int area = winSize.width * winSize.height;
        int width = intImg->width;
        int height = intImg->height;
        int thresh = area / opt_factor;

        int y1 = range.start * stripSize;
        int y2 = min(range.end * stripSize, processingRectSize.height);
        for( int y = y1; y < y2; y += yStep )
        {
            for( int x = 0; x < processingRectSize.width; x += yStep )
            {
                if ( (!mask.empty()) && (mask.at<uchar>(Point(x,y))==0)) {
                    continue;
                }

                if(opt)
                {
                    int origX = x * scalingFactor;
                    int origY = x * scalingFactor;
                    int origX2 = origX + winSize.width;
                    int origY2 = origY + winSize.height;

                    int integ = intImg->data[origY2*width+origX2]
                        - intImg->data[origY*width+origX2]
                        - intImg->data[origY2*width+origX]
                        + intImg->data[origY*width+origX];
                    if(integ < thresh)
                        continue;
                }

                double gypWeight;
                int result = classifier->runAt(evaluator, Point(x, y), gypWeight);

#if defined (LOG_CASCADE_STATISTIC)

                logger.setPoint(Point(x, y), result);
#endif
                if( rejectLevels )
                {
                    if( result == 1 )
                        result =  -(int)classifier->data.stages.size();
                    if( classifier->data.stages.size() + result < 4 )
                    {
                        mtx->lock();
                        rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
                        rejectLevels->push_back(-result);
                        levelWeights->push_back(gypWeight);
                        mtx->unlock();
                    }
                }
                else if( result > 0 )
                {
                    mtx->lock();
                    rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),
                                               winSize.width, winSize.height));
                    mtx->unlock();
                }
                if( result == 0 )
                    x += yStep;
            }
        }
    }

    MaskCascadeClassifier* classifier;
    vector<Rect>* rectangles;
    Size processingRectSize;
    int stripSize, yStep;
    double scalingFactor;
    vector<int> *rejectLevels;
    vector<double> *levelWeights;
    Mat mask;
    Mutex* mtx;

};


bool MaskCascadeClassifier::detectSingleScale( Size winSize, const Mat& image, int stripCount, Size processingRectSize,
                                           int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
{
    cout << "detect single scale" << endl;
    if (featureEvaluator->getFeatureType() == cv::FeatureEvaluator::HAAR)
        cout << "is haar feature" << endl;

    MHaarEvaluator* ev = (MHaarEvaluator*)(featureEvaluator.obj);
    cout << "ev:" << ev << endl;
    if( !ev->setWinSize( winSize ) )
        return false;

#if defined (LOG_CASCADE_STATISTIC)
    logger.setImage(image);
#endif

    cout << "detect single scale 2" << endl;

    Mat currentMask;
    if (!maskGenerator.empty()) {
        currentMask=maskGenerator->generateMask(image);
    }

    cout << "detect single scale 3" << endl;

    vector<Rect> candidatesVector;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    Mutex mtx;
    if( outputRejectLevels )
    {
        parallel_for_(Range(0, stripCount), MaskCascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
            candidatesVector, rejectLevels, levelWeights, true, currentMask, &mtx));
        levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
        weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
    }
    else
    {
         parallel_for_(Range(0, stripCount), MaskCascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
            candidatesVector, rejectLevels, levelWeights, false, currentMask, &mtx));
    }
    candidates.insert( candidates.end(), candidatesVector.begin(), candidatesVector.end() );

#if defined (LOG_CASCADE_STATISTIC)
    logger.write();
#endif

    cout << "detect single scale 4" << endl;

    return true;
}


void MaskCascadeClassifier::detectMultiScale( const Mat& image, const IntImage* ii, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    //cout << "MaskCascadeClassifier::detectMultiScale start" << endl;
    intImg = ii;
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, flags, minObjectSize, maxObjectSize, false );
    //cout << "MaskCascadeClassifier::detectMultiScale end" << endl;
}

void MaskCascadeClassifier::detectMultiScale( const Mat& image, vector<Rect>& objects,
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    cout << "MaskCascadeClassifier::detectMultiScale start" << endl;
    const double GROUP_EPS = 0.2;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;

    if( isOldFormatCascade() )
    {
        MemStorage storage(cvCreateMemStorage(0));
        CvMat _image = image;
        CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
                                              minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
        objects.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
        return;
    }

    objects.clear();

    if (!maskGenerator.empty()) {
        maskGenerator->initializeMask(image);
    }


    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();

    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }

    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
    vector<Rect> candidates;

    if( !featureEvaluator->setImage( grayImage, data.origWinSize ) )
    {
        cerr << "Error: set image for feature evaluator failed !" << endl;
        return;
    }


    for( double factor = 1; ; factor *= scaleFactor )
    {
        Size originalWindowSize = getOriginalWindowSize();

        Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
        Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
        Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );

        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;

        //Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        //resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

        int yStep;
        if( getFeatureType() == cv::FeatureEvaluator::HOG )
        {
            yStep = 4;
        }
        else
        {
            yStep = factor > 2. ? 1 : 2;
        }

        int stripCount, stripSize;

        const int PTS_PER_THREAD = 1000;
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;

        if( !detectSingleScale( windowSize, grayImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
            rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }


    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());

    if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, minNeighbors, GROUP_EPS );
    }
    cout << "MaskCascadeClassifier::detectMultiScale end" << endl;
}


bool MHaarEvaluator::setImage( const Mat &image, Size _origWinSize )
{
    int rn = image.rows+1, cn = image.cols+1;
    origWinSize = _origWinSize;

    if (image.cols < origWinSize.width || image.rows < origWinSize.height)
        return false;

    if( sum0.rows < rn || sum0.cols < cn )
    {
        sum0.create(rn, cn, CV_32S);
        sqsum0.create(rn, cn, CV_64F);
        if (hasTiltedFeatures)
            tilted0.create( rn, cn, CV_32S);
    }
    sum = Mat(rn, cn, CV_32S, sum0.data);
    sqsum = Mat(rn, cn, CV_64F, sqsum0.data);

    if( hasTiltedFeatures )
    {
        tilted = Mat(rn, cn, CV_32S, tilted0.data);
        integral(image, sum, sqsum, tilted);
    }
    else
        integral(image, sum, sqsum);
    return true;
}


bool MHaarEvaluator::setWinSize(Size sz)
{
    cout << "setWinSize" << endl;
    curSize = sz;
    factor = (double)sz.width / origWinSize.width;
    sqfactor = factor*factor;

    normrect = Rect(1, 1, origWinSize.width-2, origWinSize.height-2);
    const int* sdata = (const int*)sum.data;
    const double* sqdata = (const double*)sqsum.data;
    size_t sumStep = sum.step/sizeof(sdata[0]);
    size_t sqsumStep = sqsum.step/sizeof(sqdata[0]);

    CV_SUM_PTRS( p[0], p[1], p[2], p[3], sdata, normrect, sumStep );
    CV_SUM_PTRS( pq[0], pq[1], pq[2], pq[3], sqdata, normrect, sqsumStep );

    size_t fi, nfeatures = features->size();

    for( fi = 0; fi < nfeatures; fi++ )
        featuresPtr[fi].updatePtrs( !featuresPtr[fi].tilted ? sum : tilted, factor);
    return true;
}

bool MHaarEvaluator::setWindow( Point pt )
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + curSize.width >= sum.cols ||
        pt.y + curSize.height >= sum.rows )
        return false;

    size_t pOffset = pt.y * (sum.step/sizeof(int)) + pt.x;
    size_t pqOffset = pt.y * (sqsum.step/sizeof(double)) + pt.x;
    int valsum = CALC_SUM(p, pOffset);
    double valsqsum = CALC_SUM(pq, pqOffset);

    double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
    if( nf > 0. )
        nf = sqrt(nf);
    else
        nf = 1.;
    varianceNormFactor = 1./nf;
    offset = (int)pOffset;

    return true;
}
