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


template<class FEval>
inline int predictOrdered2( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nstages = (int)cascade.data.stages.size();
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    for( int si = 0; si < nstages; si++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[si];
        int wi, ntrees = stage.ntrees;
        sum = 0;

        for( wi = 0; wi < ntrees; wi++ )
        {
            CascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
            int idx = 0, root = nodeOfs;

            do
            {
                CascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
                double val = featureEvaluator(node.featureIdx);
                idx = val < node.threshold ? node.left : node.right;
            }
            while( idx > 0 );
            sum += cascadeLeaves[leafOfs - idx];
            nodeOfs += weak.nodeCount;
            leafOfs += weak.nodeCount + 1;
        }
        if( sum < stage.threshold )
            return -si;
    }
    return 1;
}


template<class FEval>
inline int predictOrderedStump2( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, double& sum )
{
    int nodeOfs = 0, leafOfs = 0;
    FEval& featureEvaluator = (FEval&)*_featureEvaluator;
    float* cascadeLeaves = &cascade.data.leaves[0];
    CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
    CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

    int nstages = (int)cascade.data.stages.size();
    for( int stageIdx = 0; stageIdx < nstages; stageIdx++ )
    {
        CascadeClassifier::Data::Stage& stage = cascadeStages[stageIdx];
        sum = 0.0;

        int ntrees = stage.ntrees;
        for( int i = 0; i < ntrees; i++, nodeOfs++, leafOfs+= 2 )
        {
            CascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
            double value = featureEvaluator(node.featureIdx);
            sum += cascadeLeaves[ value < node.threshold ? leafOfs : leafOfs + 1 ];
        }

        if( sum < stage.threshold )
            return -stageIdx;
    }

    return 1;
}

//optimize version, only resize detect windows in stead of resizing full image
int CascadeClassifier::runAt2( Ptr<FeatureEvaluator>& evaluator, Point pt, double& weight )
{
    CV_Assert( oldCascade.empty() );

    assert( data.featureType == FeatureEvaluator::HAAR ||
            data.featureType == FeatureEvaluator::LBP ||
            data.featureType == FeatureEvaluator::HOG );

    if( !evaluator->setWindow(pt) )
        return -1;
    if( data.isStumpBased )
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrderedStump<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategoricalStump<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrderedStump<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
    else
    {
        if( data.featureType == FeatureEvaluator::HAAR )
            return predictOrdered<HaarEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::LBP )
            return predictCategorical<LBPEvaluator>( *this, evaluator, weight );
        else if( data.featureType == FeatureEvaluator::HOG )
            return predictOrdered<HOGEvaluator>( *this, evaluator, weight );
        else
            return -2;
    }
}


bool MaskCascadeClassifier::detectSingleScale( const Mat& image, int stripCount, Size processingRectSize,
                                           int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           vector<int>& levels, vector<double>& weights, bool outputRejectLevels )
{
    if( !featureEvaluator->setImage( image, data.origWinSize ) )
        return false;

#if defined (LOG_CASCADE_STATISTIC)
    logger.setImage(image);
#endif

    Mat currentMask;
    if (!maskGenerator.empty()) {
        currentMask=maskGenerator->generateMask(image);
    }

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

        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

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

        if( !detectSingleScale( scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
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
}
