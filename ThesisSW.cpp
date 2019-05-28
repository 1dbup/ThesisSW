/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2017-2019 David Burke (dburke.ceng@gmail.com). All rights reserved.
 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

//#include "ui_QMoviePlayer.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctype.h>
#include <unistd.h> // needed for getopt()
#include <pthread.h>
//#include <chrono>

using namespace cv;
using namespace std;
using std::string;
using std::ostringstream;

const int maxCPUIndex = 18;
const int maxGPUIndex = 9;
int Rows ;  // Whole array row and col. Only set these while testing without image
int Cols ;
bool structInit = 0; //flag to indicate structure hasn't yet been initialised
int Top, Bottom, Left, Right;
bool needBorder =0;
int LowerThreshPosn = 40;
int UpperThreshPosn = 90;
int CPUFrequency = 18;
int PowerPerfQual = 3;

//Default set up values, modified by getopt if reqd
int subArraySizeX = 2;
int subArraySizeY = 2;
int WGSizeX       = 6;
int WGSizeY       = 6;

//Stats
int Level0Count, Level1Count, Level2Count;
int ConvOp = 0;
    
//Thresholding    
int l0Average,l1Average,l2Average;
int l0Target, l1Target, l2Target;
int PPQChangeFlag;
int poolTotal; //Total Pool rectangles

//Timing
double tickFreq, currentTick, oldTick, nextTick;
bool secondInt = 0;
double frameTime, lastFrameTime;
double frameInt;
bool getVidFrame = 0;
int fps;

//CPU and GPU frequency change
int CPUFreqChangeFlag =0;

int camW = 800;
int camH = 600;
	/* Choice of following:-
 320 x 240
 640 x 480
 800 x 448
 800 x 600
 848 x 480
 960 x 720
1280 x 720
1280 x 800
*/


Mat Pool; // this will be a src/worgroupSize pooled value of max, min or mean deviation
Mat sigPool; //identical size to Pool but with significance threshold levels
Mat dst;  // This will be filtered (src size) image processed on approx results in Pool
cv::Mat src; // Source gray scale file
Mat PoolHist; //Pool histogram

int WGXSubArrays = 64;//X
int WGYSubArrays = 64;//Y
int WGXStep =0;
int WGYStep = 0;

// Rectangle defiinition for work-groups
cv::Rect srcDstRect = cv::Rect(WGXStep, WGYStep, WGXSubArrays, WGYSubArrays);
  
/** Gaussian 1x1 matrix */
cv::Mat gaussian1x1 = (cv::Mat_<char>(1,1) << 1);

/** Gaussian 3x3 matrix */
cv::Mat gaussian3x3 = (cv::Mat_<char>(3,3) << 1, 2, 1,
                                      2, 4, 2,
                                      1, 2, 1);

/** Gaussian 5x5 matrix */
cv::Mat gaussian5x5 = (cv::Mat_<char>(5,5) << 1,  4,  6,  4, 1,
                                      4, 16, 24, 16, 4,
                                      6, 24, 36, 24, 6,
                                      4, 16, 24, 16, 4,
                                      1,  4,  6,  4, 1);

/** Sharpen 3x3 Matrix */
cv::Mat Sharpen3x3 = (cv::Mat_<char>(3,3) << 0, -1,  0,
                                            -1,  5, -1,
                                             0, -1,  0)/2;
											 
/** Sharpen 5x5 Matrix*/
cv::Mat Sharpen5x5 = (cv::Mat_<char>(5,5) <<-1,-1,-1,-1,-1,
                                            -1, 2, 2, 2,-1,
                                            -1, 2, 8, 2,-1,
                                            -1, 2, 2, 2,-1,
                                            -1,-1,-1,-1,-1)/2;
// define class structure of set-up variables, used by multiple methods
struct ApproxValues {
	//Variables that can be set before opening an image
    int SubArraySize[2] ; //0= Rows and 1=Col default value y,x
    int SAOffset[2] ; // x,y format
	int workGroupSize[2]; // default value for Compute lib Mali6xx y,x

    //flags for pooling sub array values into NDrange values
    bool maxPool ; //Max pooling required
    bool meanPool ; //Mean Pooling required - redundant
    bool dynRangePool ; //Dynamic range Pooling (Max-min) required
	
	//variables that can only be set after opening an image source
	int srcSize[2]; //input image size
	int dstSize[2]; // processed image size same as src
	int adjustedDims[2]; //extra rows and columns to be added if not div worgroup = integer
	int PoolSize[2]; //= srcSize/WGRange
	int PoolDims[2]; //Pool dimensions related to image / Workgroupsize
	int maxSig;
	
} ASVars;

//DVFS frequencies for Odroid XU4 processor and GPU
int freq[maxGPUIndex+1] = {177,177,266,266,350,350,420,480,543,600};
int freq0[maxCPUIndex+1] = {200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000};
// char array for script commands      
char script[30];

// Methods follow from here

// Method to vary CPU frequency of Odroid XU4
int adjustCPUfreq(void){

	//ind =  slider; //sigRatio/10;
	
	//indCPU =  ind ;//nindexCPU + ind;
	//if (ind >= maxGPUIndex){ ind = maxGPUIndex; }
	if (CPUFrequency >= maxCPUIndex) CPUFrequency = maxCPUIndex;
	//password needs to be set to sudo password in sprintf line below
	#ifdef __arm__
	sprintf(script,"echo ""password"" | sudo -S cpufreq-set -f %d -c %d", freq0[CPUFrequency],6);
    system(script);	
    #endif
	
	return freq0[CPUFrequency];
}


// Method to initialise the structure to default and/or command line option values
int initialiseStruct(void)
{ 	
	int offset;
	offset = 1;
	
	//change to vars dependent on arg parsing. Note that we use powers of 2 so that
	// left/right shift can be used instead of dividing integers
	ASVars.SubArraySize[0] 		= subArraySizeX; //X need this in powers of 2
	ASVars.SubArraySize[1] 		= subArraySizeY; //Y need this in powers of 2
	ASVars.SAOffset[0]			= (1 <<subArraySizeX)-1; //X
	ASVars.SAOffset[1]			= (1 <<subArraySizeY)-1; //Y
	ASVars.workGroupSize[0]		= WGSizeX; //X need this in powers of 2
	ASVars.workGroupSize[1]		= WGSizeY; //Y need this in powers of 2

	ASVars.maxPool   			= 1; // max poooling by default
	ASVars.meanPool  			= 0;
	ASVars.dynRangePool			= 0;
	
	ASVars.maxSig   			= 255; //set no of normalisation levels. 
	                                   //Future use as Threshold
	// set vars to image frame size after opening
	ASVars.srcSize[0]		= Rows;
	ASVars.srcSize[1]		= Cols;
	
	ASVars.PoolSize[0] 		=Rows/ASVars.workGroupSize[0]; //X
	ASVars.PoolSize[1] 		=Cols/ASVars.workGroupSize[1]; //Y

    // Check subarray size is not > the workgroup size
	if(ASVars.SubArraySize[0] > ASVars.workGroupSize[0] || ASVars.SubArraySize[0] > ASVars.workGroupSize[0])
	{
		cerr << "SubArray greater than WorkGroup "<<endl;
		return -1;
	}
	structInit = 1; //set flag to show we have been initialised
}


// Method to check and adjust the image dimensions of the input file/video 
// and or subtract rows and columns if not divisible by Workgroup size

void checkImageDims(void)
{
int newCols, newRows;
int WGwidth = 1 << ASVars.workGroupSize[0];
int WGheight =1 << ASVars.workGroupSize[1];
int widthDiv, heightDiv;

cerr<<"WGwidth "<<WGwidth<< " WGheight "<<WGheight<<endl;
needBorder = 0;
widthDiv = (int) Cols >> ASVars.workGroupSize[0];

// check column width is integer div otherwise add extra columns
if((widthDiv << ASVars.workGroupSize[0])!= Cols)
    {
    newCols = ((widthDiv+1) * WGwidth) - Cols;
    Left = (int) newCols >> 1;
    Right = newCols - Left; //takes care of odd number of cols
    Cols +=newCols;
    needBorder =1;
    cerr <<" Col adjust "<<newCols<<" L "<<Left<<" R "<<Right<<endl;
    }
heightDiv = (int) Rows >>ASVars.workGroupSize[1];

// now do the same checkexercise (integer div) for rows
if((heightDiv << ASVars.workGroupSize[1]) != Rows)
    {
    //cerr<<"Ceil result "<< (double)( Rows/WGheight)<<endl;
    newRows = ((heightDiv+1) * WGheight) - Rows;
    Top = (int) newRows >> 1;
    Bottom=newRows - Top; //Take care of odd number of rows
    Rows += newRows;
    needBorder =1;
    cerr<<"Row Adjust "<<newRows<<" T "<<Top<<" B "<<Bottom<<endl;
    }
}


// Method to Pool the Approximate Significance (deviation) array into Workgroup based value
// Options are to use a mean, max or range value pooling

void ASPool( InputArray _ApxAbsDArray,
			  OutputArray _WGPool,
			  OutputArray _sigPool )
{
    //number of sub arrays in workgroup ************* Careful- powers of two
    int WGXSubArrays = 1<< (ASVars.workGroupSize[0]-ASVars.SubArraySize[0]);//X
    int WGYSubArrays = 1<< (ASVars.workGroupSize[1]-ASVars.SubArraySize[1]);//Y
    int WGXStep, WGYStep;
    int poolXStep, poolYStep;

    Mat Approx = _ApxAbsDArray.getMat(); //input image
    Mat Pool = _WGPool.getMat(); //output array
    Mat sigPool = _sigPool.getMat(); //output significance level array

    cv::Size ApproxSize = Approx.size();
    int	AppxWidth = ApproxSize.width;
    int	AppxHeight = ApproxSize.height;

    cv::Size poolSize = Pool.size();
    int	PoolWidth = poolSize.width;
    int	PoolHeight = poolSize.height;
    
    double ApproxMin;
    double ApproxMax;
    uchar dynVal;
    Scalar Mean;
    Point PoolXY (poolXStep,poolYStep);

    // following is just for initialisation but should be calculated based on number of values 
    //in appxArray per Workgroup size = WG/SASize
    WGXStep = 32; 
    WGYStep = 32;
    Level0Count = 0;
    Level1Count = 0;
    Level2Count = 0;

    //save Pool dimensions to structure for later processing
    ASVars.PoolDims[0] = PoolWidth;
    ASVars.PoolDims[1] = PoolHeight;

    // set up cv rectangle size for workgroup size and step size
    cv::Rect ApproxRect = cv::Rect(WGXStep, WGYStep, WGXSubArrays , WGYSubArrays);
    cv::Mat ApproxRectArea = Approx(ApproxRect);

    //for each workgroup in Pool (height vs width)
    // Step size in the following is 2^(Workgroupsize - SubArraySize)
    // so for example, a workgroup of 64x 64 with sub array 4x4 is
    // 2^(6-2) ie 1 << (workGroupSize - SubArraySize)
    for (WGYStep = 0, poolYStep = 0 ; WGYStep < AppxHeight; WGYStep += WGYSubArrays, poolYStep++)
    {
	
	    for (WGXStep = 0, poolXStep =0; WGXStep < AppxWidth; WGXStep+= WGXSubArrays, poolXStep++)
		{
		
		    ApproxRect = Rect(WGXStep,WGYStep,WGXSubArrays , WGYSubArrays);
		
            ApproxRectArea = Approx(ApproxRect);
            PoolXY = Point(poolXStep,poolYStep);
		
		    if(ASVars.meanPool == 1)
			{
			    Mean = cv::mean(ApproxRectArea);
			    Pool.at<uchar>(PoolXY) =(uchar) Mean[0];
			}
		    if(ASVars.maxPool == 1)
			{
			    minMaxLoc(ApproxRectArea, NULL, &ApproxMax,NULL,NULL);
			    Pool.at<uchar>(PoolXY) =(uchar) ApproxMax; 
                if(ApproxMax >= UpperThreshPosn)
                {
                    ConvOp = 2;
                    Level2Count++;
                }
                else if (ApproxMax >= LowerThreshPosn)
                {
                    ConvOp = 1;
                    Level1Count++;
                }
                else
                {
                    ConvOp = 0;
                    Level0Count++;
                }
                sigPool.at<uchar>(poolYStep,poolXStep) =(uchar) ConvOp;
            }
		    if(ASVars.dynRangePool == 1)
			{
			    minMaxLoc(ApproxRectArea, &ApproxMin, &ApproxMax,NULL,NULL);
			    dynVal = (uchar) (ApproxMax - ApproxMin);
			    Pool.at<uchar>(PoolXY) =(uchar) dynVal; 
                if(dynVal >= UpperThreshPosn)
                {
                    ConvOp = 2;
                    Level2Count++;
                }
                else if (dynVal >= LowerThreshPosn)
                {
                    ConvOp = 1;
                    Level1Count++;
                }
                else
                {
                    ConvOp = 0;
                    Level0Count++;
                }
                sigPool.at<uchar>(poolYStep,poolXStep) =(uchar) ConvOp;
            }
		}
    }	
}


// Method to investigate Approximate sub-sampling to generate local mean
// along with absolute deviation
void ApproxAbsDev(
					InputArray _image,
					OutputArray _ApxAbsDArray )
{
unsigned long int Sum1, Sum2;
int RowCount, ColCount,RowBefore,ThisRow,x,y,xx,yy,i,j,ii,jj;
ushort UL,UR,LL,LR;
int Sum =0;
int ct = 0;
int Even;
int Mean1, Mean2,LocalMean=127, TwoRowMean;

//For now, start with a 1 in 16 selection eg (1,1) out of (0,0;3,3)
int StartRow = ASVars.SAOffset[1], StartCol = ASVars.SAOffset[0] ;
int MaskXStep= 1<< ASVars.SubArraySize[0];  
int MaskYStep = 1<< ASVars.SubArraySize[1];

// following needed for rect fill eg(0,0 to 3,3)
int XStep=MaskXStep-1,YStep=MaskYStep-1; 

// for masking Y address strides during rectangle fill
int YMask =0xFFFF ^ YStep; 


//Need to march along rows to maximise economy in accessing main memory

// write values to ApxAbsDArray, row major.
Mat image = _image.getMat(); //input image
Mat ApxAbsDArray = _ApxAbsDArray.getMat(); //output array
	
// get sizes from image Matrix 
cv::Size size =image.size();

int width = (size.width);
int height = (size.height);

int MaskWidth = width/MaskXStep;
int MaskDepth = height/MaskYStep;
int AbsWrX,AbsWrY;
int AbsDevValue ;
int GenDevFlag = 0;
int thickness = -1; // fill rectangle with no lines around
int LocalMeanFlag =1;

unsigned short Pixels[2][MaskWidth]; // Masked values from image
unsigned short AbsDev[2][MaskWidth]; //Equivalent deviation results

double RunTime,t, pyrTime, AccumRunTime1,AccumRunTime2, RunTime1,RunTime2;


AccumRunTime1 = 0;
AccumRunTime2 = 0;

// for whole array
for(RowCount = StartRow, y =0; RowCount< height; RowCount += MaskYStep, y++)
	{
	RunTime = (double)getTickCount();

	//First load first/next two relevant lines into an width/n X 2 array, 
	//calculate sum on the fly
	// Need to do rows in pairs for 2 x 2 Deviation
	Even = y & 0x01; // alternate rows
	switch (Even)
		{
		case 0:
			{
			for(ColCount = StartCol, x = 0; ColCount< width; ColCount += MaskXStep, x++)
				{
				//Read image line and store in local Nx2 array, 
				//generate line sum for line average
				Pixels[0][x] =  image.at<uchar>(RowCount,ColCount);

				}
			//Need to save this striding row count for odd storage
			RowBefore = RowCount;	
			
			GenDevFlag = 0; // Don't generate deviation until second line dealt with
			break;
			}
		
		case 1:
			{
			for(ColCount = StartCol, x = 0; ColCount< width; ColCount += MaskXStep, x++)
				{
				Pixels[1][x] = image.at<uchar>(RowCount,ColCount);
				}
			//Need to save this row count for later storage
			ThisRow = RowCount;				
			GenDevFlag = 1; // Now we can generate the deviation figures
			break;
			}
		default:
			{
			// Shouldn't get here
			cerr << "Unexpected case in ApproxAbsDev "<<endl;
		
			}
		}
	if(GenDevFlag == 1)
		{
		//Initially generate mean for these lines
		//then calculate abs dev from mean
		//Store results in a width/divN x 2 temporary array. 
		for(i=0; i<MaskWidth; i += 2)
			{
			UL = Pixels[0][i];
			UR = Pixels[0][i+1];
			LL = Pixels[1][i];
			LR = Pixels[1][i+1];
			LocalMean =(UL+UR+LL+LR) >> 2; // valid only for div 4
			
            // Fill values into temporary two line array then line fill to main memory
            // to get max memory write efficiency 
			AbsDev[0][i]   = abs(UL-LocalMean);
			AbsDev[0][i+1] = abs(UR-LocalMean);
			AbsDev[1][i]   = abs(LL-LocalMean);
			AbsDev[1][i+1] = abs(LR-LocalMean);
			}


		// accumlate run time for this section and setup for next
		
		AccumRunTime1 += ((double)getTickCount() - RunTime);
		RunTime = (double)getTickCount();

		
		//Write the calculated abs dev to rectangles in Mask array, same size as Image
		//Lastrow/col is offset by startrow/col so should start at 0,0 instead of 1,1
				
		if(Even==1)
			{
			
			for(AbsWrY=(RowBefore & YMask), j =0; AbsWrY<=(ThisRow & YMask); AbsWrY +=MaskXStep, j++)
				{
				// need to fill <from>and <to> for x and y
				for(AbsWrX = 0, i=0; AbsWrX<width; AbsWrX += MaskXStep,i++)
					{
					yy = AbsWrY >> ASVars.SubArraySize[0];
                    xx = AbsWrX >> ASVars.SubArraySize[1] ;
					
					//Here is were we store the values in the 1/n size array ************
					ApxAbsDArray.at<uchar>(yy, xx) = AbsDev[j][i]; 
					}
				}
			}
			
		GenDevFlag = 0;
		AccumRunTime2 += ((double)getTickCount() - RunTime);

		}		
	//Ready to run window masks
	}
	RunTime2 =AccumRunTime2*1000/getTickFrequency();
	RunTime =RunTime1+RunTime2;
	
	t= RunTime +pyrTime;
	//Approximation complete
}


//Method to perform 1x1, 3x3, 5x5 kernel filters to areas of image where
//corresponding Significance Pool value is below, between or above Lower and upper thresholds

int ComputeASperSig(
        InputArray _sigPoolValues,
        OutputArray _destArray)
{

    int borderType = cv::BORDER_REPLICATE;
    int imgsrc =0;
    double start, elapsed;
    double start0, accum0, elapsed0;
    double start1, accum1, elapsed1;
    double start2, accum2, elapsed2;
    double start3, accum3, elapsed3;
    double start4, accum4, elapsed4;
    int srcXStep, srcYStep;
    int poolXStep,poolYStep;
    int Check0Count, Check1Count, Check2Count;
   
    Mat sigPool = _sigPoolValues.getMat(); //output array
    Mat dstConv = _destArray.getMat();
     
    Check0Count = 0;
    Check1Count = 0;
    Check2Count = 0;
    
    WGXStep =0;
    WGYStep = 0;
    WGXSubArrays = 1<< (ASVars.workGroupSize[0]);//X
    WGYSubArrays = 1<< (ASVars.workGroupSize[1]);//Y
   
    srcDstRect.width = WGXSubArrays;
    srcDstRect.height = WGYSubArrays;
    cv::Mat src_WGRange = src(srcDstRect);
    cv::Mat dst_WGRange = dstConv(srcDstRect);
   
	
    // The purpose of this demo is to check syntax of convolving workgroups with 
    // different patterns 1x1, 3x3, 5x5
    // 
    
    //Need to lookup the Pool Threshold value and ConvOp acordingly
    
    for(WGYStep=0, poolYStep=0; WGYStep< Rows; WGYStep += WGYSubArrays, poolYStep++) 
    
    {
        //For now just do 4 of each, 1x1, 3x3 and 5x5 across the image
        ConvOp = 0;
                
        for(WGXStep=0, poolXStep=0; WGXStep < Cols ; WGXStep += WGXSubArrays, poolXStep++)
        {
            srcDstRect = Rect(WGXStep, WGYStep,WGXSubArrays,WGYSubArrays);
            
            //Extract Pool value for this workgroup and set ConvOp to 0:1:2
            // dependent on \Threshold value
            Point PoolXY (poolXStep,poolYStep);
            ConvOp =(uchar) sigPool.at<uchar>( PoolXY); 
           
            
            // just need to perform different ops sequentially & repeat
            
            switch(ConvOp)
            {
                case 0:
    	        //Just for now lets do a copy instead of 1x1 filter
				src(srcDstRect).copyTo(dstConv(srcDstRect));
					    	                 
    	        break;
	
	            case 1:
	            // 3x3 filter
                cv::filter2D(src(srcDstRect), 
                             dstConv(srcDstRect),
                             CV_8U,
                             Sharpen3x3);
                break;
                
                case 2:
		        //5x5 Convolution
                cv::Sobel(src(srcDstRect),
                            dstConv(srcDstRect),
                            CV_8U,1,1,5,1,0,BORDER_DEFAULT);
                break;
    	    }
	    }
    }
    //All sectors done
}


// Method 
void adjust_thresholds(InputArray _Pool)
{
	Mat Pool = _Pool.getMat(); //Input significance level array
	
	
	// Large images have a large number of pool values so need a variable threshold to
	// move quickly into the level0-2 values 
	
	// Create a histogram of the Pool values. Then we can count down from 255 and sum
	// the number of values. When the sum is nearest l2Target or l1Target the histogram
	// value we have reached is the new threshold.
	
	// setup for histogram method
	int histSize =256;
	float range[] = {0, 256}; //upper boundary is exclusive
	const float* histRange = {range};
	bool uniform = true; bool accumulate = false;
	
	calcHist( &Pool, 1, 0, Mat(), PoolHist, 1, &histSize, 0);
	
	
	// Now need to count down from 255 -> 0 and sum hist bins for 2nd and first thresholds
	// need to do this for l2 first & set level2 threshold, then do l1 & set level 1 
	// threshold
	int i;
	int sum2 =0;
	int sum1 =0;
	int sum0 = 0;
	int lastSum1 =0;
	int lastSum2 = 0;
	int sumTotal = 0;
	int level = 2;
	
	for(i = 255; i > 0; i--) // watch this we are counting down - not up
	{
		switch(level)
		{
			case 2: //High significance
				sum2 += (unsigned int) PoolHist.at<float>(i);
				if(sum2 >= l2Target)    //for now lets keep it simple
				{
					//cerr<<sum2<<" "<<lastSum2<<" "<<UpperThreshPosn<<" ";
					// we can make this check for the nearest to target, above or below and 
					// use nearest
					if(abs(l2Target-sum2) <= abs(l2Target - lastSum2))
						UpperThreshPosn = i; 	//set threshold
					else
					{
						i++; //increment to last value
						UpperThreshPosn = i; 
						sum2 = lastSum2;
					}	
					level = 1;				// now do Lower thresh
				}else
					lastSum2 = sum2;
				break;
				
			case 1: // Medium significance
				sum1 += (unsigned int)PoolHist.at<float>(i);
				if(sum1 >= l1Target)    //for now lets keep it simple
				{
					LowerThreshPosn = i; 	//set threshold
					if(abs(l1Target - sum1) <= abs(l1Target - lastSum1))
						LowerThreshPosn = i; 	//set threshold
					else
					{
						i++; //increment to last value
						LowerThreshPosn = i;
						sum1 = lastSum1;
					}
					level = 0;				// all done
				}else
					lastSum1 = sum1;
				break;
			case 0: // Low significance
				sum0 += (unsigned int)PoolHist.at<float>(i);
				break; //do nothing
		}
	}
	// set track sliders to new threshold position
	setTrackbarPos("Thld 1", "Sliders",  LowerThreshPosn); 
	setTrackbarPos("Thld 2", "Sliders",  UpperThreshPosn);
}


// Method to calculate target number of workgroups for the percentage levels set by the PPQ value
void changePPQVars(void)
{
    // change percentages to %ge of poolTotal global
    switch (PowerPerfQual)
	{
        case 0: //Power concious 90;5;5
        {
            l0Target = floor(0.9 * poolTotal);              //90%
            l1Target = floor(0.05 * poolTotal);             //5%
            l2Target = poolTotal -(l0Target + l1Target);    //5%
            break;
        }
        case 1: // Low Performance 80;10;10
        {
            l0Target = floor(0.8 * poolTotal);              //80
            l1Target = floor(0.1 * poolTotal);              //10
            l2Target = poolTotal -(l0Target + l1Target);    //10
            break;
        }
        case 2: // Medium Performance 70;15;15
        {
            l0Target = floor(0.7 * poolTotal);              //70
            l1Target = floor(0.15 * poolTotal);             //15
            l2Target = poolTotal -(l0Target + l1Target);    //15
            break;
        }
        case 3:  // High Performance 60;20;20
        {
            l0Target = floor(0.6 * poolTotal);              //60
            l1Target = floor(0.2 * poolTotal);              //20
            l2Target = poolTotal -(l0Target + l1Target);    //20
            break;
        }
        case 4: //Quality performance 50;30;20
        {
            l0Target = floor(0.5 * poolTotal);              //50
            l1Target = floor(0.3 * poolTotal);              //30
            l2Target = poolTotal -(l0Target + l1Target);    //20
            break;
        }
         case 5: //Demo 3x3 on full image
        {
            l0Target = floor(0.0 * poolTotal);     //0
            l1Target = floor(1.0 * poolTotal);     //100
            l2Target = poolTotal -(l0Target + l1Target);     //0
            break;
        }
        case 6: //Demo 5x5 on full image
        {
            l0Target = floor(0.0 * poolTotal);     //0
            l1Target = floor(0.0 * poolTotal);     //0
            l2Target = poolTotal -(l0Target + l1Target);     //100
            break;
        }
      
    }
}


 
// Main starts here  ********************************************************

int main(int argc, char* const* argv)

{
    //Getopt vars
    int c;
    int index;
    char *cvalue = NULL;
    // set default image filename	
    const char* filename = "/usr/share/backgrounds/Cedar_Wax_Wing_by_Raymond_Lavoie.jpg";
 
    //This Programs variables
    double* maxVal;
	int CannyMaxPosn = 150;
	int CannyMinPosn = 80;
	int ApproxThreshPosn = 60;
    double LoopStartTime, LoopTime, SigApproxStartTime, SigApproxTime,Slack;
    double runTime; 
    int l0_sum,l1_sum,l2_sum;
    int  frameAverage = 3; //Powers of 2 for right shift
    bool gotImage =  0;
    bool image_flag = 0;
    bool video_flag = 0;
    bool camera_flag = 0;
	bool noWin_flag = 0;
    bool debug_flag = 0;
    bool startup = 0;
    bool gotFrame = 0;

    Mat frame;
    
    void PPQEvent(int, void*);
    void CPUFreqEvent(int, void*);
    int  OneSecTimer(void);
    int frameTimer(void);
    
    char tellback[80];
   	char freqtb[80];

    l0Average=0;
    l1Average=0;
    l2Average=0;
    l0_sum=0;
    l1_sum=0;
    l2_sum=0;
    l0Target = 200;
    l1Target = 20;
    l2Target = 20;

	// Initialise tick counters
	currentTick = getTickCount(); 
	oldTick = currentTick ;
	tickFreq = getTickFrequency();
	
	// setup default call <image> if no options given
	if(argc == 0) {
		printf("%s",filename);
		image_flag = 1;
		}
	//Deal with command line calling options
  	while ((c = getopt (argc, argv, "i:v:ctqoxsmMdlhwn")) != -1)
    switch (c)
    {
		case 'h': // help output
			printf("\n\nPrint this help info \n");
           	printf("Call with selections of following arguments \n");
			printf(" -i image <filepath/filename> Path & alternate file to <default> \n");
			printf(" -v video <filepath/filename> file sequence for input \n");
			printf(" -c Use camera input instead of file \n");
           	printf(" -t (two)  2x2 sub array size \n");
			printf(" -q Quad,  4x4 sub array size \n");
			printf(" -o Octal, 8x8 sub array size \n");
			printf(" -x (hex), 16x16 sub array size \n");
			printf(" -s small WorkGroup size, 16x16 \n");
			printf(" -m medium WorkGroup size 32x32 \n");
			printf(" -l Large WorkGroup size 64x64 \n"); 
			printf(" -w Use extra windows for debug of Pooling \n");
			printf(" -n No windows, for performance testing \n");
			printf(" -M Significance based on max value in Workgroup \n");
			printf(" -d Significance based on dynamic range value in Workgroup \n");
           	printf("Default run action is -cqlM \n");
			//spare letters abefgjkruyz
			return 0;
		case 't':
	    	//use 2x2
	       	subArraySizeX 		= 1; //X need this in powers of 2
	       	subArraySizeY 		= 1; //Y need this in powers of 2
			break;
		case 'q':
			//use 4x4
	       	subArraySizeX 		= 2; //X need this in powers of 2
	       	subArraySizeY 		= 2; //Y need this in powers of 2
			break;
		case 'o':
	    	//use 8x8
	    	subArraySizeX 		= 3; //X need this in powers of 2
	       	subArraySizeY 		= 3; //Y need this in powers of 2
	    	break;
		case 'x':
			//use 16x16
        	subArraySizeX 		= 4; //X need this in powers of 2
        	subArraySizeY 		= 4; //Y need this in powers of 2
			break;
		case 'i': //use static image file
	    	if(video_flag == 0 && camera_flag == 0) //first there
		    	cvalue = optarg;
			filename = cvalue;
			image_flag = 1;
			break;
		case 'v': // Run using Video file 
			if( image_flag == 0 && camera_flag == 0)
		    	video_flag =1;
			cvalue = optarg;
			filename = cvalue;
			break;
		case 'c': //Use camera input
	    	if( video_flag == 0 && image_flag == 0)
		    	camera_flag =1;
			break;
		case 's':
			//WorkGroup size =16x16
        	WGSizeX		= 4; //X need this in powers of 2
        	WGSizeY		= 4; //Y need this in powers of 2
			break;
		case 'm':
			//WorkGroup size =32x32
        	WGSizeX		= 5; //X need this in powers of 2
        	WGSizeY 	= 5; //Y need this in powers of 2
			break;
		case 'l':
			//WorkGroup size =64x64
        	WGSizeX		= 6; //X need this in powers of 2
        	WGSizeY		= 6; //Y need this in powers of 2
       	break;
		case 'w':
			debug_flag = 1; // We want to display extra debug stuff 
			break; 
		case 'n':
			// We want switch off window display during performance tests
			noWin_flag = 1;  
			break; 
		case 'M':
			// We want to Pool value based on the Maximum value
			ASVars.maxPool = 1;
			ASVars.dynRangePool = 0;
			break; 
		case 'd':
			//We want to pool significance based on dynamic range, max-min
			ASVars.dynRangePool = 1;
			ASVars.maxPool = 0;
			
			break; 
		case '?':
			if (optopt == 'i')
				fprintf (stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint (optopt))
				fprintf (stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf (stderr,"Unknown option character `\\x%x'.\n",optopt);
			return 1;
		default:
			abort ();
		}

		
 	for (index = optind; index < argc; index++)
   		printf ("Non-option argument %s\n", argv[index]);
   	// end of getopt() options
  
    // setup image source dependent on options used
  	VideoCapture cap(0); // open the default camera 0 on XU4
  	//VideoCapture cap(1); // open the default camera 1 on Toshiba
  	VideoCapture file(filename);

  	if(camera_flag == 1)
  	{
    	if(!cap.isOpened())  // check if we succeeded to open camera stream
    		return -1;
      
		cap.set(CV_CAP_PROP_FRAME_WIDTH, camW);
    	cap.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
     	Cols = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    	Rows = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    	fps = cap.get(CV_CAP_PROP_FPS);
		cerr <<"Camera Width "<<Cols<<" Height "<< Rows <<" fps "<<fps <<endl; 

  	}
  	else if(video_flag == 1)
  	{
    	//Need to find what video formats and decoders are required
       	// open the file 
       	if(!file.isOpened())  // check if we succeeded
       		return -1;
       
       	Cols = file.get(CV_CAP_PROP_FRAME_WIDTH);
       	Rows = file.get(CV_CAP_PROP_FRAME_HEIGHT);
       	fps  = file.get(CV_CAP_PROP_FPS);
	   	cerr <<"Video Width "<<Cols<<" Height "<< Rows <<" fps "<<fps<<endl; 
        
  	}
  	else //if(image_flag == 1) and default
  	{
       	if(!file.isOpened())  // check if we succeeded
       		return -1;
  
  	   	frame = imread( filename, IMREAD_COLOR);

		cv::cvtColor(frame, src, cv::COLOR_BGR2GRAY);
	
	   	cv::Size OrigSize = frame.size();
       	Cols = OrigSize.width;
	   	Rows = OrigSize.height;
	   	cerr <<"ImageWidth "<<Cols<<" Image Height "<< Rows <<endl; 
  	}    
        
	namedWindow("Sliders",WINDOW_NORMAL);
	//Disable windows if reqd for performance
    if(noWin_flag ==0 )
	{
    	namedWindow("src",WINDOW_NORMAL);
    	namedWindow("Dst",WINDOW_NORMAL);
    	namedWindow("frame",WINDOW_NORMAL);

    	if(debug_flag == 1) //if extra windows are required for debug
    	{
			namedWindow("Approx",WINDOW_NORMAL);
			namedWindow("Pool",WINDOW_NORMAL);
			namedWindow("Pool T'hold",WINDOW_NORMAL);
		}
	}

    // End of option stuff now for the real stuff **************************
	PPQEvent(PowerPerfQual,NULL);    	
	if(debug_flag == 1)
    	createTrackbar("Approx thresh", "Sliders",&ApproxThreshPosn,255);
    		
	createTrackbar("Thld 2", "Sliders",&UpperThreshPosn,255);
	createTrackbar("Thld 1", "Sliders",&LowerThreshPosn,255);
	createTrackbar("PPQ", "Sliders",&PowerPerfQual,6,PPQEvent);
	createTrackbar("CPU-freq","Sliders", &CPUFrequency,18,CPUFreqEvent);
	tickFreq = getTickFrequency();
	
	// Here is the place to initialise the structure now we have opened the 
	// source
	if ( structInit == 0)
	{
		if(initialiseStruct() == -1)
			return -1;
	}	
	
	// Now check if borders are needed and update Rows & Cols
	checkImageDims();
	cerr<<"Cols "<<Cols<<" Rows "<<Rows<<endl;
	
	// Save adjusted Dims in structure for future use to create output dst array
	ASVars.adjustedDims[0] = Cols;
	ASVars.adjustedDims[1] = Rows;

    // setup Mats based on image size, sub array size, workgroup size  and adjusted size
	cv::Mat ApproxSig(Rows >> ASVars.SubArraySize[1], Cols >> ASVars.SubArraySize
	[0], CV_8U);
	cv::Mat dst(ASVars.adjustedDims[1],ASVars.adjustedDims[0], CV_8U);
	cerr<< "ApproxSig size "<< ApproxSig.size()<<endl;
	
	// Pool matrices & vars
	cv:Mat Pool( (Rows >> ASVars.workGroupSize[1]), (Cols >> ASVars.workGroupSize[0]),CV_8U);
	Mat sigPool( (Rows >> ASVars.workGroupSize[1]), (Cols >> ASVars.workGroupSize[0]),CV_8U);
	cv::Size PoolSize = Pool.size();
    int	PoolWidth = PoolSize.width;
    int	PoolHeight = PoolSize.height;
    poolTotal = PoolWidth * PoolHeight;
	cerr<< "Pool size "<< Pool.size()<<" Total "<<(poolTotal)<<endl;
	
    int oldPPQvalue = 0;
    lastFrameTime = getTickCount();
    if (image_flag ==1) // set image recycle time to 10fps for static image
       	fps = 10; // default will be overidden by video or camera
 	
 	frameInt =1.000/fps;
   	cerr<<"fps "<<fps<< " Interval "<< frameInt<<endl;
   	LoopTime =0;
   	gotImage =0;
   	gotFrame =0;
	CPUFrequency = 2000;
	sprintf(freqtb,"CPU freq %d MHz ",CPUFrequency);
	if (video_flag ==1)
	{
	    cerr<<"space to continue"<<endl;
   	    while(waitKey(100) != 32);
    }
    
    // now enter the main loop for processing images, forever until <esc> is pressed
    for(;;)
    {

    	//Frame timer will be more frequent than 1 second, check frame time before 1 sec timer
    	frameTimer();

       	OneSecTimer();
       	
        LoopStartTime = (double)getTickCount();
       	
       	//every second we need to update the threshold values
       	if(secondInt == 1)
       	{
           	adjust_thresholds(Pool);
           	// Has PPQ level changed in the last second?
           	if(PowerPerfQual != oldPPQvalue)
           	{
               	changePPQVars();
               	oldPPQvalue = PowerPerfQual;
           	}
           	secondInt=0;
       	}    
       	if( PPQChangeFlag == 1 ) // PPQ slider has been moved
       	{
           	changePPQVars();
           	PPQChangeFlag = 0;
       	}   
       	if(camera_flag == 1) //Only get frames during video file or camera use
       	{
           	cap >> frame; // get a new frame from camera
           	if(!frame.data ) 
           	{
           		cerr<<"Failed to get image from camera"<<endl;
           		break;
           	}
           	cvtColor(frame, src, COLOR_BGR2GRAY);
           	gotFrame = 1;
       	} 
       	else if((video_flag ==1) && (getVidFrame==1)) // when dealing with video file
       	{
           	file >> frame;
           	if(!frame.data ) 
           	{
           		cerr<<"Failed to get frame from video file"<<endl;
           		break;
           	}
           	cvtColor(frame,src, COLOR_BGR2GRAY);
           	gotFrame = 1;
       	}else if((image_flag == 1) && (gotImage ==0)) //when dealing with static image
       	{
       		frame = imread( filename, IMREAD_COLOR);
	    	cv::cvtColor(frame, src, COLOR_BGR2GRAY);
           	gotImage = 1;
           	fps = 10; ///dummy fps tp calc times etc
           	frameInt =1.000/fps;
       	}
       	LoopTime += ((double)getTickCount() - LoopStartTime)*1000/getTickFrequency();

       	if(gotImage || gotFrame) // have we got new camera frame or has time expired 
       	                         // for next video  frame
       	{
       		//Start timing after image is obtained,video frame fps limits the overall time
       		SigApproxStartTime = (double)getTickCount();
       		// Video file may need border adjusted every frame but image file only once
       		if(needBorder == 1)
           	{
           		cv::copyMakeBorder( src, src, Top, Bottom, Left, Right, BORDER_DEFAULT);
           	}
           	
           	//start of image processing, create approximate significance from source image
			ApproxAbsDev(src, ApproxSig);
			
			// now fill Pool matrix with mean or max resolution
			ASPool(ApproxSig, Pool, sigPool);
   
    		// Now use Pool area to select what kernel filter is applied at levels 
    		// between T'holds 1 & 2
    		ComputeASperSig(sigPool,dst);
    		
    		// generate and display tell-back info over image
    		if (Cols<2000) { // small frame size
	    		cv::putText(dst,tellback, Point(40,80), FONT_HERSHEY_COMPLEX_SMALL,1,0,2);
	    		cv::putText(dst,freqtb, Point(40,110), FONT_HERSHEY_COMPLEX_SMALL,1,0,2);
  			} else { // large frame size needs larger text
	    		cv::putText(dst,tellback, Point(10,50), FONT_HERSHEY_COMPLEX,1,0,2);
	    		cv::putText(dst,freqtb, Point(10,80), FONT_HERSHEY_COMPLEX,1,0,2);
	    	}
	    	
       		// Computation part is finished, keep window display stuff out of timing
			SigApproxTime = ((double)getTickCount() - SigApproxStartTime)*1000/getTickFrequency();
	    
    		if(debug_flag == 1) // debug will update pool image Mats
    		{
           		imshow("Pool", Pool);		    
           		cv::threshold(Pool, Pool, ApproxThreshPosn,255,THRESH_BINARY);
       		}
		
			//Remove Windows display for performance testing
			if(noWin_flag ==0 )
			{        
				imshow("Dst", dst);
				imshow("src", src);        		
				imshow("frame", frame);
				
       			//Enable the following on debug
       			if(debug_flag == 1)
       			{
		    		imshow("src", src);
   					imshow("Approx", ApproxSig);
    				imshow("frame", frame);
         			imshow("Pool T'hold", Pool);
    			}
			}
	    
       		
       		// This is the averaging process for thresholds, averaged over 8 frames
       		l0_sum =l0_sum+ Level0Count-(l0Average);
       		l1_sum =l1_sum+ Level1Count-(l1Average);
       		l2_sum =l2_sum+ Level2Count-(l2Average);
       		l0Average = l0_sum >> frameAverage;
       		l1Average = l1_sum >> frameAverage;
       		l2Average = l2_sum >> frameAverage;
       		

       		// Calculate slack time
       		Slack = (frameInt*1000)-(LoopTime+SigApproxTime);
       		if(CPUFreqChangeFlag ==1) // need to check slack doesn't go -ve
       		{
       			#ifdef __arm__
       			CPUFrequency = adjustCPUfreq();
       			#endif
       			CPUFreqChangeFlag = 0;
       			//cerr<<"Adjust freq "<<CPUFrequency<<endl;
       			sprintf(freqtb,"CPU freq %d MHz ",CPUFrequency/1000);
       		}
       		printf("\rTgt 0-2: %d %d %d  0-2 Avg: %3d %3d %3d  Proc Time %2.1f Slack %2.1f   ", l0Target, l1Target, l2Target, (l0Average), (l1Average), (l2Average), SigApproxTime, Slack);
       		
			sprintf(tellback,"0-2 Tgt: %d %d %d Avg: %3d %3d %3d Time %4.1f Slack %4.1f     ", l0Target, l1Target, l2Target, (l0Average), (l1Average), (l2Average), SigApproxTime, Slack);       		
       		       		
       		gotFrame = 0;
       		
		}
       	LoopTime =0;
 
       	if(waitKey(1) == 27) // has <esc> been pressed - stop processing
       	{
       	 	setTrackbarPos("CPU-freq", "Sliders",  maxCPUIndex); //set CPU frequency back to max
       		#ifdef __arm__
       		adjustCPUfreq();		
       		#endif
       	 	
       	 	break; 
       	} 
	}
   	// the camera will be deinitialized automatically in VideoCapture destructor
   	cerr<<endl;
   	destroyAllWindows();
   	return 0;
} // end of main()

//Slider bar event handling starts here


void ApproxThreshEvent(int ApproxThreshPosn, void*)
{
	// image is updated in for(::) loop
}
void Thresh1Event(int LowerThreshPosn, void*)
{
	// Threshold 1 level is set by main program
}
void Thresh2Event(int UpperThreshPosn, void*)
{
	// Threshold 2 level is set by main program
}
void PPQEvent(int PowerPerfQual, void*)
{
	// Set flag so image is updated in for(::) loop
	PPQChangeFlag = 1;
	//cerr<<"PPQ "<<PowerPerfQual;
	
}

void CPUFreqEvent(int CPUFrequency, void*)
{
	// Set flag so frequency is updated in for(::) loop
	CPUFreqChangeFlag = 1;
	//cerr<<"PPQ "<<PowerPerfQual;
	
}

// Following timers need to be utilising synchro C++11, but not available in OpenCV-3.X
int OneSecTimer(void)
{
	
	secondInt = 0;
    currentTick = getTickCount(); 
    if(((currentTick - oldTick)/getTickFrequency()) > 1)
    {
        oldTick = currentTick;
        //cerr<< ".";
        return secondInt = 1;       
    }

}
int frameTimer(void)
{
    getVidFrame =0;
    frameTime = getTickCount();
    if((( frameTime - lastFrameTime)/getTickFrequency())> frameInt)
    {
        lastFrameTime = frameTime;
        //cerr<< "-";
        return getVidFrame = 1;
    }
}


