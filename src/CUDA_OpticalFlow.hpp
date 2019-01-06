#ifndef __CUDA_OPTICALFLOW_HPP
#define __CUDA_OPTICALFLOW_HPP

#include <iostream>  
#include <fstream>  
 
#include <opencv2/opencv.hpp>

//!linux 
#include <sys/stat.h> 
#include <unistd.h>  
#include <sys/types.h>
#include <dirent.h>

using namespace std;  
using namespace cv;  

#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/utility.hpp>  
#include "opencv2/cudaoptflow.hpp"  
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaarithm.hpp"
using namespace cv::cuda;

#else

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv::gpu;

#endif 
 

//#define TEST_SPEED   //!compare the five methods 
 
 
 

class CUDA_OpticalFlow  
{  
public:  
	CUDA_OpticalFlow(void)  
 
	{  
#if CV_MAJOR_VERSION >= 3
		brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
        lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
        farn = cuda::FarnebackOpticalFlow::create();
        tvl1 = cuda::OpticalFlowDual_TVL1::create();
#else
       
       
#endif
 
	}    
	CUDA_OpticalFlow(cv::Size size)  
	{  
    	img_size = size;  
   		d_flowx.create(size, CV_32FC1);  
   	    d_flowy.create(size, CV_32FC1);  
#if CV_MAJOR_VERSION >= 3
		brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
        lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));
        farn = cuda::FarnebackOpticalFlow::create();
        tvl1 = cuda::OpticalFlowDual_TVL1::create();
#else
   
        
#endif
	}  

    ~CUDA_OpticalFlow()
	{

	}  


    cv::Size img_size;  
    GpuMat d_flowx;     //d_x
    GpuMat d_flowy;     //d_y
    GpuMat d_flow;
 

   
    //!methods classes
#if CV_MAJOR_VERSION >= 3
    Ptr<cuda::BroxOpticalFlow> brox;
    Ptr<cuda::DensePyrLKOpticalFlow> lk;
    Ptr<cuda::FarnebackOpticalFlow> farn;
    Ptr<cuda::OpticalFlowDual_TVL1> tvl1;
    FastOpticalFlowBM  fastBM;  
#else







#endif
 

    //!five methods
	void  calcflow_Brox(GpuMat& pre, GpuMat& current)  
	{  

        GpuMat d_frame0f;  
        GpuMat d_frame1f;  
        pre.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);  
        current.convertTo(d_frame1f, CV_32F, 1.0 / 255.0); 
#if CV_MAJOR_VERSION >= 3 
        brox->calc(d_frame0f, d_frame1f, d_flow);
#else
        BroxOpticalFlow  brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
		brox(d_frame0f, d_frame1f, d_flowx, d_flowy);
#endif

 
	}  
	void  calcflow_LK(GpuMat& pre, GpuMat& current)  
	{  
#if CV_MAJOR_VERSION >= 3 
        lk->calc(pre, current, d_flow );  
#else
        PyrLKOpticalFlow lk; lk.winSize = Size(7, 7);
		lk.dense(pre, current, d_flowx, d_flowy);
#endif
		
	}  
	void  calcflow_Farn(GpuMat& pre, GpuMat& current)  
	{  
#if CV_MAJOR_VERSION >= 3 
    	farn->calc(pre, current, d_flow );  
#else
        FarnebackOpticalFlow farn;
		farn(pre,current,d_flowx,d_flowy);
#endif

	}  
	void calcflow_TVL1(GpuMat& pre, GpuMat& current)  
	{  
#if CV_MAJOR_VERSION >= 3 
    	tvl1->calc(pre, current, d_flow ); 
#else
        OpticalFlowDual_TVL1_GPU tvl1;
		tvl1(pre,current,d_flowx,d_flowy);
#endif

	}  
	void  calcflow_fastBM(GpuMat& pre, GpuMat& current)  
	{  
   		GpuMat buf; 
        FastOpticalFlowBM fastBM; 
    	calcOpticalFlowBM(pre, current, Size(7, 7), Size(1, 1), Size(21, 21), false, d_flowx, d_flowy, buf); 
		 
	} 
 void  compute(Mat& pre, Mat& current, GpuMat& dst,const std::string & method = "Brox")   //!result :dst
 {  
    //!load image to GpuMat
    GpuMat d_frame0(pre);  
    GpuMat d_frame1(current);  
    int64 start;double timeSec;  
#ifdef TEST_SPEED
    start = getTickCount(); 
    calcflow_Brox(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "Brox : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_fastBM(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "fastBM : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_LK(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "LK : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_Farn(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "Farn : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
   calcflow_TVL1(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "TVL1 : " << timeSec << " sec" << endl; 
 #else   
 

    if(method == "Brox")
	{
		calcflow_Brox(d_frame0,d_frame1);
 
	}
    else if(method == "LK")
	{
		calcflow_LK(d_frame0,d_frame1); 
	}
    else if(method == "Farn")
	{
		calcflow_Farn(d_frame0,d_frame1); 
	}
    else if(method == "fastBM")
	{
		calcflow_fastBM(d_frame0,d_frame1); 
	}
    else if(method == "TVL1")
	{
		calcflow_TVL1(d_frame0,d_frame1); 	
	}

    dst = d_flow;
 

 #endif 

 }
 void  compute(Mat& pre, Mat& current ,const std::string & method = "Brox")   //!result :dst
 {  
    //!load image to GpuMat
    GpuMat d_frame0(pre);  
    GpuMat d_frame1(current);  
    int64 start;double timeSec;  
#ifdef TEST_SPEED
    start = getTickCount(); 
    calcflow_Brox(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "Brox : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_fastBM(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "fastBM : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_LK(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "LK : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
    calcflow_Farn(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "Farn : " << timeSec << " sec" << endl; 
    start = getTickCount(); 
   calcflow_TVL1(d_frame0,d_frame1); 
        timeSec = (getTickCount() - start) / getTickFrequency(); 
        cout << "TVL1 : " << timeSec << " sec" << endl; 
 #else   
 

    if(method == "Brox")
	{
		calcflow_Brox(d_frame0,d_frame1);
 
	}
    else if(method == "LK")
	{
		calcflow_LK(d_frame0,d_frame1); 
	}
    else if(method == "Farn")
	{
		calcflow_Farn(d_frame0,d_frame1); 
	}
    else if(method == "fastBM")
	{
		calcflow_fastBM(d_frame0,d_frame1); 
	}
    else if(method == "TVL1")
	{
		calcflow_TVL1(d_frame0,d_frame1); 	
	}
 
 #endif 

 }


    //!show 
	void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy,Mat& dst, float maxmotion = -1)  
	{  
   	   dst.create(d_flowx.size(), CV_8UC3);  
       dst.setTo(Scalar::all(0));  
   	   // determine motion range:  
       float maxrad = maxmotion;  
  
	   if (maxmotion <= 0)  
       {  
      	  maxrad = 1;  
       	  for (int y = 0; y < flowx.rows; ++y)  
          {  
           	 for (int x = 0; x < flowx.cols; ++x)  
            {  
                Point2f u(((Mat_<float>)flowx)(y, x), flowy(y, x));  
  
                if (!isFlowCorrect(u))  
                    continue;  
                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));  
            }  
        }  
      }  
  
     for (int y = 0; y < flowx.rows; ++y)  
     {  
        for (int x = 0; x < flowx.cols; ++x)  
        {  
            Point2f u(flowx(y, x), flowy(y, x));  
  
            if (isFlowCorrect(u))  
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);  
        }  
     }  
  
	} 
#if CV_MAJOR_VERSION >= 3
 //! Write a 3-channel jpg image (flow_x, flow_y, flow_magnitude) in 0-255 range  
 void writeFlowMergedJpg(string &name, const GpuMat& d_flow)
 {
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat flowmag;
    computeFlowMagnitude(flowx, flowy, flowmag);

    Mat flowx_n, flowy_n, flowmag_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowmag, flowmag_n, 0, 255, NORM_MINMAX, CV_8UC1);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    Mat flow;
    vector<Mat> array_to_merge;
    array_to_merge.push_back(flowx_n);
    array_to_merge.push_back(flowy_n);
    array_to_merge.push_back(flowmag_n);
    cv::merge(array_to_merge, flow);

    imwrite(name+".jpg", flow, compression_params);
 }

 //! Write two 1-channel jpg images (flow_x and flow_y) in 0-255 range (input flow is gpumat) 
  vector<double> writeFlowJpg(string & name, const GpuMat& d_flow)
 {
    // Split flow into x and y components in CPU
    GpuMat planes[2];
    cuda::split(d_flow, planes);
    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    // Normalize optical flows in range [0, 255]
    Mat flowx_n, flowy_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);

    // Save optical flows (x, y) as jpg images
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    imwrite(name+"_x.jpg", flowx_n, compression_params);
    imwrite(name+"_y.jpg", flowy_n, compression_params);

    // Return normalization elements
    vector<double> mm_frame;
    vector<double> temp = getMM(flowx);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());
    temp = getMM(flowy);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());

    return mm_frame;
 }
#endif

  vector<double> writeFlowJpg(string & name, const GpuMat& d_flowx,const GpuMat & d_flowy)
 {
    // Split flow into x and y components in CPU
    Mat flowx(d_flowx);
    Mat flowy(d_flowy);

    // Normalize optical flows in range [0, 255]
    Mat flowx_n, flowy_n;
    cv::normalize(flowx, flowx_n, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::normalize(flowy, flowy_n, 0, 255, NORM_MINMAX, CV_8UC1);

    // Save optical flows (x, y) as jpg images
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    imwrite(name+"_x.jpg", flowx_n, compression_params);
    imwrite(name+"_y.jpg", flowy_n, compression_params);

    // Return normalization elements
    vector<double> mm_frame;
    vector<double> temp = getMM(flowx);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());
    temp = getMM(flowy);
    mm_frame.insert(mm_frame.end(), temp.begin(), temp.end());

    return mm_frame;
 }
/* Write raw optical flow values into txt file
Example usage:
    writeFlowRaw<float>(name+"_x_raw.txt", flowx);
    writeFlowRaw<int>(name+"_x_raw_n.txt", flowx_n);
*/
 template <typename T>
 void writeFlowRaw(string & name, const Mat& flow)
 {
    ofstream file;
    file.open(name.c_str());
    for(int y=0; y<flow.rows; ++y)
    {
        for(int x=0; x<flow.cols; ++x)
        {
            file << flow.at<T>(y, x) << " ";
        }
        file << endl;
    }
    file.close();
 }

 //min_x max_x min_y max_y
 void writeMM(string name, vector<double> mm)
 {
    ofstream file;
    file.open(name.c_str());
    for(int i=0; i<mm.size(); i++)
    {
        file << mm[i] << " ";
    }
    file.close();
 }

   //min_x max_x min_y max_y (one line per frame)
  void writeMM(string & name, vector<vector<double> > &mm)
 {
    ofstream file;
    file.open(name.c_str());
    for(int i=0; i<mm.size(); i++)
    {
        for(int j=0; j<mm[i].size(); j++)
        {
            file << mm[i][j] << " ";
        }
        file << endl;
    }
    file.close();
 } 
 

private:  

	Vec3b computeColor(float fx, float fy)  
	{  
    	static bool first = true;  
  
    	// relative lengths of color transitions:  
    	// these are chosen based on perceptual similarity  
    	// (e.g. one can distinguish more shades between red and yellow  
   		//  than between yellow and green)  
   	    const int RY = 15;  
		const int YG = 6;  
		const int GC = 4;  
		const int CB = 11;  
		const int BM = 13;  
    	const int MR = 6;  
    	const int NCOLS = RY + YG + GC + CB + BM + MR;  
    	static Vec3i colorWheel[NCOLS];  
  
    	if (first)  
    	{  
        	int k = 0;  
  
        	for (int i = 0; i < RY; ++i, ++k)  
           	  colorWheel[k] = Vec3i(255, 255 * i / RY, 0);  
  
       		for (int i = 0; i < YG; ++i, ++k)  
              colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);  
  
        	for (int i = 0; i < GC; ++i, ++k)  
              colorWheel[k] = Vec3i(0, 255, 255 * i / GC);  
  
        	for (int i = 0; i < CB; ++i, ++k)  
           	  colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);  
  
       		for (int i = 0; i < BM; ++i, ++k)  
              colorWheel[k] = Vec3i(255 * i / BM, 0, 255);  
  
        	for (int i = 0; i < MR; ++i, ++k)  
              colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);  
  
       		 first = false;  
    	}  
  
   		 const float rad = sqrt(fx * fx + fy * fy);  
         const float a = atan2(-fy, -fx) / (float) CV_PI;  
  
   		 const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);  
   		 const int k0 = static_cast<int>(fk);  
   		 const int k1 = (k0 + 1) % NCOLS;  
   		 const float f = fk - k0;  
  
    	 Vec3b pix;  
  
    	 for (int b = 0; b < 3; b++)  
    	{  
        	 const float col0 = colorWheel[k0][b] / 255.0f;  
      		 const float col1 = colorWheel[k1][b] / 255.0f;  
  
       		 float col = (1 - f) * col0 + f * col1;  
  
        	if (rad <= 1)  
           	  col = 1 - rad * (1 - col); // increase saturation with radius  
      		else  
          	  col *= .75; // out of range  
  
        	pix[2 - b] = static_cast<uchar>(255.0 * col);  
    	}  
  
    	return pix;  
	}  

    inline bool isFlowCorrect(Point2f u)  
    {  
        return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;  
    }  
 
 //! Compute the magnitude of flow given x and y components  
 void computeFlowMagnitude(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst)
 {
    dst.create(flowx.size(), CV_32FC1);
    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (!isFlowCorrect(u))
                continue;

            dst.at<float>(y, x) = sqrt(u.x * u.x + u.y * u.y);
        }
    }
 }


  //!get min max
  vector<double> getMM(const Mat& flow) 
 {
    double min, max;
    cv::minMaxLoc(flow, &min, &max);
    vector<double> mm;
    mm.push_back(min);
    mm.push_back(max);
    return mm;
 }


 
};

#endif
