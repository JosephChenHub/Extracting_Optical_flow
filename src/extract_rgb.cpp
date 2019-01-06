#include <iostream>
#include <fstream>


#include "CUDA_OpticalFlow.hpp"
#include "utils.hpp"
#include <pthread.h> //pthreads

using namespace std;
using namespace cv;

struct Args {
    int width;
    int height;
    long tid;
    string outputPath;
    vector<string> *files;
};

const int NUM_TASKS = 8; // 8 processors
#define THREAD_CHECK(x) do { \
    if(x) { \
        cout << "Error: return code from pthread  is " << x << endl;\
        return -1; \
    }} while(0)


void *extract_frames(void * args) {
    int width = ((Args *)args)->width;
    int height = ((Args *)args)->height;
    long tid = ((Args *)args)->tid;
    string outputPath = ((Args *)args)->outputPath;
    vector<string> files = *(((Args *)args)->files);
    int count = 0;
    float progress = 0.0; 
	for(int i = 0; i < files.size(); ++i) {
		std::string directory(files[i]); 
		directory.erase(directory.end()-4,directory.end());   //!remove '.avi' or '.mpg'
        count ++;
        progress =  count*100.0f/files.size(); 
	 	std::cout<<"process video files:"<<files[i]<<"  thread:"<<tid << \
            " progress:"<<progress<<"%"<<std::endl;
		if(outputPath != "./" && !outputPath.empty()) {
			directory.insert(1,"/"+outputPath);
			if(access(directory.c_str(),F_OK) != 0) {
				makedirs(directory.c_str());
			}
		}
		const char * dir = directory.c_str();
		if(access(dir,F_OK) != 0) {
			mkdir(dir,0777);
			std::cout<<"mkdir:"<<dir<<std::endl; 
		} 
		//!now extract frames
		cv::VideoCapture capture(files[i]);
		if(!capture.isOpened())
		{
			std::cout<<"fail to open video:"<<files[i]<<std::endl;
		//	errorCnt++;
			continue;
		} 
	    Mat currentFrame, frame;
		int cnt = 0;
 		long frames = capture.get(CV_CAP_PROP_FRAME_COUNT);
		for(int k = 0; k < frames; ++k) {
			if(!capture.read(frame)) {
				break;
			}
			cv::resize(frame,currentFrame,cv::Size(width,height));
			//!name
            std::string str;
		    char  s_cnt[5]; 
		    sprintf(s_cnt,"%04d",cnt);
			str = s_cnt;
            std::string name(dir);  
			name += "/image_" + str + ".jpg";
            cv::imwrite(name, currentFrame);
            cnt++;
        }
		capture.release();
	  }
    pthread_exit(NULL);
}



int main(int argc, const char* argv[]) {
	struct timeval t_start,t_end; 
	long cost_time = 0; 
	static int errorCnt = 0;
    ///parse args using opencv's cmd
    int width = 240;
    int height = 240;
    std::string inputPath = "./";
    std::string outputPath = "./";
    std::string rm = "false";
    std::string method = "Brox";

    const char * keys = "{ w  |   256  |  width  }"
						"{ h  |   256  |  height }"
						"{ i  |   ./   |  input path}" 
						"{ o  |   ./   |  output path}";

    cv::CommandLineParser cmd(argc, argv, keys);
    if(argc > 1) {
  	 	width = cmd.get<int>("w");
   		height = cmd.get<int>("h");
   		inputPath = cmd.get<std::string>("i");
  	    outputPath = cmd.get<std::string>("o");
	}
	gettimeofday(&t_start, NULL); 
	long start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 
	std::vector<std::string> files;
    globFiles(inputPath,files);  
    std::cout<<"video files size:"<<files.size()<<std::endl;
 	std::cout<<"OPENCV_VERSION:"<<CV_VERSION<<std::endl;
    std::cout<<"start processing videos"<<std::endl;
    /// split files into NUM_TASKS parts
    Args args[NUM_TASKS];
    vector<string> split_files[NUM_TASKS];
    int len = files.size() / NUM_TASKS;
    for(int i = 0; i < NUM_TASKS; ++i) {
        int start = 0 + len * i;
        int end = start + len;
        if (end >= files.size()) {end = files.size();}
        if (i == NUM_TASKS-1) {
            end = files.size();
        }
        vector<string>::iterator first = files.begin() + start;
        vector<string>::iterator last = files.begin() + end;
        split_files[i] = vector<string>(first, last);
        args[i].files = &split_files[i];
    }
    pthread_t threads[NUM_TASKS];
    for(int i = 0; i < NUM_TASKS; ++i) {
        args[i].tid = i;
        args[i].width = width;
        args[i].height = height;
        args[i].outputPath = outputPath;
        THREAD_CHECK(pthread_create(&threads[i], NULL, extract_frames, (void *)&args[i]));
    }
    for(int i = 0; i < NUM_TASKS; ++i) {
        THREAD_CHECK(pthread_join(threads[i], NULL));
    }

	gettimeofday(&t_end, NULL); 
	long end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
	cost_time = end - start;
	if(errorCnt != 0) {
	   std::cout<<"error occurred!"<<std::endl;	
	}
 	std::cout<<"extracting done!"<<std::endl;
	std::cout<<"cost-time:"<<cost_time/1000.0<<"sec"<<std::endl;


		
	return 0;
}


