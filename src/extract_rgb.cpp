#include <iostream>
#include <fstream>


#include "CUDA_OpticalFlow.hpp"
#include "utils.hpp"
#include <pthread.h> //pthreads
#include <semaphore.h>
#include <boost/filesystem.hpp>
#include <queue>
#include <cstdlib>

using namespace std;
using namespace cv;
namespace bf=boost::filesystem;

struct Args {
    int width;
    int height;
    long tid;
    string inputPath;  // e.g. /home/joseph/hdd/DiDeMo/ *.mp4
    string outputPath; // e.g. /home/joseph/hdd/DiDeMo_rgb 
    string cachePath;  // e.g. /home/joseph/ssd/DiDeMo_cache ==> outputPath
    vector<string> *files;
};

const int NUM_TASKS = 4;    //! processors
pthread_mutex_t Mutex_imgs; //! pthread_mutex_lock/unlock
pthread_mutex_t Mutex_flag;
pthread_mutex_t Mutex_ssd;
sem_t sem_img; 
sem_t sem_ssd;
bool flagFinished = false;
map<string, cv::Mat> Imgs; //! shared mem.
queue<string> folders;    

#define DIRECTLY_WRITE 
#define USE_SSD_CACHE


#define THREAD_CHECK(x) do { \
    if(x) { \
        cout << "Error: return code from pthread  is " << x << endl;\
        return -1; \
    }} while(0)



void *write_images(void *args) {
#ifdef DIRECTLY_WRITE
#ifdef USE_SSD_CACHE
    long tid = ((Args *)args)->tid;
    string outputPath = ((Args *)args)->outputPath;
    string cachePath = ((Args *)args)->cachePath;
    cout << "write_images called from " << tid << endl;
    ///move files from SSD to HDD 
    string cache_dir, out_dir;
    while(!flagFinished) {
        sem_wait(&sem_ssd); //! wait 
        pthread_mutex_lock(&Mutex_ssd);
        while(!folders.empty()) {
//            cache_dir = folders.pop_front(); //! cache + "/xxxx"
            cache_dir = folders.front();
            folders.pop();
            int pos = cache_dir.rfind("/");
            if(pos == string::npos) {
                cout << "cannot match cache path:"<< cache_dir << endl;
                pthread_exit(NULL);
            }
            string sub_dir(cache_dir.begin()+pos, cache_dir.end());
            out_dir = outputPath + sub_dir;
            //! cache_dir-> out_dir
//            cout << "cache:" << cache_dir << "out_dir:" << out_dir << endl;
            string cmd = "mv " + cache_dir + "  " + out_dir;
            system(cmd.c_str());
        }
        pthread_mutex_unlock(&Mutex_ssd);
    }    
#endif
#else    
    map<string, cv::Mat>::iterator iter;
    while(!flagFinished) {
        sem_wait(&sem_img); //! wait for the semaphore
        pthread_mutex_lock(&Mutex_imgs); //! STL map ...
        for(iter = Imgs.begin(); iter != Imgs.end(); ) {
            string name = iter->first;
            cv::Mat img = iter->second;
            cv::imwrite(name, img);
            Imgs.erase(iter++);
        }
        pthread_mutex_unlock(&Mutex_imgs);
    }
    for(iter = Imgs.begin(); iter != Imgs.end(); ){
        cv::imwrite(iter->first, iter->second);
        Imgs.erase(iter++);
    }
#endif
    cout << "===>write images exit!" << endl;
    pthread_exit(NULL);
}

void *extract_frames(void * args) {
    int width = ((Args *)args)->width;
    int height = ((Args *)args)->height;
    long tid = ((Args *)args)->tid;
    string outputPath = ((Args *)args)->outputPath;
    vector<string> files = *(((Args *)args)->files);
    string inputPath = ((Args *)args)->inputPath;
    string cachePath = ((Args *)args)->cachePath;

    int count = 0;
    float progress = 0.0; 
    string directory, out_dir; 
	for(size_t i = 0; i < files.size(); ++i) {
		directory = files[i];  //! inputPath + "/" + "xxx.mp4" 
        int pos = directory.rfind("/");
        if(pos == string::npos) {
            cout << "Error occurred:" << directory << endl;
            pthread_exit(NULL);
        }
        string sub_dir(directory.begin() + pos, directory.end() - 4);
#ifdef USE_SSD_CACHE        
        out_dir = cachePath + sub_dir; //! cachePath + "/" + "xxx"
#else
        out_dir = outputPath + sub_dir;
#endif        
		const char * dir = out_dir.c_str();
		if(access(dir, F_OK) != 0) {
			makedirs(dir);
		}

        count ++;
        progress =  count*100.0f/files.size(); 
	 	std::cout<<"process video files:"<<files[i]<<"  thread:"<<tid << \
            " progress:"<<progress<<"%"<<std::endl;
		//!now extract frames
		cv::VideoCapture capture(files[i]);
		if(!capture.isOpened())
		{
			std::cout<<"fail to open video:"<<files[i]<<std::endl;
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
		    char  s_cnt[5]; 
		    sprintf(s_cnt,"%04d",cnt);
			string str(s_cnt);
            string name(out_dir);
			name += "/image_" + str + ".jpg";
#ifdef DIRECTLY_WRITE            
            cv::imwrite(name, currentFrame); //! write to SSD then copy to HDD
#else
            pthread_mutex_lock(&Mutex_imgs);            
            Imgs.insert(make_pair(name, currentFrame));
            pthread_mutex_unlock(&Mutex_imgs);
            sem_post(&sem_img);
#endif            
            cnt++;
        }
		capture.release();
        ///
#ifdef USE_SSD_CACHE
        pthread_mutex_lock(&Mutex_ssd);
        folders.push(out_dir); //! out_dir
        pthread_mutex_unlock(&Mutex_ssd);
        sem_post(&sem_ssd);
#endif        
	  }
    pthread_mutex_lock(&Mutex_flag);
    flagFinished = true;
    pthread_mutex_unlock(&Mutex_flag);
    cout << "---->extract frames exit!" << endl;
    pthread_exit(NULL);
}



int main(int argc, char* argv[]) {
	struct timeval t_start,t_end; 
	long cost_time = 0; 
	static int errorCnt = 0;
    ///parse args using opencv's cmd
    int width = 240;
    int height = 240;
    std::string inputPath = "./";
    std::string outputPath = "./";
    std::string cachePath = "./";
    std::string rm = "false";
    std::string method = "Brox";
    const char * keys = "{ w  |   256  |  width  }"
						"{ h  |   256  |  height }"
						"{ i  |   ./   |  input path}" 
						"{ o  |   ./   |  output path}"
                        "{ c  |   ./   |  cache path}";

    cv::CommandLineParser cmd(argc, argv, keys);
    if(argc > 1) {
  	 	width = cmd.get<int>("w");
   		height = cmd.get<int>("h");
   		inputPath = cmd.get<std::string>("i");
  	    outputPath = cmd.get<std::string>("o");
        cachePath = cmd.get<std::string>("c");
	}
	if(access(outputPath.c_str(), F_OK) != 0) {
		makedirs(outputPath.c_str());
	}
	if(access(cachePath.c_str(), F_OK) != 0) {
		makedirs(cachePath.c_str());
	}

	gettimeofday(&t_start, NULL); 
	long time_start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 
	std::vector<std::string> files;
    globFiles(inputPath,files);  
    std::cout<<"video files size:"<<files.size()<<std::endl;
 	std::cout<<"OPENCV_VERSION:"<<CV_VERSION<<std::endl;
    std::cout<<"start processing videos"<<std::endl;
    /// split files into NUM_TASKS parts
    Args args[NUM_TASKS*2];
    vector<string> split_files[NUM_TASKS];
    int len = files.size() / NUM_TASKS;
    for(int i = 0; i < NUM_TASKS; ++i) {
        long start = 0 + len * i;
        long end = start + len;
        if (end >= files.size()) {end = files.size();}
        if (i == NUM_TASKS-1) {
            end = files.size();
        }
        vector<string>::iterator first = files.begin() + start;
        vector<string>::iterator last = files.begin() + end;
        split_files[i] = vector<string>(first, last);
        args[i].files = &split_files[i];
    }
    pthread_t threads[NUM_TASKS*2]; //double threads
    if(sem_init(&sem_img, 0, 0) == -1) {
        cout << "semaphore initialization failed!" << endl;
        return -1;
    }
    if(sem_init(&sem_ssd, 0, 0) == -1) {
        cout << "semaphore sem_ssd initialization failed!" << endl;
        return -1;
    }
    pthread_mutex_init(&Mutex_imgs, NULL);
    pthread_mutex_init(&Mutex_flag, NULL);
    pthread_mutex_init(&Mutex_ssd,  NULL);

    for(int i = 0; i < NUM_TASKS; ++i) {
        args[i].tid = i;
        args[i].width = width;
        args[i].height = height;
        args[i].inputPath  = inputPath;
        args[i].outputPath = outputPath;
        args[i].cachePath =  cachePath;
        THREAD_CHECK(pthread_create(&threads[i], NULL, extract_frames, (void *)&args[i]));
        long tid = i+ NUM_TASKS;
        args[tid].tid = tid;
        args[tid].cachePath = cachePath;
        args[tid].outputPath = outputPath;
        THREAD_CHECK(pthread_create(&threads[i+NUM_TASKS], NULL, write_images, (void *)&args[tid]));
    }
    for(int i = 1; i < NUM_TASKS; ++i) {
        THREAD_CHECK(pthread_join(threads[i], NULL));
        THREAD_CHECK(pthread_join(threads[i+NUM_TASKS], NULL));
    }

	gettimeofday(&t_end, NULL); 
	long time_end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
	cost_time = time_end - time_start;
	if(errorCnt != 0) {
	   std::cout<<"error occurred!"<<std::endl;	
	}
 	std::cout<<"extracting done!"<<std::endl;
	std::cout<<"cost-time:"<<cost_time/1000.0<<"sec"<<std::endl;

    pthread_mutex_destroy(&Mutex_imgs);
    pthread_mutex_destroy(&Mutex_flag);
    pthread_mutex_destroy(&Mutex_ssd);
    sem_destroy(&sem_img);
    sem_destroy(&sem_ssd);


	return 0;
}


