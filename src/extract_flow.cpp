#include <iostream>
#include <fstream>


#include "CUDA_OpticalFlow.hpp"

#include <boost/mpi.hpp> //Boost MPI
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include "utils.hpp"

using namespace std;
namespace mpi=boost::mpi;


//#define OUTPUT_MIN_MAX
void extract_flow(vector<string> files, string outputPath, string data_name,  int width, int height,
        string method);

#define MPI_MASTER 0
#define GPU_NUMS   2

int main(int argc, char* argv[]) {
    ///MPI initialization
    mpi::environment env(argc, argv);
    mpi::communicator world;
    const int taskID = world.rank();
    const int numTasks = world.size();
    string hostName = env.processor_name();
    cout << "taskID:" << taskID << " of numTasks:" << numTasks 
        <<" running on node:" << hostName << endl;

	struct timeval t_start,t_end; 
	long cost_time = 0; 
	static int errorCnt = 0;
    
    int gpuIDs[] = {0,1,2,3};
    int gpuID;
    int width = 256;
    int height = 256;
    string inputPath = "./";
    string outputPath = "./";
    string rm = "false";
    string method = "TVL1";
    string cachePath;
    
    const char * keys = "{ g|   0    |  gpuID  }"
						"{ w  |   256  |  width  }"
						"{ h  |   256  |  height }"
						"{ i  |   ./   |  input path}" 
						"{ o  |   ./   |  output path}" 
						"{ r  |   false|  remove file}"
						"{ m  |  TVL1  |  five methods}"
                        "{ c  |   ./   |  cache path}";

    cv::CommandLineParser cmd(argc, argv, keys);
    if(argc > 1) {
  	 	gpuID = cmd.get<int>("g");
  	 	width = cmd.get<int>("w");
   		height = cmd.get<int>("h");
   		inputPath = cmd.get<std::string>("i");
  	    outputPath = cmd.get<std::string>("o");
		method = cmd.get<std::string>("m");
        cachePath = cmd.get<std::string>("c");
	}
    cout << "width:" << width << "height:" << height << "inputPath:"
        << inputPath << "outputPath:" << outputPath << " method:"
        << method << endl;

	/// get start time 
	gettimeofday(&t_start, NULL); 
	long start = ((long)t_start.tv_sec)*1000+(long)t_start.tv_usec/1000; 

	vector<string>files;
 
//    setDevice(gpuID); //gpuID
    vector<string> split_files;
    if(taskID == MPI_MASTER) {
        globFiles(inputPath,files);   
        if (files.size() == 0) {
            cout << "No video files!!!" << endl;
            return -1;
        } 
        cout<<"video files size:"<<files.size()<<endl;
     	cout<<"OPENCV_VERSION:"<<CV_VERSION<<endl;
        int len = files.size() / numTasks;
        vector<string>::iterator first, last;
        for(int i = 0; i < numTasks; ++i) {
            int start = 0 + len * i;
            int end = start + len;
            if(end >= files.size()) {end = files.size();}
            if(i == numTasks - 1) {
                end = files.size();
            }
            first = files.begin() + start;
            last  = files.begin() + end;

            vector<string> sendBuf;
            sendBuf.reserve(len);
            sendBuf = vector<string>(first, last);

            cout << "sendBuf:" << sendBuf.size() << endl;
            if(i == 0) {
                move(sendBuf.begin(), sendBuf.end(), back_inserter(split_files));
            } else {
                world.send(i, 0, sendBuf); //! send to other nodes
            }  
         }
    }
    if(taskID != MPI_MASTER) {
        world.recv(0, 0, split_files);
    }
    int gpu_id = gpuIDs[taskID];
    if(gpu_id >= GPU_NUMS) {
        gpu_id %= GPU_NUMS;
    }
    setDevice(gpu_id);
    cout << "ID:" << taskID <<" gpu_id:" << gpu_id << " len(split_files):" << split_files.size()
         << "split_files[0]:" << split_files[0] << endl;


    extract_flow(split_files, outputPath, cachePath, width, height, method);

	//!end time
	gettimeofday(&t_end, NULL); 
	long end = ((long)t_end.tv_sec)*1000+(long)t_end.tv_usec/1000; 
	cost_time = end - start;
	if(errorCnt != 0)
	{
	   std::cout<<"error occurred!"<<std::endl;	
	}
	
 	std::cout<<"extracting done!"<<std::endl;
	std::cout<<"cost-time:"<<cost_time/1000.0<<"sec"<<std::endl;

		
	return 0;
}


void extract_flow(vector<string> files, string outputPath, string cachePath, int width, int height,
        string method) {
    static int count = 0;
    float progress = 0.0; 
	Mat currentFrame,preFrame,frame;
    string cache_dir, out_dir;
	for(int i = 0; i < files.size(); ++i) {
        CUDA_OpticalFlow m_flow(Size(width,height));
		string directory(files[i]);  //! path + "/xxx.ext"
        int pos = directory.rfind("/");
        if(pos == string::npos) {
            cout << "Cannot find the last /:" << directory << endl;
            return;
        }
        string sub_str(directory.begin()+pos, directory.end()-4);
        cache_dir = cachePath + sub_str;
        if(access(cache_dir.c_str(), F_OK) != 0) {
            makedirs(cache_dir.c_str());
        }
		const char * dir = cache_dir.c_str();
        

        count ++;
        progress =  count*100.0f/files.size(); 
	 	cout<<"process video files:"<<files[i]<<" progress:"<<progress<<"%"<<endl;
		cv::VideoCapture capture(files[i]);
		if(!capture.isOpened()) {
			std::cout<<"fail to open video:"<<files[i]<<std::endl;
//			errorCnt++;
			continue;
		} 
		int cnt = 0;
 		long frames_cnt = capture.get(CV_CAP_PROP_FRAME_COUNT);
	    capture.read(frame);  //read the first frame
		cv::resize(frame,currentFrame,cv::Size(width,height));
        cv::cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);
#ifdef OUTPUT_MIN_MAX
        std::vector<std::vector<double> > mm;
#endif 
		for(int k = 1; k < frames_cnt; ++k) {
			if(!capture.read(frame)) {
				break;
			}
            currentFrame.copyTo(preFrame);   //pre = current
			cv::resize(frame,currentFrame,cv::Size(width,height));
            cv::cvtColor(currentFrame, currentFrame, CV_BGR2GRAY);
			//!name
            std::string str;
		    char  s_cnt[5]; 
		    sprintf(s_cnt,"%04d",cnt);
			str = s_cnt;
            std::string name(dir);  
			name += "/image_" + str;      //！extension is included in writeFlowJpg
 			//!compute
            m_flow.compute(preFrame,currentFrame,method);  //!default method = "TVL1"
#if CV_MAJOR_VERSION >= 3
			 std::vector<double> mm_frame = m_flow.writeFlowJpg(name, m_flow.d_flow); //!write to JPG
#else
			std::vector<double> mm_frame = m_flow.writeFlowJpg(name, m_flow.d_flowx,m_flow.d_flowy); //!write to JPG
#endif
#ifdef OUTPUT_MIN_MAX            
            mm.push_back(mm_frame);   
#endif             
            cnt ++;
        }
#ifdef OUPUT_MIN_MAX        
        string output_mm(dir);
		output_mm += "/op_minmax.txt";    
		m_flow.writeMM(output_mm, mm);
        cout<<"output_MM:"<<output_mm<<endl;
#endif         
		capture.release();
	  }
}

