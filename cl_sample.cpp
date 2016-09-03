#include <iostream>
#include <CL/cl.hpp>
#include <vector>
#include<string>
#include <opencv2/opencv.hpp>
//#include<opencv2/ocl/ocl.hpp>

using namespace std;
int main(){
	////get all platforms (drivers)
	//cv::ocl::PlatformsInfo platformsInfo;
	//cv::ocl::getOpenCLPlatforms(platformsInfo);
	//cout <<"total platforms "<< platformsInfo.size() << endl;
	//for (int i = 0; i < platformsInfo.size(); i++)
	//{
	//	cout << platformsInfo[i]->platformName << endl;
	//}
	//cv::Mat im = cv::imread("C:/Users/zhaolei/Desktop/hehe.jpg");
	//cv::namedWindow("hehe");
	//cv::imshow("hehe", im);
	//cv::waitKey(1000);
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0){
	std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0){
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";


	cl::Context context({ default_device });

	cl::Program::Sources sources;

	// kernel calculates for each element C=A+B
	std::string kernel_code =
		"   void kernel simple_add(__global const float* A,__global float* C){       "
		
		"C[get_global_id(0)]=sqrt(A[get_global_id(0)]); "
		"   }                                                                               ";
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS){
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}


	// create buffers on the device
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * 10);
	//cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float) * 10);

	float A[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
	//int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * 10, A);
	//queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);


	//run the kernel
	cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
	simple_add(buffer_A,  buffer_C);

	//alternative way to run the kernel
	/*cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
	kernel_add.setArg(0,buffer_A);
	kernel_add.setArg(1,buffer_B);
	kernel_add.setArg(2,buffer_C);
	queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
	queue.finish();*/

	float C[10];
	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * 10, C);

	std::cout << " result: \n";
	for (int i = 0; i<10; i++){
		std::cout << C[i] << " ";
	}

	return 0;
}