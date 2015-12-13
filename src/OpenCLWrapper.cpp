
#include "OpenCLWrapper.hpp"

OpenCLWrapper::OpenCLWrapper(void)
{
	initialized = false;
}

OpenCLWrapper::~OpenCLWrapper(void)
{
	if (initialized)
		cleanDeviceMemory();
}

cl_int
OpenCLWrapper::init(void)
{
	cl_int				err;

	num_entries = 1;
	err = clGetPlatformIDs(num_entries, &platformID, &num_platforms);
	if (err != CL_SUCCESS)
		return (printError(std::ostringstream().flush() << "Error: Failed to retrieve platform id ! " << err, EXIT_FAILURE));
	err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &clDeviceId, 0);
	if (err != CL_SUCCESS)
		return (printError(std::ostringstream().flush() << "Error: Failed to create a device group ! " << err, EXIT_FAILURE));
#ifdef __APPLE__
	CGLContextObj			kCGLContext = CGLGetCurrentContext();
	CGLShareGroupObj		kCGLShareGroup = CGLGetShareGroup(kCGLContext);
	cl_context_properties	props[] =
	{
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties)kCGLShareGroup,
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformID,
		0
	};
#else
	cl_context_properties props[] =
	{
		CL_GL_CONTEXT_KHR,
		(cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR,
		(cl_context_properties)glXGetCurrentDisplay(),
		0
	};
#endif
	clContext = clCreateContext(props, 1, &clDeviceId, 0, 0, &err);
	if (!clContext || err != CL_SUCCESS)
		return (printError("Error: Failed to create a compute context !", EXIT_FAILURE));
	clCommands = clCreateCommandQueue(clContext, clDeviceId, 0, &err);
	if (!clCommands || err != CL_SUCCESS)
		return (printError("Error: Failed to create a command queue !", EXIT_FAILURE));
	initialized = true;
	return (CL_SUCCESS);
}

cl_int
OpenCLWrapper::initKernels(std::vector<std::string> const &kernelFiles,
							std::vector<std::string> const &kernelNames,
							std::string const &options)
{
	int							err;
	size_t						len;
	char						buffer[2048];
	std::string					file_content;
	char						*file_string;
	size_t						i;

	if (kernelFiles.size() != kernelNames.size())
		return (printError("Error: kernel names and files must be of the same size !", EXIT_FAILURE));
	programNumber = kernelFiles.size();
	for (i = 0; i < programNumber; ++i)
	{
		file_content = getFileContents(kernelFiles[i]);
		file_string = (char *)file_content.c_str();
		clPrograms[i] = clCreateProgramWithSource(clContext, 1, (char const **)&file_string, 0, &err);
		if (!clPrograms[i] || err != CL_SUCCESS)
		{
			return (printError(std::ostringstream().flush()
								<< "Error Failed to create compute "
								<< kernelNames[i]
								<< " program !",
								EXIT_FAILURE));
		}
		err = clBuildProgram(clPrograms[i], 0, 0, options.c_str(), 0, 0);
		if (err != CL_SUCCESS)
		{
			std::cerr << "Error: Failed to build program executable ! " << err << std::endl;
			clGetProgramBuildInfo(clPrograms[i], clDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			std::cerr << buffer << std::endl;
			return (EXIT_FAILURE);
		}
		clKernels[i] = clCreateKernel(clPrograms[i], kernelNames[i].c_str(), &err);
		if (!clKernels[i] || err != CL_SUCCESS)
			return (printError("Error: Failed to create compute kernel !", EXIT_FAILURE));
		err = clGetKernelWorkGroupInfo(clKernels[i], clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local[i], NULL);
		if (err != CL_SUCCESS)
			return (printError(std::ostringstream().flush() << "Error: Failed to retrieve kernel work group info! " << err, EXIT_FAILURE));
	}
	return (CL_SUCCESS);
}

cl_int
OpenCLWrapper::cleanDeviceMemory(void)
{
	cl_int			err;
	size_t			i;

	for (i = 0; i < programNumber; ++i)
	{
		err = clReleaseProgram(clPrograms[i]);
		if (err != CL_SUCCESS)
			return (printError("Error: Failed to release program !", EXIT_FAILURE));
		err = clReleaseKernel(clKernels[i]);
		if (err != CL_SUCCESS)
			return (printError("Error: Failed to release kernel !", EXIT_FAILURE));
	}
	err = clReleaseCommandQueue(clCommands);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to release command queue !", EXIT_FAILURE));
	err = clReleaseContext(clContext);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to release context !", EXIT_FAILURE));
	return (CL_SUCCESS);
}

cl_int
OpenCLWrapper::getOpenCLInfo(void)
{
	static int const	size = 2048;
	char				buffer[size];
	cl_int				err;

	err = clGetPlatformInfo(platformID, CL_PLATFORM_VERSION, size, buffer, 0);
	if (err != CL_SUCCESS)
		std::cerr << "Failed to retrieve platform version!" << std::endl;
	else
		std::cerr << "VERSION: " << buffer << std::endl;
	err = clGetPlatformInfo(platformID, CL_PLATFORM_NAME, size, buffer, 0);
	if (err != CL_SUCCESS)
		std::cerr << "Failed to retrieve platform name!" << std::endl;
	else
		std::cerr << buffer << std::endl;
	err = clGetPlatformInfo(platformID, CL_PLATFORM_VENDOR, size, buffer, 0);
	if (err != CL_SUCCESS)
		std::cerr << "Failed to retrieve platform vendor!" << std::endl;
	else
		std::cerr << buffer << std::endl;
	err = clGetPlatformInfo(platformID, CL_PLATFORM_EXTENSIONS, size, buffer, 0);
	if (err != CL_SUCCESS)
		std::cerr << "Failed to retrieve platform extensions!" << std::endl;
	else
		std::cerr << buffer << std::endl;
	return (CL_SUCCESS);
}

cl_int
Core::launchKernelsUpdate(void)
{
	cl_int			err;

	err = clEnqueueAcquireGLObjects(clCommands, 1, &dp, 0, 0, 0);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to acquire GL Objects !", EXIT_FAILURE));
	err = clSetKernelArg(clKernels[UPDATE_KERNEL], 0, sizeof(cl_mem), &dp);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to set kernel arguments !", EXIT_FAILURE));
	err = clEnqueueNDRangeKernel(clCommands, clKernels[UPDATE_KERNEL], 1, 0, &global, &local[UPDATE_KERNEL], 0, 0, 0);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to launch update kernel !", EXIT_FAILURE));
	err = clEnqueueReleaseGLObjects(clCommands, 1, &dp, 0, 0, 0);
	if (err != CL_SUCCESS)
		return (printError("Error: Failed to release GL Objects !", EXIT_FAILURE));
	clFinish(clCommands);
	return (CL_SUCCESS);
}
