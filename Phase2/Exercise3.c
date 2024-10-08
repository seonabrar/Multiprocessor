#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng/lodepng.h"
#include <OpenCL/cl.h>

// Function to read image
unsigned char* ReadImage(const char* filename, unsigned* width, unsigned* height) {
    unsigned error;
    unsigned char* image;
    error = lodepng_decode32_file(&image, width, height, filename);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
        return NULL;
    }
    return image;
}

// Function to write image
void WriteImage(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    unsigned error;
    error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error) {
        fprintf(stderr, "Error %u: %s\n", error, lodepng_error_text(error));
    }
}

// Function to measure execution time
double MeasureTime(clock_t start, clock_t end) {
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main() {
    const char* inputFilename = "im0.png";
    const char* outputFilename = "image_0_bw_opencl.png";

    unsigned char* image;
    unsigned width, height;

    // Read image
    image = ReadImage(inputFilename, &width, &height);
    if (image == NULL) {
        return 1;
    }

    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform\n");
        return 1;
    }

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting device\n");
        return 1;
    }

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating context\n");
        return 1;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue\n");
        return 1;
    }

    // Load kernel source
    FILE* file = fopen("resize_kernel.cl", "r");
    if (!file) {
        fprintf(stderr, "Error loading kernel source\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t source_size = ftell(file);
    rewind(file);
    char* source = (char*)malloc(source_size + 1);
    if (!source) {
        fprintf(stderr, "Error allocating memory for kernel source\n");
        return 1;
    }
    fread(source, 1, source_size, file);
    fclose(file);
    source[source_size] = '\0';

    // Load grayscale kernel source
    FILE* grayscaleFile = fopen("grayscale_kernel.cl", "r");
    if (!grayscaleFile) {
        fprintf(stderr, "Error loading grayscale kernel source\n");
        return 1;
    }
    fseek(grayscaleFile, 0, SEEK_END);
    size_t grayscaleSourceSize = ftell(grayscaleFile);
    rewind(grayscaleFile);
    char* grayscaleSource = (char*)malloc(grayscaleSourceSize + 1);
    if (!grayscaleSource) {
        fprintf(stderr, "Error allocating memory for grayscale kernel source\n");
        return 1;
    }
    fread(grayscaleSource, 1, grayscaleSourceSize, grayscaleFile);
    fclose(grayscaleFile);
    grayscaleSource[grayscaleSourceSize] = '\0';

    // Load filter kernel source
    FILE* filterFile = fopen("filter_kernel.cl", "r");
    if (!filterFile) {
        fprintf(stderr, "Error loading filter kernel source\n");
        return 1;
    }
    fseek(filterFile, 0, SEEK_END);
    size_t filterSourceSize = ftell(filterFile);
    rewind(filterFile);
    char* filterSource = (char*)malloc(filterSourceSize + 1);
    if (!filterSource) {
        fprintf(stderr, "Error allocating memory for filter kernel source\n");
        return 1;
    }
    fread(filterSource, 1, filterSourceSize, filterFile);
    fclose(filterFile);
    filterSource[filterSourceSize] = '\0';


    // Create program
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program\n");
        return 1;
    }

    // Create grayscale program
    cl_program grayscaleProgram = clCreateProgramWithSource(context, 1, (const char**)&grayscaleSource, &grayscaleSourceSize, &err);
    if (!grayscaleProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating grayscale program\n");
        return 1;
    }

    // Create filter program
    cl_program filterProgram = clCreateProgramWithSource(context, 1, (const char**)&filterSource, &filterSourceSize, &err);
    if (!filterProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating filter program\n");
        return 1;
    }


    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");
        return 1;
    }

    // Build grayscale program
    err = clBuildProgram(grayscaleProgram, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building grayscale program\n");
        return 1;
    }

    // Build filter program
    err = clBuildProgram(filterProgram, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building filter program\n");
        return 1;
    }


    // Create kernel
    kernel = clCreateKernel(program, "resizeImage", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernel\n");
        return 1;
    }

    // Create grayscale kernel
    cl_kernel grayscaleKernel = clCreateKernel(grayscaleProgram, "grayscaleImage", &err);
    if (!grayscaleKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating grayscale kernel\n");
        return 1;
    }

    // Create filter kernel
    cl_kernel filterKernel = clCreateKernel(filterProgram, "applyFilter", &err);
    if (!filterKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating filter kernel\n");
        return 1;
    }

    // Create input and output buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        width * height * sizeof(cl_uchar4), image, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         (width / 4) * (height / 4) * sizeof(cl_uchar4), NULL, &err);
    if (!inputBuffer || !outputBuffer || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffers\n");
        return 1;
    }



    // Define variables to hold the values
    cl_int newWidth = width / 4;
    cl_int newHeight = height / 4;

    // Create output buffer for grayscale image
    cl_mem grayscaleOutputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                newWidth * newHeight * sizeof(cl_uchar), NULL, &err);
    if (!grayscaleOutputBuffer || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating grayscale output buffer\n");
        return 1;
    }

    // Create output buffer for filtered image
    cl_mem filteredOutputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                newWidth * newHeight * sizeof(cl_uchar), NULL, &err);
    if (!filteredOutputBuffer || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating filtered output buffer\n");
        return 1;
    }
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &newWidth);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &newHeight);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel arguments\n");
        return 1;
    }

    // Set grayscale kernel arguments
    err = clSetKernelArg(grayscaleKernel, 0, sizeof(cl_mem), &outputBuffer); // Use resized image as input
    err |= clSetKernelArg(grayscaleKernel, 1, sizeof(cl_mem), &grayscaleOutputBuffer);
    err |= clSetKernelArg(grayscaleKernel, 2, sizeof(cl_int), &newWidth);
    err |= clSetKernelArg(grayscaleKernel, 3, sizeof(cl_int), &newHeight);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting grayscale kernel arguments\n");
        return 1;
    }

    // Set filter kernel arguments
    err = clSetKernelArg(filterKernel, 0, sizeof(cl_mem), &grayscaleOutputBuffer); // Use grayscale image as input
    err |= clSetKernelArg(filterKernel, 1, sizeof(cl_mem), &filteredOutputBuffer);
    err |= clSetKernelArg(filterKernel, 2, sizeof(cl_int), &newWidth);
    err |= clSetKernelArg(filterKernel, 3, sizeof(cl_int), &newHeight);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting filter kernel arguments\n");
        return 1;
    }

    // Define event variables
    cl_event kernel_event, grayscale_event, filter_event;

    // Enqueue kernel
    size_t globalWorkSize[2] = { width / 4, height / 4 };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing kernel\n");
        return 1;
    }

    // Enqueue grayscale kernel
    size_t grayscaleGlobalWorkSize[2] = { newWidth, newHeight };
    err = clEnqueueNDRangeKernel(queue, grayscaleKernel, 2, NULL, grayscaleGlobalWorkSize, NULL, 0, NULL, &grayscale_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing grayscale kernel\n");
        return 1;
    }

    // Enqueue filter kernel
    size_t filterGlobalWorkSize[2] = { newWidth, newHeight };
    err = clEnqueueNDRangeKernel(queue, filterKernel, 2, NULL, filterGlobalWorkSize, NULL, 0, NULL, &filter_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing filter kernel\n");
        return 1;
    }

        // Wait for all events to complete
    clWaitForEvents(1, &kernel_event);
    clWaitForEvents(1, &grayscale_event);
    clWaitForEvents(1, &filter_event);

    // Read back the result
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, (width / 4) * (height / 4) * sizeof(cl_uchar4), image, 0, NULL, NULL);


    // Read back the grayscale result
    clEnqueueReadBuffer(queue, grayscaleOutputBuffer, CL_TRUE, 0, newWidth * newHeight * sizeof(cl_uchar), image, 0, NULL, NULL);

    // Read back the filtered result
    clEnqueueReadBuffer(queue, filteredOutputBuffer, CL_TRUE, 0, newWidth * newHeight * sizeof(cl_uchar), image, 0, NULL, NULL);

    // Save the resulting filtered image
    WriteImage(outputFilename, image, newWidth, newHeight);
    // Get profiling information
    cl_ulong kernel_start_time, kernel_end_time, grayscale_start_time, grayscale_end_time, filter_start_time, filter_end_time;

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start_time, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end_time, NULL);

    clGetEventProfilingInfo(grayscale_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &grayscale_start_time, NULL);
    clGetEventProfilingInfo(grayscale_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &grayscale_end_time, NULL);

    clGetEventProfilingInfo(filter_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &filter_start_time, NULL);
    clGetEventProfilingInfo(filter_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &filter_end_time, NULL);

    // Calculate execution time
    double kernel_execution_time = (kernel_end_time - kernel_start_time) * 1e-9; // Convert nanoseconds to seconds
    double grayscale_execution_time = (grayscale_end_time - grayscale_start_time) * 1e-9;
    double filter_execution_time = (filter_end_time - filter_start_time) * 1e-9;

    printf("Kernel Execution Time: %f seconds\n", kernel_execution_time);
    printf("Grayscale Kernel Execution Time: %f seconds\n", grayscale_execution_time);
    printf("Filter Kernel Execution Time: %f seconds\n", filter_execution_time);


    // Release grayscale resources
    clReleaseMemObject(grayscaleOutputBuffer);
    clReleaseKernel(grayscaleKernel);
    clReleaseProgram(grayscaleProgram);

    // Release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Release filter resources
    clReleaseMemObject(filteredOutputBuffer);
    clReleaseKernel(filterKernel);
    clReleaseProgram(filterProgram);

    // Clean up grayscale memory
    free(grayscaleSource);

    // Free memory
    free(image);
    free(source);

    // Clean up filter memory
    free(filterSource);

    return 0;
}
