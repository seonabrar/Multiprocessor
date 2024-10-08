#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "lodepng/lodepng.h"
#include <sys/time.h> // For gettimeofday on Linux
#include <OpenCL/cl.h>

int MAXDISP = 65; // Maximum disparity (downscaled)
int MINDISP = 0;
const uint32_t BSX = 9; // Window size on X-axis (width)
const uint32_t BSY = 9; // Window size on Y-axis (height)
const int THRESHOLD = 2;// Threshold for cross-checkings
const int NEIBSIZE = 256; // Size of the neighborhood for occlusion-filling
const uint32_t BSIZE = 315;

cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };
cl_image_desc desc;

// Function to read image
uint8_t *ReadImage(const char *filename, uint32_t *width, uint32_t *height)
{
    uint32_t error;
    uint8_t *image;
    error = lodepng_decode32_file(&image, width, height, filename);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        return NULL;
    }
    return image;
}

void normalize_dmap(uint8_t *arr, uint32_t w, uint32_t h)
{
    uint8_t max = 0;
    uint8_t min = UCHAR_MAX;
    int32_t imsize = w * h;
    uint32_t i;
    for (i = 0; i < imsize; i++)
    {
        if (arr[i] > max)
            max = arr[i];
        if (arr[i] < min)
            min = arr[i];
    }

    for (i = 0; i < imsize; i++)
    {
        arr[i] = (uint8_t)(255 * (arr[i] - min) / (max - min));
    }
}


// Function to check OpenCL errors
void checkError(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (error code: %d)\n", message, err);
        exit(EXIT_FAILURE);
    }
}
// Function to print device information
void printDeviceInfo(cl_device_id device) {
    cl_device_local_mem_type localMemType;
    cl_ulong localMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxClockFrequency;
    cl_ulong maxConstantBufferSize;
    size_t maxWorkGroupSize;
    size_t maxWorkItemSizes[3];
    cl_int err;

    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_LOCAL_MEM_TYPE)");

    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_LOCAL_MEM_SIZE)");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_COMPUTE_UNITS)");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_CLOCK_FREQUENCY)");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_WORK_GROUP_SIZE)");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);
    checkError(err, "clGetDeviceInfo (CL_DEVICE_MAX_WORK_ITEM_SIZES)");

    // Print device information
    printf("CL_DEVICE_LOCAL_MEM_TYPE: %d\n", localMemType);
    printf("CL_DEVICE_LOCAL_MEM_SIZE: %llu bytes\n", localMemSize);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", maxComputeUnits);
    printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", maxClockFrequency);
    printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %llu bytes\n", maxConstantBufferSize);
    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %zu\n", maxWorkGroupSize);
    printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: [%zu, %zu, %zu]\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);
}


int32_t main(int32_t argc, char **argv)
{
    const char* inputFilename1 = "im0.png"; // Left image filename
    const char* inputFilename2 = "im1.png"; // Right image filename
    const char* outputFilename = "depthmap.png"; // Output filename for the disparity map

    uint8_t *OriginalImageL; // Left image
    uint8_t *OriginalImageR; // Right image
    uint8_t *Disparity;
    uint32_t Error; // Error code

    uint32_t Width, Height;
    uint32_t w1, h1;
    uint32_t w2, h2;
    uint32_t imsize;

    struct timespec start, finish;
    double elapsed;

    struct timeval start_time, end_time; // Variables to hold start and end timestamps
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_platforms, num_devices;
    cl_int err;
    cl_context context;
    cl_command_queue queue;
    int tmp;
    /// Reading the images into memory
    OriginalImageL = ReadImage(inputFilename1, &w1, &h1);
    OriginalImageR = ReadImage(inputFilename2, &w2, &h2);

    if (!OriginalImageL || !OriginalImageR)
    {
        return -1;
    }

    // Checking whether the sizes of images correspond to each other
    if ((w1 != w2) || (h1 != h2))
    {
        printf("The sizes of the images do not match!\n");
        return -1;
    }



    Width = w1 / 4;
    Height = h1 / 4;
    imsize = Width*Height;

    // Resizing
    gettimeofday(&start_time, NULL); // Record start time

           // Get platform
    err = clGetPlatformIDs(1, &platform_id, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform\n");
        return 1;
    }

    // Get device
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting device\n");
        return 1;
    }

    // Create context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating context\n");
        return 1;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue\n");
        return 1;
    }
    // Print device info
    printf("GPU Device Info:\n");
    printDeviceInfo(device_id);

    // Load resize_grayscale kernel source
    FILE* resizeGreyscaleFile = fopen("resize_greyscale.cl", "r");
    if (!resizeGreyscaleFile) {
        fprintf(stderr, "Error loading resize_grayscale kernel source\n");
        return 1;
    }
    fseek(resizeGreyscaleFile, 0, SEEK_END);
    size_t resizeGreyscaleSourceSize = ftell(resizeGreyscaleFile);
    rewind(resizeGreyscaleFile);
    char* resizeGreyscaleSource = (char*)malloc(resizeGreyscaleSourceSize + 1);
    if (!resizeGreyscaleSource) {
        fprintf(stderr, "Error allocating memory for resize_grayscale kernel source\n");
        return 1;
    }
    fread(resizeGreyscaleSource, 1, resizeGreyscaleSourceSize, resizeGreyscaleFile);
    fclose(resizeGreyscaleFile);
    resizeGreyscaleSource[resizeGreyscaleSourceSize] = '\0';

    // Load zncc kernel source
    FILE* znccFile = fopen("zncc.cl", "r");
    if (!znccFile) {
        fprintf(stderr, "Error loading zncc kernel source\n");
        return 1;
    }
    fseek(znccFile, 0, SEEK_END);
    size_t znccSourceSize = ftell(znccFile);
    rewind(znccFile);
    char* znccSource = (char*)malloc(znccSourceSize + 1);
    if (!znccSource) {
        fprintf(stderr, "Error allocating memory for zncc kernel source\n");
        return 1;
    }
    fread(znccSource, 1, znccSourceSize, znccFile);
    fclose(znccFile);
    znccSource[znccSourceSize] = '\0';

    // Load cross_check kernel source
    FILE* crossCheckFile = fopen("cross_check.cl", "r");
    if (!crossCheckFile) {
        fprintf(stderr, "Error loading cross_check kernel source\n");
        return 1;
    }
    fseek(crossCheckFile, 0, SEEK_END);
    size_t crossCheckSourceSize = ftell(crossCheckFile);
    rewind(crossCheckFile);
    char* crossCheckSource = (char*)malloc(crossCheckSourceSize + 1);
    if (!crossCheckSource) {
        fprintf(stderr, "Error allocating memory for cross_check kernel source\n");
        return 1;
    }
    fread(crossCheckSource, 1, crossCheckSourceSize, crossCheckFile);
    fclose(crossCheckFile);
    crossCheckSource[crossCheckSourceSize] = '\0';

    // Load occlusion kernel source
    FILE* occlusionFile = fopen("occlusion.cl", "r");
    if (!occlusionFile) {
        fprintf(stderr, "Error loading occlusion kernel source\n");
        return 1;
    }
    fseek(occlusionFile, 0, SEEK_END);
    size_t occlusionSourceSize = ftell(occlusionFile);
    rewind(occlusionFile);
    char* occlusionSource = (char*)malloc(occlusionSourceSize + 1);
    if (!occlusionSource) {
        fprintf(stderr, "Error allocating memory for occlusion kernel source\n");
        return 1;
    }
    fread(occlusionSource, 1, occlusionSourceSize, occlusionFile);
    fclose(occlusionFile);
    occlusionSource[occlusionSourceSize] = '\0';

    // Work group size
    const size_t wgSize[] = {3, 21};

    // Global size
    const size_t globalSize[] = {504, 735};

    const size_t wgSize1D[] = {wgSize[0]*wgSize[1]};
    // Global size
    const size_t globalSize1D[] = {globalSize[0]*globalSize[1]};


    cl_mem ImageL = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!ImageL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem ImageR = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!ImageL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem dDisparityLR = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dDisparityLR || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem dDisparityRL = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dDisparityRL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem dDisparityLRCC = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dDisparityLRCC || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem dDisparity = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dDisparity || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    // Create resize_greyscale program
    cl_program resizeGreyscaleProgram = clCreateProgramWithSource(context, 1, (const char**)&resizeGreyscaleSource, &resizeGreyscaleSourceSize, &err);
    if (!resizeGreyscaleProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating resize_greyscale program\n");
        return 1;
    }

    // Create zncc program
    cl_program znccProgram = clCreateProgramWithSource(context, 1, (const char**)&znccSource, &znccSourceSize, &err);
    if (!znccProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating zncc program\n");
        return 1;
    }

    // Create cross_check program
    cl_program crossCheckProgram = clCreateProgramWithSource(context, 1, (const char**)&crossCheckSource, &crossCheckSourceSize, &err);
    if (!crossCheckProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating cross_check program\n");
        return 1;
    }

    // Create occlusion program
    cl_program occlusionProgram = clCreateProgramWithSource(context, 1, (const char**)&occlusionSource, &occlusionSourceSize, &err);
    if (!occlusionProgram || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating occlusion program\n");
        return 1;
    }

    // Build resize_greyscale program
    err = clBuildProgram(resizeGreyscaleProgram, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building reizegrayscaleprogram\n");
        return 1;
    }

    // Build zncc program
    err = clBuildProgram(znccProgram, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building zncc program\n");
        return 1;
    }

    // Build cross_check program
    err = clBuildProgram(crossCheckProgram, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building crossCheck program\n");
        return 1;
    }

    // Build occlusion program
    err = clBuildProgram(occlusionProgram, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building occlusion program\n");
        return 1;
    }

    // Create resize_greyscale kernel
    cl_kernel resizeGreyscaleKernel = clCreateKernel(resizeGreyscaleProgram, "resize_greyscale", &err);
    if (!resizeGreyscaleKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating resize_greyscale kernel\n");
        return 1;
    }

    // Create zncc kernel
    cl_kernel znccKernel = clCreateKernel(znccProgram, "zncc", &err);
    if (!znccKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating zncc kernel\n");
        return 1;
    }

    // Create crossCheck kernel
    cl_kernel crossCheckKernel = clCreateKernel(crossCheckProgram, "cross_check", &err);
    if (!crossCheckKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating crossCheck kernel\n");
        return 1;
    }

    // Create occlusion kernel
    cl_kernel occlusionKernel = clCreateKernel(occlusionProgram, "occlusion", &err);
    if (!occlusionKernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating occlusion kernel\n");
        return 1;
    }


    Disparity = (uint8_t*) malloc(Width*Height); 
     
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = w1;
    desc.image_height = h1;
    desc.image_depth = 8;
    desc.image_row_pitch = w1 * 4;

    cl_mem dOriginalImageL = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
    &format, &desc, OriginalImageL, &err);
    if (!dOriginalImageL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating image\n");
        return 1;
    }
    
    cl_mem dOriginalImageR = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
    &format, &desc, OriginalImageR, &err);
    if (!dOriginalImageR || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating image\n");
        return 1;
    }
    size_t work_units[2] = {w1/4, h1/4};

        // Define event variables
    cl_event resize_greyscale_event, zncc_event1,zncc_event2, cross_check_event, occlusion_event;

    err = clSetKernelArg(resizeGreyscaleKernel, 0, sizeof(dOriginalImageL), &dOriginalImageL);
    err |= clSetKernelArg(resizeGreyscaleKernel, 1, sizeof(dOriginalImageR), &dOriginalImageR);
    err |= clSetKernelArg(resizeGreyscaleKernel, 2, sizeof(ImageL), &ImageL);
    err |= clSetKernelArg(resizeGreyscaleKernel, 3, sizeof(ImageR), &ImageR);
    err |= clSetKernelArg(resizeGreyscaleKernel, 4, sizeof(Width), &Width);
    err |= clSetKernelArg(resizeGreyscaleKernel, 5, sizeof(Height), &Height);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting resizeGreyscale kernel arguments\n");
        return 1;
    }

    // Enqueue kernel
  // Enqueue resize_greyscale kernel
    err = clEnqueueNDRangeKernel(queue, resizeGreyscaleKernel, 2, NULL, work_units, NULL, 0, NULL, &resize_greyscale_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to execute resize Greyscale kernel\n");
        return 1;
    } 
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, ImageL, CL_TRUE,  0, Width*Height, Disparity, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        fprintf(stderr, "'resize_kernel': Failed to send the data to host !\n");
        abort();
    }


    // Set zncc kernel arguments
    err = clSetKernelArg(znccKernel, 0, sizeof(ImageL), &ImageL); 
    err |= clSetKernelArg(znccKernel, 1, sizeof(ImageR), &ImageR);
    err |= clSetKernelArg(znccKernel, 2, sizeof(dDisparityLR), &dDisparityLR);
    err |= clSetKernelArg(znccKernel, 3, sizeof(Width), &Width);
    err |= clSetKernelArg(znccKernel, 4, sizeof(Height), &Height);
    err |= clSetKernelArg(znccKernel, 5, sizeof(int), &BSX);
    err |= clSetKernelArg(znccKernel, 6, sizeof(int), &BSY);
    err |= clSetKernelArg(znccKernel, 7, sizeof(MINDISP), &MINDISP);
    err |= clSetKernelArg(znccKernel, 8, sizeof(int), &MAXDISP);    
    err |= clSetKernelArg(znccKernel, 9, sizeof(BSIZE), &BSIZE);
    err |= clSetKernelArg(znccKernel, 10, sizeof(imsize), &imsize);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting zncc kernel arguments\n");
        return 1;
    }

    // Enqueue zncc kernel
    err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, (const size_t*)&globalSize,  (const size_t*)&wgSize, 0, NULL, &zncc_event1);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing zncc kernel\n");
        return 1;
    }




    // Disparity RL zncc kernel call
    tmp = MINDISP;
    MINDISP = -MAXDISP;
    MAXDISP = tmp;
    // Set zncc kernel arguments
    err = clSetKernelArg(znccKernel, 0, sizeof(ImageL), &ImageL); 
    err |= clSetKernelArg(znccKernel, 1, sizeof(ImageR), &ImageR);
    err |= clSetKernelArg(znccKernel, 2, sizeof(dDisparityRL), &dDisparityRL);
    err |= clSetKernelArg(znccKernel, 3, sizeof(Width), &Width);
    err |= clSetKernelArg(znccKernel, 4, sizeof(Height), &Height);
    err |= clSetKernelArg(znccKernel, 5, sizeof(int), &BSX);
    err |= clSetKernelArg(znccKernel, 6, sizeof(int), &BSY);
    err |= clSetKernelArg(znccKernel, 7, sizeof(MINDISP), &MINDISP);
    err |= clSetKernelArg(znccKernel, 8, sizeof(int), &MAXDISP);    
    err |= clSetKernelArg(znccKernel, 9, sizeof(BSIZE), &BSIZE);
    err |= clSetKernelArg(znccKernel, 10, sizeof(imsize), &imsize);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting zncc kernel arguments\n");
        return 1;
    }

    // Enqueue zncc kernel
    err = clEnqueueNDRangeKernel(queue, znccKernel, 2, NULL, (const size_t*)&globalSize,  (const size_t*)&wgSize, 0, NULL, &zncc_event2);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing zncc kernel\n");
        return 1;
    }

    err = clSetKernelArg(crossCheckKernel, 0, sizeof(dDisparityLR), &dDisparityLR);
    err |= clSetKernelArg(crossCheckKernel, 1, sizeof(dDisparityRL), &dDisparityRL);
    err |= clSetKernelArg(crossCheckKernel, 2, sizeof(dDisparityLRCC), &dDisparityLRCC);
    err |= clSetKernelArg(crossCheckKernel, 3, sizeof(imsize), &imsize);
    err |= clSetKernelArg(crossCheckKernel, 4, sizeof(THRESHOLD), &THRESHOLD);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting crossCheck kernel arguments\n");
        return 1;
    }

    // Enqueue cross_check kernel
    err = clEnqueueNDRangeKernel(queue, crossCheckKernel, 1, NULL, (const size_t*)&globalSize1D,  (const size_t*)&wgSize1D, 0, NULL, &cross_check_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing crossCheck kernel\n");
        return 1;
    }



    err = clSetKernelArg(occlusionKernel, 0, sizeof(dDisparityLRCC), &dDisparityLRCC);
    err |= clSetKernelArg(occlusionKernel, 1, sizeof(dDisparity), &dDisparity);
    err |= clSetKernelArg(occlusionKernel, 2, sizeof(Width), &Width);
    err |= clSetKernelArg(occlusionKernel, 3, sizeof(Height), &Height);
    err |= clSetKernelArg(occlusionKernel, 4, sizeof(NEIBSIZE), &NEIBSIZE);
    err |= clSetKernelArg(occlusionKernel, 5, sizeof(imsize), &imsize);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting crossCheck kernel arguments\n");
        return 1;
    }


    // Enqueue occlusion kernel
    err = clEnqueueNDRangeKernel(queue, occlusionKernel, 1, NULL, (const size_t*)&globalSize,  (const size_t*)&wgSize, 0, NULL, &occlusion_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing crossCheck kernel\n");
        return 1;
    }



    //normalize_dmap(Disparity, Width, Height);
    
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("Elapsed time: %.4lf s.\n", elapsed);
    

    Error = lodepng_encode_file("depthmap.png", Disparity, Width, Height, LCT_GREY, 8);
    if(Error){
        printf("Error in saving of the disparity %u: %s\n", Error, lodepng_error_text(Error));
        return -1;
    }   


    gettimeofday(&end_time, NULL); // Record end time
    double algorithm_time = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_usec - start_time.tv_usec) / 1000000.0; // Calculate execution time

    printf("Algorithm time: %.6f seconds\n", algorithm_time);

    // Release grayscale resources
    clReleaseMemObject(dOriginalImageL);  
    clReleaseMemObject(dOriginalImageR);      
    clReleaseMemObject(ImageL);    
    clReleaseMemObject(ImageR);    
    clReleaseMemObject(dDisparityLR);    
    clReleaseMemObject(dDisparityRL);    
    clReleaseMemObject(dDisparityLRCC);    
    clReleaseMemObject(dDisparity);
    clReleaseKernel(resizeGreyscaleKernel);    
    clReleaseKernel(znccKernel);    
    clReleaseKernel(occlusionKernel);    
    clReleaseKernel(crossCheckKernel);    
    clReleaseProgram(resizeGreyscaleProgram);    
    clReleaseProgram(znccProgram);    
    clReleaseProgram(occlusionProgram);    
    clReleaseProgram(crossCheckProgram);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);



    free(OriginalImageR);
    free(OriginalImageL);
    free(ImageR);
    free(ImageL);
    free(Disparity);

    return 0;
}
