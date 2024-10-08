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
#define MAXDISP 65 // Maximum disparity (downscaled)
#define MINDISP 0

#define BSX 9 // Window size on X-axis (width)
#define BSY 9 // Window size on Y-axis (height)

#define THRESHOLD 2 // Threshold for cross-checking

#define NEIBSIZE 256 // Size of the neighborhood for occlusion-filling
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

// Function to write image
void WriteImage(const char *filename, const cl_mem *image, uint32_t width, uint32_t height)
{
    uint32_t error;
    error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }
}



uint8_t *CALCZNCC(const uint8_t *left, const uint8_t *right, uint32_t w, uint32_t h, int32_t bsx, int32_t bsy, int32_t mind, int32_t maxd)
{
    /* Disparity map computation */
    int32_t imsize = w * h; // Size of the image
    int32_t bsize = bsx * bsy; // Block size

    uint8_t *dmap = (uint8_t *)malloc(imsize); // Memory allocation for the disparity map
    int32_t i, j;     // Indices for rows and colums respectively
    int32_t i_b, j_b; // Indices within the block
    int32_t ind_l, ind_r; // Indices of block values within the whole image
    int32_t d;             // Disparity value
    float cl, cr;          // centered values of a pixel in the left and right images;

    float lbmean, rbmean; // Blocks means for left and right images
    float lbstd, rbstd;   // Left block std, Right block std
    float current_score;  // Current ZNCC value

    int32_t best_d;
    float best_score;

    for (i = 0; i < h; i++)
    {
        for (j = 0; j < w; j++)
        {
            // Searching for the best d for the current pixel
            best_d = maxd;
            best_score = -1;
            for (d = mind; d <= maxd; d++)
            {
                // Calculating the blocks' means
                lbmean = 0;
                rbmean = 0;
                for (i_b = -bsy / 2; i_b < bsy / 2; i_b++)
                {
                    for (j_b = -bsx / 2; j_b < bsx / 2; j_b++)
                    {
                        // Borders checking
                        if (!(i + i_b >= 0) || !(i + i_b < h) || !(j + j_b >= 0) || !(j + j_b < w) || !(j + j_b - d >= 0) || !(j + j_b - d < w))
                        {
                            continue;
                        }
                        // Calculatiing indices of the block within the whole image
                        ind_l = (i + i_b) * w + (j + j_b);
                        ind_r = (i + i_b) * w + (j + j_b - d);
                        // Updating the blocks' means
                        lbmean += left[ind_l];
                        rbmean += right[ind_r];
                    }
                }
                lbmean /= bsize;
                rbmean /= bsize;

                // Calculating ZNCC for given value of d
                lbstd = 0;
                rbstd = 0;
                current_score = 0;

                // Calculating the nomentaor and the standard deviations for the denominator
                for (i_b = -bsy / 2; i_b < bsy / 2; i_b++)
                {
                    for (j_b = -bsx / 2; j_b < bsx / 2; j_b++)
                    {
                        // Borders checking
                        if (!(i + i_b >= 0) || !(i + i_b < h) || !(j + j_b >= 0) || !(j + j_b < w) || !(j + j_b - d >= 0) || !(j + j_b - d < w))
                        {
                            continue;
                        }
                        // Calculatiing indices of the block within the whole image
                        ind_l = (i + i_b) * w + (j + j_b);
                        ind_r = (i + i_b) * w + (j + j_b - d);

                        cl = left[ind_l] - lbmean;
                        cr = right[ind_r] - rbmean;
                        lbstd += cl * cl;
                        rbstd += cr * cr;
                        current_score += cl * cr;
                    }
                }
                // Normalizing the denominator
                current_score /= sqrt(lbstd) * sqrt(rbstd);
                // Selecting the best disparity
                if (current_score > best_score)
                {
                    best_score = current_score;
                    best_d = d;
                }
            }
            dmap[i * w + j] = (uint8_t)abs(best_d); // Considering both Left to Right and Right to left disparities
        }
    }

    return dmap;
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

uint8_t *CrossCheck(const uint8_t *map1, const uint8_t *map2, uint32_t imsize, uint32_t dmax, uint32_t threshold)
{
    uint8_t *map = (uint8_t *)malloc(imsize);
    uint32_t idx;

    for (idx = 0; idx < imsize; idx++)
    {
        if (abs((int32_t)map1[idx] - map2[idx]) > threshold) // Remember about the trick for Rigth to left disprity in zncc!!
            map[idx] = 0;
        else
            map[idx] = map1[idx];
    }
    return map;
}

uint8_t *OcclusionFill(const uint8_t *map, uint32_t w, uint32_t h, uint32_t nsize)
{
    int32_t imsize = w * h; // Size of the image

    uint8_t *result = (uint8_t *)malloc(imsize);
    int32_t i, j;     // Indices for rows and colums respectively
    int32_t i_b, j_b; // Indices within the block
    int32_t ind_neib; // Index in the nighbourhood
    int32_t ext;
    bool stop; // Stop flag for nearest neighbor interpolation

    for (i = 0; i < h; i++)
    {
        for (j = 0; j < w; j++)
        {
            // If the value of the pixel is zero, perform the occlusion filling by nearest neighbour interpolation
            result[i * w + j] = map[i * w + j];
            if (map[i * w + j] == 0)
            {

                // Spreading search of non-zero pixel in the neighborhood i,j
                stop = false;
                for (ext = 1; (ext <= nsize / 2) && (!stop); ext++)
                {
                    for (j_b = -ext; (j_b <= ext) && (!stop); j_b++)
                    {
                        for (i_b = -ext; (i_b <= ext) && (!stop); i_b++)
                        {
                            // Cehcking borders
                            if (!(i + i_b >= 0) || !(i + i_b < h) || !(j + j_b >= 0) || !(j + j_b < w) || (i_b == 0 && j_b == 0))
                            {
                                continue;
                            }
                            // Calculatiing indices of the block within the whole image
                            ind_neib = (i + i_b) * w + (j + j_b);
                            //If we meet a nonzero pixel, we interpolate and quite from this loop
                            if (map[ind_neib] != 0)
                            {
                                result[i * w + j] = map[ind_neib];
                                stop = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
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
    printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu bytes\n", localMemSize);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", maxComputeUnits);
    printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", maxClockFrequency);
    printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %lu bytes\n", maxConstantBufferSize);
    printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %zu\n", maxWorkGroupSize);
    printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: [%zu, %zu, %zu]\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);
}
// Function to check OpenCL errors
void checkError(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (error code: %d)\n", message, err);
        exit(EXIT_FAILURE);
    }
}
int32_t main(int32_t argc, char **argv)
{
    const char* inputFilename1 = "im0.png"; // Left image filename
    const char* inputFilename2 = "im1.png"; // Right image filename
    const char* outputFilename = "depthmap.png"; // Output filename for the disparity map

    uint8_t *OriginalImageL; // Left image
    uint8_t *OriginalImageR; // Right image
    uint8_t *DisparityLR;
    uint8_t *DisparityRL;
    uint8_t *DisparityLRCC;
    uint8_t *Disparity;


    uint32_t Width, Height;
    uint32_t w1, h1;
    uint32_t w2, h2;
    uint32_t imsize;

    struct timeval start_time, end_time; // Variables to hold start and end timestamps


    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_platforms, num_devices;
    cl_int err;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

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

    // Load kernel source
    FILE* file = fopen("resize_greyscale.cl", "r");
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

    // Create program
    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program\n");
        return 1;
    }

        // Build program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");
        return 1;
    }

        // Create kernel
    kernel = clCreateKernel(program, "resizegrayImage", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernel\n");
        return 1;
    }

    Width = w1 / 4;
    Height = h1 / 4;
    cl_mem dImageL = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dImageL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }

    cl_mem dImageR = clCreateBuffer(context, CL_MEM_READ_WRITE, Width*Height, 0, &err);
    if (!dImageL || err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer\n");
        return 1;
    }
    Disparity = (uint8_t*) malloc(Width*Height); 
     
    clock_gettime(CLOCK_MONOTONIC, &err);
    

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
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dOriginalImageL);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dOriginalImageR);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &dImageL);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &dImageR);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &Width);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &Height);
    // Resizing
    gettimeofday(&start_time, NULL); // Record start time

    // Define event variables
    cl_event kernel_event;

    // Enqueue kernel
    const size_t globalSize[] = {atoi(argv[5]), atoi(argv[6])};

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, (const size_t*)&globalSize, NULL, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueueing kernel\n");
        return 1;
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error waiting for commands to finish\n");
        return 1;
    }


    

        // Wait for all events to complete
    clWaitForEvents(1, &kernel_event);
    clEnqueueReadBuffer(queue, dImageL, CL_TRUE, 0, Width*Height, Disparity, 0, NULL, NULL);

    // Calculating the disparity maps
    printf("Computing maps with zncc...\n");
    DisparityLR = CALCZNCC(dImageL, dImageR, Width, Height, BSX, BSY, MINDISP, MAXDISP);
    DisparityRL = CALCZNCC(dImageR, dImageL, Width, Height, BSX, BSY, -MAXDISP, MINDISP);
    // Cross-checking
    printf("Performing cross-checking...\n");
    DisparityLRCC = CrossCheck(DisparityLR, DisparityRL, Width * Height, MAXDISP, THRESHOLD);
    // Occlusion-filling
    printf("Performing occlusion-filling...\n");
    Disparity = OcclusionFill(DisparityLRCC, Width, Height, NEIBSIZE);
    // Normalization
    printf("Performing maps normalization...\n");
    normalize_dmap(Disparity, Width, Height);
     gettimeofday(&end_time, NULL); // Record end time
    double algorithm_time = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_usec - start_time.tv_usec) / 1000000.0; // Calculate execution time

    printf("Algorithm time: %.6f seconds\n", algorithm_time);

    normalize_dmap(DisparityLR, Width, Height);
    normalize_dmap(DisparityRL, Width, Height);

    // Saving the results
    WriteImage("resized_left.png", dImageL, Width, Height);
    WriteImage("resized_right.png", dImageR, Width, Height);
    WriteImage("depthmap_before_post_procLR.png", DisparityLR, Width, Height);
    WriteImage("depthmap_before_post_procRL.png", DisparityRL, Width, Height);
    WriteImage("depthmap.png", Disparity, Width, Height);

        // Release OpenCL resources
    clReleaseMemObject(dImageL);
    clReleaseMemObject(dImageR);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(OriginalImageR);
    free(OriginalImageL);
    free(Disparity);
    free(DisparityLR);
    free(DisparityRL);
    free(DisparityLRCC);

    return 0;
}
