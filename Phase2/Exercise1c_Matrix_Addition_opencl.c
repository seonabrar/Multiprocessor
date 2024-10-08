#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    // Matrix dimensions
    int rows = 100;
    int cols = 100;
    size_t datasize = sizeof(float) * rows * cols;

    // Allocate memory for matrices
    float *matrix_1 = (float *)malloc(datasize);
    float *matrix_2 = (float *)malloc(datasize);
    float *result = (float *)malloc(datasize);

    // Initialize matrices
    for (int i = 0; i < rows * cols; ++i) {
        matrix_1[i] = i;
        matrix_2[i] = i;
    }

    // Load the kernel source code
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("add_matrix.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device
    cl_mem matrix_1_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &ret);
    cl_mem matrix_2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &ret);
    cl_mem result_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &ret);

    // Copy matrices to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, matrix_1_mem_obj, CL_TRUE, 0, datasize, matrix_1, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, matrix_2_mem_obj, CL_TRUE, 0, datasize, matrix_2, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "add_matrix", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_1_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_2_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&result_mem_obj);

    // Execute the OpenCL kernel
    size_t global_item_size[2] = {rows, cols};
    size_t local_item_size[2] = {1, 1};
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, &event);

    // Wait for the command queue to finish
    ret = clFinish(command_queue);

    // Calculate execution time
    cl_ulong time_start, time_end;
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double total_time = (time_end - time_start)*10^-9;

    printf("Execution time: %0.3f seconds\n", total_time);

    // Read the result from the device
    ret = clEnqueueReadBuffer(command_queue, result_mem_obj, CL_TRUE, 0, datasize, result, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(matrix_1_mem_obj);
    ret = clReleaseMemObject(matrix_2_mem_obj);
    ret = clReleaseMemObject(result_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(matrix_1);
    free(matrix_2);
    free(result);
    return 0;
}
