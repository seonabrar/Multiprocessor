#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

int main() {
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id *platforms;

    // Get the number of available OpenCL platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    printf("Number of OpenCL platforms detected: %d\n", num_platforms);

    // Allocate memory to store platform information
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);

    // Get platform IDs
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform IDs\n");
        return EXIT_FAILURE;
    }

    // Iterate through each platform and display information
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_info[1024];
        size_t info_size;

        // Get platform vendor
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_info), platform_info, &info_size);
        printf("Platform %d Vendor: %s\n", i + 1, platform_info);

        // Get platform name
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_info), platform_info, &info_size);
        printf("Platform %d Name: %s\n", i + 1, platform_info);

        // Get platform profile
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(platform_info), platform_info, &info_size);
        printf("Platform %d Profile: %s\n", i + 1, platform_info);

        // Get platform version
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_info), platform_info, &info_size);
        printf("Platform %d Version: %s\n", i + 1, platform_info);

        // Get the number of devices available in the platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        printf("Number of devices in Platform %d: %d\n", i + 1, num_devices);

        // Get device IDs
        cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        // Iterate through each device and display information
        for (cl_uint j = 0; j < num_devices; j++) {
            // Get device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(platform_info), platform_info, &info_size);
            printf("Device %d Name: %s\n", j + 1, platform_info);

            // Get device hardware version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(platform_info), platform_info, &info_size);
            printf("Device %d Hardware Version: %s\n", j + 1, platform_info);

            // Get device driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(platform_info), platform_info, &info_size);
            printf("Device %d Driver Version: %s\n", j + 1, platform_info);

            // Get device OpenCL version
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(platform_info), platform_info, &info_size);
            printf("Device %d OpenCL Version: %s\n", j + 1, platform_info);

            // Get device parallel compute units
            cl_uint num_compute_units;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_compute_units, NULL);
            printf("Device %d Parallel Compute Units: %d\n", j + 1, num_compute_units);
        }
        printf("\n");

        free(devices);
    }

    free(platforms);

    return 0;
}
