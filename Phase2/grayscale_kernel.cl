__kernel void grayscaleImage(__global const uchar4* inputImage,
                             __global uchar* outputImage,
                             const int width,
                             const int height) {
    int x = get_global_id(0);  // Thread index in x direction
    int y = get_global_id(1);  // Thread index in y direction

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel = inputImage[index];
        
        float grayValue = 0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z;
        
        outputImage[index] = (uchar)grayValue;
    }
}
