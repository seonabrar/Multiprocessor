__kernel void resizeImage(__global const uchar4* inputImage,
                          __global uchar4* outputImage,
                          const int inputWidth,
                          const int inputHeight,
                          const int outputWidth,
                          const int outputHeight) {
    int x = get_global_id(0);  // Thread index in x direction
    int y = get_global_id(1);  // Thread index in y direction

    if (x < outputWidth && y < outputHeight) {
        float scaleX = (float)inputWidth / (float)outputWidth;
        float scaleY = (float)inputHeight / (float)outputHeight;

        float srcX = (x + 0.5f) * scaleX - 0.5f;
        float srcY = (y + 0.5f) * scaleY - 0.5f;

        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = srcX - x0;
        float dy = srcY - y0;

        uchar4 pixel00 = inputImage[y0 * inputWidth + x0];
        uchar4 pixel01 = inputImage[y0 * inputWidth + x1];
        uchar4 pixel10 = inputImage[y1 * inputWidth + x0];
        uchar4 pixel11 = inputImage[y1 * inputWidth + x1];

        uchar4 interpolatedPixel;
        interpolatedPixel.x = (1.0f - dx) * (1.0f - dy) * pixel00.x + dx * (1.0f - dy) * pixel01.x +
                              (1.0f - dx) * dy * pixel10.x + dx * dy * pixel11.x;
        interpolatedPixel.y = (1.0f - dx) * (1.0f - dy) * pixel00.y + dx * (1.0f - dy) * pixel01.y +
                              (1.0f - dx) * dy * pixel10.y + dx * dy * pixel11.y;
        interpolatedPixel.z = (1.0f - dx) * (1.0f - dy) * pixel00.z + dx * (1.0f - dy) * pixel01.z +
                              (1.0f - dx) * dy * pixel10.z + dx * dy * pixel11.z;
        interpolatedPixel.w = (1.0f - dx) * (1.0f - dy) * pixel00.w + dx * (1.0f - dy) * pixel01.w +
                              (1.0f - dx) * dy * pixel10.w + dx * dy * pixel11.w;

        outputImage[y * outputWidth + x] = interpolatedPixel;
    }
}
