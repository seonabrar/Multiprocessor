__kernel void applyFilter(__global const uchar* inputImage,
                          __global uchar* outputImage,
                          const int width,
                          const int height) {
    const int filterSize = 5;
    const int filter[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    };

    const int filterSum = 25; // Sum of all elements in the filter

    int x = get_global_id(0);
    int y = get_global_id(1);

    // Apply filter to each pixel
    if (x < width && y < height) {
        int sum = 0;
        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                int cx = clamp(x - filterSize / 2 + fx, 0, width - 1);
                int cy = clamp(y - filterSize / 2 + fy, 0, height - 1);
                sum += inputImage[cy * width + cx] * filter[fy][fx];
            }
        }
        outputImage[y * width + x] = sum / filterSum;
    }
}
