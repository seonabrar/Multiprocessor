__kernel void resize_and_grayscale(
    __global const uchar4 *input,
    __global uchar *output,
    const uint input_width,
    const uint input_height,
    const uint output_width,
    const uint output_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Ensure that we do not go out of bounds
    if (x >= output_width || y >= output_height)
        return;

    int ix = x * 4;
    int iy = y * 4;

    // Accumulate values in these variables
    float gray = 0.0;
    int count = 0;

    // Loop to average the pixels in the 4x4 block
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            int px = ix + dx;
            int py = iy + dy;
            if (px < input_width && py < input_height) {
                int idx = py * input_width + px;
                uchar4 pixel = input[idx];
                // Convert to grayscale using the luminosity method
                gray += 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                count++;
            }
        }
    }

    // Write the averaged gray value to the output image
    if (count > 0) {
        gray /= count;
        output[y * output_width + x] = (uchar)gray;
    }
}