
#include "/usr/local/opt/libomp/include/omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "lodepng/lodepng.h"
#include <sys/time.h> // For gettimeofday on Linux

#define MAXDISP 65 // Maximum disparity (downscaled)
#define MINDISP 0

#define BSX 9 // Window size on X-axis (width)
#define BSY 9 // Window size on Y-axis (height)

#define THRESHOLD 2 // Threshold for cross-checking

#define NEIBSIZE 256 // Size of the neighborhood for occlusion-filling

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
void WriteImage(const char *filename, const uint8_t *image, uint32_t width, uint32_t height)
{
    uint32_t error;
    error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error)
    {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }
}

void resizegray(const uint8_t *imageL, const uint8_t *imageR, uint8_t *resizedL, uint8_t *resizedR, uint32_t w, uint32_t h)
{
    /* Downscaling and conversion to 8bit grayscale image */

    int32_t i, j;     // Indices of the resized image
    int32_t new_w = w / 4, new_h = h / 4; // Width and height of the downscaled image
    int32_t orig_i, orig_j;               // Indices of the original image

    // Iterating through the pixels of the downscaled image
    for (i = 0; i < new_h; i++)
    {
        for (j = 0; j < new_w; j++)
        {
            // Calculating corresponding indices in the original image
            orig_i = (4 * i - 1 * (i > 0));
            orig_j = (4 * j - 1 * (j > 0));
            // Grayscaling
            resizedL[i * new_w + j] = 0.2126 * imageL[orig_i * (4 * w) + 4 * orig_j] + 0.7152 * imageL[orig_i * (4 * w) + 4 * orig_j + 1] + 0.0722 * imageL[orig_i * (4 * w) + 4 * orig_j + 2];
            resizedR[i * new_w + j] = 0.2126 * imageR[orig_i * (4 * w) + 4 * orig_j] + 0.7152 * imageR[orig_i * (4 * w) + 4 * orig_j + 1] + 0.0722 * imageR[orig_i * (4 * w) + 4 * orig_j + 2];
        }
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
    #pragma omp parallel for private(i, j, i_b, j_b, ind_l, ind_r, d, lbmean, rbmean, lbstd, rbstd, current_score, best_d, best_score, cl, cr) shared(left, right, dmap)

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
    uint8_t *ImageL; // Left image
    uint8_t *ImageR; // Right image

    uint32_t Width, Height;
    uint32_t w1, h1;
    uint32_t w2, h2;
    struct timeval start_time, end_time; // Variables to hold start and end timestamps
  
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
    // Resizing
    gettimeofday(&start_time, NULL); // Record start time

    ImageL = (uint8_t *)malloc(Width * Height); // Memory pre-allocation for the resized image
    ImageR = (uint8_t *)malloc(Width * Height); // Memory pre-allocation for the resized image
    resizegray(OriginalImageL, OriginalImageR, ImageL, ImageR, Width * 4, Height * 4); // Left Image

    // Calculating the disparity maps
    printf("Computing maps with zncc...\n");
    DisparityLR = CALCZNCC(ImageL, ImageR, Width, Height, BSX, BSY, MINDISP, MAXDISP);
    DisparityRL = CALCZNCC(ImageR, ImageL, Width, Height, BSX, BSY, -MAXDISP, MINDISP);
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
    WriteImage("resized_left.png", ImageL, Width, Height);
    WriteImage("resized_right.png", ImageR, Width, Height);
    WriteImage("depthmap_before_post_procLR.png", DisparityLR, Width, Height);
    WriteImage("depthmap_before_post_procRL.png", DisparityRL, Width, Height);
    WriteImage("depthmap.png", Disparity, Width, Height);

    free(OriginalImageR);
    free(OriginalImageL);
    free(ImageR);
    free(ImageL);
    free(Disparity);
    free(DisparityLR);
    free(DisparityRL);
    free(DisparityLRCC);

    return 0;
}
