#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng/lodepng.h"

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

// Function to resize image to 1/16 of original size
unsigned char* ResizeImage(const unsigned char* image, unsigned width, unsigned height, unsigned* newWidth, unsigned* newHeight) {
    *newWidth = width / 4;
    *newHeight = height / 4;
    unsigned char* resizedImage = (unsigned char*)malloc((*newWidth) * (*newHeight) * 4 * sizeof(unsigned char));
    if (resizedImage == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for resized image\n");
        return NULL;
    }
    for (unsigned y = 0; y < *newHeight; ++y) {
        for (unsigned x = 0; x < *newWidth; ++x) {
            unsigned index = (y * 4 * width + x * 4) * 4;
            unsigned newIndex = (y * (*newWidth) + x) * 4;
            resizedImage[newIndex] = image[index];
            resizedImage[newIndex + 1] = image[index + 1];
            resizedImage[newIndex + 2] = image[index + 2];
            resizedImage[newIndex + 3] = image[index + 3];
        }
    }
    return resizedImage;
}

// Function to convert image to grayscale
unsigned char* GrayScaleImage(const unsigned char* image, unsigned width, unsigned height) {
    unsigned char* grayImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (grayImage == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for grayscale image\n");
        return NULL;
    }
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            unsigned index = (y * width + x) * 4;
            unsigned char r = image[index];
            unsigned char g = image[index + 1];
            unsigned char b = image[index + 2];
            grayImage[y * width + x] = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
    }
    return grayImage;
}

// Function to apply a 5x5 moving filter to grayscale image
unsigned char* ApplyFilter(const unsigned char* image, unsigned width, unsigned height) {
    unsigned char* filteredImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (filteredImage == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for filtered image\n");
        return NULL;
    }
    int filter[5][5] = {
        {1,  4,  6,  4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1,  4,  6,  4, 1}
    };
    int filterSum = 256;
    for (unsigned y = 2; y < height - 2; ++y) {
        for (unsigned x = 2; x < width - 2; ++x) {
            int sum = 0;
            for (int fy = -2; fy <= 2; ++fy) {
                for (int fx = -2; fx <= 2; ++fx) {
                    sum += image[(y + fy) * width + (x + fx)] * filter[fy + 2][fx + 2];
                }
            }
            filteredImage[y * width + x] = sum / filterSum;
        }
    }
    return filteredImage;
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
    const char* outputFilename = "image_0_bw.png";

    unsigned char* image;
    unsigned width, height;

    // Read image
    clock_t start = clock();
    image = ReadImage(inputFilename, &width, &height);
    clock_t end = clock();
    printf("Time taken to read image: %lf seconds\n", MeasureTime(start, end));

    if (image == NULL) {
        return 1;
    }

    unsigned newWidth, newHeight;

    // Resize image
    start = clock();
    unsigned char* resizedImage = ResizeImage(image, width, height, &newWidth, &newHeight);
    end = clock();
    printf("Time taken to resize image: %lf seconds\n", MeasureTime(start, end));
    if (resizedImage == NULL) {
        free(image);
        return 1;
    }

    // Convert image to grayscale
    start = clock();
    unsigned char* grayImage = GrayScaleImage(resizedImage, newWidth, newHeight);
    end = clock();
    printf("Time taken to convert image to grayscale: %lf seconds\n", MeasureTime(start, end));
    if (grayImage == NULL) {
        free(image);
        free(resizedImage);
        return 1;
    }

    // Apply filter to grayscale image
    start = clock();
    unsigned char* filteredImage = ApplyFilter(grayImage, newWidth, newHeight);
    end = clock();
    printf("Time taken to apply filter: %lf seconds\n", MeasureTime(start, end));
    if (filteredImage == NULL) {
        free(image);
        free(resizedImage);
        free(grayImage);
        return 1;
    }

    // Write resulting image
    start = clock();
    WriteImage(outputFilename, filteredImage, newWidth, newHeight);
    end = clock();
    printf("Time taken to write image: %lf seconds\n", MeasureTime(start, end));

    // Free memory
    free(image);
    free(resizedImage);
    free(grayImage);
    free(filteredImage);

    return 0;
}
