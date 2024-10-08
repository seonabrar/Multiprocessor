#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // For gettimeofday function

// Function to add two matrices element-wise
void add_Matrix(float *matrix_1, float *matrix_2, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Perform element-wise addition
            result[i * cols + j] = matrix_1[i * cols + j] + matrix_2[i * cols + j];
        }
    }
}

int main() {
    int rows = 100;
    int cols = 100;
    float *matrix_1 = (float *)malloc(rows * cols * sizeof(float));
    float *matrix_2 = (float *)malloc(rows * cols * sizeof(float));
    float *result = (float *)malloc(rows * cols * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_1[i * cols + j] = (float)(rand() % 100); // Example: Random values between 0 and 99
            matrix_2[i * cols + j] = (float)(rand() % 100); // Example: Random values between 0 and 99
        }
    }

    // Perform matrix addition and measure execution time
    struct timeval start, end;
    gettimeofday(&start, NULL); // Start time

    add_Matrix(matrix_1, matrix_2, result, rows, cols);

    gettimeofday(&end, NULL); // End time
    double execution_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Display execution time
    printf("Host Execution Time: %.6f seconds\n", execution_time);

    // Free dynamically allocated memory
    free(matrix_1);
    free(matrix_2);
    free(result);

    return 0;
}
