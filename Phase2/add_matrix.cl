__kernel void add_matrix(__global const float *matrix_1, __global const float *matrix_2, __global float *result, const int rows, const int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int index = i * cols + j;
    if (i < rows && j < cols) {
        result[index] = matrix_1[index] + matrix_2[index];
    }
}

