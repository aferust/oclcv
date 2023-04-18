
// compile time constants for speed:
// filter_depth, filter_width, filter_height

__kernel void convolution(__global const float *input,
                           __global const float *filter,
                           __global float *output,
                           const int input_width,
                           const int input_height,
                           const int input_depth,
                           const int filter_width,
                           const int filter_height)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int gid_z = get_global_id(2);

    float sum = 0.0f;

    const int filter_depth = input_depth;
 
    for (int z = 0; z < filter_depth; z++) {
        for (int y = 0; y < filter_height; y++) {
            for (int x = 0; x < filter_width; x++) {
                int input_x = gid_x + x - filter_width / 2;
                int input_y = gid_y + y - filter_height / 2;
                int input_z = gid_z + z - filter_depth / 2; 

                if (input_x < 0 || input_x >= input_width ||
                    input_y < 0 || input_y >= input_height ||
                    input_z < 0 || input_z >= input_depth) {
                    // out of bounds
                    continue;
                }

                int input_index = input_z * input_height * input_width +
                                  input_y * input_width + input_x;
                int filter_index = z * filter_height * filter_width +
                                   y * filter_width + x;             

                sum += input[input_index] * filter[filter_index];
            }
        }
    }

    int output_index = gid_z * input_height * input_width +
                       gid_y * input_width + gid_x;
    output[output_index] = sum;
}