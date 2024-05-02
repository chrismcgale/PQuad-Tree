#include <iostream>
#include <vector>
#include <random>

#include "pquad_tree.cuh"

// Generate random points biased towards a focal point
generate_random_points(Points& points, int num_points, float focal_x, float focal_y, float bias) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(focal_x - bias, focal_x + bias);

    for (int i = 0; i < num_points; ++i) {
        points->set_point(dist(gen), dist(gen))
    }
}

int main() {
    const int num_points = 100;
    const float focal_x = 0.0f;
    const float focal_y = 0.0f;
    const float bias = 0.1f;
    const int max_depth = 8; // Limited by max recursion depth (26) and the heap running out of space. 8 allocates 1Mb to our tree.

    Points h_points(num_points);
    Points* d_points;
    
    generate_random_points(h_points, num_points, focal_x, focal_y, bias);

    cudaMalloc((void**)&d_points, sizeof(h_points));
    cudaMemcpy(d_points, &h_points, sizeof(h_points), cudaMemcpyHostToDevice);

    Quadtree_Node* nodes = new Quadtree_Node[pow(4, max_depth)];
    cudaMalloc((void**)&nodes, sizeof(Quadtree_Node) * num_points);
   
    Parameters params(max_depth, 2);

    build_quadtree_kernel<<<1, 1, 8 * sizeof(int)>>>(nodes, points, params);

    cudaFree(nodes);
    cudaFree(points);

    delete [] nodes;
    delete [] points
}