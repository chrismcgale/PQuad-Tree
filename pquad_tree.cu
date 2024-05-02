#include "pquad_tree.cuh"


// Check the number of points and its depth
__device__ bool check_num_points_and_depth(Quadtree_Node &nde, Points *points, int num_points, Parameters params) {
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node) {
        // Stop the recursion here. Make sure points[0] contains all the points
        if (params.point_selector == 1) {
            int it = node.points_begin(), end = node.points_end();
            for (it += threadIdx.x; it < end ; it += blockDim.x) {
                points[0].set_point(it, points[1].get_point(it));
            }
        }
        return true;
    }
    return false;
}

// Count the number of points in each quadrant
__device__ void count_point_in_children(const Points &in_points, int* smem, int range_begin, int range_end, float2 centre) {
    // Init shared memory
    if (threadIdx.x < 4) smem[threadIdx.x] = 0;
    __syncthreads();

    for (int iter=range_begin+ threadIdx.x; iter < range_end; iter+=blockDim.x) {
        float2 p = in_points.get_point(iter);
        if (p.x < centre.x && p.y >= centre.y)
            atomicAdd(&smem[0], 1);
        if (p.x >= centre.x && p.y >= centre.y)
            atomicAdd(&smem[1], 1);
        if (p.x < centre.x && p.y < centre.y)
            atomicAdd(&smem[2], 1);
        if (p.x >= centre.x && p.y < centre.y)
            atomicAdd(&smem[3], 1);
    }
    __syncthreads();
}

// Scan quadrants results to obtain reordering offset
__device__ void scan_for_offsets(int node_points_begin, int* smem) {
    int* smem2 = &smem[4];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++) smem2[i] = i ==0 ? 0 : smem2[i-1] + smem[i-1];
        for (int i = 0; i < 4; i++) smem2[i] += node_points_begin;
    }
    __syncthreads();
}

// Reorder points to group by quadrant
__device__ void reorder_points(Points &out_points, const Points &in_points, int* smem, int range_begin, int range_end, float2 centre) {
    int* smem2 = &smem[4];

    for (int iter = range_begin+threadIdx.x; iter < range_end; iter+=blockDim.x) {
        int dest;
        float2 p = in_points.get_point(iter);
        if (p.x < centre.x && p.y >= centre.y)
            atomicAdd(&smem[0], 1);
        if (p.x >= centre.x && p.y >= centre.y)
            atomicAdd(&smem[1], 1);
        if (p.x < centre.x && p.y < centre.y)
            atomicAdd(&smem[2], 1);
        if (p.x >= centre.x && p.y < centre.y)
            atomicAdd(&smem[3], 1);

        // Move point
        out_points.set_point(dest, p);
    }
    __syncthreads();
}

// Prepare children launch
__device__ void prepare_children(Quadtree_Node *children, Quadtree_Node &node, const Bounding_Box &bbox, int *smem) {
    int child_offset = 4*node.id(); // The offsets of the children at their level

    // Set IDs
    children[child_offset+0].set_id(4*node.id() + 0);
    children[child_offset+1].set_id(4*node.id() + 1);
    children[child_offset+2].set_id(4*node.id() + 2);
    children[child_offset+3].set_id(4*node.id() + 3);

    // Points of the bounding box
    const float2 = &p_min = bbox.get_min();
    const float2 = &p_max = bbox.get_max();

    children[child_offset+0].set_bounding_box(p_min.x, centre.y, centre.x, p_max.y);
    children[child_offset+1].set_bounding_box(centre.x, centre.y, p_max.x, p_max.y);
    children[child_offset+2].set_bounding_box(p_min.x, p_min.y, centre.x, centre.y);
    children[child_offset+3].set_bounding_box(centre.x, p_min.y, p_max.x, centre.y);

    // Set the ranges of the children
    children[child_offset+0].set_range(node.points_begin(), smem[4 + 0]);
    children[child_offset+1].set_range(smem[4 + 0], smem[4 + 1]);
    children[child_offset+2].set_range(smem[4 + 1], smem[4 + 2]);
    children[child_offset+3].set_range(smem[4 + 2], smem[4 + 3]);
}


__global__ void build_quadtree_kernel(Quadtree_Node *nodes, Points *points, Parameters params) {
    __shared__ int smem[8]; // Stores the number of points in each quadrant

    // The current node
    Quadtree_Node &node = nodes[blockIdx.x];
    node.set_id(node.is() + blockIdx.x);
    int num_points = node.num_points();

    // Check the number of points and its depth
    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) return;

    const Bounding_Box &bbox = node.bounding_box();
    float2 centre;
    bbox.compute_centre(centre);

    int range_begin = node.points_begin();
    int range_end = node.points_end();
    const Points &in_points = points[params.point_selector]; // Input points
    const Points &out_points = points[(params.point_selector + 1) % 2]; // Output points

    // Count the number of points in each child
    count_point_in_children(in_points, smem, range_begin, range_end, centre);

    // Scan the quadrants' results to know the reordering offset
    scan_for_offsets(node.points_begin(), smem);

    // Move points
    reorder_points(out_points, in_points, smem, range_begin, range_end, centre);

    // Launch new blocks
    if (threadIdx.x == blockDim.x-1) {
        // The children
        Quadtree_Node *children = &nodes[params.num_nodes_at_level];

        // Prepare children launch
        prepare_children(children, node, bbox, smem);

        // Launch 4 children
        build_quadtree_kernel<<<4, blockDim.x, 8 * sizeof(int)>>>(children, points, Parameters(params));
    }
};
