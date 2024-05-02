class Points {
    float *m_x;
    float *m_y;

  public:
    // Constructor
    __host__ __device__ Points(int num_points) {
        m_x = new float[num_points];
        m_y = new float[num_points];
    }

    // Get a point by index
    __host__ __device__ __forceinline__ float2 get_point(int idx) const {
        return make_float2(m_x[idx], m_y[idx]);
    }

    // Set a point
    __host__ __device__ __forceinline__ void set_point(int idx, const float2 &p) const {
        m_x[idx] = p.x;
        m_y[idx] = p.y;
    }

    // Destructor
    ~Points() {
        delete [] m_x;
        delete [] m_y;
    }
};

class Bounding_Box {
    float2 m_p_min;
    float2 m_p_max;

  public:
    // Constructor
    __host__ __device__ Bounding_Box() {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }

    // Compute centre
    __host__ __device__ void compute_centre(float2 &centre) {
        centre.x = 0.5f * (m_p_min.x + m_p_max.x);
        centre.y = 0.5f * (m_p_min.y + m_p_max.y);
    }

    __host__ __device__ __forceinline__ const float2 &get_max() const {
        return m_p_max;
    }

    __host__ __device__ __forceinline__ const float2 &get_min() const {
        return m_p_min;
    }

    // Check if contains a point
    __host__ __ __device__ bool contains(const float2 &p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y && p.y < m_p_max.y;
    }

    // Change the box
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

class Quadtree_Node {
    int m_id;
    Bounding_Box m_bounding_box;
    int m_begin, m_end;

public:
    __host__ __device__ Quadtree_Node() : m_id(0), m_begin(0), m_end(0) {}

    __host__ __device__ int id() const {
        return m_id;
    }

    __host__ __device__ void set_id(int new_id) {
        m_id = new_id;
    }

    __host__ __device__ __forceinline__ const Bounding_Box &bounding_box() const {
        return m_bounding_box;
    }

    __host__ __device__ __forceinline__ void set_bounding_box(float min_x, float min_y, float max_x, float max_y) {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }

    // Return the number of points in the tree
    __host__ __device__ __forceinline__ int num_points() const {
        return m_end - m_begin;
    }

    __host__ __device__ __forceinline__ int points_begin() const {
        return m_begin;
    }

    __host__ __device__ __forceinline__ int points_end() const {
        return m_end;
    }

    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        m_begin = begin;
        m_end = end;
    }
};

struct Parameters {
    int point_selector;
    int num_nodes_at_level;
    int depth;

    const int max_depth;
    const int min_points_per_node;

    // Constructor
    __host__ __device__ Parameters(int max_depth, int min_points_per_node) : point_selector(0), num_nodes_at_level(1), depth(0), max_depth(max_depth), min_points_per_node(min_points_per_node) {}

    // Copy Constructor
    __host__ __device__ Parameters(const Parameters &params) : point_selector((params.point_selector + 1) % 2), num_nodes_at_level(4 * params.num_nodes_at_level), depth(params.depth + 1), max_depth(params.max_depth), min_points_per_node(params.min_points_per_node) {}
};
