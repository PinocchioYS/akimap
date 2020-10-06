#ifndef AKIMAP_AKIMAP_H
#define AKIMAP_AKIMAP_H

#include <unordered_map>
#include <utils/RTree.h>

#include <akimap/akimap_key.h>
#include <akimap/akimap_node.h>

class AKIMap {
public:
    // Constructor & Destructor
    AKIMap(double _resolution, double _free_sample_distance);
    virtual ~AKIMap();

    /*
     * Update an occupancy grid from a point cloud with its sensor origin.
     *
     * @param pointcloud: a scan consisting of points in the world coordinate; [3xP] matrix
     * @param origin: a sensor origin in the world coordinate; [3x1] column vector
     */
    void insert_pointcloud(const Eigen::MatrixXd& _pointcloud, const Eigen::Vector3d& _origin);

    /*
     * Make a set of occupancy samples from the given sensor measurement.
     *
     * @param pointcloud: a sensor measurement; [3xP] matrix
     * @param origin: a sensor origin; [3x1] column vector
     * @return samples: occupancy samples; [3xN] matrix
     * @return labels: occupancy labels; [Nx1] column vector; 0(free), 1(occupied)
     */
    void make_occupancy_samples(const Eigen::MatrixXd& _pointcloud, const Eigen::Vector3d& _origin,
                                Eigen::MatrixXd& _samples, Eigen::VectorXi& _labels);

    /*
     * Compute a bandwidth matrix of kernel at each occupancy sample.
     *
     * @param samples: occupancy samples; [3xN] matrix
     * @param labels: occupancy labels; [Nx1] column vector; 0(free), 1(occupied)
     * @return transforms: decomposed components of bandwidths; [(3x3)xN] matrix
     * @return weights: kernel weights; [Nx1] column vector
     */
    void compute_kernel_bandwidths(const Eigen::MatrixXd& _samples, const Eigen::VectorXi& _labels,
                                   Eigen::MatrixXd& _transforms, Eigen::VectorXd& _weights);

    /*
     * Refine a kernel bandwidth - local optimization of kernel scale.
     *
     * @param center: a center point of kernel; [3x1] column vector
     * @param positives: positive neighbor points of the kernel center; [3xPOS] matrix
     * @param negatives: negative neighbor points of the kernel center; [3xNEG] matrix
     * @param init_transform: a initial transform composing a covariance matrix; [3x3] matrix
     * @param init_weight: a kernel weight
     * @return MIN_SCALE: the minimum bound of scale optimization
     */
    double optimize_kernel_scale(const Eigen::Vector3d& _center, const Eigen::MatrixXd& _positives, const Eigen::MatrixXd& _negatives,
                                 const Eigen::Matrix3d& _init_transform, const double _init_weight, const double _MIN_SCALE);

    /*
     * Update the occupancy estimations of cells from occupancy samples using adaptive kernels.
     *
     * @param samples: occupancy samples; [3xN] matrix
     * @param labels: occupancy labels; [Nx1] column vector; 0(free), 1(occupied)
     * @param transforms: decomposed components of adaptive bandwidths; [(3x3)xN] matrix
     * @param weight: kernel weights; [Nx1] column vector
     */
    void update_adaptive_kernel_estimations(const Eigen::MatrixXd& _samples, const Eigen::VectorXi& _labels,
                                            const Eigen::MatrixXd& _transforms, const Eigen::VectorXd& _weights);

    /*
     * Get an array to access all the nodes.
     *
     * @return node array
     */
    const std::unordered_map<AKIMapKey, AKIMapNode*, AKIMapKey::KeyHash>& get_nodes() const { return node_array; }


    /*
     * Convert types of data between continuous and discretized spaces.
     */
    inline AKIMapKey::key_type coordinate_to_key(double _coordinate) const {
        return AKIMapKey::key_type(((int)std::floor(RESOLUTION_FACTOR * _coordinate)) + TREE_MAX_VAL);
    }
    inline AKIMapKey coordinate_to_key(double _x, double _y, double _z) const {
        return AKIMapKey(coordinate_to_key(_x), coordinate_to_key(_y), coordinate_to_key(_z));
    }
    inline AKIMapKey coordinate_to_key(const Eigen::Vector3d& _coordinate) const {
        return coordinate_to_key(_coordinate(0), _coordinate(1), _coordinate(2));
    }

    inline double key_to_coordinate(AKIMapKey::key_type _key) const {
        return ((double)((int)_key - TREE_MAX_VAL) + 0.5) * RESOLUTION;
    }
    inline Eigen::Vector3d key_to_coordinate(AKIMapKey::key_type _kx, AKIMapKey::key_type _ky, AKIMapKey::key_type _kz) const {
        return { key_to_coordinate(_kx), key_to_coordinate(_ky), key_to_coordinate(_kz) };
    }
    inline Eigen::Vector3d key_to_coordinate(const AKIMapKey& _key) const {
        return key_to_coordinate(_key[0], _key[1], _key[2]);
    }

    inline AKIMapKey::key_type key_to_blockkey(AKIMapKey::key_type _key) const {
        return (AKIMapKey::key_type)((((unsigned)((int)_key - TREE_MAX_VAL) >> BLOCK_DEPTH) << BLOCK_DEPTH) + TREE_MAX_VAL);
    }
    inline AKIMapKey key_to_blockkey(const AKIMapKey& _key) const {
        return AKIMapKey(key_to_blockkey(_key[0]), key_to_blockkey(_key[1]), key_to_blockkey(_key[2]));
    }
    inline AKIMapKey::key_type coordinate_to_blockkey(double _coordinate) const {
        return key_to_blockkey(coordinate_to_key(_coordinate));
    }
    inline AKIMapKey coordinate_to_blockkey(const Eigen::Vector3d& _point) const {
        return AKIMapKey(coordinate_to_blockkey(_point(0)), coordinate_to_blockkey(_point(1)), coordinate_to_blockkey(_point(2)));
    }
    inline AKIMapKey coordinate_to_blockkey(double _x, double _y, double _z) const {
        return AKIMapKey(coordinate_to_blockkey(_x), coordinate_to_blockkey(_y), coordinate_to_blockkey(_z));
    }

protected:
    std::unordered_map<AKIMapKey, AKIMapNode*, AKIMapKey::KeyHash> node_array;

    double RESOLUTION;              // [m]
    double RESOLUTION_FACTOR;
    double FREE_SAMPLE_DISTANCE;    // [m]

    RTree rtree;
    double MAX_SEARCH_RANGE;        // [m]
    const double MAX_SENSING_RANGE = 8.0;   // [m]

    unsigned int BLOCK_DEPTH;
    unsigned int NUM_OF_CELLS_IN_BLOCK;
    Eigen::Vector3d BLOCK_TO_TEST_MIN;
    Eigen::Vector3d BLOCK_TO_TEST_MAX;
    Eigen::MatrixXd BLOCK_TO_TEST_POINTS;

    const unsigned int MAX_EPOCH     = 100;
    const double       TOLERANCE     = 0.001;
    const double       LEARNING_RATE = 0.01;

    const Eigen::Matrix3d COV_NOISE = Eigen::Matrix3d::Identity() * 0.001;

    const int TREE_MAX_VAL = 32768;

    const Eigen::MatrixXd TRANSFORMED_KERNEL_SUPPORT_BBX = (Eigen::MatrixXd(3, 8) <<
            1.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0,
            1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,
            1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0).finished();

    inline double get_search_radius(double _distance);
};

#endif //AKIMAP_AKIMAP_H
