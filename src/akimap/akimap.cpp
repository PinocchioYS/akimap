#include <akimap/akimap.h>

#include <unordered_set>
#include <random>

#ifdef OPENMP
#include <omp.h>
#endif

#include <flann/flann.h>

AKIMap::AKIMap(double _resolution, double _free_sample_distance)
: RESOLUTION(_resolution), RESOLUTION_FACTOR(1.0f/_resolution), FREE_SAMPLE_DISTANCE(_free_sample_distance), MAX_SEARCH_RANGE(_free_sample_distance), BLOCK_DEPTH(2)
{
    NUM_OF_CELLS_IN_BLOCK = (unsigned int)pow(2, BLOCK_DEPTH);

    BLOCK_TO_TEST_MIN = Eigen::Vector3d::Constant(0.0);
    BLOCK_TO_TEST_MAX = Eigen::Vector3d::Constant((NUM_OF_CELLS_IN_BLOCK-1) * RESOLUTION);

    BLOCK_TO_TEST_POINTS = Eigen::MatrixXd(3, (unsigned int)pow(NUM_OF_CELLS_IN_BLOCK, 3));
    Eigen::VectorXd test_point_list = Eigen::VectorXd::LinSpaced(NUM_OF_CELLS_IN_BLOCK, BLOCK_TO_TEST_MIN[0], BLOCK_TO_TEST_MAX[0]);
    unsigned int insertion = 0;
    for(int x = 0; x < NUM_OF_CELLS_IN_BLOCK; x++) {
        for(int y = 0; y < NUM_OF_CELLS_IN_BLOCK; y++) {
            for(int z = 0; z < NUM_OF_CELLS_IN_BLOCK; z++) {
                BLOCK_TO_TEST_POINTS.col(insertion++) << test_point_list[x], test_point_list[y], test_point_list[z];
            }
        }
    }
}

AKIMap::~AKIMap()
{
    for(const auto& node : node_array)
        delete node.second;
}

void AKIMap::insert_pointcloud(const Eigen::MatrixXd& _pointcloud, const Eigen::Vector3d& _origin)
{
    // 1. Make the occupancy samples    ================================================================================
    Eigen::MatrixXd samples;
    Eigen::VectorXi labels;
    make_occupancy_samples(_pointcloud, _origin, samples, labels);

    // 2. Optimize the adaptive kernel bandwidths    ===================================================================
    Eigen::MatrixXd transforms; // NOTE: distance of two points using bandwidth matrix == distance of two points in transformed space
    Eigen::VectorXd weights;
    compute_kernel_bandwidths(samples, labels, transforms, weights);

    // 3. Update the kernel estimations from the occupancy samples    ==================================================
    update_adaptive_kernel_estimations(samples, labels, transforms, weights);
}

void AKIMap::make_occupancy_samples(const Eigen::MatrixXd& _pointcloud, const Eigen::Vector3d& _origin,
                                    Eigen::MatrixXd& _samples, Eigen::VectorXi& _labels)
{
    // 0. Initialize a down sampler
    std::unordered_set<AKIMapKey, AKIMapKey::KeyHash> down_sampler;

    // 1. Generate the occupied samples
    Eigen::MatrixXd occupieds(3, _pointcloud.cols());
    {
        unsigned int insertion = 0;
        for(unsigned int i = 0; i < _pointcloud.cols(); i++) {
            const Eigen::Vector3d& point = _pointcloud.col(i);
            const AKIMapKey& key = coordinate_to_key(point);
            if(down_sampler.find(key) == down_sampler.end()) {
                occupieds.col(insertion++) = point;
                down_sampler.insert(key);
            }
        }
        occupieds.conservativeResize(Eigen::NoChange, insertion);
    }

    // 2. Generate the free samples
    Eigen::MatrixXd frees;
    {
        // Initialize a random generator and the size of free samples
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, FREE_SAMPLE_DISTANCE);

        Eigen::MatrixXd direction = occupieds.colwise() - _origin;
        Eigen::VectorXd length = direction.colwise().norm().array();
        direction.colwise().normalize();
        Eigen::VectorXi num_of_free_samples = (length / FREE_SAMPLE_DISTANCE).cast<int>();

        frees.resize(3, num_of_free_samples.sum() + occupieds.cols() + 1);

        // Insert the origin into free samples
        unsigned int insertion = 0;
        down_sampler.insert(coordinate_to_key(_origin));
        frees.col(insertion++) = _origin;

        // Make a random sample on a sensor ray
        for(unsigned int i = 0; i < occupieds.cols(); i++) {
            double dist = FREE_SAMPLE_DISTANCE * 0.5;
            while (dist < length(i)) {
                double random_sample_dist = dist + dis(gen);
                if(random_sample_dist >= length(i))
                    break;

                Eigen::Vector3d point = _origin + random_sample_dist * direction.col(i);
                const AKIMapKey& key = coordinate_to_key(point);
                if(down_sampler.find(key) == down_sampler.end()) {
                    frees.col(insertion++) = point;
                    down_sampler.insert(key);
                }

                dist += FREE_SAMPLE_DISTANCE;
            }
        }
        frees.conservativeResize(Eigen::NoChange, insertion);
    }

    // 3. Merge the occupied and free samples
    _samples.resize(3, frees.cols() + occupieds.cols());
    _samples.leftCols(frees.cols()) = frees;
    _samples.rightCols(occupieds.cols()) = occupieds;
    _labels.resize(_samples.cols());
    _labels.head(frees.cols()).setZero();
    _labels.tail(occupieds.cols()).setOnes();
}

void AKIMap::compute_kernel_bandwidths(const Eigen::MatrixXd& _samples, const Eigen::VectorXi& _labels,
                                       Eigen::MatrixXd& _transforms, Eigen::VectorXd& _weights)
{
    // 0. Compute the covariance matrix using only positive neighbors
    unsigned int NUM_OF_SAMPLES = _samples.cols();
    unsigned int NUM_OF_OCCUPIEDS = _labels.sum();
    unsigned int NUM_OF_FREES = NUM_OF_SAMPLES - NUM_OF_OCCUPIEDS;
    _transforms.resize(3, 3 * NUM_OF_SAMPLES);
    _weights.resize(NUM_OF_SAMPLES);

    // 0. Initialize K-D trees of occupied and free samples
    flann::Matrix<double> flann_samples[2] = { flann::Matrix<double>(const_cast<double*>(_samples.data()), (size_t)NUM_OF_FREES, (size_t)_samples.rows()),
                                               flann::Matrix<double>(const_cast<double*>(_samples.data()+_samples.rows()*NUM_OF_FREES), (size_t)NUM_OF_OCCUPIEDS, (size_t)_samples.rows()) };
    flann::Index<flann::L2<double>>* kdtree[2] = { new flann::Index<flann::L2<double>>(flann::KDTreeSingleIndexParams()),
                                                   new flann::Index<flann::L2<double>>(flann::KDTreeSingleIndexParams()) };
    kdtree[0]->buildIndex(flann_samples[0]);
    kdtree[1]->buildIndex(flann_samples[1]);

    const Eigen::Vector3d& origin = _samples.col(0);
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(unsigned int i = 0; i < _samples.cols(); i++) {
        const Eigen::Vector3d& point = _samples.col(i);
        const int pos_label = _labels[i];
        const int neg_label = 1 - pos_label;

        // NOTE: FLANN K-D Tree requires a search radius as squared value
        flann::Matrix<double> flann_query(const_cast<double*>(point.data()), 1, 3);
        double squared_search_radius = std::pow(get_search_radius((point - origin).norm()), 2);

        Eigen::MatrixXd positives;
        {
            std::vector<std::vector<int>> pos_idx;
            std::vector<std::vector<double>> pos_d;
            kdtree[pos_label]->radiusSearch(flann_query, pos_idx, pos_d, (float)squared_search_radius, flann::SearchParams(128, 0.0, false));

            // Minimum support region
            if (pos_idx[0].size() <= 2) {
                _transforms.block<3, 3>(0, 3 * i) = Eigen::Matrix3d::Identity() * RESOLUTION_FACTOR;
                _weights(i) = std::pow(RESOLUTION_FACTOR, 3);
                continue;
            }

            positives.resize(3, pos_idx[0].size());
            const auto& flann_positive_samples = flann_samples[pos_label];
            for (unsigned int n = 0; n < pos_idx[0].size(); n++) {
                const int& idx = pos_idx[0][n];
                positives.col(n) << flann_positive_samples[idx][0], flann_positive_samples[idx][1], flann_positive_samples[idx][2];
            }
        }

        // Decompose the covariance into the bandwidth matrix
        Eigen::MatrixXd centered = positives.transpose().rowwise() - point.transpose();
        Eigen::Matrix3d cov = (centered.adjoint() * centered) / (positives.cols() - 2) + COV_NOISE;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
        Eigen::Vector3d length = eig.eigenvalues().cwiseSqrt();
        Eigen::Matrix3d transform = eig.eigenvectors() * length.cwiseInverse().asDiagonal();
        double weight = 1.0 / length.prod();

        // SCALE OPTIMIZATION ==========================================================================================
        Eigen::MatrixXd negatives;
        {
            std::vector<std::vector<int>> neg_idx;
            std::vector<std::vector<double>> neg_d;
            kdtree[neg_label]->radiusSearch(flann_query, neg_idx, neg_d, (float)squared_search_radius, flann::SearchParams(128, 0.0, false));

            if(neg_idx[0].empty()) {
                _transforms.block<3, 3>(0, 3*i) = transform;
                _weights(i) = weight;
                continue;
            }

            negatives.resize(3, neg_idx[0].size());
            const auto& flann_negative_samples = flann_samples[neg_label];
            for(unsigned int n = 0; n < neg_idx[0].size(); n++) {
                const int& idx = neg_idx[0][n];
                negatives.col(n) << flann_negative_samples[idx][0], flann_negative_samples[idx][1], flann_negative_samples[idx][2];
            }
        }

        double opt_scale = optimize_kernel_scale(point, positives, negatives, transform, weight, RESOLUTION / length(2));
        _transforms.block<3, 3>(0, 3 * i) = transform / opt_scale;
        _weights(i) = weight / std::pow(opt_scale, 3);

    }

    delete kdtree[0];
    delete kdtree[1];
}

double AKIMap::optimize_kernel_scale(const Eigen::Vector3d& _center, const Eigen::MatrixXd& _positives, const Eigen::MatrixXd& _negatives,
                                     const Eigen::Matrix3d& _init_transform, const double _init_weight, const double _MIN_SCALE)
{
    // 0. Pre-compute distances from kernel center to test points
    Eigen::VectorXd init_r, target;
    int NUM_OF_POSITIVES = 0, NUM_OF_NEGATIVES = 0;
    {
        Eigen::MatrixXd test_samples(3, _negatives.cols() + _positives.cols());
        test_samples.leftCols(_negatives.cols()) = _negatives;
        test_samples.rightCols(_positives.cols()) = _positives;

        Eigen::MatrixXd centered = test_samples.colwise() - _center;
        Eigen::VectorXd r = (_init_transform.transpose() * centered).colwise().norm();

        init_r.resize(r.rows());
        int insertion = 0;
        for(int i = 0; i < r.rows(); i++) {
            if(r[i] < 1.0) {
                init_r[insertion++] = r[i];
                i < _negatives.cols() ? NUM_OF_NEGATIVES++ : NUM_OF_POSITIVES++;
            }
        }
        init_r.conservativeResize(insertion);

        if(NUM_OF_NEGATIVES == 0 || NUM_OF_POSITIVES == 0)
            return 1.0;

        target.resize(init_r.rows());
        target.head(NUM_OF_NEGATIVES).setZero();
        const Eigen::VectorXd& positive_r = init_r.tail(NUM_OF_POSITIVES);
        target.tail(NUM_OF_POSITIVES) = (((2.0 + (positive_r * 2.0 * 3.1415926).array().cos()) * (1.0 - positive_r.array()) / 3.0) +
                                         (positive_r * 2.0 * 3.1415926).array().sin() / (2.0 * 3.1415926));
        target.tail(NUM_OF_POSITIVES).cwiseMax(0.0);
    }

    double scale = 1.0;
    for(int epoch = 1; epoch <= MAX_EPOCH; epoch++) {
        Eigen::VectorXd scaled_r = init_r / scale;                  // r(s) = r_init / s
        Eigen::VectorXd rotated_r = scaled_r * 2.0 * 3.1415926;     // 2 * pi * r(s)
        Eigen::VectorXd sin_value = rotated_r.array().sin();        // sin ( 2 * pi * r(s) ) == sin_v(s)
        Eigen::VectorXd cos_value = rotated_r.array().cos();        // cos ( 2 * pi * r(s) ) == cos_v(s)

        // k(s) = ( 2.0 + cos_v(s) ) * ( 1.0 - r(s) ) / 3.0 + sin_v(s) / ( 2 * pi )
        Eigen::VectorXd scaled_k_value = ((2.0 + cos_value.array()) * (1.0 - scaled_r.array()) / 3.0).array() + (sin_value / (2.0 * 3.1415926)).array();
        scaled_k_value = scaled_k_value.cwiseMax(0.0);

        // Scaled weight and gradient of scaled weight
        double scaled_w = _init_weight / std::pow(scale, 3);     // w(s)  = w0 / s^3
        double scaled_w_grad = -1.5 * scaled_w / scale;             // w(s)' = (w0 / 2) * (-3 / s^4) = -1.5 * w0 / s^4 = -1.5 * w(s) / s

        // k(s)' = ( 1.0 - r(s) ) * ( 2 * pi * r(s) ) * sin_v(s) / ( 3.0 * s )
        //       + ( 2.0 + cos_v(s) ) * r(s) / ( 3.0 * s )
        //       - ( r(s) * cos_v(s) ) / s
        Eigen::VectorXd scaled_k_grad_1 = Eigen::VectorXd((-scaled_r).array() + 1.0).cwiseProduct(rotated_r).cwiseProduct(sin_value) / (3.0 * scale);
        Eigen::VectorXd scaled_k_grad_2 = Eigen::VectorXd(cos_value.array() + 2.0).cwiseProduct(scaled_r) / (3.0 * scale);
        Eigen::VectorXd scaled_k_grad_3 = scaled_r.cwiseProduct(cos_value) / scale;
        Eigen::VectorXd scaled_k_grad = scaled_k_grad_1 + scaled_k_grad_2 - scaled_k_grad_3;    // k(s)'

        Eigen::VectorXd estimation_error = target - scaled_k_value;                                     // err(s) = t - k(s)
        double kernel_gradient_sum = (estimation_error.cwiseProduct(scaled_k_grad)).sum();              // SUM( err(s) * k(s)' )
        double inference_error_square_sum = Eigen::VectorXd(estimation_error.array().square()).sum();   // SUM( err(s)^2 )

        // Gradient(s) = w(s)' * SUM( e(s)^2 ) - w(s) * SUM( e(s) * k(s)' )
        double gradient = (scaled_w_grad * inference_error_square_sum - scaled_w * kernel_gradient_sum);
        double error = 0.5 * scaled_w * inference_error_square_sum;

        double new_scale = std::min(1.0, std::max(scale - LEARNING_RATE * gradient / epoch, _MIN_SCALE));

        // Early stop of optimization
        if(std::abs(gradient) < TOLERANCE || std::abs(error) < TOLERANCE || std::abs(scale - new_scale) < TOLERANCE)
            break;

        scale = new_scale;
    }

    return scale;
}

void AKIMap::update_adaptive_kernel_estimations(const Eigen::MatrixXd& _samples, const Eigen::VectorXi& _labels,
                                                const Eigen::MatrixXd& _transforms, const Eigen::VectorXd& _weights)
{
    // 0. Initialize a R-Tree consisting of anisotropic kernel supports ================================================
    // NOTE: a bounding box of kernel support becomes a search region of R-Tree.
    for(int i = 0; i < _samples.cols(); i++) {
        const Eigen::Matrix3d& inv_transform = _transforms.block<3, 3>(0, 3*i).transpose().inverse();
        Eigen::MatrixXd kernel_support_bbx = (inv_transform * TRANSFORMED_KERNEL_SUPPORT_BBX).colwise() + _samples.col(i);
        Eigen::Vector3d min_bbx = kernel_support_bbx.rowwise().minCoeff();
        Eigen::Vector3d max_bbx = kernel_support_bbx.rowwise().maxCoeff();
        rtree.Insert(min_bbx, max_bbx, i);
    }

    // 1. Find a set of the blocks including the test cells to be updated ==============================================
    std::vector<AKIMapKey> test_block_keys;
    {
        std::vector<AKIMapKey> block_keys;
        AKIMapKey lim_min_key = coordinate_to_blockkey(_samples.rowwise().minCoeff().array() - MAX_SEARCH_RANGE);
        AKIMapKey lim_max_key = coordinate_to_blockkey(_samples.rowwise().maxCoeff().array() + MAX_SEARCH_RANGE);

        for(AKIMapKey::key_type kx = lim_min_key[0]; kx <= lim_max_key[0]; kx += NUM_OF_CELLS_IN_BLOCK) {
            for(AKIMapKey::key_type ky = lim_min_key[1]; ky <= lim_max_key[1]; ky += NUM_OF_CELLS_IN_BLOCK) {
                for(AKIMapKey::key_type kz = lim_min_key[2]; kz <= lim_max_key[2]; kz += NUM_OF_CELLS_IN_BLOCK) {
                    block_keys.emplace_back(kx, ky, kz);
                }
            }
        }

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int i = 0; i < block_keys.size(); i++) {
            const AKIMapKey& block_key = block_keys[i];
            Eigen::Vector3d block_offset = key_to_coordinate(block_key);
            Eigen::Vector3d test_min = BLOCK_TO_TEST_MIN + block_offset;
            Eigen::Vector3d test_max = BLOCK_TO_TEST_MAX + block_offset;

            if(rtree.HasData(test_min, test_max))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_block_keys.push_back(block_key);
            }
        }
    }

    // 2. Update the kernel estimations to the test cells ==============================================================
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i = 0; i < test_block_keys.size(); i++) {
        // Compute the test points, which are centers of cells, associated with a block
        const AKIMapKey& block_key = test_block_keys[i];
        Eigen::MatrixXd test_points = BLOCK_TO_TEST_POINTS.colwise() + key_to_coordinate(block_key);

        // Update the kernel estimations to each test cell
        for(int j = 0; j < test_points.cols(); j++) {
            const Eigen::Vector3d& test_point = test_points.col(j);

            // Search the occupancy samples that affects the kernel estimations to the cell
            std::vector<int> indices;
            rtree.Search(test_point, test_point, indices);
            if(indices.empty())
                continue;

            // Compute the kernel estimations from the occupancy samples
            int num_of_samples = indices.size();
            Eigen::VectorXd weights_of_samples(num_of_samples);
            Eigen::VectorXd labels_of_samples(num_of_samples);
            Eigen::VectorXd kernel_values(num_of_samples);
            for(int n = 0; n < kernel_values.rows(); n++) {
                const int& index = indices[n];
                kernel_values(n)      = (_transforms.block<3, 3>(0, 3*index).transpose() * (test_point - _samples.col(index))).norm();
                weights_of_samples(n) = _weights[index];
                labels_of_samples(n)  = (double)_labels[index];
            }

            // A sparse kernel with an adaptive bandwidth
            kernel_values = (((2.0 + (kernel_values * 2.0 * 3.1415926).array().cos()) * (1.0 - kernel_values.array()) / 3.0) +
                             (kernel_values * 2.0 * 3.1415926).array().sin() / (2.0 * 3.1415926)).matrix();
            kernel_values = kernel_values.cwiseMax(0.0);
            // Occupancy estimation
            double value = kernel_values.transpose() * weights_of_samples.cwiseProduct(labels_of_samples);
            double signal = kernel_values.transpose() * weights_of_samples;

            if(signal > 0.0) {
                AKIMapKey key = coordinate_to_key(test_point);
                AKIMapNode* node = nullptr;
#ifdef OPENMP
#pragma omp critical
#endif
                {
                    if(node_array.find(key) == node_array.end()) {
                        node = new AKIMapNode();
                        node_array.insert(std::pair<AKIMapKey, AKIMapNode*>(key, node));
                    }
                    else{
                        node = node_array[key];
                    }
                }
                node->add_kernel_estimation((float)value, (float)signal);
            }
        }
    }

    rtree.RemoveAll();
}

inline double AKIMap::get_search_radius(double _distance)
{
    return std::min((MAX_SEARCH_RANGE - RESOLUTION) / MAX_SENSING_RANGE * _distance + RESOLUTION, MAX_SEARCH_RANGE);
}