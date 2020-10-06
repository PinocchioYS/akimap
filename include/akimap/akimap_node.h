#ifndef AKIMAP_AKIMAP_NODE_H
#define AKIMAP_AKIMAP_NODE_H

#include <fstream>

class AKIMapNode {
public:
    // Constructor and Destructor
    AKIMapNode() : AKIMapNode(0.001, 0.001) {}
    AKIMapNode(float _alpha, float _beta) : alpha(_alpha), beta(_beta) {}
//    AKIMapNode(const AKIMapNode& _other) : alpha(_other.alpha), beta(_other.beta) {}

    /*
     * Update the kernel estimation from occupancy sample
     *
     * @param value: kernel estimation value
     * @param signal: kernel estimation signal
     */
    void add_kernel_estimation(const float& _value, const float& _signal) { alpha += _value;   beta += (_signal - _value); }

    /*
     * Compute the occupancy probability of the node
     *
     * @return occupancy probability
     */
    double get_occupancy() const { return alpha / (alpha + beta); }

    // IO operations
    void write(std::ofstream& _stream) {
        _stream.write((char*)&alpha, sizeof(alpha));
        _stream.write((char*)&beta, sizeof(beta));
    }

    void read(std::ifstream& _stream) {
        _stream.read((char*)&alpha, sizeof(alpha));
        _stream.read((char*)&beta, sizeof(beta));
    }

protected:
    float alpha;    // Kernel estimation from occupied sample
    float beta;     // Kernel estimation from free sample
};

#endif //AKIMAP_AKIMAP_NODE_H
