#ifndef RTREE_H
#define RTREE_H

#include <Eigen/Dense>
#include <utility>
#include <vector>

#define RTREE_USE_SPHERICAL_VOLUME // Better split classification, may be slower on some systems

/// \class RTree
/// Implementation of RTree, a multidimensional bounding rectangle tree.
/// Templated C++ version by Greg Douglas at Auran (http://www.auran.com)
/// http://superliminal.com/sources/sources.htm
///
/// The implementation is originated from the templated version.
/// This simplified version includes some functions that are used in AKIMap, and
/// the re-implementation uses Eigen library (http://eigen.tuxfamily.org) without a template class.
///
/// DATATYPE: int
/// ELEMTYPE: double
/// NUMDIMS: 3
/// ELEMTYPEREAL: double
///
/// NOTES: Inserting and removing data requires the knowledge of its constant Minimal Bounding Rectangle.
///        This version uses new/delete for nodes, but using a fixed size allocator can improve efficiency.
///
class RTree {
protected:
    struct Rect;
    struct Node;
    struct Branch;

public:
    static const int MAXNODES = 8;
    static const int MINNODES = MAXNODES / 2;

public:
    RTree();
    virtual ~RTree();

    /// Insert entry
    /// \param a_min Min of bounding rect
    /// \param a_max Max of bounding rect
    /// \param a_dataId Positive Id of data. Maybe zero, but negative numbers not allowed.
    void Insert(const Eigen::Vector3d& a_min, const Eigen::Vector3d& a_max, const int& a_dataId);

    /// Find all within search rectangle
    /// \param a_min Min of search bounding rect
    /// \param a_max Max of search bounding rect
    /// \param a_context Ids of entries found
    /// \return Returns the number of entries found
    int Search(const Eigen::Vector3d& a_min, const Eigen::Vector3d& a_max, std::vector<int>& a_context) const;

    /// Check the existence of data within search rectangle
    /// \param a_min Min of search bounding rect
    /// \param a_max Max of search bounding rect
    /// \return Returns the existence of data found
    bool HasData(const Eigen::Vector3d& a_min, const Eigen::Vector3d& a_max) const;

    /// Remove all entries from tree
    void RemoveAll();

protected:

    /// Minimal bounding rectangle (3-dimensional)
    struct Rect {
        Rect() : m_min(0.0, 0.0, 0.0), m_max(0.0, 0.0, 0.0) {}
        Rect(Eigen::Vector3d _min, Eigen::Vector3d _max) : m_min(std::move(_min)), m_max(std::move(_max)) {}

        Eigen::Vector3d m_min;    ///< Min dimensions of bounding box
        Eigen::Vector3d m_max;    ///< Max dimensions of bounding box
    };

    /// May be data or may be another subtree
    /// The parents level determines this.
    /// If the parents level is 0, then this is data
    struct Branch {
        Branch() : m_rect(), m_child(nullptr), m_data(-1) {}
        Branch(const Eigen::Vector3d& _min, const Eigen::Vector3d& _max, int _id) : m_rect(_min, _max), m_child(nullptr), m_data(_id) {}

        Rect  m_rect;    ///< Bounds
        Node* m_child;   ///< Child node
        int   m_data;    ///< Data Id
    };

    /// Node for each branch level
    struct Node {
        explicit Node(int _level) : m_count(0), m_level(_level) {}

        bool IsInternalNode()   { return (m_level > 0); }   // Not a leaf, but a internal node
        bool IsLeaf()           { return (m_level == 0); }  // A leaf, contains data

        int m_count;                 ///< Count
        int m_level;                 ///< Leaf is zero, others positive
        Branch m_branch[MAXNODES];   ///< Branch
    };

    /// Variables for finding a split partition
    struct PartitionVars {
        enum { NOT_TAKEN = -1 }; // indicates that position

        int m_partition[MAXNODES+1];
        int m_total;
        int m_minFill;
        int m_count[2];
        Rect m_cover[2];
        double m_area[2];

        Branch m_branchBuf[MAXNODES+1];
        int m_branchCount;
        Rect m_coverSplit;
        double m_coverSplitArea;
    };

    bool InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level);
    bool InsertRect(const Branch& a_branch, Node** a_root, int a_level);

    bool HasData(Node* a_node, Rect* a_rect) const;

    void Search(Node* a_node, Rect* a_rect, std::vector<int>& a_context) const;

    Rect NodeCover(Node* a_node);
    bool AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode);
    int PickBranch(const Rect* a_rect, Node* a_node);
    void SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode);
    void GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars);
    void ChoosePartition(PartitionVars* a_parVars, int a_minFill);
    void LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars);
    void InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill);
    void PickSeeds(PartitionVars* a_parVars);
    void Classify(int a_index, int a_group, PartitionVars* a_parVars);

    bool Overlap(Rect* a_rectA, Rect* a_rectB) const;
    void RemoveAllRec(Node* a_node);
    void Reset();

    inline Rect CombineRect(const Rect* a_rectA, const Rect* a_rectB);
    inline double CalcRectVolume(const Rect* a_rect);

    Node* m_root;                                  ///< Root of tree

    const double m_unitSphereVolume = 4.188790;    ///< Unit sphere constant for required number of dimensions
};

#endif //RTREE_H

