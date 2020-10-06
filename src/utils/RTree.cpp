#include <utils/RTree.h>

RTree::RTree()
{
    m_root = new Node(0);
}

RTree::~RTree()
{
    Reset(); // Free, or reset node memory
}

void RTree::Insert(const Eigen::Vector3d& a_min, const Eigen::Vector3d& a_max, const int& a_dataId)
{
    Branch branch(a_min, a_max, a_dataId);

    InsertRect(branch, &m_root, 0);
}

int RTree::Search(const Eigen::Vector3d& a_min, const Eigen::Vector3d& a_max, std::vector<int>& a_context) const
{
    Rect rect(a_min, a_max);

    Search(m_root, &rect, a_context);

    return (int)a_context.size();
}

bool RTree::HasData(const Eigen::Vector3d &a_min, const Eigen::Vector3d &a_max) const
{
    Rect rect(a_min, a_max);

    return HasData(m_root, &rect);
}

void RTree::RemoveAll()
{
    // Delete all existing nodes
    Reset();
    m_root = new Node(0);
}

void RTree::Reset()
{
    // Delete all existing nodes
    RemoveAllRec(m_root);
}

void RTree::RemoveAllRec(Node* a_node)
{
    if(a_node->IsInternalNode()) { // This is an internal node in the tree
        for(int index = 0; index < a_node->m_count; ++index) {
            RemoveAllRec(a_node->m_branch[index].m_child);
        }
    }
    delete a_node;
}

// Inserts a new data rectangle into the index structure.
// Recursively descends tree, propagates splits back up.
// Returns 0 if node was not split.  Old node updated.
// If node was split, returns 1 and sets the pointer pointed to by
// new_node to point to the new node.  Old node updated to become one of two.
// The level argument specifies the number of steps up from the leaf
// level to insert; e.g. a data rectangle goes in at level = 0.
bool RTree::InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level)
{
    // recurse until we reach the correct level for the new record. data records
    // will always be called with a_level == 0 (leaf)
    if(a_node->m_level > a_level) {
        // Still above level for insertion, go down tree recursively
        Node* otherNode;

        // find the optimal branch for this record
        int index = PickBranch(&a_branch.m_rect, a_node);

        // recursively insert this record into the picked branch
        bool childWasSplit = InsertRectRec(a_branch, a_node->m_branch[index].m_child, &otherNode, a_level);
        if (!childWasSplit) {
            // Child was not split. Merge the bounding box of the new record with the
            // existing bounding box
            a_node->m_branch[index].m_rect = CombineRect(&a_branch.m_rect, &(a_node->m_branch[index].m_rect));
            return false;
        }
        else {
            // Child was split. The old branches are now re-partitioned to two nodes
            // so we have to re-calculate the bounding boxes of each node
            a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
            Branch branch;
            branch.m_child = otherNode;
            branch.m_rect = NodeCover(otherNode);

            // The old node is already a child of a_node. Now add the newly-created
            // node to a_node as well. a_node might be split because of that.
            return AddBranch(&branch, a_node, a_newNode);
        }
    }
    else {
        // We have reached level for insertion. Add rect, split if necessary
        return AddBranch(&a_branch, a_node, a_newNode);
    }
}


// Insert a data rectangle into an index structure.
// InsertRect provides for splitting the root;
// returns 1 if root was split, 0 if it was not.
// The level argument specifies the number of steps up from the leaf
// level to insert; e.g. a data rectangle goes in at level = 0.
// InsertRect2 does the recursion.
//
bool RTree::InsertRect(const Branch& a_branch, Node** a_root, int a_level)
{
    Node* newNode;
    if(InsertRectRec(a_branch, *a_root, &newNode, a_level)) { // Root split
        // Grow tree taller and new root
        Node* newRoot = new Node((*a_root)->m_level + 1);

        Branch branch;

        // add old root node as a child of the new root
        branch.m_rect = NodeCover(*a_root);
        branch.m_child = *a_root;
        AddBranch(&branch, newRoot, nullptr);

        // add the split node as a child of the new root
        branch.m_rect = NodeCover(newNode);
        branch.m_child = newNode;
        AddBranch(&branch, newRoot, nullptr);

        // set the new root as the root node
        *a_root = newRoot;

        return true;
    }

    return false;
}

// Find the smallest rectangle that includes all rectangles in branches of a node.
RTree::Rect RTree::NodeCover(Node* a_node)
{
    Eigen::MatrixXd bbx_min(3, a_node->m_count);
    Eigen::MatrixXd bbx_max(3, a_node->m_count);
    for(int index = 0; index < a_node->m_count; index++) {
        bbx_min.col(index) = a_node->m_branch[index].m_rect.m_min;
        bbx_max.col(index) = a_node->m_branch[index].m_rect.m_max;
    }
    return { bbx_min.rowwise().minCoeff(), bbx_max.rowwise().maxCoeff() };
}

// Add a branch to a node.  Split the node if necessary.
// Returns FALSE if node not split.  Old node updated.
// Returns TRUE if node split, sets *new_node to address of new node.
// Old node updated, becomes one of two.
bool RTree::AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode)
{
    if(a_node->m_count < MAXNODES) { // Split won't be necessary
        a_node->m_branch[a_node->m_count] = *a_branch;
        ++a_node->m_count;
        return false;
    }
    else {
        SplitNode(a_node, a_branch, a_newNode);
        return true;
    }
}

// Pick a branch.  Pick the one that will need the smallest increase
// in area to accomodate the new rectangle.  This will result in the
// least total area for the covering rectangles in the current node.
// In case of a tie, pick the one which was smaller before, to get
// the best resolution when searching.
int RTree::PickBranch(const Rect* a_rect, Node* a_node)
{
    bool firstTime = true;
    double increase;
    double bestIncr = -1.0f;
    double area;
    double bestArea = -1.0f;
    int best = -1;
    Rect tempRect;

    for(int index = 0; index < a_node->m_count; ++index) {
        Rect* curRect = &a_node->m_branch[index].m_rect;
        area = CalcRectVolume(curRect);
        tempRect = CombineRect(a_rect, curRect);
        increase = CalcRectVolume(&tempRect) - area;

        if((increase < bestIncr) || firstTime) {
            best = index;
            bestArea = area;
            bestIncr = increase;
            firstTime = false;
        }
        else if((increase == bestIncr) && (area < bestArea)) {
            best = index;
            bestArea = area;
            bestIncr = increase;
        }
    }
    return best;
}

// Combine two rectangles into larger one containing both
RTree::Rect RTree::CombineRect(const Rect* a_rectA, const Rect* a_rectB)
{
    return { a_rectA->m_min.cwiseMin(a_rectB->m_min), a_rectA->m_max.cwiseMax(a_rectB->m_max) };
}

// Split a node.
// Divides the nodes branches and the extra one between two nodes.
// Old node is one of the new ones, and one really new one is created.
// Tries more than one method for choosing a partition, uses best result.
void RTree::SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode)
{
    // Could just use local here, but member or external is faster since it is reused
    PartitionVars localVars;
    PartitionVars* parVars = &localVars;

    // Load all the branches into a buffer, initialize old node
    GetBranches(a_node, a_branch, parVars);

    // Find partition
    ChoosePartition(parVars, MINNODES);

    // Create a new node to hold (about) half of the branches
    *a_newNode = new Node(a_node->m_level);

    // Put branches from buffer into 2 nodes according to the chosen partition
    a_node->m_count = 0;
    LoadNodes(a_node, *a_newNode, parVars);
}

// Use one of the methods to calculate rectangle volume
double RTree::CalcRectVolume(const Rect* a_rect)
{
#ifdef RTREE_USE_SPHERICAL_VOLUME
    Eigen::Vector3d size = (a_rect->m_max - a_rect->m_min) * 0.5;
    double radius = std::sqrt(size.squaredNorm());
    return (radius * radius * radius * m_unitSphereVolume);   // 4.188790 == 4*pi/3
#else
    // Calculate the n-dimensional volume of a rectangle
    return (a_rect->m_max - a_rect->m_min).prod(); // Faster but can cause poor merges
#endif
}

// Load branch buffer with branches from full node plus the extra branch.
void RTree::GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars)
{
    // Load the branch buffer
    for(int index = 0; index < MAXNODES; ++index) {
        a_parVars->m_branchBuf[index] = a_node->m_branch[index];
    }
    a_parVars->m_branchBuf[MAXNODES] = *a_branch;
    a_parVars->m_branchCount = MAXNODES + 1;

    // Calculate rect containing all in the set
    a_parVars->m_coverSplit = a_parVars->m_branchBuf[0].m_rect;
    for(int index = 1; index < MAXNODES+1; ++index) {
        a_parVars->m_coverSplit = CombineRect(&a_parVars->m_coverSplit, &a_parVars->m_branchBuf[index].m_rect);
    }
    a_parVars->m_coverSplitArea = CalcRectVolume(&a_parVars->m_coverSplit);
}


// Method #0 for choosing a partition:
// As the seeds for the two groups, pick the two rects that would waste the
// most area if covered by a single rectangle, i.e. evidently the worst pair
// to have in the same group.
// Of the remaining, one at a time is chosen to be put in one of the two groups.
// The one chosen is the one with the greatest difference in area expansion
// depending on which group - the rect most strongly attracted to one group
// and repelled from the other.
// If one group gets too full (more would force other group to violate min
// fill requirement) then other group gets the rest.
// These last are the ones that can go in either group most easily.
void RTree::ChoosePartition(PartitionVars* a_parVars, int a_minFill)
{
    double biggestDiff = -1.0f;
    int group = -1, chosen = -1, betterGroup = -1;

    InitParVars(a_parVars, a_parVars->m_branchCount, a_minFill);
    PickSeeds(a_parVars);

    while (((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
           && (a_parVars->m_count[0] < (a_parVars->m_total - a_parVars->m_minFill))
           && (a_parVars->m_count[1] < (a_parVars->m_total - a_parVars->m_minFill)))
    {
        biggestDiff = -1.0;
        for(int index = 0; index < a_parVars->m_total; ++index) {
            if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index]) {
                Rect* curRect = &a_parVars->m_branchBuf[index].m_rect;
                Rect rect0 = CombineRect(curRect, &a_parVars->m_cover[0]);
                Rect rect1 = CombineRect(curRect, &a_parVars->m_cover[1]);
                double growth0 = CalcRectVolume(&rect0) - a_parVars->m_area[0];
                double growth1 = CalcRectVolume(&rect1) - a_parVars->m_area[1];
                double diff = growth1 - growth0;

                if(diff >= 0) {
                    group = 0;
                }
                else {
                    group = 1;
                    diff = -diff;
                }

                if(diff > biggestDiff) {
                    biggestDiff = diff;
                    chosen = index;
                    betterGroup = group;
                }
                else if((diff == biggestDiff) && (a_parVars->m_count[group] < a_parVars->m_count[betterGroup])) {
                    chosen = index;
                    betterGroup = group;
                }
            }
        }

        Classify(chosen, betterGroup, a_parVars);
    }

    // If one group too full, put remaining rects in the other
    if((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total) {
        if(a_parVars->m_count[0] >= a_parVars->m_total - a_parVars->m_minFill) {
            group = 1;
        }
        else {
            group = 0;
        }

        for(int index=0; index<a_parVars->m_total; ++index) {
            if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index]) {
                Classify(index, group, a_parVars);
            }
        }
    }
}

// Copy branches from the buffer into two nodes according to the partition.
void RTree::LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars)
{
    for(int index = 0; index < a_parVars->m_total; ++index) {
        int targetNodeIndex = a_parVars->m_partition[index];
        Node* targetNodes[] = { a_nodeA, a_nodeB };

        // It is assured that AddBranch here will not cause a node split.
        AddBranch(&a_parVars->m_branchBuf[index], targetNodes[targetNodeIndex], nullptr);
    }
}


// Initialize a PartitionVars structure.
void RTree::InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill)
{
    a_parVars->m_count[0] = a_parVars->m_count[1] = 0;
    a_parVars->m_area[0] = a_parVars->m_area[1] = 0.0;
    a_parVars->m_total = a_maxRects;
    a_parVars->m_minFill = a_minFill;
    for(int index=0; index < a_maxRects; ++index)
        a_parVars->m_partition[index] = PartitionVars::NOT_TAKEN;
}

void RTree::PickSeeds(PartitionVars* a_parVars)
{
    int seed0 = -1, seed1 = -1;
    double worst, waste;
    double area[MAXNODES+1];

    for(int index = 0; index < a_parVars->m_total; ++index) {
        area[index] = CalcRectVolume(&a_parVars->m_branchBuf[index].m_rect);
    }

    worst = -a_parVars->m_coverSplitArea - 1;
    for(int indexA = 0; indexA < a_parVars->m_total-1; ++indexA) {
        for(int indexB = indexA+1; indexB < a_parVars->m_total; ++indexB) {
            Rect oneRect = CombineRect(&a_parVars->m_branchBuf[indexA].m_rect, &a_parVars->m_branchBuf[indexB].m_rect);
            waste = CalcRectVolume(&oneRect) - area[indexA] - area[indexB];
            if(waste > worst) {
                worst = waste;
                seed0 = indexA;
                seed1 = indexB;
            }
        }
    }

    Classify(seed0, 0, a_parVars);
    Classify(seed1, 1, a_parVars);
}

// Put a branch in one of the groups.
void RTree::Classify(int a_index, int a_group, PartitionVars* a_parVars)
{
    a_parVars->m_partition[a_index] = a_group;

    // Calculate combined rect
    if (a_parVars->m_count[a_group] == 0)
        a_parVars->m_cover[a_group] = a_parVars->m_branchBuf[a_index].m_rect;
    else
        a_parVars->m_cover[a_group] = CombineRect(&a_parVars->m_branchBuf[a_index].m_rect, &a_parVars->m_cover[a_group]);

    // Calculate volume of combined rect
    a_parVars->m_area[a_group] = CalcRectVolume(&a_parVars->m_cover[a_group]);

    ++a_parVars->m_count[a_group];
}

// Decide whether two rectangles overlap.
bool RTree::Overlap(Rect* a_rectA, Rect* a_rectB) const
{
    for(int index = 0; index < 3; ++index) {
        if (a_rectA->m_min[index] > a_rectB->m_max[index] ||
            a_rectB->m_min[index] > a_rectA->m_max[index])
        {
            return false;
        }
    }
    return true;
}

// Search in an index tree or subtree for all data rectangles that overlap the argument rectangle.
void RTree::Search(Node* a_node, Rect* a_rect, std::vector<int>& a_context) const
{
    if(a_node->IsInternalNode()) {
        // This is an internal node in the tree
        for(int index = 0; index < a_node->m_count; ++index) {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect)) {
                Search(a_node->m_branch[index].m_child, a_rect, a_context);
            }
        }
    }
    else {
        // This is a leaf node
        for(int index = 0; index < a_node->m_count; ++index) {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect)) {
                a_context.push_back(a_node->m_branch[index].m_data);
            }
        }
    }
}

bool RTree::HasData(Node* a_node, Rect* a_rect) const
{
    if(a_node->IsInternalNode()) {
        // This is an internal node in the tree
        for(int index = 0; index < a_node->m_count; ++index) {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect)) {
                if(HasData(a_node->m_branch[index].m_child, a_rect))
                    return true;
            }
        }
    }
    else {
        // This is a leaf node
        for(int index = 0; index < a_node->m_count; ++index) {
            if(Overlap(a_rect, &a_node->m_branch[index].m_rect)) {
                return true;
            }
        }
    }
    return false;
}
