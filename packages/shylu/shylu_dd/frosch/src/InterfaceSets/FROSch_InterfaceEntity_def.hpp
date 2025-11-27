// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_INTERFACEENTITY_DEF_HPP
#define _FROSCH_INTERFACEENTITY_DEF_HPP

#include "FROSch_Output.h"
#include "FROSch_Tools_decl.hpp"
#include "Kokkos_MathematicalConstants.hpp"
#include "Teuchos_ArrayRCPDecl.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_ScalarTraitsDecl.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include "Xpetra_MultiVector_decl.hpp"
#include <FROSch_InterfaceEntity_decl.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <unordered_set>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO>
    bool Node<SC,LO,GO>::operator< (const Node &n) const
    {
        return NodeIDGlobal_<n.NodeIDGlobal_;
    }

    template <class SC,class LO,class GO>
    bool Node<SC,LO,GO>::operator== (const Node &n) const
    {
        return NodeIDGlobal_==n.NodeIDGlobal_;
    }


    template <class SC,class LO,class GO,class NO>
    InterfaceEntity<SC,LO,GO,NO>::InterfaceEntity(EntityType type,
                                                  UN dofsPerNode,
                                                  UN multiplicity,
                                                  const int *subdomains,
                                                  EntityFlag flag) :
    Type_ (type),
    Flag_ (flag),
    SubdomainsVector_ (multiplicity),
    DofsPerNode_ (dofsPerNode),
    Multiplicity_ (multiplicity)
    {
        for (UN i=0; i<multiplicity; i++) {
            SubdomainsVector_[i] = subdomains[i];
        }
        sortunique(SubdomainsVector_);

        Ancestors_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        Offspring_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        Roots_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
    }

    template <class SC,class LO,class GO,class NO>
    InterfaceEntity<SC,LO,GO,NO>::~InterfaceEntity()
    {

    } // Do we need sth here?

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(LO nodeIDGamma,
                                              LO nodeIDLocal,
                                              GO nodeIDGlobal,
                                              UN nDofs,
                                              const LOVecPtr dofsGamma,
                                              const LOVecPtr dofsLocal,
                                              const GOVecPtr dofsGlobal)
    {
        FROSCH_ASSERT(nDofs<=DofsPerNode_,"nDofs>NumDofs_.");

        FROSCH_ASSERT(dofsGamma.size()==nDofs,"dofIDs.size()!=nDofs");
        FROSCH_ASSERT(dofsLocal.size()==nDofs,"dofIDs.size()!=nDofs");
        FROSCH_ASSERT(dofsGlobal.size()==nDofs,"dofIDs.size()!=nDofs");

        Node<SC,LO,GO> node;

        node.NodeIDGamma_ = nodeIDGamma;
        node.NodeIDLocal_ = nodeIDLocal;
        node.NodeIDGlobal_ = nodeIDGlobal;

        node.DofsGamma_.deepCopy(dofsGamma());
        node.DofsLocal_.deepCopy(dofsLocal());
        node.DofsGlobal_.deepCopy(dofsGlobal());

        NodeVector_.push_back(node);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(const NodePtr &node)
    {
        return addNode(*node);
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addNode(const Node<SC,LO,GO> &node)
    {
        FROSCH_ASSERT(node.DofsGamma_.size()<=DofsPerNode_,"node.DofsGamma_ is too large.");
        FROSCH_ASSERT(node.DofsLocal_.size()<=DofsPerNode_,"node.DofsLocal_ is too large.");
        FROSCH_ASSERT(node.DofsGlobal_.size()<=DofsPerNode_,"node.DofsGlobal_ is too large.");

        NodeVector_.push_back(node);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetGlobalDofs(UN iD,
                                                      UN nDofs,
                                                      UN *dofIDs,
                                                      GO *dofsGlobal)
    {
        FROSCH_ASSERT(iD<getNumNodes(),"iD=>getNumNodes()");
        FROSCH_ASSERT(nDofs<=DofsPerNode_,"nDofs>DofsPerNode_.");

        for (unsigned i=0; i<nDofs; i++) {
            if (dofIDs[i]<=DofsPerNode_) {
                NodeVector_[iD].DofsGlobal_[dofIDs[i]] = dofsGlobal[dofIDs[i]];
            } else {
                FROSCH_ASSERT(false,"dofIDs[i] is out of range.");
            }
        }

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::removeNode(UN iD)
    {
        FROSCH_ASSERT(iD<getNumNodes(),"iD=>getNumNodes()");
        NodeVector_.erase(NodeVector_.begin()+iD);
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::removeNode(const Node<SC, LO, GO> &node, const int rank)
    {
        int printRank = 0;
        auto it = std::lower_bound(NodeVector_.begin(), NodeVector_.end(), node);
        // lower_bound returns the location of the first element greater than or equal to the input. It does not check
        // whether they are equal so we have to do that here.
        if (it != NodeVector_.end() && *it == node) {
            if (rank == printRank) {
                std::cout << "==> in removeNode removing (" << this->getGlobalNodeID(std::distance(NodeVector_.begin(), it)) << ", " << this->getLocalNodeID(std::distance(NodeVector_.begin(), it)) << ")" << std::endl << std::flush;
                if (node == *it) {
                    std::cout << "\t yes the nodes are equal" << std::endl << std::flush;
                } else {
                    std::cout << "\t no the nodes are not equal" << std::endl << std::flush;
                }               
            }
            NodeVector_.erase(it);
            return 0;
        } else {
            if (rank == printRank) {
                std::cout << "==> in removeNode but node not found" << std::endl << std::flush;
            }
            return -1;
        }
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::sortByGlobalID()
    {
        sortunique(NodeVector_);
        return 0;
    }

    template <class SC, class LO, class GO, class NO> int InterfaceEntity<SC, LO, GO, NO>::sortUniqueByLocalID() {
        // Sort not unique
        std::sort(NodeVector_.begin(), NodeVector_.end(),
                  [](const Node<SC, LO, GO> &a, const Node<SC, LO, GO> &b) { return a.NodeIDLocal_ < b.NodeIDLocal_; });
        // Move duplicate objects to end of vector
        auto it = std::unique(
            NodeVector_.begin(), NodeVector_.end(),
            [](const Node<SC, LO, GO> &a, const Node<SC, LO, GO> &b) { return a.NodeIDLocal_ == b.NodeIDLocal_; });
        // Remove the duplicate objects
        NodeVector_.erase(it, NodeVector_.end());
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::reindexGammaID(){
        for (int i = 0; i < NodeVector_.size(); i++){
            NodeVector_[i].NodeIDGamma_ = i;
            for (int j = 0; j < DofsPerNode_; j++) {
                NodeVector_[i].DofsGamma_[j] = i*DofsPerNode_ + j;
            }
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setUniqueID(GO uniqueID)
    {
        UniqueID_ = uniqueID;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setLocalID(LO localID)
    {
        LocalID_ = localID;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setRootID(LO rootID)
    {
        RootID_ = rootID;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setLeafID(LO leafID)
    {
        LeafID_ = leafID;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::setUniqueIDToFirstGlobalID()
    {
        UniqueID_ = NodeVector_[0].NodeIDGlobal_;
        return 0;
    }

    template <class SC, class LO, class GO, class NO>
    int InterfaceEntity<SC, LO, GO, NO>::setGammaIDs(const int i, const LO gammaNodeID) {
        NodeVector_[i].NodeIDGamma_ = gammaNodeID;
        for (int j = 0; j < DofsPerNode_; j++) {
            NodeVector_[i].DofsGamma_[j] = gammaNodeID * DofsPerNode_ + j;
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetEntityType(EntityType type)
    {
        Type_ = type;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::resetEntityFlag(EntityFlag flag)
    {
        Flag_ = flag;
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::findAncestorsInSet(EntitySetPtr entitySet)
    {
        EntitySetPtr ancestors(new EntitySet<SC,LO,GO,NO>(*entitySet));
        IntVec tmpVector;
        // Remove all entities that are not an ancestor, either because they have a lower or equal multiplicity (just being sure),
        // or because they are not part of all the same subdomains (the "normal" case). An ancestor should belong to a superset of subdomains.
        for (UN i=0; i<Multiplicity_; i++) {
            UN length = ancestors->getNumEntities();
            for (UN j=0; j<length; j++) {
                tmpVector = ancestors->getEntity(length-1-j)->getSubdomainsVector();
                if (ancestors->getEntity(length-1-j)->getMultiplicity()<=this->getMultiplicity() || !binary_search(tmpVector.begin(),tmpVector.end(),SubdomainsVector_[i])) {
                    ancestors->removeEntity(length-1-j);
                }
            }
        }

        for (UN i=0; i<ancestors->getNumEntities(); i++) {
            Ancestors_->addEntity(ancestors->getEntity(i));
        }
        Ancestors_->sortUnique();

        // this is offspring of each ancestor
        for (UN i=0; i<Ancestors_->getNumEntities(); i++) {
            InterfaceEntityPtr thisEntity = rcpFromRef(*this);
            FROSCH_ASSERT(!thisEntity.is_null(),"FROSch::InterfaceEntity: thisEntity.is_null()");
            Ancestors_->getEntity(i)->addOffspring(thisEntity);
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearAncestors()
    {
        Ancestors_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::addOffspring(InterfaceEntityPtr interfaceEntity)
    {
        Offspring_->addEntity(interfaceEntity);
        Offspring_->sortUnique();
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearOffspring()
    {
        Offspring_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::findRoots()
    {
        // PURPOSE: Recursively find the root entities (entities with no ancestors) in the hierarchy.
        // Root entities are the "top-level" entities that have no parents in the inheritance tree.
        // The hierarchy is ordered from bottom to top: roots -> leaves, where entities belonging
        // to more subdomains are ancestors of entities belonging to fewer subdomains.
        // This results in ancestors being subsets of offspring (e.g., vertices -> edges -> faces).
        if (Roots_->getNumEntities()) {
            FROSCH_ASSERT(Ancestors_->getNumEntities()!=0,"Ancestors_->getNumEntities()==0");
            return Roots_;
        }
        for (UN i=0; i<Ancestors_->getNumEntities(); i++) {
            EntitySetPtr tmpRoots = Ancestors_->getEntity(i)->findRoots();
            if (tmpRoots.is_null()) {
                FROSCH_ASSERT(Ancestors_->getEntity(i)->getAncestors()->getNumEntities()==0,"EntityVector_[i]->getAncestors()->getNumEntities()!=0");
                Roots_->addEntity(Ancestors_->getEntity(i));
            } else {
                FROSCH_ASSERT(Ancestors_->getEntity(i)->getAncestors()->getNumEntities()!=0,"EntityVector_[i]->getAncestors()->getNumEntities()==0");
                FROSCH_ASSERT(tmpRoots->getNumEntities()>0,"tmpRoots->getNumEntities()<=0");
                Roots_->addEntitySet(tmpRoots);
            }
        }
        // STEP 3: Remove duplicates and return results
        Roots_->sortUnique();
        if (Roots_->getNumEntities()) {
            FROSCH_ASSERT(Ancestors_->getNumEntities()!=0,"Ancestors_->getNumEntities()==0");
            return Roots_;
        } else {
            // No roots found (this entity has no ancestors)
            return null;
        }
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::clearRoots()
    {
        Roots_.reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int InterfaceEntity<SC,LO,GO,NO>::computeDistancesToRoots(UN dimension,
                                                              ConstXMultiVectorPtr &nodeList,
                                                              DistanceFunction distanceFunction)
    {
        if (Roots_->getNumEntities()>0) {
            DistancesVector_.resize(getNumNodes());
            for (UN i=0; i<getNumNodes(); i++) {
                DistancesVector_[i].resize(Roots_->getNumEntities()+1,numeric_limits<SC>::max());
            }

            switch (distanceFunction) {
                case ConstantDistanceFunction:
                    for (UN i=0; i<NodeVector_.size(); i++) {
                        for (UN j=0; j<Roots_->getNumEntities(); j++) {
                            DistancesVector_[i][j] = ScalarTraits<SC>::one(); // AH 08/08/2019 TODO: Make a MultiVector out of this and use putScalar()
                        }
                    }
                    break;
                case InverseEuclideanDistanceFunction:
                    FROSCH_ASSERT(!nodeList.is_null(),"FROSch::InterfaceEntity: The inverse euclidean distance cannot be calculated without coordinates of the nodes!");
                    FROSCH_ASSERT(dimension==nodeList->getNumVectors(),"FROSch::InterfaceEntity: Inconsistent Dimension.");
                    for (UN i=0; i<Roots_->getNumEntities(); i++) {
                        for (UN j=0; j<Roots_->getEntity(i)->getNumNodes(); j++) {
                            // Coordinates of the nodes of the coarse node
                            SCVecPtr CN(dimension);
                            for (UN k=0; k<dimension; k++) {
                                CN[k] = nodeList->getData(k)[Roots_->getEntity(i)->getLocalNodeID(j)];
                            }
                            for (UN k=0; k<NodeVector_.size(); k++) {
                                // Coordinates of the nodes of the entity
                                SCVecPtr E(dimension);
                                SC distance = ScalarTraits<SC>::zero();
                                // Compute quadratic distance
                                for (UN l=0; l<dimension; l++) {
                                    distance += (nodeList->getData(l)[this->getLocalNodeID(k)]-CN[l]) * (nodeList->getData(l)[this->getLocalNodeID(k)]-CN[l]);
                                }
                                // Compute inverse euclidean distance
                                distance = sqrt(distance);
                                // Keep the min. distance over all nodes in the root entity (loop j)
                                DistancesVector_[k][i] = min(DistancesVector_[k][i],distance);
                            }
                        }
                    }
                    for (UN i=0; i<NodeVector_.size(); i++) {
                        for (UN j=0; j<Roots_->getNumEntities(); j++) {
                            DistancesVector_[i][j] = ScalarTraits<SC>::one()/DistancesVector_[i][j];
                        }
                    }
                    break;
                default:
                    FROSCH_ASSERT(false,"FROSch::InterfaceEntity: Specify a valid Distance Function.");
            }

            // The last "column" stores the sum of the distances from node i to all of the roots.
            // Required for the inverst Euclidean formulation.
            for (UN i=0; i<NodeVector_.size(); i++) {
                DistancesVector_[i][Roots_->getNumEntities()] = ScalarTraits<SC>::zero();
                for (UN j=0; j<Roots_->getNumEntities(); j++) {
                    DistancesVector_[i][Roots_->getNumEntities()] += DistancesVector_[i][j];
                }
            }
        }
        return 0;
    }

    template <class SC, class LO, class GO, class NO>
    typename InterfaceEntity<SC, LO, GO, NO>::InterfaceEntityPtr
    InterfaceEntity<SC, LO, GO, NO>::divideEntity(ConstXMatrixPtr matrix, int pID) {
        InterfaceEntityPtr entity(
            new InterfaceEntity<SC, LO, GO, NO>(Type_, DofsPerNode_, Multiplicity_, &(SubdomainsVector_[0]), Flag_));
        if (getNumNodes() >= 2) {
            sortByGlobalID();
            GOVecPtr mapVector(getNumNodes());
            for (UN i = 0; i < getNumNodes(); i++) {
                // DofsGamma_ contains the dof IDs of dofs belonging to this node with a local enumeration of all dofs
                // in the interface on this subdomain. Here we only take the first one since we are interested in node
                // connectivity encoded in matrix, which is redundant across dofs for each node.
                mapVector[i] = NodeVector_[i].DofsGamma_[0];
            }
            // I -> inside entity. O -> outside entity.
            XMatrixPtr matII, matIO, matOI, matOO;
            // mapVector has been built to contain all dofs from this entity that lie on Gamma.
            // matrix has been built to contain all dofs in this subdomain that lie on gamma.
            // Here we split matrix into components contained in mapVector, outside of mapVector and coupling terms
            // between the two. The subsequent iteration on matII serves to split this entity into two according to a
            // direct connection with the first node.
            BuildSubmatrices(matrix, mapVector(), matII, matIO, matOI, matOO, false, pID);

            XVectorPtr iterationVector = VectorFactory<SC, LO, GO, NO>::Build(matII->getRowMap());
            // This propagates non-zero entries to all nodes that are connected with the first node.
            // Thus, the entity is split into a connected entity containing the first node and a, possibly partially,
            // connected entity disconnected from the first node.
            iterationVector->getDataNonConst(0)[0] = ScalarTraits<SC>::one();
            for (UN i = 0; i < getNumNodes() - 1; i++) {
                matII->apply(*iterationVector, *iterationVector);
            }

            Teuchos::Array<UN> nodesToDelete;
            for (UN i = 0; i < getNumNodes(); i++) {
                if (fabs(iterationVector->getData(0)[i]) < 1.0e-10) {
                    entity->addNode(getNode(i));
                    // Can't delete nodes here as this would break the order in this->NodeVector_
                    nodesToDelete.push_back(i);
                }
            }
            for (UN i = 0; i < nodesToDelete.size(); i++) {
                // After every remove there is one node less. Shift the index by i to compensate.
                removeNode(nodesToDelete[i] - i);
            }
        }
        return entity;
    }

    template <class SC, class LO, class GO, class NO>
    int InterfaceEntity<SC, LO, GO, NO>::computeDistancesOnBoundary(UN dimension, ConstXMultiVectorPtr &nodeList,
                                                                    ArrayRCP<EntitySetPtr> entitySetVector,
                                                                    const int pID) {
        FROSCH_ASSERT(!nodeList.is_null(),
                      "FROSch::InterfaceEntity: The inverse euclidean distance to the Dirichlet entity cannot be "
                      "calculated without coordinates of the nodes!");
        FROSCH_ASSERT(dimension == nodeList->getNumVectors() && dimension > 1,
                      "FROSch::InterfaceEntity: Inconsistent Dimension.");
        FROSCH_ASSERT(
            Flag_ == DirichletFlag || Flag_ == DoNothingFlag,
            "FROSch::InterfaceEntity: computeDistancesOnBoundary() can only be called on boundary InterfaceEntities.");

        // Hijack/overwrite the distancesVector with distance to Dirichlet node for all roots. We one vector in the
        // multivector for each root since in the InterfacePartitionOfUnity classes an entry is built for each root.
        DistancesVector_.resize(getNumNodes());
        for (UN i = 0; i < getNumNodes(); i++) {
            DistancesVector_[i].resize(Roots_->getNumEntities() + 1, numeric_limits<SC>::max());
        }

        // If this is a Dirichlet entity, the distance calculated in RGDSWInterfacePartitionOfUnity should be zero. We
        // set all the distances to zero, and the sum of the distances to all roots (last column) to one as this is the
        // denominator in the inverse Euclidean distance.
        if (Flag_ == DirichletFlag) {
            for (UN i = 0; i < NodeVector_.size(); i++) {
                for (UN j = 0; j < Roots_->getNumEntities(); j++) {
                    DistancesVector_[i][j] = ScalarTraits<SC>::zero();
                }
                DistancesVector_[i][Roots_->getNumEntities()] = ScalarTraits<SC>::one();
            }
        } else {

            auto dirichletEntities = rcp(new EntitySet<SC, LO, GO, NO>(BoundaryType));
            auto boundaryEntities = entitySetVector[1];
            // Extract Dirichlet entities (if they exist)
            for (int i = 0; i < boundaryEntities->getNumEntities(); i++) {
                if (boundaryEntities->getEntity(i)->getEntityFlag() == DirichletFlag) {
                    dirichletEntities->addEntity(boundaryEntities->getEntity(i));
                }
            }
            // If there are zero or more than one Dirichlet entities in the local subdomain, set the distance to
            // one/num. coarse basis functions everywhere since it does not make sense to compute the inverse Euclidean
            // distance to a Dirichlet entity in this case.

            if (dirichletEntities->getNumEntities() != 1) {
                if (dirichletEntities->getNumEntities() > 1) {
                    FROSCH_WARNING("FROSch_InterfaceEntity", true,
                                   "There are more than one Dirichlet entities in this subdomain. Setting do nothing "
                                   "interface to one.");
                }

                // Each entry in Roots_ corresponds to a coarse basis function. Since every one will contribute to the
                // value on the do nothing entity, divide by the number of roots to get one in total.
                for (UN i = 0; i < NodeVector_.size(); i++) {
                    for (UN j = 0; j < Roots_->getNumEntities(); j++) {
                        DistancesVector_[i][j] = ScalarTraits<SC>::one() / Roots_->getNumEntities();
                    }
                    // Note that the last entry in DistancesVector_[i] usually contains the sum of all the distances
                    // between node i and all the roots. Setting this value to one, ensures that 1/1 = 1 results from
                    // the inverse Euclidean calculation in the RGDSWInterfacePartitionOfUnity
                    DistancesVector_[i][Roots_->getNumEntities()] = ScalarTraits<SC>::one();
                }
            } else {
                // In this case there is exactly one Dirichlet and one interface entity higher up the hierarchy as
                // neighbors

                // Find the interface neighbor that is furthest down the hierarchy e.g. a face instead of an edge
                // contained by the face. Not sure if this should be the other way around, but a face contains a
                // superset of the nodes of a contained edge, so the min. distance to the face will be smaller than the
                // min. distance to the edge.
                InterfaceEntityPtr neighbor;
                FROSCH_ASSERT(std::is_sorted(NodeVector_.begin(), NodeVector_.end()),
                              "FROSch::InterfaceEntity: the intersection algorithm requires sorted node vectors")
                // Only care about entities that are part of at least two subdomains i.e. "real" interface entities.
                int multiplicity = 2;
                // Go up the interface entity hierarchy
                // TODO: [KH] this works in 2D with one subdomain per rank, because a do nothing entity with one
                // Dirichlet neighbor can only have one other end that shares a node with an interface entity higher up
                // the hierarchy. In 3D multiple neighbors are possible. In this case we either have to calculate the
                // inverse Euclidean distances to all neighbors and add them, or take the min. distance for each node.
                while (neighbor.is_null() && multiplicity < entitySetVector.size()) {
                    // At each level we check every entity
                    int i = 0;
                    while (neighbor.is_null() && i < entitySetVector[multiplicity]->getNumEntities()) {
                        auto nodeVec = entitySetVector[multiplicity]->getEntity(i)->getConstNodeVectorRef();
                        // Efficient O(N + M) search for common nodes. Assumes that the NodeVectors are sorted
                        FROSCH_ASSERT(
                            std::is_sorted(nodeVec.begin(), nodeVec.end()),
                            "FROSch::InterfaceEntity: the intersection algorithm requires sorted node vectors")
                        auto it1 = nodeVec.begin();
                        auto it2 = NodeVector_.begin();
                        while (it1 != nodeVec.end() && it2 != NodeVector_.end()) {
                            if (*it1 < *it2) {
                                it1++;
                            } else if (*it2 < *it1) {
                                it2++;
                            } else {
                                neighbor = entitySetVector[multiplicity]->getEntity(i);
                                break;
                            }
                        }
                        i++;
                    }
                    multiplicity++;
                }

                FROSCH_ASSERT(!neighbor.is_null(),
                              "FROSch::InterfaceEntity: no neighbor entity found for do nothing entity");

                SCVecPtr dirichletDistance = ArrayRCP<SC>(NodeVector_.size(), numeric_limits<SC>::max());
                // Iterate over the whole Dirichlet entity and keep the min. distance
                for (UN i = 0; i < dirichletEntities->getEntity(0)->getNumNodes(); i++) {
                    // Coordinates of node j of the Dirichlet entity
                    SCVecPtr dirichletNode(dimension);
                    for (UN j = 0; j < dimension; j++) {
                        dirichletNode[j] = nodeList->getData(j)[dirichletEntities->getEntity(0)->getLocalNodeID(i)];
                    }
                    for (UN j = 0; j < NodeVector_.size(); j++) {
                        SC distance = ScalarTraits<SC>::zero();
                        // Compute quadratic distance between node j in the Dirichlet entity and all the nodes in
                        // this entity
                        for (UN k = 0; k < dimension; k++) {
                            distance += (nodeList->getData(k)[this->getLocalNodeID(j)] - dirichletNode[k]) *
                                        (nodeList->getData(k)[this->getLocalNodeID(j)] - dirichletNode[k]);
                        }
                        distance = sqrt(distance);
                        // Keep the min. distance over all nodes in the Dirichlet entity (loop i)
                        dirichletDistance[j] = min(dirichletDistance[j], distance);
                    }
                }

                // Iterate over the whole neighbor entity and keep the min. distance
                for (UN i = 0; i < neighbor->getNumNodes(); i++) {
                    // Coordinates of node j of the Dirichlet entity
                    SCVecPtr neighborNode(dimension);
                    for (UN j = 0; j < dimension; j++) {
                        neighborNode[j] = nodeList->getData(j)[neighbor->getLocalNodeID(i)];
                    }
                    for (UN j = 0; j < NodeVector_.size(); j++) {
                        SC distance = ScalarTraits<SC>::zero();
                        // Compute quadratic distance between node j in the Dirichlet entity and all the nodes in
                        // this entity
                        for (UN k = 0; k < dimension; k++) {
                            distance += (nodeList->getData(k)[this->getLocalNodeID(j)] - neighborNode[k]) *
                                        (nodeList->getData(k)[this->getLocalNodeID(j)] - neighborNode[k]);
                        }
                        distance = sqrt(distance);
                        // Keep the min. distance over all nodes in the neighbor entity (loop i)
                        DistancesVector_[j][0] = min(DistancesVector_[j][0], distance);
                    }
                }

                // Convert the distance to an inverse distance
                for (UN i = 0; i < NodeVector_.size(); i++) {
                    dirichletDistance[i] = ScalarTraits<SC>::one() / dirichletDistance[i];
                    // neighborDistance might be zero since boundary entities are constructed to overlap with their
                    // neighbors
                    if (DistancesVector_[i][0] < 10 * numeric_limits<SC>::min()) {
                        DistancesVector_[i][0] = ScalarTraits<SC>::zero();
                    } else {
                        DistancesVector_[i][0] = ScalarTraits<SC>::one() / DistancesVector_[i][0];
                    }
                    // For every root there is a coarse basis function, each of which requires this distance, so we copy
                    // the distance to entries for all other roots.
                    for (UN j = 1; j < Roots_->getNumEntities(); j++) {
                        DistancesVector_[i][j] = DistancesVector_[i][0];
                    }
                }

                // The last "column" stores the sum of the distances from node i to the Dirichlet entity and the
                // neighbor interface entity. Required for the inverse Euclidean formulation.
                for (UN i = 0; i < NodeVector_.size(); i++) {
                    DistancesVector_[i][Roots_->getNumEntities()] = DistancesVector_[i][0] + dirichletDistance[i];
                }
            }
        }
        return 0;
    }

    /////////////////
    // Get Methods //
    /////////////////

    template <class SC,class LO,class GO,class NO>
    EntityType InterfaceEntity<SC,LO,GO,NO>::getEntityType() const
    {
        return Type_;
    }

    template <class SC,class LO,class GO,class NO>
    EntityFlag InterfaceEntity<SC,LO,GO,NO>::getEntityFlag() const
    {
        return Flag_;
    }

    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getDofsPerNode() const
    {
        return DofsPerNode_;
    }

    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getMultiplicity() const
    {
        return Multiplicity_;
    }

    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getUniqueID() const
    {
        return UniqueID_;
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalID() const
    {
        return LocalID_;
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getRootID() const
    {
        return RootID_;
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLeafID() const
    {
        return LeafID_;
    }

    template <class SC,class LO,class GO,class NO>
    const Node<SC,LO,GO>& InterfaceEntity<SC,LO,GO,NO>::getNode(UN iDNode) const
    {
        return NodeVector_[iDNode];
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getGammaNodeID(UN iDNode) const
    {
        return NodeVector_[iDNode].NodeIDGamma_;
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalNodeID(UN iDNode) const
    {
        return NodeVector_[iDNode].NodeIDLocal_;
    }

    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getGlobalNodeID(UN iDNode) const
    {
        return NodeVector_[iDNode].NodeIDGlobal_;
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getGammaDofID(UN iDNode, UN iDDof) const
    {
        return NodeVector_[iDNode].DofsGamma_[iDDof];
    }

    template <class SC,class LO,class GO,class NO>
    LO InterfaceEntity<SC,LO,GO,NO>::getLocalDofID(UN iDNode, UN iDDof) const
    {
        return NodeVector_[iDNode].DofsLocal_[iDDof];
    }

    template <class SC,class LO,class GO,class NO>
    GO InterfaceEntity<SC,LO,GO,NO>::getGlobalDofID(UN iDNode, UN iDDof) const
    {
        return NodeVector_[iDNode].DofsGlobal_[iDDof];
    }

    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::IntVec & InterfaceEntity<SC,LO,GO,NO>::getSubdomainsVector() const
    {
        return SubdomainsVector_;
    }

    template <class SC,class LO,class GO,class NO>
    typename InterfaceEntity<SC,LO,GO,NO>::UN InterfaceEntity<SC,LO,GO,NO>::getNumNodes() const
    {
        return NodeVector_.size();
    }

    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getAncestors() const
    {
        return Ancestors_;
    }

    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getOffspring() const
    {
        return Offspring_;
    }

    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::EntitySetPtr InterfaceEntity<SC,LO,GO,NO>::getRoots() const
    {
        return Roots_;
    }

    template <class SC,class LO,class GO,class NO>
    SC InterfaceEntity<SC,LO,GO,NO>::getDistanceToRoot(UN iDNode,
                                                       UN iDRoot) const
    {
        FROSCH_ASSERT(iDNode<getNumNodes(),"iDNode>=getNumNodes()");
        FROSCH_ASSERT(iDRoot<Roots_->getNumEntities()+1,"iDNode>=Roots_->getNumEntities()+1");
        return DistancesVector_[iDNode][iDRoot];
    }

    template <class SC,class LO,class GO,class NO>
    const typename InterfaceEntity<SC,LO,GO,NO>::NodeVec& InterfaceEntity<SC,LO,GO,NO>::getConstNodeVectorRef() const{
        return NodeVector_;
    }

    template <class SC,class LO,class GO,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                  RCP<InterfaceEntity<SC,LO,GO,NO> > iEb)
    {
        return iEa->getUniqueID()<iEb->getUniqueID();
    }

    template <class SC,class LO,class GO,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                RCP<InterfaceEntity<SC,LO,GO,NO> > iEb)
    {
        return iEa->getUniqueID()==iEb->getUniqueID();
    }
}

#endif
