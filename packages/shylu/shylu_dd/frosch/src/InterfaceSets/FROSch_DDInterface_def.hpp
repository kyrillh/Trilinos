// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_DDINTERFACE_DEF_HPP
#define _FROSCH_DDINTERFACE_DEF_HPP

#include "FROSch_EntitySet_decl.hpp"
#include "FROSch_InterfaceEntity_decl.hpp"
#include "FROSch_Output.h"
#include "FROSch_Tools_decl.hpp"
#include "Teuchos_ArrayViewDecl.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_VerbosityLevel.hpp"
#include <FROSch_DDInterface_decl.hpp>
#include <FROSch_EntitySet_def.hpp>
#include <FROSch_ExtractSubmatrices_def.hpp>
#include <algorithm>
#include <iterator>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    DDInterface<SC,LO,GO,NO>::DDInterface(UN dimension,
                                          UN dofsPerNode,
                                          ConstXMapPtr localToGlobalMap,
                                          Verbosity verbosity,
                                          UN levelID,
                                          CommunicationStrategy commStrategy) :
    MpiComm_ (localToGlobalMap->getComm()),
    Dimension_ (dimension),
    DofsPerNode_ (dofsPerNode),
    NumMyNodes_ (localToGlobalMap->getLocalNumElements()),
    NodesMap_ (localToGlobalMap),
    CommStrategy_ (commStrategy),
    /* Verbose_ (MpiComm_->getRank()==0), */
    Verbose_ (false),
    Verbosity_ (verbosity),
    LevelID_ (levelID),
    HaveDirichletEntities_(false)
    {
        FROSCH_DETAILTIMER_START_LEVELID(dDInterfaceTime,"DDInterface::DDInterface");
        FROSCH_ASSERT(((Dimension_==2)||(Dimension_==3)),"FROSch::DDInterface: Only dimension 2 and 3 are available");

        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface" << endl;

        IntVecVecPtr componentsSubdomains;
        IntVecVec componentsSubdomainsUnique;

        // Builds the following lists:
        // - componentsSubdomains: one entry for each node. Each entry is a list of subdomains that contains the node.
        // - componentsSubdomainsUnique: one entry for each equivalence class in the subdomain. e.g. all nodes on an
        // interface edge are condensed into an equivalence class belonging to the subdomains [1, 2]
        communicateLocalComponents(componentsSubdomains,componentsSubdomainsUnique);

        // Uses both of these lists to generate entity sets. An entity is a collection of nodes that can be grouped
        // together in an equivalence class e.g. all the nodes on the interface edge between subdomains 1 and 2. Entity
        // sets are built to gather all entities in which nodes belong to the same number of subdomains. Here first two
        // entities are built, one for interior and one for interface nodes. Then the interface nodes are further placed
        // into entity set's e.g. for vertices, edges and faces.
        identifyLocalComponents(componentsSubdomains, componentsSubdomainsUnique);
    }

    template <class SC,class LO,class GO,class NO>
    DDInterface<SC,LO,GO,NO>::~DDInterface()
    {

    } // Do we need sth here?

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::resetGlobalDofs(ConstXMapPtrVecPtr dofsMaps)
    {
        FROSCH_DETAILTIMER_START_LEVELID(resetGlobalDofsTime,"DDInterface::resetGlobalDofs");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Resetting Global IDs" << endl;

        // PURPOSE: Update global DOF IDs for all entities when new DOF maps are provided.
        // This is needed when the DOF numbering changes or when switching between
        // different DOF orderings (e.g., node-wise vs. dimension-wise).

        // STEP 1: Update global DOF IDs for all entities in EntitySetVector_
        // EntitySetVector_[l] contains entities belonging to exactly l subdomains
        for (UN l=0; l<EntitySetVector_.size(); l++) {
            for (UN i=0; i<EntitySetVector_[l]->getNumEntities(); i++) {
                for (UN j=0; j<EntitySetVector_[l]->getEntity(i)->getNumNodes(); j++) {
                    LO localID = EntitySetVector_[l]->getEntity(i)->getLocalNodeID(j);
                    UNVecPtr dofIDs(DofsPerNode_);
                    GOVecPtr dofsGlobal(DofsPerNode_);
                    for (UN k=0; k<DofsPerNode_; k++) {
                        dofIDs[k] = k;
                        dofsGlobal[k] = dofsMaps[k]->getGlobalElement(localID);
                    }
                    EntitySetVector_[l]->getEntity(i)->resetGlobalDofs(j,DofsPerNode_,&(dofIDs[0]),&(dofsGlobal[0]));
                }
            }
        }

        // STEP 2: Update global DOF IDs for Interface entities
        // Interface entities contain nodes shared between multiple subdomains
        for (UN i=0; i<Interface_->getNumEntities(); i++) {
            for (UN j=0; j<Interface_->getEntity(i)->getNumNodes(); j++) {
                LO localID = Interface_->getEntity(i)->getLocalNodeID(j);
                UNVecPtr dofIDs(DofsPerNode_);
                GOVecPtr dofsGlobal(DofsPerNode_);
                for (UN k=0; k<DofsPerNode_; k++) {
                    dofIDs[k] = k;
                    dofsGlobal[k] = dofsMaps[k]->getGlobalElement(localID);
                }
                Interface_->getEntity(i)->resetGlobalDofs(j,DofsPerNode_,&(dofIDs[0]),&(dofsGlobal[0]));
            }
        }

        // STEP 3: Update global DOF IDs for Interior entities
        // Interior entities contain nodes belonging to only one subdomain
        for (UN i=0; i<Interior_->getNumEntities(); i++) {
            for (UN j=0; j<Interior_->getEntity(i)->getNumNodes(); j++) {
                LO localID = Interior_->getEntity(i)->getLocalNodeID(j);
                UNVecPtr dofIDs(DofsPerNode_);
                GOVecPtr dofsGlobal(DofsPerNode_);
                for (UN k=0; k<DofsPerNode_; k++) {
                    dofIDs[k] = k;
                    dofsGlobal[k] = dofsMaps[k]->getGlobalElement(localID);
                }
                Interior_->getEntity(i)->resetGlobalDofs(j,DofsPerNode_,&(dofIDs[0]),&(dofsGlobal[0]));
            }
        }

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::removeDirichletNodes(GOVecView dirichletBoundaryDofs)
    {
        FROSCH_DETAILTIMER_START_LEVELID(removeDirichletNodesTime,"DDInterface::removeDirichletNodes");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Removing Dirichlet Nodes from the domain decomposition interface" << endl;

        // PURPOSE: Remove nodes with Dirichlet boundary conditions from all interface entities.
        // Dirichlet nodes are constrained and should not participate in the coarse space
        // construction, so they are removed from the domain decomposition interface.

        // STEP 1: Remove Dirichlet nodes from all entity sets
        // Iterate through all entity sets organized by multiplicity
        for (UN l=0; l<EntitySetVector_.size(); l++) {
            EntitySetVector_[l]->removeNodesWithDofs(dirichletBoundaryDofs);
        }
        
        // STEP 2: Clean up empty entities that may have been created
        // After removing Dirichlet nodes, some entities may become empty
        removeEmptyEntities();
        
        // STEP 3: Reset unique IDs for sorting
        // Update unique IDs for all remaining entities to ensure proper sorting
        for (UN l=0; l<EntitySetVector_.size(); l++) {
            EntitySetVector_[l]->setUniqueIDToFirstGlobalNodeID();
        }
        return 0;
    }

    // Part of the boundary framework
    template <class SC, class LO, class GO, class NO>
    void DDInterface<SC, LO, GO, NO>::addBoundaryNodes(const GOVecView boundaryDofs, const EntityFlag type,
                                                       const int dofOffset) {
        FROSCH_ASSERT(type == DirichletFlag || type == CustomBCFlag,
                      "addBoundaryNodes() is only for adding Dirichlet or do nothing boundaries")
        // If type == CustomBCFlag and HaveDirichletEntities_ == false we don't add custom boundary since we only want to
        // modify it in conjuction with a Dirichlet entity
        if (boundaryDofs.size() > 0 && (type != CustomBCFlag || HaveDirichletEntities_)) {

            // Change the entity type of boundary entity set from default
            EntitySetVector_[1]->resetEntityType(BoundaryType);
            const auto interface = Interface_->getEntity(0);
            const int numBoundaryNodes = boundaryDofs.size() / DofsPerNode_;

            // Build a sorted list of global Dirichlet boundary node indices
            // This assumes that the boundary dofs are provided in a node-wise ordering
            Teuchos::Array<GO> boundaryNodes(numBoundaryNodes);
            for (LO i = 0; i < numBoundaryNodes; i++) {
                boundaryNodes[i] = (boundaryDofs[i * DofsPerNode_] - dofOffset) / DofsPerNode_;
            }
            sortunique(boundaryNodes);

            // Add boundary nodes to interface
            for (LO i = 0; i < boundaryNodes.size(); i++) {
                // *The interface/interior ID does not matter at this point. The subdomain solver assumes that both ID
                // local and interface/interior ID are constructed in ascending order relative to each other i.e. when
                // one ID is sorted in ascending order, this also sorts the other --> build the tmpEntity here and sort
                // its nodes by local ID afterwards since we are building using global ID
                LO nodeIDBndry = 0;
                // Global ID across all nodes
                GO nodeIDGlobal = boundaryNodes[i];
                // ID local to the subdomain across interior and interface
                LO nodeIDLocal = NodesMap_->getLocalElement(nodeIDGlobal);

                LOVecPtr dofsI(DofsPerNode_);
                LOVecPtr dofsLocal(DofsPerNode_);
                GOVecPtr dofsGlobal(DofsPerNode_);
                // Assumes node-wise ordering. Same is done in identifyLocalComponents().
                for (UN k = 0; k < DofsPerNode_; k++) {
                    dofsI[k] = DofsPerNode_ * nodeIDBndry + k;
                    dofsLocal[k] = DofsPerNode_ * nodeIDLocal + k;
                    dofsGlobal[k] = DofsPerNode_ * nodeIDGlobal + k;
                }
                FROSCH_ASSERT(nodeIDLocal >= 0, "The global interface node " + std::to_string(nodeIDGlobal) +
                                                    " does not lie in subdomain " +
                                                    std::to_string(this->MpiComm_->getRank()))
                interface->addNode(nodeIDBndry, nodeIDLocal, nodeIDGlobal, DofsPerNode_, dofsI, dofsLocal, dofsGlobal);
            }
            // reindex GammaID so that GammaID and local ID are ordered in the same way. See comment*
            interface->sortUniqueByLocalID();
            interface->reindexGammaID();

            // Create an entity just for the boundary nodes
            constexpr UN multiplicity = 1;
            IntVec subdomains({MpiComm_->getRank()});
            RCP<InterfaceEntity<SC, LO, GO, NO>> tmpEntity;
            if (type == DirichletFlag) {
                tmpEntity = Teuchos::rcp(new InterfaceEntity<SC, LO, GO, NO>(BoundaryType, DofsPerNode_, multiplicity,
                                                                             subdomains.data(), DirichletFlag));
            } else if (type == CustomBCFlag) {
                tmpEntity = Teuchos::rcp(new InterfaceEntity<SC, LO, GO, NO>(BoundaryType, DofsPerNode_, multiplicity,
                                                                             subdomains.data(), CustomBCFlag));
            }
 
            // Check vectors are sorted for subset operations below
            for (int i = 2; i < EntitySetVector_.size(); i++) {
                for (int j = 0; j < EntitySetVector_[i]->getNumEntities(); j++) {
                    auto tmpEntityNodesVec = EntitySetVector_[i]->getEntity(j)->getConstNodeVectorRef();
                    FROSCH_ASSERT(std::is_sorted(tmpEntityNodesVec.begin(), tmpEntityNodesVec.end()),
                                  "FROSch::DDInterface: std::includes requires sorted vectors.")
                    FROSCH_ASSERT(std::is_sorted(boundaryNodes.begin(), boundaryNodes.end()),
                                  "FROSch::DDInterface: std::includes requires sorted vectors.")
                }
            }
            // Now check whether there are any entities in equivalence classes higher than 1, that lie completely in the
            // Dirichlet boundary. E.g. in 2D backward-facing step a regular decomposition will result in an interface
            // vertex at the corner of the step because it belongs to three subdomains.
            // If such an entity is found, move it to the equivalence class 1, together with all of the other boundary
            // entities.
            auto& boundaryEntityVector = EntitySetVector_[1]->getEntityVector();
            for (int i = 2; i < EntitySetVector_.size(); i++) {
                // Get the entities in the current set.
                auto& tmpEntityVector = EntitySetVector_[i]->getEntityVector();
                // If an entity lies comletely in the Dirichlet boundary, it's moved to the equivalence class 1.
                // Stable partition moves all entries to the back that fail the test, maintaing relative order.
                // std::includes returns true if first set is a superset of the second set.
                auto newEnd = std::stable_partition(
                    tmpEntityVector.begin(), tmpEntityVector.end(),
                    [&](Teuchos::RCP<InterfaceEntity<SC, LO, GO, NO>> entity) {
                        std::vector<GO> nodeVec(entity->getConstNodeVectorRef().length());
                        for (int j = 0; j < entity->getConstNodeVectorRef().length(); j++) {
                            nodeVec[j] = entity->getConstNodeVectorRef()[j].NodeIDGlobal_;
                        }
                        FROSCH_ASSERT(std::is_sorted(nodeVec.begin(), nodeVec.end()), "nodeVec must be sorted!")
                        auto result = !std::includes(boundaryNodes.begin(), boundaryNodes.end(), nodeVec.begin(), nodeVec.end());
                        return result;
                    });

                for (auto it = newEnd; it != tmpEntityVector.end(); it++) {
                    // We need to change the type of the entities that we move.
                    (*it)->resetEntityFlag(type);
                    (*it)->resetEntityType(BoundaryType);
                    FROSCH_ASSERT((*it)->getSubdomainsVector().size() == i,
                                  "FROSCH::DDInterface: An entity was found in the wrong equivalence class")
                    // Finally, we need to also remove the nodes of the entities to be moved from the boundary nodes
                    // vector. It will be used to build a boundary entity continaining remaining boundary nodes.
                    // Boundary entities should not contain duplicate nodes.
                    auto newBoundaryNodesEnd = std::stable_partition(boundaryNodes.begin(), boundaryNodes.end(), [&](const auto &a) {
                        std::vector<GO> nodeVec((*it)->getConstNodeVectorRef().length());
                        for (int j = 0; j < (*it)->getConstNodeVectorRef().length(); j++) {
                            nodeVec[j] = (*it)->getConstNodeVectorRef()[j].NodeIDGlobal_;
                        }
                        FROSCH_ASSERT(std::is_sorted(nodeVec.begin(), nodeVec.end()), "nodeVec must be sorted!")
                        return !std::binary_search(nodeVec.begin(), nodeVec.end(), a);
                    });
                    boundaryNodes.resize(newBoundaryNodesEnd - boundaryNodes.begin());
                }
                std::move(newEnd, tmpEntityVector.end(), std::back_inserter(boundaryEntityVector));
                tmpEntityVector.resize(newEnd - tmpEntityVector.begin(),
                                       Teuchos::RCP<InterfaceEntity<SC, LO, GO, NO>>{});
            }

            // Here we check two things for each node in the equivalence classes higher than 1 i.e. not the boundary:
            // if the node is a boundary node, we extract it into a new entity with the same multiplicity etc. and
            // remove its ID from the boundaryNodes vector. Otherwise we update it's GammaID_. All of the GammaID_'s
            // have been shuffled around by adding the boundary nodes to the single "interface" interface entity that
            // contains all nodes in the interface.
            auto interfaceNodes = interface->getConstNodeVectorRef();
            FROSCH_ASSERT(std::is_sorted(boundaryNodes.begin(), boundaryNodes.end()),
                            "FROSch::DDInterface: boundaryNodes need to be sorted for binary search.")
            FROSCH_ASSERT(std::is_sorted(interfaceNodes.begin(), interfaceNodes.end(), [](const auto &a, const auto &b){ return a.NodeIDLocal_ < b.NodeIDLocal_;}),
                            "FROSch::DDInterface: interfaceNodes need to be sorted by localy ID for binary search.")
            for (int i = 2; i < EntitySetVector_.size(); i++) {
                auto tmpEntitySet = Teuchos::rcp(new EntitySet<SC, LO, GO, NO>(BoundaryType));
                // Goes through all nodes in the entities of the current entity set. If they are in boundaryNodes
                // vector, they are moved to a new boundary entity and also removed from boundaryNodes vector.
                EntitySetVector_[i]->moveNodesWithIDsToBoundary(boundaryNodes, tmpEntitySet);
                EntitySetVector_[1]->addEntitySet(tmpEntitySet);
                for (int j = 0; j < EntitySetVector_[i]->getNumEntities(); j++) {
                    // The remaining nodes are not in the boundary and need their gammaIDs updated
                    for (int k = 0; k < EntitySetVector_[i]->getEntity(j)->getNumNodes(); k++) {
                        // Find the current node in interface using binary search. This is probably more efficient
                        // than building a hashed map std::unordered_map since we are finding e.g. 2400 elements for
                        // subdomains with 640000 in 3D.
                        auto interfaceNodeIt = std::lower_bound(
                            interfaceNodes.begin(), interfaceNodes.end(), EntitySetVector_[i]->getEntity(j)->getNode(k),
                            [](const auto &a, const auto &b) { return a.NodeIDLocal_ < b.NodeIDLocal_; });

                        EntitySetVector_[i]->getEntity(j)->setGammaIDs(
                            k, interface->getGammaNodeID(std::distance(interfaceNodes.begin(), interfaceNodeIt)));
                    }
                }
            }

            // Build the tmpEntity with the new gammaIDs and remove nodes from Interior_
            auto boundaryNodesIt = boundaryNodes.begin();
            // removeNode() uses binary search on the globalID so requires the nodes to be sorted by globalID
            Interior_->getEntity(0)->sortByGlobalID();
            // interface needs to be globally sorted because of how we step through it below 
            interface->sortByGlobalID();
            // boundary nodes should all be in the interface at this point. Note the && boundaryNodesIt guard:
            // once every boundary node has been matched, dereferencing boundaryNodesIt is undefined.
            for (int i = 0; i < interface->getNumNodes() && boundaryNodesIt != boundaryNodes.end(); i++) {
                // Only need to deal with nodes actually in the boundary
                // The boundary nodes were added to the interface above.
                // That's why at this stage they are in the interface and in the interior.
                if (*boundaryNodesIt == interface->getGlobalNodeID(i)) {
                    tmpEntity->addNode(interface->getNode(i));
                    // If node is not in the entity, removeNode does nothing and returns -1.
                    Interior_->getEntity(0)->removeNode(interface->getNode(i));
                    boundaryNodesIt++;
                }
            }



            // Restore local-ID order on interface.
            interface->sortUniqueByLocalID();

            // Interior_ Gamma IDs need updating since nodes were removed.
            Interior_->getEntity(0)->reindexGammaID();

            // Add the boundary entity to this->EntitySetVector_. It is not split into strictly connected entities here
            // i.e. entities in which the union of the support of associated finitie element basis functions forms a
            // connected set. This is done in a later call to sortInterface()
            EntitySetVector_[1]->addEntity(tmpEntity);
            if (type == DirichletFlag) {
                HaveDirichletEntities_ = true;
            }
        }
    }

    template <class SC, class LO, class GO, class NO>
    int DDInterface<SC, LO, GO, NO>::divideUnconnectedEntities(ConstXMatrixPtr matrix) {
        FROSCH_DETAILTIMER_START_LEVELID(divideUnconnectedEntitiesTime,"DDInterface::divideUnconnectedEntities");
        // matrix is typically K_ i.e. the global system matrix
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Decomposing unconnected interface components" << endl;

        // PURPOSE: Split interface entities that are not properly connected based on matrix connectivity.
        // This ensures that each interface entity forms a connected component, which is
        // important for the effectiveness of domain decomposition methods.

        // STEP 1: Extract interface DOF indices for connectivity analysis
        // Create a list of global DOF IDs for all interface nodes
        std::vector<GO> tmpGammaIndices;
        for (UN k=0; k<DofsPerNode_; k++) {
            for (UN i=0; i<Interface_->getEntity(0)->getNumNodes(); i++) {
                tmpGammaIndices.push_back(Interface_->getEntity(0)->getGammaDofID(i,k));
            }
        }
        auto maxDof = std::max_element(tmpGammaIndices.begin(), tmpGammaIndices.end());

       GOVecPtr indicesGammaDofs((*maxDof) + 1);
        for (UN k=0; k<DofsPerNode_; k++) {
            for (UN i=0; i<Interface_->getEntity(0)->getNumNodes(); i++) {
                indicesGammaDofs[Interface_->getEntity(0)->getGammaDofID(i,k)] = Interface_->getEntity(0)->getGlobalDofID(i,k);
            }
        }

        // STEP 2: Create map and extract subdomain matrix for interface
        // Extract the submatrix corresponding to interface DOFs for connectivity analysis
        const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
        XMapPtr map = MapFactory<LO,GO,NO>::Build(matrix->getRowMap()->lib(),INVALID,indicesGammaDofs(),0,MpiComm_);
        matrix = FROSch::ExtractLocalSubdomainMatrix(matrix.getConst(),map.getConst(),ScalarTraits<SC>::one());

        // STEP 3: Divide unconnected entities in all entity sets
        // Use matrix connectivity to split entities that are not connected
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetVector_[i]->divideUnconnectedEntities(matrix);
        }

        /*
        LO numSeparateEdges = Edges_->divideUnconnectedEntities(matrix,MpiComm_->getRank());
        LO numSeparateFaces = Faces_->divideUnconnectedEntities(matrix,MpiComm_->getRank());

        if (Verbose_ && Verbosity_==All) {
            cout << "\n\
            --------------------------------------------\n\
            # separate edges:     --- " << numSeparateEdges << "\n\
            # separate faces:     --- " << numSeparateFaces << "\n\
            --------------------------------------------\n";
        }
        */

        // STEP 4: Clean up and finalize
        // Remove any empty entities created during the division process
        removeEmptyEntities();

        // STEP 5: Reset unique IDs for proper sorting
        // Update unique IDs for all entities to ensure they can be sorted correctly
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetVector_[i]->setUniqueIDToFirstGlobalNodeID();
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::flagEntities(ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(flagEntitiesTime,"DDInterface::flagEntities");
        
        // PURPOSE: Assign geometric flags to interface entities based on their properties.
        // These flags (DefaultFlag, StraightFlag, ShortFlag, NodeFlag) are used
        // later to classify entities into vertices, edges, and faces.

        // STEP 1: Flag nodes and short entities
        // Assign flags based on entity properties without geometric information
        // We skip entries 0 and 1 in EntitySetVector_ as these are empty/boundary entities with their own flags
        for (UN l=2; l<EntitySetVector_.size(); l++) {
            EntitySetVector_[l]->flagNodes();           // Flag single-node entities
            EntitySetVector_[l]->flagShortEntities();  // Flag entities that are too short
        }
        
        // STEP 2: Flag straight entities (requires geometric information)
        // Use node coordinates to identify straight entities (e.g., straight edges)
        if (!nodeList.is_null()) {
            for (UN l=2; l<EntitySetVector_.size(); l++) {
                EntitySetVector_[l]->flagStraightEntities(Dimension_,nodeList);
            }
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::removeEmptyEntities()
    {
        FROSCH_DETAILTIMER_START_LEVELID(removeEmptyEntitiesTime,"DDInterface::removeEmptyEntities");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Removing empty interface components" << endl;

        // PURPOSE: Remove entities that have no nodes from all entity sets.
        // Empty entities can be created during processing (e.g., after removing
        // Dirichlet nodes) and should be cleaned up to maintain data integrity.

        // Remove empty entities from all entity sets organized by multiplicity
        for (UN l=0; l<EntitySetVector_.size(); l++) {
            EntitySetVector_[l]->removeEmptyEntities();
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::sortVerticesEdgesFaces(ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(sortVerticesEdgesFacesTime,"DDInterface::sortVerticesEdgesFaces");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Sorting interface components" << endl;

        // Clear EntitySets if non-empty
        if (Vertices_->getNumEntities()>0) Vertices_.reset(new EntitySet<SC,LO,GO,NO>(VertexType));
        if (ShortEdges_->getNumEntities()>0) ShortEdges_.reset(new EntitySet<SC,LO,GO,NO>(EdgeType));
        if (StraightEdges_->getNumEntities()>0) StraightEdges_.reset(new EntitySet<SC,LO,GO,NO>(EdgeType));
        if (Edges_->getNumEntities()>0) Edges_.reset(new EntitySet<SC,LO,GO,NO>(EdgeType));
        if (Faces_->getNumEntities()>0) Faces_.reset(new EntitySet<SC,LO,GO,NO>(FaceType));

        flagEntities(nodeList);

        // Make sure that we do not sort any empty entities
        removeEmptyEntities();

        for (UN l=0; l<EntitySetVector_.size(); l++) {
            switch (l) {
                case 0:
                    FROSCH_ASSERT(EntitySetVector_[l]->getNumEntities()==0,"FROSch::DDInterface: This case is impossible.");
                    break;
                case 1:
                    // Test to see if boundary interface entities have been built properly
                    for (UN i=0; i<EntitySetVector_[l]->getNumEntities(); i++) {
                        auto flag = EntitySetVector_[l]->getEntity(i)->getEntityFlag();
                        FROSCH_ASSERT(flag == DirichletFlag || flag == CustomBCFlag,"FROSch::DDInterface: EntitySetVector_[1] contains non-boundary entities.")
                    }
                    break;
                case 2:
                    for (UN i=0; i<EntitySetVector_[l]->getNumEntities(); i++) {
                        switch (EntitySetVector_[l]->getEntity(i)->getEntityFlag()) {
                            case DefaultFlag: // By default, an entity which belongs to 2 subdomains is a face
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(FaceType);
                                Faces_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case StraightFlag: // If an entity is straight, it is always a straight edge
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(EdgeType);
                                StraightEdges_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case ShortFlag: // If an entity is a short, it is always a short edge
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(EdgeType);
                                ShortEdges_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case NodeFlag: // If an entity is a node, it is always a vertex
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(VertexType);
                                Vertices_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            default:
                                break;
                        }
                    }
                    break;
                default:
                    for (UN i=0; i<EntitySetVector_[l]->getNumEntities(); i++) {
                        switch (EntitySetVector_[l]->getEntity(i)->getEntityFlag()) {
                            case DefaultFlag: // By default, an entity which belongs to more than 2 subdomains is an edge
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(EdgeType);
                                Edges_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case StraightFlag:  // If an entity is straight, it is always a straight edge
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(EdgeType);
                                StraightEdges_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case ShortFlag: // If an entity is a short, it is always a short edge
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(EdgeType);
                                ShortEdges_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            case NodeFlag: // If an entity is a node, it is always a vertex
                                EntitySetVector_[l]->getEntity(i)->resetEntityType(VertexType);
                                Vertices_->addEntity(EntitySetVector_[l]->getEntity(i));
                                break;
                            default:
                                break;
                        }
                    }
                    break;
            }

        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::buildEntityMaps(bool buildVerticesMap,
                                                  bool buildShortEdgesMap,
                                                  bool buildStraightEdgesMap,
                                                  bool buildEdgesMap,
                                                  bool buildFacesMap,
                                                  bool buildRootsMap,
                                                  bool buildLeafsMap)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildEntityMapsTime,"DDInterface::buildEntityMaps");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Building global interface component maps" << endl;

        // PURPOSE: Build global maps for interface entities to enable parallel communication.
        // These maps assign unique global IDs to entities across all processors,
        // which is essential for coarse space construction and parallel operations.

        // STEP 1: Build entity maps for each entity type based on requested flags
        // Each entity set gets its own global map for parallel communication
        if (buildVerticesMap) Vertices_->buildEntityMap(NodesMap_);
        if (buildShortEdgesMap) ShortEdges_->buildEntityMap(NodesMap_);
        if (buildStraightEdgesMap) StraightEdges_->buildEntityMap(NodesMap_);
        if (buildEdgesMap) Edges_->buildEntityMap(NodesMap_);
        if (buildFacesMap) Faces_->buildEntityMap(NodesMap_);
        if (buildRootsMap) Roots_->buildEntityMap(NodesMap_);
        if (buildLeafsMap) Leafs_->buildEntityMap(NodesMap_);

        if (Verbosity_==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
            // Count entities
            GOVec globalVec(7);
            LOVec localVec(7);
            LOVec sumVec(7);
            SCVec avgVec(7);
            LOVec minVec(7);
            LOVec maxVec(7);
            if (buildVerticesMap) {
                globalVec[0] = Vertices_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || Vertices_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[0] += 1;
                }
                if (globalVec[0]<0) globalVec[0] = 0;
                localVec[0] = (LO) max((LO) Vertices_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[0],ptr(&sumVec[0]));
                avgVec[0] = max(sumVec[0]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[0],ptr(&minVec[0]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[0],ptr(&maxVec[0]));
            } else {
                globalVec[0] = -1;
                localVec[0] = -1;
                avgVec[0] = -1;
                minVec[0] = -1;
                maxVec[0] = -1;
                sumVec[0] = -1;
            }
            if (buildShortEdgesMap) {
                globalVec[1] = ShortEdges_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || ShortEdges_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[1] += 1;
                }
                if (globalVec[1]<0) globalVec[1] = 0;
                localVec[1] = (LO) max((LO) ShortEdges_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[1],ptr(&sumVec[1]));
                avgVec[1] = max(sumVec[1]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[1],ptr(&minVec[1]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[1],ptr(&maxVec[1]));
            } else {
                globalVec[1] = -1;
                localVec[1] = -1;
                avgVec[1] = -1;
                minVec[1] = -1;
                maxVec[1] = -1;
                sumVec[1] = -1;
            }
            if (buildStraightEdgesMap) {
                globalVec[2] = StraightEdges_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || StraightEdges_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[2] += 1;
                }
                if (globalVec[2]<0) globalVec[2] = 0;
                localVec[2] = (LO) max((LO) StraightEdges_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[2],ptr(&sumVec[2]));
                avgVec[2] = max(sumVec[2]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[2],ptr(&minVec[2]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[2],ptr(&maxVec[2]));
            } else {
                globalVec[2] = -1;
                localVec[2] = -1;
                avgVec[2] = -1;
                minVec[2] = -1;
                maxVec[2] = -1;
                sumVec[2] = -1;
            }
            if (buildEdgesMap) {
                globalVec[3] = (LO) Edges_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || Edges_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[3] += 1;
                }
                if (globalVec[3]<0) globalVec[3] = 0;
                localVec[3] = max((LO) Edges_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[3],ptr(&sumVec[3]));
                avgVec[3] = max(sumVec[3]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[3],ptr(&minVec[3]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[3],ptr(&maxVec[3]));
            } else {
                globalVec[3] = -1;
                localVec[3] = -1;
                avgVec[3] = -1;
                minVec[3] = -1;
                maxVec[3] = -1;
                sumVec[3] = -1;
            }
            if (buildFacesMap) {
                globalVec[4] = (LO) Faces_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || Faces_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[4] += 1;
                }
                if (globalVec[4]<0) globalVec[4] = 0;
                localVec[4] = max((LO) Faces_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[4],ptr(&sumVec[4]));
                avgVec[4] = max(sumVec[4]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[4],ptr(&minVec[4]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[4],ptr(&maxVec[4]));
            } else {
                globalVec[4] = -1;
                localVec[4] = -1;
                avgVec[4] = -1;
                minVec[4] = -1;
                maxVec[4] = -1;
                sumVec[4] = -1;
            }
            if (buildRootsMap) {
                globalVec[5] = Roots_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || Roots_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[5] += 1;
                }
                if (globalVec[5]<0) globalVec[5] = 0;
                localVec[5] = (LO) max((LO) Roots_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[5],ptr(&sumVec[5]));
                avgVec[5] = max(sumVec[5]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[5],ptr(&minVec[5]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[5],ptr(&maxVec[5]));
            } else {
                globalVec[5] = -1;
                localVec[5] = -1;
                avgVec[5] = -1;
                minVec[5] = -1;
                maxVec[5] = -1;
                sumVec[5] = -1;
            }
            if (buildLeafsMap) {
                globalVec[6] = Leafs_->getEntityMap()->getMaxAllGlobalIndex();
                if (NodesMap_->lib()==UseEpetra || Leafs_->getEntityMap()->getGlobalNumElements()>0) {
                    globalVec[6] += 1;
                }
                if (globalVec[6]<0) globalVec[6] = 0;
                localVec[6] = (LO) max((LO) Leafs_->getEntityMap()->getLocalNumElements(),(LO) 0);
                reduceAll(*this->MpiComm_,REDUCE_SUM,localVec[6],ptr(&sumVec[6]));
                avgVec[6] = max(sumVec[6]/double(MpiComm_->getSize()),0.0);
                reduceAll(*MpiComm_,REDUCE_MIN,localVec[6],ptr(&minVec[6]));
                reduceAll(*MpiComm_,REDUCE_MAX,localVec[6],ptr(&maxVec[6]));
            } else {
                globalVec[6] = -1;
                localVec[6] = -1;
                avgVec[6] = -1;
                minVec[6] = -1;
                maxVec[6] = -1;
                sumVec[6] = -1;
            }

            for (UN i=0; i<globalVec.size(); i++) {
                if (globalVec[i]<0) {
                    globalVec[i] = -1;
                }
            }

            NumEntity_ = globalVec;

            if (Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| "
                << left << setw(74) << "> Interface Statistics " << right << setw(8) << "(Level " << setw(2) << LevelID_ << ")"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "========================================================================================="
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(41) << "Interface communication strategy" << right
                << " | " << setw(41) << CommStrategy_
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << " " << right
                << " | " << setw(10) << "total"
                << " | " << setw(10) << "avg"
                << " | " << setw(10) << "min"
                << " | " << setw(10) << "max"
                << " | " << setw(10) << "global sum"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Vertices" << right
                << " | "; globalVec[0]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[0]; cout
                << " | "; avgVec[0]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[0]; cout
                << " | "; minVec[0]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[0]; cout
                << " | "; maxVec[0]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[0]; cout
                << " | "; sumVec[0]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[0]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Short edges" << right
                << " | "; globalVec[1]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[1]; cout
                << " | "; avgVec[1]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[1]; cout
                << " | "; minVec[1]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[1]; cout
                << " | "; maxVec[1]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[1]; cout
                << " | "; sumVec[1]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[1]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Straight edges" << right
                << " | "; globalVec[2]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[2]; cout
                << " | "; avgVec[2]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[2]; cout
                << " | "; minVec[2]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[2]; cout
                << " | "; maxVec[2]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[2]; cout
                << " | "; sumVec[2]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[2]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Edges" << right
                << " | "; globalVec[3]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[3]; cout
                << " | "; avgVec[3]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[3]; cout
                << " | "; minVec[3]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[3]; cout
                << " | "; maxVec[3]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[3]; cout
                << " | "; sumVec[3]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[3]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Faces" << right
                << " | "; globalVec[4]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[4]; cout
                << " | "; avgVec[4]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[4]; cout
                << " | "; minVec[4]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[4]; cout
                << " | "; maxVec[4]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[4]; cout
                << " | "; sumVec[4]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[4]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Roots" << right
                << " | "; globalVec[5]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[5]; cout
                << " | "; avgVec[5]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[5]; cout
                << " | "; minVec[5]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[5]; cout
                << " | "; maxVec[5]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[5]; cout
                << " | "; sumVec[5]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[5]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Leafs" << right
                << " | "; globalVec[6]<0 ? cout << setw(10) << " " : cout << setw(10) << globalVec[6]; cout
                << " | "; avgVec[6]<0 ? cout << setw(10) << " " : cout << setw(10) << setprecision(5) << avgVec[6]; cout
                << " | "; minVec[6]<0 ? cout << setw(10) << " " : cout << setw(10) << minVec[6]; cout
                << " | "; maxVec[6]<0 ? cout << setw(10) << " " : cout << setw(10) << maxVec[6]; cout
                << " | "; sumVec[6]<0 ? cout << setw(10) << " " : cout << setw(10) << sumVec[6]; cout
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << endl;
            }
        }

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::buildEntityHierarchy()
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildEntityHierarchyTime,"DDInterface::buildEntityHierarchy");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Building hierarchy of interface components" << endl;

        // Build hierarchy
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            for (UN j=i+1; j<EntitySetVector_.size(); j++) {
                // Look for ancestors in the sets with higher multiplicity -> edge is ancestor of a face
                // This also sets the offspring of all the ancestor entities -> we only need to traverse the tree once.
                EntitySetVector_[i]->findAncestorsInSet(EntitySetVector_[j]);
            }
        }

        // Find roots i.e. ancestors that don't have ancestors themselves
        // NOTE: [KH] In the boundary framework, boundary entities might not have any roots in their ancestors e.g.
        // RGDSW root lies inside the global domain. Nevertheless Roots_ stores these internal entities to indiciate to
        // which coarse basis function the boundary entity belongs.
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetPtr tmpRoots = EntitySetVector_[i]->findRoots();
            Roots_->addEntitySet(tmpRoots);
        }
        Roots_->sortUnique();
        Roots_->setRootID();

        // Find Leafs i.e. offspring that don't have offspring themselves
        // NOTE: [KH] The boundary framework might have broken the usage of leaves (leafs) in GDSW-star.
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetPtr tmpLeafs = EntitySetVector_[i]->findLeafs();
            Leafs_->addEntitySet(tmpLeafs);
        }
        Leafs_->sortUnique();
        Leafs_->setLeafID();
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::computeDistancesToRoots(UN dimension,
                                                          ConstXMultiVectorPtr &nodeList,
                                                          DistanceFunction distanceFunction)
    {
        FROSCH_DETAILTIMER_START_LEVELID(computeDistancesToRootsTime,"DDInterface::computeDistancesToRoots");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Computing distances to the coarse nodes" << endl;

        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetVector_[i]->computeDistancesToRoots(dimension,nodeList,distanceFunction);
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::identifyConnectivityEntities(UNVecPtr multiplicities,
                                                               EntityFlagVecPtr flags)
    {
        FROSCH_DETAILTIMER_START_LEVELID(identifyConnectivityEntitiesTime,"DDInterface::identifyConnectivityEntities");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Preparing subdomain graph" << endl;

        if (multiplicities.is_null()) {
            multiplicities = UNVecPtr(1,2);
        }
        if (flags.is_null()) {
            flags = EntityFlagVecPtr(4);
            flags[0] = DefaultFlag;
            flags[1] = StraightFlag;
            flags[2] = ShortFlag;
            flags[3] = NodeFlag;
        }

        for (UN j=0; j<multiplicities.size(); j++) {
            for (UN i=0; i<EntitySetVector_[multiplicities[j]]->getNumEntities(); i++) {
                if (binary_search(flags.begin(),flags.end(),EntitySetVector_[multiplicities[j]]->getEntity(i)->getEntityFlag())) {
                    ConnectivityEntities_->addEntity(EntitySetVector_[multiplicities[j]]->getEntity(i));
                }
            }
        }
        return 0;
    }

    // Part of the boundary framework
    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::computeDistancesOnBoundary(UN dimension, ConstXMultiVectorPtr &nodeList){
        // Calculate distances
        for (int i = 0; i < EntitySetVector_[1]->getNumEntities(); i++){
            EntitySetVector_[1]->getEntity(i)->computeDistancesOnBoundary(dimension, nodeList, EntitySetVector_);
        }
        return 0;
    }

    // Part of the boundary framework
    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::computeDistancesToDirichletBoundary(UN dimension, ConstXMultiVectorPtr &nodeList){
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetVector_[i]->computeDistancesToDirichletBoundary(dimension, nodeList, EntitySetVector_);
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::UN DDInterface<SC,LO,GO,NO>::getDimension() const
    {
        return Dimension_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::UN DDInterface<SC,LO,GO,NO>::getDofsPerNode() const
    {
        return DofsPerNode_;
    }

    template <class SC,class LO,class GO,class NO>
    LO DDInterface<SC,LO,GO,NO>::getNumMyNodes() const
    {
        return NumMyNodes_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getVertices() const
    {
        return Vertices_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getShortEdges() const
    {
        return ShortEdges_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getStraightEdges() const
    {
        return StraightEdges_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getEdges() const
    {
        return Edges_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getFaces() const
    {
        return Faces_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getInterface() const
    {
        return Interface_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getInterior() const
    {
        return Interior_;
    }
 
    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getRoots() const
    {
        return Roots_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getLeafs() const
    {
        return Leafs_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetPtrConstVecPtr & DDInterface<SC,LO,GO,NO>::getEntitySetVector() const
    {
        return EntitySetVector_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::GOVec DDInterface<SC,LO,GO,NO>::getNumEnt() const
    {
        return NumEntity_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::EntitySetConstPtr & DDInterface<SC,LO,GO,NO>::getConnectivityEntities() const
    {
        return ConnectivityEntities_;
    }

    template <class SC,class LO,class GO,class NO>
    typename DDInterface<SC,LO,GO,NO>::ConstXMapPtr DDInterface<SC,LO,GO,NO>::getNodesMap() const
    {
        return NodesMap_.getConst();
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::communicateLocalComponents(IntVecVecPtr &componentsSubdomains,
                                                             IntVecVec &componentsSubdomainsUnique)
    {
        FROSCH_DETAILTIMER_START_LEVELID(communicateLocalComponentsTime,"DDInterface::communicateLocalComponents");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Communicating nodes" << endl;

        // PURPOSE: This function determines which subdomains each node belongs to in a domain decomposition.
        // It communicates information across processors to identify shared nodes (interface nodes) and
        // interior nodes. The result is stored in componentsSubdomains where each entry contains
        // the list of subdomain IDs (processor ranks) that own each node.

        // Handle Epetra compatibility issue with CreateOneToOneMap strategy
        if (NodesMap_->lib() == UseEpetra && CommStrategy_ == CreateOneToOneMap) {
            FROSCH_WARNING("FROSch::DDInterface",Verbose_,"CreateOneToOneMap communication strategy does not work for Epetra => Switching to CommCrsGraph.");
            CommStrategy_ = CommCrsGraph;
        }

        // Different communication strategies for determining subdomain ownership
        const GO INVALID = Teuchos::OrdinalTraits<GO>::invalid();
        switch (CommStrategy_) {
            case CommCrsMatrix:
                {
                    // STRATEGY 1: Use CRS Matrix for communication
                    // This strategy creates a communication matrix where each row represents a node
                    // and each column represents a subdomain. The matrix entries indicate ownership.
                    
                    // Create a unique map for nodes (removes duplicates across processors)
                    UniqueNodesMap_ = BuildUniqueMap<LO,GO,NO>(NodesMap_);
                    
                    // Create communication matrix with estimated 10 non-zeros per row
                    // (assuming each node belongs to at most 10 subdomains)
                    RCP<Matrix<SC,LO,GO,NO> > commMat = MatrixFactory<SC,LO,GO,NO>::Build(NodesMap_,10);
                    RCP<Matrix<SC,LO,GO,NO> > commMatTmp = MatrixFactory<SC,LO,GO,NO>::Build(UniqueNodesMap_,10);
                    XExportPtr commExporter = ExportFactory<LO,GO,NO>::Build(NodesMap_,UniqueNodesMap_);

                    // Insert processor rank (PID) for each local node into the communication matrix
                    Array<SC> one(1,ScalarTraits<SC>::one());
                    Array<GO> myPID(1,MpiComm_->getRank());
                    for (int i=0; i<NumMyNodes_; i++) {
                        // For each local node, insert the current processor rank
                        commMat->insertGlobalValues(NodesMap_->getGlobalElement(i),myPID(),one());
                    }
                    
                    // Create domain map for the communication matrix
                    XMapPtr domainMap = MapFactory<LO,GO,NO>::Build(NodesMap_->lib(),INVALID,myPID(),0,NodesMap_->getComm());

                    // Complete the matrix construction and perform communication
                    commMat->fillComplete(domainMap,NodesMap_);
                    commMatTmp->doExport(*commMat,*commExporter,INSERT);  // Export to unique map
                    commMatTmp->fillComplete(domainMap,UniqueNodesMap_);
                    commMat = MatrixFactory<SC,LO,GO,NO>::Build(NodesMap_,LO(0));
                    commMat->doImport(*commMatTmp,*commExporter,INSERT);  // Import back to original map

                    // Extract subdomain information for each node
                    componentsSubdomains = IntVecVecPtr(NumMyNodes_);

                    ArrayView<const GO> indices;
                    ArrayView<const SC> values;
                    for (LO i=0; i<NumMyNodes_; i++) {
                        // Get all processor ranks that own this node
                        commMat->getGlobalRowView(NodesMap_->getGlobalElement(i),indices,values);
                        componentsSubdomains[i].resize(indices.size());
                        for (LO j=0; j<indices.size(); j++) {
                            componentsSubdomains[i][j] = as<int>(indices[j]);
                        }
                    }
                }
                break;

            case CommCrsGraph:
                {
                    // STRATEGY 2: Use CRS Graph for communication (more memory efficient)
                    // Similar to CommCrsMatrix but uses graph structure instead of matrix values
                    
                    // Create unique map for nodes
                    UniqueNodesMap_ = BuildUniqueMap<LO,GO,NO>(NodesMap_);

                    // Create communication graph with estimated 10 connections per node
                    XCrsGraphPtr commGraph = CrsGraphFactory<LO,GO,NO>::Build(NodesMap_,10);
                    XCrsGraphPtr commGraphTmp = CrsGraphFactory<LO,GO,NO>::Build(UniqueNodesMap_,10);
                    XExportPtr commExporter = ExportFactory<LO,GO,NO>::Build(NodesMap_,UniqueNodesMap_);

                    // Insert processor rank for each local node
                    Array<GO> myPID(1,MpiComm_->getRank());
                    for (int i=0; i<NumMyNodes_; i++) {
                        commGraph->insertGlobalIndices(NodesMap_->getGlobalElement(i),myPID());
                    }
                    XMapPtr domainMap = MapFactory<LO,GO,NO>::Build(NodesMap_->lib(),INVALID,myPID(),0,NodesMap_->getComm());

                    // Complete graph construction and perform communication
                    commGraph->fillComplete(domainMap,NodesMap_);
                    commGraphTmp->doExport(*commGraph,*commExporter,INSERT);
                    commGraphTmp->fillComplete(domainMap,UniqueNodesMap_);
                    commGraph = CrsGraphFactory<LO,GO,NO>::Build(NodesMap_);
                    commGraph->doImport(*commGraphTmp,*commExporter,INSERT);

                    // Extract subdomain information for each node
                    componentsSubdomains = IntVecVecPtr(NumMyNodes_);

                    ArrayView<const GO> indices;
                    for (LO i=0; i<NumMyNodes_; i++) {
                        // Get all processor ranks connected to this node
                        commGraph->getGlobalRowView(NodesMap_->getGlobalElement(i),indices);
                        componentsSubdomains[i].resize(indices.size());
                        for (LO j=0; j<indices.size(); j++) {
                            componentsSubdomains[i][j] = as<int>(indices[j]);
                        }
                    }
                }
                break;

            case CreateOneToOneMap:
                {
                    // STRATEGY 3: Create one-to-one map with tie-breaking
                    // This strategy uses a sophisticated tie-breaking mechanism to assign
                    // each shared node to exactly one processor, creating a unique ownership
                    
                    // Create tie-breaking object that handles conflicts when multiple processors
                    // claim ownership of the same node
                    RCP<LowerPIDTieBreak<LO,GO,NO> > lowerPIDTieBreak(new LowerPIDTieBreak<LO,GO,NO>(MpiComm_,NodesMap_,Dimension_,LevelID_));
                    
                    // Build unique map using tie-breaking strategy
                    UniqueNodesMap_ = BuildUniqueMap<LO,GO,NO>(NodesMap_,true,lowerPIDTieBreak);
                    
                    // Send ownership data back to original processors
                    lowerPIDTieBreak->sendDataToOriginalMap();
                    
                    // Get the final subdomain assignments
                    componentsSubdomains = lowerPIDTieBreak->getComponents();
                }
                break;

            default:
                FROSCH_ASSERT(false,"FROSch::DDInterface: Specify a valid communication strategy.");
        }

        // Post-process the results: create unique sorted lists and handle edge cases
        componentsSubdomainsUnique = IntVecVec(NumMyNodes_);
        for (LO i=0; i<NumMyNodes_; i++) {
            // Sort and remove duplicates from the subdomain list for each node
            sortunique(componentsSubdomains[i]);
            
            // Handle special case: if no subdomains are found (shouldn't happen in normal cases),
            // assign the node to the current processor
            if (componentsSubdomains[i].size() == 0) {
                componentsSubdomains[i].push_back(MpiComm_->getRank());
            }
            
            // Store the processed result
            componentsSubdomainsUnique[i] = componentsSubdomains[i];
        }

        // Sort the unique components list globally
        // This now contains one entry for each unique equivalence class in the subdomain e.g. an interface edge between
        // processors 1 and 2 will be contained here as a single entry [1, 2]. In contrast, componentsSubdomains will
        // contain such entries for each node in the interface edge.
        sortunique(componentsSubdomainsUnique);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int DDInterface<SC,LO,GO,NO>::identifyLocalComponents(IntVecVecPtr &componentsSubdomains,
                                                          IntVecVec &componentsSubdomainsUnique)
    {
        FROSCH_DETAILTIMER_START_LEVELID(identifyLocalComponentsTime,"DDInterface::identifyLocalComponents");
        //if (Verbose_ && Verbosity_==All) cout << "FROSch::DDInterface : Classifying interface components based on equivalence classes" << endl;

        // PURPOSE: This function classifies nodes into interface components (vertices, edges, faces) based on
        // their subdomain multiplicity. It groups nodes with the same subdomain membership into equivalence
        // classes and creates the appropriate interface entities.

        // STEP 1: Analyze equivalence classes and determine multiplicity
        // componentsSubdomainsUnique contains unique combinations of subdomain IDs (equivalence classes). This is in
        // contrast to componentsSubdomains which contains one entry per node in the subdomain. Each entry is a vector
        // containing the subdomains the node is part of. By removing duplicates componentsSubdomainsUnique is built,
        // exactly the equivalence classes in the subdomain.
        UNVecPtr componentsMultiplicity(componentsSubdomainsUnique.size());
        IntVecVecPtr components(componentsSubdomainsUnique.size());        // Local node indices for each equivalence class
        IntVecVecPtr componentsGamma(componentsSubdomainsUnique.size());    // Interface node indices for each equivalence class
        UN maxMultiplicity = 0;
        
        // Calculate multiplicity (number of subdomains) for each equivalence class
        for (UN i=0; i<componentsSubdomainsUnique.size(); i++) {
            componentsMultiplicity[i] = componentsSubdomainsUnique[i].size();
            maxMultiplicity = max(maxMultiplicity,componentsMultiplicity[i]);
        }
        
        // Create EntitySetVector indexed by multiplicity (0 to maxMultiplicity)
        // EntitySetVector[i] contains all entities that belong to exactly i subdomains
        EntitySetVector_ = EntitySetPtrVecPtr(maxMultiplicity+1);
        for (UN i=0; i<maxMultiplicity+1; i++) {
            EntitySetVector_[i].reset(new EntitySet<SC,LO,GO,NO>(DefaultType));
        }

        // STEP 2: Map each node to its equivalence class
        // For each local node, find which equivalence class it belongs to
        typename IntVecVec::iterator classIterator;
        LOVecPtr localComponentIndices(NumMyNodes_);
        for (int i=0; i<NumMyNodes_; i++) {
            // Use binary search to find the equivalence class for this node's subdomain list
            classIterator = lower_bound(componentsSubdomainsUnique.begin(),componentsSubdomainsUnique.end(),componentsSubdomains[i]);
            localComponentIndices[i] = classIterator - componentsSubdomainsUnique.begin();
        }

        // STEP 3: Create Interior and Interface entities
        // Interior: nodes belonging to only one subdomain (multiplicity = 1)
        // Interface: nodes belonging to multiple subdomains (multiplicity > 1)
        LO tmp1 = 0; // The interface and interior have multiplicity 0 in our construction
        int *tmp2 = NULL;
        RCP<InterfaceEntity<SC,LO,GO,NO> > interior(new InterfaceEntity<SC,LO,GO,NO>(InteriorType,DofsPerNode_,tmp1,tmp2));
        RCP<InterfaceEntity<SC,LO,GO,NO> > interface(new InterfaceEntity<SC,LO,GO,NO>(InterfaceType,DofsPerNode_,tmp1,tmp2));
        
        for (LO i=0; i<NumMyNodes_; i++) {
            if (componentsMultiplicity[localComponentIndices[i]] == 1) {
                // INTERIOR NODE: belongs to only one subdomain
                LO nodeIDI = interior->getNumNodes();
                LO nodeIDLocal = i;
                GO nodeIDGlobal = NodesMap_->getGlobalElement(nodeIDLocal);
                LOVecPtr dofsI(DofsPerNode_);
                LOVecPtr dofsLocal(DofsPerNode_);
                GOVecPtr dofsGlobal(DofsPerNode_);
                for (UN k=0; k<DofsPerNode_; k++) {
                    dofsI[k] = DofsPerNode_*nodeIDI+k;
                    dofsLocal[k] = DofsPerNode_*nodeIDLocal+k;
                    dofsGlobal[k] = DofsPerNode_*nodeIDGlobal+k;
                }
                interior->addNode(nodeIDI,nodeIDLocal,nodeIDGlobal,DofsPerNode_,dofsI,dofsLocal,dofsGlobal);
            } else {
                // INTERFACE NODE: belongs to multiple subdomains
                FROSCH_ASSERT(componentsMultiplicity[localComponentIndices[i]]>1,"FROSch::DDInterface: There cannot be any nodes with multiplicity 0.");
                LO nodeIDGamma = interface->getNumNodes();
                LO nodeIDLocal = i;
                GO nodeIDGlobal = NodesMap_->getGlobalElement(nodeIDLocal);
                LOVecPtr dofsGamma(DofsPerNode_);
                LOVecPtr dofsLocal(DofsPerNode_);
                GOVecPtr dofsGlobal(DofsPerNode_);
                for (UN k=0; k<DofsPerNode_; k++) {
                    dofsGamma[k] = DofsPerNode_*nodeIDGamma+k;
                    dofsLocal[k] = DofsPerNode_*nodeIDLocal+k;
                    dofsGlobal[k] = DofsPerNode_*nodeIDGlobal+k;
                }
                interface->addNode(nodeIDGamma,nodeIDLocal,nodeIDGlobal,DofsPerNode_,dofsGamma,dofsLocal,dofsGlobal);

                // Track which local nodes belong to each equivalence class
                components[localComponentIndices[i]].push_back(i);
                componentsGamma[localComponentIndices[i]].push_back(interface->getNumNodes()-1);
            }
        }
        
        // Add the interior and interface entities to the entity sets
        Interior_->addEntity(interior);
        Interface_->addEntity(interface);

        // STEP 4: Create interface entities for each equivalence class
        // Each equivalence class becomes an interface entity (vertex, edge, or face)
        for (UN i=0; i<componentsSubdomainsUnique.size(); i++) {
            FROSCH_ASSERT(componentsMultiplicity[i]>0,"FROSch::DDInterface: There cannot be any component with multiplicity 0.");
            
            // Create interface entity with the subdomain information
            RCP<InterfaceEntity<SC,LO,GO,NO> > tmpEntity(new InterfaceEntity<SC,LO,GO,NO>(VertexType,DofsPerNode_,componentsMultiplicity[i],&(componentsSubdomainsUnique[i][0])));
            LO nodeIDGamma;
            LO nodeIDLocal;
            GO nodeIDGlobal;
            LOVecPtr dofsGamma(DofsPerNode_);
            LOVecPtr dofsLocal(DofsPerNode_);
            GOVecPtr dofsGlobal(DofsPerNode_);

            // Sort the local node indices for this equivalence class
            sortunique(components[i]);

            // Add all nodes belonging to this equivalence class to the interface entity
            for (UN j=0; j<components[i].size(); j++) {
                nodeIDGamma = componentsGamma[i][j];      // ID amongst all nodes on the interface in this subdomain across all equivalence classes
                nodeIDLocal = components[i][j];            // ID amongst all nodes in this subdomain across all equivalence classes
                nodeIDGlobal = NodesMap_->getGlobalElement(nodeIDLocal);  // Global node ID
                for (UN k=0; k<DofsPerNode_; k++) {
                    dofsGamma[k] = DofsPerNode_*nodeIDGamma+k;
                    dofsLocal[k] = DofsPerNode_*nodeIDLocal+k;
                    dofsGlobal[k] = DofsPerNode_*nodeIDGlobal+k;
                }

                tmpEntity->addNode(nodeIDGamma,nodeIDLocal,nodeIDGlobal,DofsPerNode_,dofsGamma,dofsLocal,dofsGlobal);
            }
            
            // Set entity type to default (will be refined later in sortVerticesEdgesFaces)
            tmpEntity->resetEntityType(DefaultType);
            
            // Add entity to the appropriate EntitySet based on multiplicity
            EntitySetVector_[componentsMultiplicity[i]]->addEntity(tmpEntity);
        }

        // STEP 5: Clean up and finalize
        // Remove any empty entities that might have been created
        removeEmptyEntities();

        // Sort the nodes in the entity by their global ID and set the entity ID to the first global node ID
        for (UN i=0; i<EntitySetVector_.size(); i++) {
            EntitySetVector_[i]->setUniqueIDToFirstGlobalNodeID();
        }
        return 0;
    }

}

#endif
