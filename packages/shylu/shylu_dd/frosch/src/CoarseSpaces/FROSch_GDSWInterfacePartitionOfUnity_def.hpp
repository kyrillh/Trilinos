// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_GDSWINTERFACEPARTITIONOFUNITY_DEF_HPP
#define _FROSCH_GDSWINTERFACEPARTITIONOFUNITY_DEF_HPP

#include <FROSch_GDSWInterfacePartitionOfUnity_decl.hpp>
#include <FROSch_InterfacePartitionOfUnity_def.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,class LO,class GO,class NO>
    GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>::GDSWInterfacePartitionOfUnity(CommPtr mpiComm,
                                                                              CommPtr serialComm,
                                                                              UN dimension,
                                                                              UN dofsPerNode,
                                                                              ConstXMapPtr nodesMap,
                                                                              ConstXMapPtrVecPtr dofsMaps,
                                                                              ParameterListPtr parameterList,
                                                                              Verbosity verbosity,
                                                                              UN levelID) :
    InterfacePartitionOfUnity<SC,LO,GO,NO> (mpiComm,serialComm,dimension,dofsPerNode,nodesMap,dofsMaps,parameterList,verbosity,levelID)
    {
        FROSCH_DETAILTIMER_START_LEVELID(gDSWInterfacePartitionOfUnityTime,"GDSWInterfacePartitionOfUnity::GDSWInterfacePartitionOfUnity");
        if (!this->ParameterList_->get("Type","Full").compare("Full")) {
            UseVertices_ = true;
            UseShortEdges_ = true;
            UseStraightEdges_ = true;
            UseEdges_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("Vertices")) {
            UseVertices_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("ShortEdges")) {
            UseShortEdges_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("StraightEdges")) {
            UseStraightEdges_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("Edges")) {
            UseEdges_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("AllEdges")) {
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoVertices")) {
            UseShortEdges_ = true;
            UseStraightEdges_ = true;
            UseEdges_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoShortEdges")) {
            UseVertices_ = true;
            UseStraightEdges_ = true;
            UseEdges_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoStraightEdges")) {
            UseVertices_ = true;
            UseShortEdges_ = true;
            UseEdges_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoEdges")) {
            UseVertices_ = true;
            UseShortEdges_ = true;
            UseStraightEdges_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoAllEdges")) {
            UseVertices_ = true;
            UseFaces_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("NoFaces")) {
            UseVertices_ = true;
            UseShortEdges_ = true;
            UseStraightEdges_ = true;
            UseEdges_ = true;
        } else if (!this->ParameterList_->get("Type","Full").compare("Custom")) {
            UseVertices_ = this->ParameterList_->sublist("Custom").get("Vertices",false);
            UseShortEdges_ = this->ParameterList_->sublist("Custom").get("Vertices",false);
            UseStraightEdges_ = this->ParameterList_->sublist("Custom").get("Vertices",false);
            UseEdges_ = this->ParameterList_->sublist("Custom").get("Vertices",false);
            UseFaces_ = this->ParameterList_->sublist("Custom").get("Vertices",false);
        } else {
            FROSCH_ASSERT(false,"FROSch::GDSWInterfacePartitionOfUnity: Specify a valid Type.");
        }
        this->LocalPartitionOfUnity_ = ConstXMultiVectorPtrVecPtr(5);
        this->PartitionOfUnityMaps_ = ConstXMapPtrVecPtr(5);
    }

    template <class SC,class LO,class GO,class NO>
    GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>::~GDSWInterfacePartitionOfUnity()
    {

    }

    template <class SC,class LO,class GO,class NO>
    int GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>::removeDirichletNodes(GOVecView dirichletBoundaryDofs,
                                                                         ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(removeDirichletNodesTime,"GDSWInterfacePartitionOfUnity::removeDirichletNodes");
        if (!dirichletBoundaryDofs.is_null()) {
            GOVec tmpDirichletBoundaryDofs(dirichletBoundaryDofs());
            sortunique(tmpDirichletBoundaryDofs);
            this->DDInterface_->removeDirichletNodes(tmpDirichletBoundaryDofs());
            this->DDInterface_->sortVerticesEdgesFaces(nodeList);
        }
        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>::sortInterface(ConstXMatrixPtr matrix,
                                                                  ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(sortInterfaceTime,"GDSWInterfacePartitionOfUnity::sortInterface");
        if (this->ParameterList_->get("Test Unconnected Interface",true)) {
            if (matrix.is_null()) {
                FROSCH_WARNING("FROSch::GDSWInterfacePartitionOfUnity",this->Verbose_,"divideUnconnectedEntities() cannot be performed without the matrix.");
            } else this->DDInterface_->divideUnconnectedEntities(matrix);
        }
        this->DDInterface_->sortVerticesEdgesFaces(nodeList);

        return 0;
    }

    template <class SC,class LO,class GO,class NO>
    int GDSWInterfacePartitionOfUnity<SC,LO,GO,NO>::computePartitionOfUnity(ConstXMultiVectorPtr nodeList)
    {
        FROSCH_DETAILTIMER_START_LEVELID(computePartitionOfUnityTime,"GDSWInterfacePartitionOfUnity::computePartitionOfUnity");
        
        // PURPOSE: Compute the GDSW (Generalized Dryja-Smith-Widlund) partition of unity functions.
        // This function constructs coarse basis functions for each interface entity type
        // (vertices, edges, faces) by assigning constant values of 1.0 to all nodes
        // belonging to each entity. This creates a simple partition of unity where
        // each entity contributes equally to the coarse space.

        // STEP 1: Get interface dimensions and setup
        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();
        UN numInterfaceDofs = dofsPerNode*this->DDInterface_->getInterface()->getEntity(0)->getNumNodes();

        // STEP 2: Build entity maps for parallel communication
        // Create global maps for each entity type that will be used in the coarse space
        this->DDInterface_->buildEntityMaps(UseVertices_,
                                            UseShortEdges_,
                                            UseStraightEdges_,
                                            UseEdges_,
                                            UseFaces_,
                                            false,  // Don't build roots map
                                            false); // Don't build leafs map

        // STEP 3: Store entity sets and their maps for each enabled entity type
        // Each entity type gets its own map for parallel communication
        if (UseVertices_) {
            Vertices_ = this->DDInterface_->getVertices();
            this->PartitionOfUnityMaps_[0] = Vertices_->getEntityMap();
        }

        if (UseShortEdges_) {
            ShortEdges_ = this->DDInterface_->getShortEdges();
            this->PartitionOfUnityMaps_[1] = ShortEdges_->getEntityMap();
        }

        if (UseStraightEdges_) {
            StraightEdges_ = this->DDInterface_->getStraightEdges();
            this->PartitionOfUnityMaps_[2] = StraightEdges_->getEntityMap();
        }

        if (UseEdges_) {
            Edges_ = this->DDInterface_->getEdges();
            this->PartitionOfUnityMaps_[3] = Edges_->getEntityMap();
        }

        if (UseFaces_) {
            Faces_ = this->DDInterface_->getFaces();
            this->PartitionOfUnityMaps_[4] = Faces_->getEntityMap();
        }

        // STEP 4: Print configuration information (if verbose output enabled)
        if (this->Verbose_) {
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| "
            << left << setw(74) << "> GDSW Interface Partition Of Unity " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")" << right
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Vertices" << right
            << " | " << setw(41) << boolalpha << UseVertices_ << noboolalpha
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Short edges" << right
            << " | " << setw(41) << boolalpha << UseShortEdges_ << noboolalpha
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Straight edges" << right
            << " | " << setw(41) << boolalpha << UseStraightEdges_ << noboolalpha
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Edges" << right
            << " | " << setw(41) << boolalpha << UseEdges_ << noboolalpha
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Faces" << right
            << " | " << setw(41) << boolalpha << UseFaces_ << noboolalpha
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
        }

        // STEP 5: Build partition of unity vectors for each entity type
        // Create a serial map for the interface DOFs (used for local computation)
        XMapPtr serialInterfaceMap = MapFactory<LO,GO,NO>::Build(this->DDInterface_->getNodesMap()->lib(),numInterfaceDofs,0,this->SerialComm_);

        // STEP 5a: Build vertices partition of unity vectors
        // Each vertex gets a constant value of 1.0 for all its nodes
        if (UseVertices_ && Vertices_->getNumEntities()>0) {
            XMultiVectorPtr tmpVector = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,Vertices_->getNumEntities());

            for (UN i=0; i<Vertices_->getNumEntities(); i++) {
                for (UN j=0; j<Vertices_->getEntity(i)->getNumNodes(); j++) {
                    for (UN k=0; k<dofsPerNode; k++) {
                        // Assign value 1.0 to all DOFs belonging to this vertex
                        tmpVector->replaceLocalValue(Vertices_->getEntity(i)->getGammaDofID(j,k),i,ScalarTraits<SC>::one());
                    }
                }
            }

            this->LocalPartitionOfUnity_[0] = tmpVector;
        }

        // STEP 5b: Build short edges partition of unity vectors
        // Each short edge gets a constant value of 1.0 for all its nodes
        if (UseShortEdges_ && ShortEdges_->getNumEntities()>0) {
            XMultiVectorPtr tmpVector = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,ShortEdges_->getNumEntities());

            for (UN i=0; i<ShortEdges_->getNumEntities(); i++) {
                for (UN j=0; j<ShortEdges_->getEntity(i)->getNumNodes(); j++) {
                    for (UN k=0; k<dofsPerNode; k++) {
                        // Assign value 1.0 to all DOFs belonging to this short edge
                        tmpVector->replaceLocalValue(ShortEdges_->getEntity(i)->getGammaDofID(j,k),i,ScalarTraits<SC>::one());
                    }
                }
            }

            this->LocalPartitionOfUnity_[1] = tmpVector;
        }

        // STEP 5c: Build straight edges partition of unity vectors
        // Each straight edge gets a constant value of 1.0 for all its nodes
        if (UseStraightEdges_ && StraightEdges_->getNumEntities()>0) {
            XMultiVectorPtr tmpVector = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,StraightEdges_->getNumEntities());

            for (UN i=0; i<StraightEdges_->getNumEntities(); i++) {
                for (UN j=0; j<StraightEdges_->getEntity(i)->getNumNodes(); j++) {
                    for (UN k=0; k<dofsPerNode; k++) {
                        // Assign value 1.0 to all DOFs belonging to this straight edge
                        tmpVector->replaceLocalValue(StraightEdges_->getEntity(i)->getGammaDofID(j,k),i,ScalarTraits<SC>::one());
                    }
                }
            }
            this->LocalPartitionOfUnity_[2] = tmpVector;
        }

        // STEP 5d: Build general edges partition of unity vectors
        // Each edge gets a constant value of 1.0 for all its nodes
        if (UseEdges_ && Edges_->getNumEntities()>0) {
            XMultiVectorPtr tmpVector = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,Edges_->getNumEntities());

            for (UN i=0; i<Edges_->getNumEntities(); i++) {
                for (UN j=0; j<Edges_->getEntity(i)->getNumNodes(); j++) {
                    for (UN k=0; k<dofsPerNode; k++) {
                        // Assign value 1.0 to all DOFs belonging to this edge
                        tmpVector->replaceLocalValue(Edges_->getEntity(i)->getGammaDofID(j,k),i,ScalarTraits<SC>::one());
                    }
                }
            }

            this->LocalPartitionOfUnity_[3] = tmpVector;
        }

        // STEP 5e: Build faces partition of unity vectors
        // Each face gets a constant value of 1.0 for all its nodes
        if (UseFaces_ && Faces_->getNumEntities()>0) {
            XMultiVectorPtr tmpVector = MultiVectorFactory<SC,LO,GO,NO>::Build(serialInterfaceMap,Faces_->getNumEntities());

            for (UN i=0; i<Faces_->getNumEntities(); i++) {
                for (UN j=0; j<Faces_->getEntity(i)->getNumNodes(); j++) {
                    for (UN k=0; k<dofsPerNode; k++) {
                        // Assign value 1.0 to all DOFs belonging to this face
                        tmpVector->replaceLocalValue(Faces_->getEntity(i)->getGammaDofID(j,k),i,ScalarTraits<SC>::one());
                    }
                }
            }

            this->LocalPartitionOfUnity_[4] = tmpVector;
        }

        return 0;
    }

}

#endif
