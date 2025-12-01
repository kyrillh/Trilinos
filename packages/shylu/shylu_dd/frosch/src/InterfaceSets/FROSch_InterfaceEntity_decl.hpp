// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_INTERFACEENTITY_DECL_HPP
#define _FROSCH_INTERFACEENTITY_DECL_HPP

#include <Xpetra_VectorFactory_fwd.hpp>

#include <FROSch_ExtractSubmatrices_decl.hpp>
#include <FROSch_Tools_decl.hpp>


namespace FROSch {
    
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class EntitySet;

    enum EntityType {DefaultType,VertexType,EdgeType,FaceType,InteriorType,InterfaceType,BoundaryType};
    enum EntityFlag {DefaultFlag,StraightFlag,ShortFlag,NodeFlag,DirichletFlag,DoNothingFlag};
    enum DistanceFunction {ConstantDistanceFunction,InverseEuclideanDistanceFunction};

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal>
    struct Node {
        LO NodeIDGamma_;
        LO NodeIDLocal_;
        GO NodeIDGlobal_;

        ArrayRCP<LO> DofsGamma_;
        ArrayRCP<LO> DofsLocal_;
        ArrayRCP<GO> DofsGlobal_;

        bool operator< (const Node &n) const;

        bool operator== (const Node &n) const;
    };

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class InterfaceEntity {

    protected:

        using XMatrix               = Xpetra::Matrix<SC,LO,GO,NO>;
        using XMatrixPtr            = RCP<XMatrix>;
        using ConstXMatrixPtr       = RCP<const XMatrix>;

        using XVector               = Xpetra::Vector<SC,LO,GO,NO>;
        using XVectorPtr            = RCP<XVector>;

        using XMultiVector          = Xpetra::MultiVector<SC,LO,GO,NO>;
        using XMultiVectorPtr       = RCP<XMultiVector>;
        using ConstXMultiVectorPtr  = RCP<const XMultiVector>;

        using EntitySetPtr          = RCP<EntitySet<SC,LO,GO,NO> >;

        using InterfaceEntityPtr    = RCP<InterfaceEntity<SC,LO,GO,NO> >;

        using NodeVec               = Array<Node<SC,LO,GO> >;
        using NodePtr               = RCP<Node<SC,LO,GO> >;
        using NodePtrVec            = Array<NodePtr>;

        using UN                    = unsigned;

        using IntVec                = Array<int>;

        using LOVecPtr              = ArrayRCP<LO>;

        using GOVec                 = Array<GO>;
        using GOVecPtr              = ArrayRCP<GO>;

        using SCVecPtr              = ArrayRCP<SC>;
        using SCVecPtrVec           = Array<SCVecPtr>;

    public:

        InterfaceEntity(EntityType type,
                        UN dofsPerNode,
                        UN multiplicity,
                        const int *subdomains,
                        EntityFlag flag = DefaultFlag);

        ~InterfaceEntity();

        int addNode(LO nodeIDGamma,
                    LO nodeIDLocal,
                    GO nodeIDGlobal,
                    UN nDofs,
                    const LOVecPtr dofsGamma,
                    const LOVecPtr dofsLocal,
                    const GOVecPtr dofsGlobal);

        int addNode(const NodePtr &node);

        int addNode(const Node<SC,LO,GO> &node);

        int resetGlobalDofs(UN iD,
                            UN nDofs,
                            UN *dofIDs,
                            GO *dofsGlobal);

        int removeNode(UN iD);

        int removeNode(const Node<SC, LO, GO>& node);

        int sortByGlobalID();

        int reindexGammaID();
        int sortUniqueByLocalID();

        int setUniqueID(GO uniqueID);

        int setLocalID(LO localID);

        int setRootID(LO rootID);
        
        int setLeafID(LO leafID);

        int setUniqueIDToFirstGlobalID();

        int setGammaIDs(const int i, const LO gammaNodeID);

        int resetEntityType(EntityType type);

        int resetEntityFlag(EntityFlag flag);

        int findAncestorsInSet(EntitySetPtr entitySet);

        int clearAncestors();

        int addOffspring(InterfaceEntityPtr interfaceEntity);

        int clearOffspring();

        EntitySetPtr findRoots();

        int clearRoots();

        int computeDistancesToRoots(UN dimension,
                                    ConstXMultiVectorPtr &nodeList = null,
                                    DistanceFunction distanceFunction = ConstantDistanceFunction);

        InterfaceEntityPtr divideEntity(ConstXMatrixPtr matrix,
                                        int pID);

        /**
         * @brief Computes the distance of DoNothing nodes to a single neighboring Dirichlet entity. This is to enable
         * modification of the RGDSW coarse space to have an inverse Euclidean drop along DoNothing boundary entities
         * that have a DoNothing and Dirichlet boundary entity as neighbors. This was first developed to improve the
         * performance of the nonlinear Schwarz solver applied to the 2D lid-driven cavity problem. If no Dirichlet and
         * DoNothing boundaries were passed, this function does nothing. If more than one Dirichlet entity is found in
         * this subdomain, the distance to set to one everywhere since a linear decrease between two Dirichlet entities
         * does not make sense.
         * @param dimension Dimension of the problem
         * @param nodeList List of the node coords on the current subdomain. Required to calculate distances
         * @param dirichletEntities A set of the Dirichlet entities on this subdomain
         */
        int computeDistancesOnBoundary(UN dimension, ConstXMultiVectorPtr &nodeList,
                                        ArrayRCP<EntitySetPtr> entitySetVector, const int pID);

        /////////////////
        // Get Methods //
        /////////////////

        EntityType getEntityType() const;

        EntityFlag getEntityFlag() const;

        UN getDofsPerNode() const;

        UN getMultiplicity() const;

        GO getUniqueID() const;

        LO getLocalID() const;

        LO getRootID() const;
        
        LO getLeafID() const;

        const Node<SC,LO,GO>& getNode(UN iDNode) const;

        LO getGammaNodeID(UN iDNode) const;

        LO getLocalNodeID(UN iDNode) const;

        GO getGlobalNodeID(UN iDNode) const;

        LO getGammaDofID(UN iDNode, UN iDDof) const;

        LO getLocalDofID(UN iDNode, UN iDDof) const;

        GO getGlobalDofID(UN iDNode, UN iDDof) const;

        const IntVec & getSubdomainsVector() const;

        UN getNumNodes() const;

        const EntitySetPtr getAncestors() const;

        const EntitySetPtr getOffspring() const;

        const EntitySetPtr getRoots() const;

        SC getDistanceToRoot(UN iDNode,
                             UN iDRoot) const;

        const NodeVec& getConstNodeVectorRef() const;

    protected:

        EntityType Type_ = DefaultType;

        EntityFlag Flag_ = DefaultFlag;

        NodeVec NodeVector_ = NodeVec(0);

        IntVec SubdomainsVector_ = IntVec(0);

        // This tree structure is ordered from bottom to top i.e. roots -> leaves. Interface entities with belonging to
        // more subdomains are further down the tree i.e. are ancestors of entities belonging to less subdomains.
        // This results in ancestors being subsets of offspring e.g. vertices -> edges -> faces.
        // Roots are ancestors that themselves don't have ancestors. While ancestors are one level down the tree, roots
        // are determined by going all the way down the tree e.g. roots of a face are the vertices around it while it's
        // ancestors are the edges.
        EntitySetPtr Ancestors_;
        EntitySetPtr Offspring_;
        EntitySetPtr Roots_;

        SCVecPtrVec DistancesVector_ = SCVecPtrVec(0); // AH 08/08/2019 TODO: make a MultiVector out of this

        UN DofsPerNode_ = 1;
        UN Multiplicity_ = 1;
        GO UniqueID_ = -1;
        LO LocalID_ = -1;
        LO RootID_ = -1;
        LO LeafID_ = -1;
    };

    template <class SC,class LO,class GO,class NO>
    bool compareInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                  RCP<InterfaceEntity<SC,LO,GO,NO> > iEb);

    template <class SC,class LO,class GO,class NO>
    bool equalInterfaceEntities(RCP<InterfaceEntity<SC,LO,GO,NO> > iEa,
                                RCP<InterfaceEntity<SC,LO,GO,NO> > iEb);

}

#endif
