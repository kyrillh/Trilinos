// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_DDINTERFACE_DECL_HPP
#define _FROSCH_DDINTERFACE_DECL_HPP

//#define INTERFACE_OUTPUT

#include <Xpetra_Operator_fwd.hpp>
#include <Xpetra_MapFactory_fwd.hpp>
#include <Xpetra_ExportFactory_fwd.hpp>
#include <Xpetra_CrsGraphFactory.hpp>

#include <FROSch_EntitySet_decl.hpp>
#include <FROSch_InterfaceEntity_decl.hpp>

#include <FROSch_ExtractSubmatrices_decl.hpp>


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    enum CommunicationStrategy {CommCrsMatrix,CommCrsGraph,CreateOneToOneMap};

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class DDInterface {

    protected:

        using CommPtr                   = RCP<const Comm<int> >;

        using XMap                      = Map<LO,GO,NO>;
        using XMapPtr                   = RCP<XMap>;
        using ConstXMapPtr              = RCP<const XMap>;
        using XMapPtrVecPtr             = ArrayRCP<XMapPtr>;
        using ConstXMapPtrVecPtr        = ArrayRCP<ConstXMapPtr>;

        using XMatrix                   = Matrix<SC,LO,GO,NO>;
        using XMatrixPtr                = RCP<XMatrix>;
        using ConstXMatrixPtr           = RCP<const XMatrix>;

        using XCrsGraph                 = CrsGraph<LO,GO,NO>;
        using XCrsGraphPtr              = RCP<XCrsGraph>;

        using XMultiVector              = MultiVector<SC,LO,GO,NO>;
        using XMultiVectorPtr           = RCP<XMultiVector>;
        using ConstXMultiVectorPtr      = RCP<const XMultiVector>;

        using XImport                   = Import<LO,GO,NO>;
        using XImportPtr                = RCP<XImport>;

        using XExport                   = Export<LO,GO,NO>;
        using XExportPtr                = RCP<XExport>;

        using EntitySetPtr              = RCP<EntitySet<SC,LO,GO,NO> >;
        using EntitySetConstPtr         = const EntitySetPtr;
        using EntitySetPtrVecPtr        = ArrayRCP<EntitySetPtr>;
        using EntitySetPtrConstVecPtr   = const EntitySetPtrVecPtr;

        using EntityFlagVecPtr          = ArrayRCP<EntityFlag>;

        using InterfaceEntityPtr        = RCP<InterfaceEntity<SC,LO,GO,NO> >;
        using InterfaceEntityPtrVecPtr  = ArrayRCP<InterfaceEntityPtr>;

        using UN                        = unsigned;
        using ConstUN                   = const UN;
        using UNVecPtr                  = ArrayRCP<UN>;

        using IntVec                    = Array<int>;
        using IntVecVec                 = Array<IntVec>;
        using IntVecVecPtr              = ArrayRCP<IntVec>;

        using LOVec                     = Array<LO>;
        using LOVecPtr                  = ArrayRCP<LO>;

        using GOVec                     = Array<GO>;
        using ConstGOVecView            = ArrayView<const GO>;
        using GOVecPtr                  = ArrayRCP<GO>;
        using GOVecView                 = ArrayView<GO>;
        using GOVecVec                  = Array<GOVec>;
        using GOVecVecPtr               = ArrayRCP<GOVec>;

        using SCVec                     = Array<SC>;
        using SCVecPtr                  = ArrayRCP<SC>;

    public:
       /**
        * Constructor for domain decomposition interface.
        * This constructor initializes the domain decomposition interface by
        * determining which nodes belong to which subdomains and creating
        * the initial interface structure.
        * @param dimension Spatial dimension of the problem (2 or 3)
        * @param dofsPerNode Number of degrees of freedom per node
        * @param localToGlobalMap Map from local to global node IDs
        * @param verbosity Level of output verbosity
        * @param levelID Level identifier for multi-level methods
        * @param commStrategy Communication strategy for subdomain detection
        */
        DDInterface(UN dimension,
                    UN dofsPerNode,
                    ConstXMapPtr localToGlobalMap,
                    Verbosity verbosity = All,
                    UN levelID = 1,
                    CommunicationStrategy commStrategy = CommCrsGraph);

        /**
         * Destructor for domain decomposition interface.
         * Cleans up all resources associated with the interface.
         */
        ~DDInterface();

        /**
         * Reset global DOF IDs for all entities in the domain decomposition interface.
         * This function updates the global DOF mappings for all interface entities,
         * interior entities, and entity sets when new DOF maps are provided.
         * @param dofsMaps Array of DOF maps, one for each DOF per node
         * @return 0 on success
         */
        int resetGlobalDofs(ConstXMapPtrVecPtr dofsMaps);

        /**
         * Remove Dirichlet boundary nodes from the domain decomposition interface.
         * This function identifies and removes nodes that are constrained by Dirichlet
         * boundary conditions from all interface entities and entity sets.
         * @param dirichletBoundaryDofs Array view of global DOF IDs on Dirichlet boundaries
         * @return 0 on success
         */
        int removeDirichletNodes(GOVecView dirichletBoundaryDofs);

        /**
         * Part of the boundary framework.
         * 
         * Add nodes to EntitySetVector_[1] and move them from Interior_ to Interface_ as specified in boundaryDofs. The
         * nodes are added to EntitySetVector_[1] as a single entity. A later call to e.g. sortInterface() is required
         * to split this into strictly connected entities i.e. entities where the union of the support of the
         * corresponding finite element basis functions forms a connected set. These can then be used to build modified
         * versions of GDSW-type coarse spaces.
         * @param boundaryDofs the global dofs that are to be added to the interface. These should be boundary dofs in node-wise ordering.
         * this is not explicitly checked by the function.
         * @param type Specifies the type of boundary that is being added i.e. Dirichlet or CustomBC.
         * @param offset the offset required to determine global node IDs from the dof IDs. Required e.g. for block
         * systems.
         */
        void addBoundaryNodes(const GOVecView boundaryDofs, const enum EntityFlag type,
                                const int dofOffset = 0);

        /**
         * Divide unconnected interface entities based on matrix connectivity.
         * This function analyzes the connectivity pattern in the provided matrix
         * and splits interface entities that are not properly connected into
         * separate, connected components.
         * @param matrix The system matrix used to determine connectivity
         * @return 0 on success
         */
        int divideUnconnectedEntities(ConstXMatrixPtr matrix);

        /**
         * Flag interface entities based on geometric properties.
         * This function analyzes the geometry of interface entities and assigns
         * flags (DefaultFlag, StraightFlag, ShortFlag, NodeFlag) based on
         * their geometric characteristics. These flags are used later for
         * classification into vertices, edges, and faces.
         * @param nodeList Optional node coordinates for geometric analysis
         * @return 0 on success
         */
        int flagEntities(ConstXMultiVectorPtr nodeList = null);

        /**
         * Remove empty entities from all entity sets.
         * This function cleans up entity sets by removing entities that have
         * no nodes (empty entities) that may have been created during
         * previous processing steps.
         * @return 0 on success
         */
        int removeEmptyEntities();

        /**
         * Sort interface entities into vertices, edges, and faces.
         * This function classifies interface entities based on their multiplicity
         * (number of subdomains they belong to) and geometric flags into
         * appropriate entity sets: Vertices_, ShortEdges_, StraightEdges_,
         * Edges_, and Faces_.
         * @param nodeList Optional node coordinates for geometric analysis
         * @return 0 on success
         */
        int sortVerticesEdgesFaces(ConstXMultiVectorPtr nodeList = null);

        /**
         * Build global maps for interface entities.
         * This function creates global maps for the specified entity types,
         * which are needed for parallel communication and coarse space
         * construction. The maps assign unique global IDs to entities
         * across all processors.
         * @param buildVerticesMap Build map for vertices
         * @param buildShortEdgesMap Build map for short edges
         * @param buildStraightEdgesMap Build map for straight edges
         * @param buildEdgesMap Build map for general edges
         * @param buildFacesMap Build map for faces
         * @param buildRootsMap Build map for root entities
         * @param buildLeafsMap Build map for leaf entities
         * @return 0 on success
         */
        int buildEntityMaps(bool buildVerticesMap = true,
                            bool buildShortEdgesMap = true,
                            bool buildStraightEdgesMap = true,
                            bool buildEdgesMap = true,
                            bool buildFacesMap = true,
                            bool buildRootsMap = false,
                            bool buildLeafsMap = false);

        /**
         * Build hierarchical relationships between interface entities.
         * This function establishes parent-child relationships between entities
         * of different dimensions (e.g., faces contain edges, edges contain vertices).
         * It also identifies root entities (highest level) and leaf entities
         * (lowest level) in the hierarchy.
         * @return 0 on success
         */
        int buildEntityHierarchy();

        /**
         * Compute distances from interface nodes to root entities.
         * This function calculates distances between interface nodes and
         * root entities (coarse nodes) using the specified distance function.
         * These distances are used in partition of unity construction.
         * @param dimension Spatial dimension of the problem
         * @param nodeList Node coordinates for distance computation
         * @param distanceFunction Function to compute distances (Euclidean, etc.)
         * @return 0 on success
         */
        int computeDistancesToRoots(UN dimension,
                                    ConstXMultiVectorPtr &nodeList = null,
                                    DistanceFunction distanceFunction = ConstantDistanceFunction);

        /**
         * This function extracts those entities which are to be used to build a connectivity graph on the subdomain
         * level. By default, we identify all entities with multiplicity 2. Afterwards, the corresponding entities can
         * be obtained using the function getConnectivityEntities().
         * If short or straight edges should be omitted, the function flagEntities() has to be called in advance.
         */
        int identifyConnectivityEntities(UNVecPtr multiplicities = null,
                                         EntityFlagVecPtr flags = null);

        /**
         * Part of the boundary framework.
         * 
         * @brief wrapper function to call FROSch_InterfaceEntity::computeDistancesOnBoundary()
         * @param dimension Dimension of the problem
         * @param nodeList The coordinates of the finite element nodes in this subdomain. Required to calculate inverse Euclidean distances.
         */
        int computeDistancesOnBoundary(UN dimension, ConstXMultiVectorPtr &nodeList);

        /**
         * Part of the boundary framework.
         * 
         * @brief wrapper function to call FROSch_EntitySet::computeDistancesToDirichletBoundary()
         * @param dimension Dimension of the problem
         * @param nodeList The coordinates of the finite element nodes in this subdomain. Required to calculate inverse Euclidean distances.
         */
        int computeDistancesToDirichletBoundary(UN dimension, ConstXMultiVectorPtr &nodeList);



        UN getDimension() const;

        UN getDofsPerNode() const;

        LO getNumMyNodes() const;

        //
        // Remove the references below?
        //

        EntitySetConstPtr & getVertices() const;

        EntitySetConstPtr & getShortEdges() const;

        EntitySetConstPtr & getStraightEdges() const;

        EntitySetConstPtr & getEdges() const;

        EntitySetConstPtr & getFaces() const;

        EntitySetConstPtr & getInterface() const;

        EntitySetConstPtr & getInterior() const;

        EntitySetConstPtr & getRoots() const;

        EntitySetConstPtr & getLeafs() const;

        EntitySetPtrConstVecPtr & getEntitySetVector() const;

        GOVec getNumEnt() const;
        
        /**
         * Get entities used for building connectivity graph on subdomain level.
         * These entities have been identified using identifyConnectivityEntities().
         * @return Const reference to connectivity entities set
         */
        EntitySetConstPtr & getConnectivityEntities() const;

        ConstXMapPtr getNodesMap() const;


    protected:

        /**
         * Communicate local components to determine subdomain ownership.
         * This function exchanges information between processors to determine
         * which subdomains each node belongs to. It supports multiple
         * communication strategies (CommCrsMatrix, CommCrsGraph, CreateOneToOneMap).
         * @param componentsSubdomains Output: For each node, list of subdomain IDs
         * @param componentsSubdomainsUnique Output: Unique equivalence classes of subdomain combinations
         * @return 0 on success
         */
        int communicateLocalComponents(IntVecVecPtr &componentsSubdomains,
                                       IntVecVec &componentsSubdomainsUnique);

        /**
         * Identify and classify local components based on subdomain multiplicity.
         * This function groups nodes into equivalence classes based on their
         * subdomain membership and creates interface entities accordingly.
         * @param componentsSubdomains Input: Subdomain lists for each node
         * @param componentsSubdomainsUnique Input: Unique equivalence classes
         * @return 0 on success
         */
        int identifyLocalComponents(IntVecVecPtr &componentsSubdomains,
                                    IntVecVec &componentsSubdomainsUnique);


        CommPtr MpiComm_;

        UN Dimension_ = 3;
        UN DofsPerNode_ = 1;
        LO NumMyNodes_ = 0;

        EntitySetPtr Vertices_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(VertexType));
        EntitySetPtr ShortEdges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr StraightEdges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr Edges_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(EdgeType));
        EntitySetPtr Faces_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(FaceType));
        EntitySetPtr Interface_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(InterfaceType));
        EntitySetPtr Interior_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(InteriorType));
        // Entities with no ancestors
        EntitySetPtr Roots_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
        // Entities with no offspring
        EntitySetPtr Leafs_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
        EntitySetPtr ConnectivityEntities_ = EntitySetPtr(new EntitySet<SC,LO,GO,NO>(DefaultType));
        EntitySetPtrVecPtr EntitySetVector_;

        ConstXMapPtr NodesMap_;
        ConstXMapPtr UniqueNodesMap_;

        CommunicationStrategy CommStrategy_ = CommCrsGraph;

        bool Verbose_ = false;

        Verbosity Verbosity_ = All;

        ConstUN LevelID_ = 1;
        GOVec NumEntity_;
        bool HaveDirichletEntities_;
    };

}

#endif
