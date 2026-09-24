// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "mpi.h"
#include <algorithm>
#include <vector>
#include <Teuchos_CommHelpers.hpp>

#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include <Teuchos_StackedTimer.hpp>

#include <Tpetra_KokkosCompat_DefaultNode.hpp>

#include "Xpetra_CrsMatrixWrap.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include <Xpetra_DefaultPlatform.hpp>

#include <FROSch_DDInterface_def.hpp>
#include <FROSch_RGDSWInterfacePartitionOfUnity_def.hpp>

#include <FROSch_Tools_decl.hpp>


using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;

// Whole, partial, and untouched entities share multiplicity two on rank zero.
// Inserting local boundary node zero shifts every existing interface Gamma ID.
int testBoundaryExtraction(const RCP<const Comm<int>> &comm)
{
    if (comm->getSize() != 4) return EXIT_FAILURE;
    const int rank = comm->getRank();
    const std::vector<std::vector<GO>> nodeLists{{0,2,4,6,8,10,12}, {2,4}, {6,8}, {10,12}};
    const std::vector<std::vector<GO>> boundaryLists{{0,2,4,6}, {2,4}, {6}, {}};
    const auto &nodes = nodeLists[rank];
    const auto &boundary = boundaryLists[rank];
    auto map = MapFactory<LO,GO,NO>::Build(UseTpetra,
        Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid(),
        ArrayView<const GO>(nodes.data(), nodes.size()), 0, comm);
    DDInterface<SC,LO,GO,NO> dd(2, 2, map, None);
    Array<GO> boundaryDofs;
    for (GO node : boundary) {
        boundaryDofs.push_back(2*node);
        boundaryDofs.push_back(2*node+1);
    }
    dd.addBoundaryNodes(boundaryDofs(), DirichletFlag, 0);

    int failures = 0;
    auto check = [&](bool condition, const char *message) {
        if (!condition) {
            ++failures;
            std::cerr << "Boundary extraction rank " << rank << ": " << message << std::endl;
        }
    };
    const auto interface = dd.getInterface()->getEntity(0);
    const auto sets = dd.getEntitySetVector();
    std::vector<int> occurrences(nodes.size(), 0);
    int boundaryEntities = 0, remainingEntities = 0;
    for (unsigned set = 1; set < sets.size(); ++set) {
        for (unsigned e = 0; e < sets[set]->getNumEntities(); ++e) {
            const auto entity = sets[set]->getEntity(e);
            if (set >= 2) {
                check(entity->getNumNodes() > 0, "empty source entity survived");
                ++remainingEntities;
            } else if (entity->getNumNodes() > 0) {
                ++boundaryEntities;
                check(entity->getEntityType() == BoundaryType && entity->getEntityFlag() == DirichletFlag,
                      "incorrect boundary type or flag");
            }
            for (unsigned n = 0; n < entity->getNumNodes(); ++n) {
                const GO global = entity->getGlobalNodeID(n);
                const auto local = std::find(nodes.begin(), nodes.end(), global);
                check(local != nodes.end(), "unexpected node");
                if (local == nodes.end()) continue;
                ++occurrences[local - nodes.begin()];
                const bool isBoundary = std::find(boundary.begin(), boundary.end(), global) != boundary.end();
                check((set == 1) == isBoundary, "incorrect boundary membership");
                std::vector<int> expectedSubdomains;
                for (int r = 0; r < 4; ++r)
                    if (std::find(nodeLists[r].begin(), nodeLists[r].end(), global) != nodeLists[r].end())
                        expectedSubdomains.push_back(r);
                const auto &subdomains = entity->getSubdomainsVector();
                check(entity->getMultiplicity() == expectedSubdomains.size(), "multiplicity changed");
                check(subdomains.size() == expectedSubdomains.size() &&
                      std::equal(subdomains.begin(), subdomains.end(), expectedSubdomains.begin()),
                      "subdomain membership changed");
                unsigned gamma = 0;
                while (gamma < interface->getNumNodes() && interface->getGlobalNodeID(gamma) != global) ++gamma;
                check(gamma < interface->getNumNodes(), "node missing from interface");
                if (gamma == interface->getNumNodes()) continue;
                check(entity->getGammaNodeID(n) == interface->getGammaNodeID(gamma), "stale Gamma node ID");
                for (unsigned dof = 0; dof < 2; ++dof) {
                    check(entity->getGammaDofID(n,dof) == interface->getGammaDofID(gamma,dof), "stale Gamma DOF ID");
                    check(entity->getLocalDofID(n,dof) == interface->getLocalDofID(gamma,dof), "local DOF changed");
                    check(entity->getGlobalDofID(n,dof) == 2*global+dof, "global DOF changed");
                }
            }
        }
    }
    for (int count : occurrences) check(count == 1, "node lost or duplicated");
    const int expectedBoundary[] = {3,1,1,0};
    const int expectedRemaining[] = {2,0,1,1};
    check(boundaryEntities == expectedBoundary[rank], "incorrect boundary entity grouping");
    check(remainingEntities == expectedRemaining[rank], "incorrect remaining entity count");
    int total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &failures, &total);
    return total == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Exercise custom-only, mixed, shared, and initially empty coarse interfaces.
int testCustomBoundary(const RCP<const Comm<int>> &comm)
{
    if (comm->getSize() != 4) return EXIT_FAILURE;
    int failures = 0;
    // Shared custom roots, isolated custom roots, mixed ordinary/custom roots, and no roots.
    for (int scenario = 0; scenario < 4; ++scenario) {
        const bool isolated = scenario == 1 || scenario == 3;
        for (int dofs = 1; dofs <= 2; ++dofs) {
            const int rank = comm->getRank();
            Array<GO> nodes;
            nodes.push_back(isolated ? 3*rank : rank);
            nodes.push_back(isolated ? 3*rank+1 : rank+1);
            nodes.push_back(20+rank);
            if (scenario == 2) {
                nodes.push_back(30+rank);
                nodes.push_back(31+rank);
            }
            auto map = MapFactory<LO,GO,NO>::Build(UseTpetra,
                OrdinalTraits<Xpetra::global_size_t>::invalid(), nodes(), 0, comm);
            ArrayRCP<RCP<const Map<LO,GO,NO>>> dofMaps(dofs);
            for (int k = 0; k < dofs; ++k) {
                Array<GO> ids;
                for (GO node : nodes) ids.push_back(dofs*node+k);
                dofMaps[k] = MapFactory<LO,GO,NO>::Build(UseTpetra,
                    OrdinalTraits<Xpetra::global_size_t>::invalid(), ids(), 0, comm);
            }
            auto params = rcp(new ParameterList);
            params->set("Test Unconnected Interface", false);
            RGDSWInterfacePartitionOfUnity<SC,LO,GO,NO> pou(comm, rcp(new SerialComm<int>),
                2, dofs, map, dofMaps, params, None);
            Array<GO> custom, dirichlet;
            for (int k = 0; k < dofs; ++k) {
                if (scenario != 3) {
                    custom.push_back(dofs*nodes[0]+k);
                    custom.push_back(dofs*nodes[1]+k);
                }
                if (dofs == 2 || scenario == 3) dirichlet.push_back(dofs*nodes[2]+k);
            }
            std::sort(custom.begin(), custom.end());
            pou.addBoundaryNodes(dirichlet(), DirichletFlag, 0);
            pou.addBoundaryNodes(custom(), CustomBCFlag, 0);
            auto coords = MultiVectorFactory<SC,LO,GO,NO>::Build(map, 2);
            for (int n = 0; n < nodes.size(); ++n) {
                coords->replaceLocalValue(n, 0, nodes[n]);
                coords->replaceLocalValue(n, 1, rank);
            }
            pou.sortInterface(null, coords);
            pou.computePartitionOfUnity(coords);
            const auto dd = pou.getDDInterface();
            const auto roots = dd->getRoots();
            const auto values = pou.getLocalPartitionOfUnity()[0];
            const auto entities = dd->getEntitySetVector()[1];
            for (unsigned e = 0; e < entities->getNumEntities(); ++e) {
                const auto entity = entities->getEntity(e);
                const bool isCustom = entity->getEntityFlag() == CustomBCFlag;
                if (isCustom && (entity->getRootID() < 0 || entity->getAncestors()->getNumEntities())) ++failures;
                if (!isCustom) {
                    // Dirichlet entities inherit ordinary interface roots, never custom roots.
                    // With no ordinary roots they remain rootless and must not become roots themselves.
                    if (entity->getRootID() != -1) ++failures;
                    unsigned expectedDirichletRoots = 0;
                    for (unsigned c = 0; c < roots->getNumEntities(); ++c) {
                        const bool ordinaryRoot = roots->getEntity(c)->getEntityFlag() != CustomBCFlag;
                        if (ordinaryRoot) ++expectedDirichletRoots;
                        bool associated = false;
                        for (unsigned r = 0; r < entity->getRoots()->getNumEntities(); ++r)
                            associated |= entity->getRoots()->getEntity(r).getRawPtr() == roots->getEntity(c).getRawPtr();
                        if (associated != ordinaryRoot) ++failures;
                    }
                    if (entity->getRoots()->getNumEntities() != expectedDirichletRoots) ++failures;
                }
                for (unsigned n = 0; n < entity->getNumNodes(); ++n) {
                    for (int k = 0; k < dofs; ++k) {
                        SC sum = 0;
                        for (unsigned c = 0; !values.is_null() && c < values->getNumVectors(); ++c) {
                            const SC value = values->getData(c)[entity->getGammaDofID(n,k)];
                            if (!isCustom && value != 0.0) ++failures;
                            sum += value;
                        }
                        if (std::abs(sum - (isCustom ? 1.0 : 0.0)) > 1.e-12) ++failures;
                    }
                }
            }
            const unsigned expectedRoots = scenario == 3 ? 0 :
                (isolated ? 1 : 2) + (scenario == 2 ? (rank == 0 || rank == 3 ? 1 : 2) : 0);
            if (roots->getNumEntities() != expectedRoots ||
                (values.is_null() ? 0 : values->getNumVectors()) != expectedRoots) ++failures;
            // Shared boundary nodes must reference the same global coarse column on both owners.
            GO localIDs[2] = {-1, -1};
            for (unsigned c = 0; c < roots->getNumEntities(); ++c) {
                const auto root = roots->getEntity(c);
                for (unsigned n = 0; n < root->getNumNodes(); ++n)
                    for (int j = 0; j < 2; ++j)
                        if (root->getGlobalNodeID(n) == nodes[j])
                            localIDs[j] = roots->getEntityMap()->getGlobalElement(c);
            }
            GO allIDs[8];
            gatherAll(*comm, 2, localIDs, 8, allIDs);
            if (!isolated)
                for (int r = 0; r < 3; ++r)
                    if (allIDs[2*r+1] != allIDs[2*(r+1)]) ++failures;
            for (unsigned c = 0; !values.is_null() && c < values->getNumVectors(); ++c) {
                SC norm = 0;
                for (SC value : values->getData(c)) norm += value*value;
                if (!(norm > 0)) ++failures;
            }
        }
    }
    int total = 0;
    reduceAll(*comm, REDUCE_SUM, 1, &failures, &total);
    if (comm->getRank() == 0) std::cout << "Custom boundary failures: " << total << std::endl;
    return total ? EXIT_FAILURE : EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;
    using namespace FROSch;

    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    bool testCustom = false;
    My_CLP.setOption("TESTCUSTOMBOUNDARY", "NOTESTCUSTOMBOUNDARY", &testCustom);
    bool testBoundary = false;
    My_CLP.setOption("TESTBOUNDARYEXTRACTION", "NOTESTBOUNDARYEXTRACTION", &testBoundary,
                     "Run the boundary extraction regression on four ranks.");

    int M = 4;
    My_CLP.setOption("M",&M,"H / h.");
    int Dimension = 3;
    My_CLP.setOption("DIM",&Dimension,"Dimension.");
    bool useEpetra = true;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useEpetra,"Use Epetra infrastructure for the linear algebra.");
    int communicationStrategy = 0;
    My_CLP.setOption("COMMSTRAT",&communicationStrategy,"Communication strategy for the domain decomposition interface.");

    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }

    if (testCustom) return testCustomBoundary(CommWorld);
    if (testBoundary) return testBoundaryExtraction(CommWorld);

    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("InterfaceSets Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    int N = 0;
    int color=1;
    if (Dimension == 2) {
        N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N) {
            color=0;
        }
    } else if (Dimension == 3) {
        N = (int) (pow(CommWorld->getSize(),1/3.) + 100*numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank()<N*N*N) {
            color=0;
        }
    } else {
        assert(false);
    }

    UnderlyingLib xpetraLib = UseTpetra;
    if (useEpetra) {
        xpetraLib = UseEpetra;
    } else {
        xpetraLib = UseTpetra;
    }

    RCP<const Comm<int> > Comm = CommWorld->split(color,CommWorld->getRank());

    if (color==0) {

        Comm->barrier(); if (Comm->getRank()==0) cout << "#############\n# Assembly #\n#############\n" << endl;
        //stackedTimer->start("Assembly Time");
        FROSCH_TIMER_START(assemblyTimer,"Assembly");

        ParameterList GaleriList;
        GaleriList.set("nx", GO(N*M));
        GaleriList.set("ny", GO(N*M));
        GaleriList.set("nz", GO(N*M));
        GaleriList.set("mx", GO(N));
        GaleriList.set("my", GO(N));
        GaleriList.set("mz", GO(N));

        RCP<const Map<LO,GO,NO> > UniqueMap;
        RCP<MultiVector<SC,LO,GO,NO> > Coordinates;
        RCP<Matrix<SC,LO,GO,NO> > K;
        if (Dimension==2) {
            UniqueMap = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian2D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("2D",UniqueMap,GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace2D",UniqueMap,GaleriList);
            K = Problem->BuildMatrix();
        } else if (Dimension==3) {
            UniqueMap = Galeri::Xpetra::CreateMap<LO,GO,NO>(xpetraLib,"Cartesian3D",Comm,GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map<LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("3D",UniqueMap,GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> > > Problem = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map<LO,GO,NO>,CrsMatrixWrap<SC,LO,GO,NO>,MultiVector<SC,LO,GO,NO> >("Laplace3D",UniqueMap,GaleriList);
            K = Problem->BuildMatrix();
        }

        FROSCH_TIMER_STOP(assemblyTimer);
        Comm->barrier(); if (Comm->getRank()==0) cout << "#############\n# Constructing Repeated Map #\n#############\n" << endl;
        FROSCH_TIMER_START(repMapTimer,"Construct RepeatedMap");

        RCP<const Map<LO,GO,NO> > RepeatedMap = BuildRepeatedMap<LO,GO,NO>(K->getCrsGraph());

        FROSCH_TIMER_STOP(repMapTimer);
        Comm->barrier(); if (Comm->getRank()==0) cout << "#############\n# Identification of Interface Sets #\n#############\n" << endl;
        FROSCH_TIMER_START(intSetsTimer,"Identification of Interface Sets");
        RCP<EntitySet<SC,LO,GO,NO> > vertices,shortEdges,straightEdges,edges,faces,interface,interior;
        //RCP<Map<LO,GO,NO> > verticesMap,shortEdgesMap,straightEdgesMap,edgesMap,facesMap;

        DDInterface<SC,LO,GO,NO> dDInterface(Dimension,1,RepeatedMap,All,(CommunicationStrategy) communicationStrategy);
        dDInterface.divideUnconnectedEntities(K);
        dDInterface.sortVerticesEdgesFaces();

        dDInterface.buildEntityHierarchy();

        dDInterface.buildEntityMaps(true,
                                    true,
                                    true,
                                    true,
                                    true,
                                    true);

        FROSCH_TIMER_STOP(intSetsTimer);
        Comm->barrier(); if (Comm->getRank()==0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }

    CommWorld->barrier();
    stackedTimer->stop("InterfaceSets Test");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report(*out,CommWorld,options);

    return(EXIT_SUCCESS);
}
