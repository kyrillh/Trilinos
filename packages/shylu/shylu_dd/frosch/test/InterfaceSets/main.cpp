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
