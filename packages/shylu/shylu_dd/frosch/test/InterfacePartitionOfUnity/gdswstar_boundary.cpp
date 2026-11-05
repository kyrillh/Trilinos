// @HEADER
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// @HEADER

#include <Tpetra_Core.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_MapFactory.hpp>
#include <FROSch_GDSWStarInterfacePartitionOfUnity_def.hpp>

using SC = double;
using LO = int;
using GO = FROSch::DefaultGlobalOrdinal;
using NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

int main(int argc, char **argv)
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    using namespace Teuchos;
    using namespace Xpetra;
    using namespace FROSch;
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize()!=4) return EXIT_FAILURE;
    const int rank = comm->getRank();
    int failures = 0;
    auto check = [&](bool condition, const char *message) {
        if (!condition) {
            ++failures;
            std::cerr << "GDSWStar boundary rank " << rank << ": " << message << std::endl;
        }
    };

    // Node 100 is a four-way root; nodes 11--13 are two-way leaves.
    // Unique Dirichlet nodes become offspring of all these ordinary entities.
    for (int scenario=0; scenario<4; ++scenario) {
        for (int dofs : {1,2}) {
            for (const std::string distance : {"Constant", "Inverse Euclidean"}) {
                for (const std::string type : {"Full", "Roots", "Leafs", "Custom"}) {
                    Array<GO> nodes{GO(10+rank), GO(11+rank), GO(100), GO(200+rank)};
                    if (scenario==2) nodes.push_back(300+rank);
                    auto map = MapFactory<LO,GO,NO>::Build(UseTpetra,
                        OrdinalTraits<global_size_t>::invalid(), nodes(), 0, comm);
                    ArrayRCP<RCP<const Map<LO,GO,NO>>> dofMaps(dofs);
                    for (int k=0; k<dofs; ++k) {
                        Array<GO> ids;
                        for (GO node : nodes) ids.push_back(dofs*node+k);
                        dofMaps[k] = MapFactory<LO,GO,NO>::Build(UseTpetra,
                            OrdinalTraits<global_size_t>::invalid(), ids(), 0, comm);
                    }
                    auto coords = MultiVectorFactory<SC,LO,GO,NO>::Build(map, 2);
                    for (int n=0; n<nodes.size(); ++n) {
                        coords->replaceLocalValue(n, 0, nodes[n]);
                        coords->replaceLocalValue(n, 1, 0.0);
                    }
                    auto params = rcp(new ParameterList);
                    params->set("Test Unconnected Interface", false);
                    params->set("Distance Function", distance);
                    // The GDSW base constructor accepts Full/Custom, but not the
                    // derived class's Roots/Leafs names. Exercise those selections
                    // through the supported Custom switches.
                    params->set("Type", type=="Full" ? "Full" : "Custom");
                    params->sublist("Custom").set("Roots", type!="Leafs").set("Leafs", type!="Roots");
                    GDSWStarInterfacePartitionOfUnity<SC,LO,GO,NO> pou(
                        comm, rcp(new SerialComm<int>), 2, dofs, map, dofMaps, params, None);
                    Array<GO> dirichlet, custom;
                    for (GO node : nodes) {
                        for (int k=0; k<dofs; ++k) {
                            if (scenario==3 || (scenario!=0 && node==200+rank))
                                dirichlet.push_back(dofs*node+k);
                            if (scenario==2 && node==300+rank) custom.push_back(dofs*node+k);
                        }
                    }
                    pou.addBoundaryNodes(dirichlet(), DirichletFlag, 0);
                    pou.addBoundaryNodes(custom(), CustomBCFlag, 0);
                    pou.sortInterface(null, coords);
                    pou.computePartitionOfUnity(coords);
                    const auto values = pou.getLocalPartitionOfUnity();
                    const auto maps = pou.getPartitionOfUnityMaps();
                    const auto dd = pou.getDDInterface();
                    const auto interface = dd->getInterface()->getEntity(0);
                    const bool useRoots = type!="Leafs", useLeafs = type!="Roots";
                    const unsigned ordinaryLeaves = rank==0 || rank==3 ? 1 : 2;
                    const unsigned leafCount = scenario==3 ? 0 : ordinaryLeaves+(scenario==2 ? 1 : 0);
                    const unsigned rootCount = scenario==3 ? 0 : 1+(scenario==2 ? 1 : 0);
                    for (int part=0; part<2; ++part) {
                        const unsigned expected = part==0 ? (useRoots ? rootCount : 0) : (useLeafs ? leafCount : 0);
                        check((values[part].is_null() ? 0 : values[part]->getNumVectors())==expected,
                              "incorrect coarse column count");
                        check((maps[part].is_null() ? 0 : maps[part]->getLocalNumElements())==expected,
                              "coarse map does not match selected entities");
                    }
                    GO leafIDs[3] = {-1,-1,-1};
                    for (unsigned n=0; n<interface->getNumNodes(); ++n) {
                        const GO node = interface->getGlobalNodeID(n);
                        const bool isDirichlet = scenario==3 || (scenario!=0 && node==200+rank);
                        const bool isLeaf = (node>=11 && node<=13) || (scenario==2 && node==300+rank);
                        for (int k=0; k<dofs; ++k) {
                            const LO row = interface->getGammaDofID(n,k);
                            SC sum = 0;
                            for (int part=0; part<2; ++part) {
                                for (unsigned c=0; !values[part].is_null() && c<values[part]->getNumVectors(); ++c) {
                                    const SC value = values[part]->getData(c)[row];
                                    if (isDirichlet) check(value==0.0, "nonzero Dirichlet trace");
                                    if (part==0 && isLeaf) check(value==0.0, "leaf also receives root trace");
                                    sum += value;
                                    if (part==1 && node>=11 && node<=13 && value==1.0)
                                        leafIDs[node-11] = maps[1]->getGlobalElement(c);
                                }
                            }
                            const SC expected = isDirichlet ? 0.0 : (isLeaf ? useLeafs : useRoots);
                            check(std::abs(sum-expected)<1.e-12, "incorrect root/leaf trace");
                        }
                    }
                    GO allIDs[12];
                    gatherAll(*comm, 3, leafIDs, 12, allIDs);
                    if (useLeafs && scenario!=3) {
                        for (int edge=0; edge<3; ++edge)
                            check(allIDs[3*edge+edge]>=0 &&
                                  allIDs[3*edge+edge]==allIDs[3*(edge+1)+edge],
                                  "shared leaf has inconsistent global coarse ID");
                    }
                    // Leaf selection ignores Dirichlet entities and offspring,
                    // but must retain the ancestor/offspring relationships.
                    const auto leaves = dd->getLeafs();
                    check(leaves->getNumEntities()==leafCount, "incorrect shared leaf count");
                    for (unsigned e=0; e<leaves->getNumEntities(); ++e) {
                        check(leaves->getEntity(e)->getEntityFlag()!=DirichletFlag,
                              "Dirichlet entity selected as a leaf");
                        check(leaves->getEntity(e)->getLeafID()==LO(e),
                              "shared leaf IDs do not match set ordering");
                    }
                    if (useLeafs)
                        check(maps[1].getRawPtr()==leaves->getEntityMap().getRawPtr(),
                              "GDSWStar does not use the shared leaf map");
                    const auto sets = dd->getEntitySetVector();
                    for (unsigned s=0; s<sets.size(); ++s) {
                        for (unsigned e=0; e<sets[s]->getNumEntities(); ++e) {
                            const auto entity = sets[s]->getEntity(e);
                            const GO node = entity->getGlobalNodeID(0);
                            const bool expectedLeaf = scenario!=3 &&
                                ((node>=11 && node<=13) || (scenario==2 && node==300+rank));
                            LO selectedID = -1;
                            for (unsigned l=0; l<leaves->getNumEntities(); ++l)
                                if (leaves->getEntity(l).getRawPtr()==entity.getRawPtr()) selectedID = LO(l);
                            check((selectedID>=0)==expectedLeaf, "incorrect shared leaf membership");
                            check(entity->getLeafID()==selectedID,
                                  "entity has an incorrect or stale leaf ID");
                            if (scenario==1 || scenario==2) {
                                if (entity->getEntityFlag()==DirichletFlag) {
                                    check(entity->getAncestors()->getNumEntities()==ordinaryLeaves+1,
                                          "Dirichlet ancestor relationships were removed");
                                } else if (entity->getEntityFlag()!=CustomBCFlag) {
                                    unsigned dirichletOffspring = 0;
                                    const auto offspring = entity->getOffspring();
                                    for (unsigned o=0; o<offspring->getNumEntities(); ++o)
                                        if (offspring->getEntity(o)->getEntityFlag()==DirichletFlag)
                                            ++dirichletOffspring;
                                    check(dirichletOffspring==1,
                                          "ordinary entity lost its Dirichlet offspring");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int total = 0;
    reduceAll(*comm, REDUCE_SUM, 1, &failures, &total);
    if (rank==0) std::cout << "GDSWStar boundary failures: " << total << std::endl;
    return total ? EXIT_FAILURE : EXIT_SUCCESS;
}
