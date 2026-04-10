// @HEADER
// *****************************************************************************
//               ShyLU: Scalable Hybrid LU Preconditioner and Solver
//
// Copyright 2011 NTESS and the ShyLU contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef _FROSCH_SOLVER_FACTORY_DECL_HPP
#define _FROSCH_SOLVER_FACTORY_DECL_HPP

#include <ShyLU_DDFROSch_config.h>

// FROSch
#include <FROSch_Solver_decl.hpp>
#if defined(HAVE_SHYLU_DDFROSCH_AMESOS) && defined(HAVE_SHYLU_DDFROSCH_EPETRA)
#include <FROSch_AmesosSolverEpetra_decl.hpp>
#endif
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <FROSch_Amesos2SolverEpetra_decl.hpp>
#endif
#include <FROSch_Amesos2SolverTpetra_decl.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_BELOS
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <FROSch_BelosSolverEpetra_decl.hpp>
#endif
#include <FROSch_BelosSolverTpetra_decl.hpp>
#endif
#include <FROSch_FROSchPreconditioner_decl.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_IFPACK2
#include <FROSch_Ifpack2PreconditionerTpetra_decl.hpp>
#endif
#ifdef HAVE_SHYLU_DDFROSCH_MUELU
#include <FROSch_MueLuPreconditioner_decl.hpp>
#endif
#if defined(HAVE_SHYLU_DDFROSCH_THYRA) && defined(HAVE_SHYLU_DDFROSCH_STRATIMIKOS)
#include <FROSch_ThyraPreconditioner_decl.hpp>
#include <FROSch_ThyraSolver_decl.hpp>
#endif


namespace FROSch {

    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC,
              class LO,
              class GO,
              class NO>
    class SolverFactory {

    protected:

        using XMatrixPtr                        = typename Solver<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr                   = typename Solver<SC,LO,GO,NO>::ConstXMatrixPtr;

        using ParameterListPtr                  = typename Solver<SC,LO,GO,NO>::ParameterListPtr;

        using SolverPtr                         = RCP<Solver<SC,LO,GO,NO> >;
#if defined(HAVE_SHYLU_DDFROSCH_AMESOS) && defined(HAVE_SHYLU_DDFROSCH_EPETRA)
        using AmesosSolverEpetraPtr             = RCP<AmesosSolverEpetra<SC,LO,GO,NO> >;
#endif
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
        using Amesos2SolverEpetraPtr            = RCP<Amesos2SolverEpetra<SC,LO,GO,NO> >;
#endif
        using Amesos2SolverTpetraPtr            = RCP<Amesos2SolverTpetra<SC,LO,GO,NO> >;
#ifdef HAVE_SHYLU_DDFROSCH_BELOS
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
        using BelosSolverEpetraPtr              = RCP<BelosSolverEpetra<SC,LO,GO,NO> >;
#endif
        using BelosSolverTpetraPtr              = RCP<BelosSolverTpetra<SC,LO,GO,NO> >;
#endif
        using FROSchPreconditionerPtr           = RCP<FROSchPreconditioner<SC,LO,GO,NO> >;
#ifdef HAVE_SHYLU_DDFROSCH_IFPACK2
        using Ifpack2PreconditionerTpetraPtr    = RCP<Ifpack2PreconditionerTpetra<SC,LO,GO,NO> >;
#endif
#ifdef HAVE_SHYLU_DDFROSCH_MUELU
        using MueLuPreconditionerPtr            = RCP<MueLuPreconditioner<SC,LO,GO,NO> >;
#endif
#if defined(HAVE_SHYLU_DDFROSCH_THYRA) && defined(HAVE_SHYLU_DDFROSCH_STRATIMIKOS)
        using ThyraPreconditionerPtr            = RCP<ThyraPreconditioner<SC,LO,GO,NO> >;
        using ThyraSolverPtr                    = RCP<ThyraSolver<SC,LO,GO,NO> >;
#endif

    public:
      /**
       * \brief Build a FROSch solver object. This wraps another Trilinos solver interface to an iterative or direct
       * solver.
       * \param k Matrix on which the solver is to be applied
       * \param parameterList Parameter list for the solver
       * \param description String describing features of the solver e.g. local subdomain solver on a certain level
       * \param initSolver Whether or not the solver object should be initialized. Was introduced for the nonlinear
       * Schwarz solver since there are use cases in which a dummy solver should be built as a placeholder for later
       * replacement.
       */
      static SolverPtr Build(ConstXMatrixPtr k, ParameterListPtr parameterList, string description,
                             const bool initSolver = true);
    };

}

#endif
