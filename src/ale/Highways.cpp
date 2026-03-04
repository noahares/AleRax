#include "AleOptimizer.hpp"

#include <IO/FileSystem.hpp>
#include <IO/Logger.hpp>
#include <optimizers/DTLOptimizer.hpp>
#include <parallelization/ParallelContext.hpp>
#include <search/SpeciesTransferSearch.hpp>

class HighwayProbasOptimizer : public FunctionToOptimize {
public:
  HighwayProbasOptimizer(AleEvaluator &evaluator,
                         const std::vector<Highway> &highways)
      : _highways(highways), _evaluator(evaluator) {}

  virtual double evaluate(Parameters &parameters) {
    assert(parameters.dimensions() == _highways.size());
    parameters.ensurePositivity();
    for (unsigned int i = 0; i < _highways.size(); ++i) {
      auto highwayCopy = _highways[i];
      highwayCopy.proba = parameters[i];
      _evaluator.addHighway(highwayCopy);
    }
    auto res = _evaluator.computeLikelihood();
    for (const auto &highway : _highways) {
      (void)(highway);
      _evaluator.removeHighway();
    }
    parameters.setScore(res);
    return res;
  }
  virtual void printIndividualContribution(Parameters &parameters, const std::string outputDir, std::vector<ScoredHighway> &scoredHighways) {
	    for (unsigned int i = 0; i < scoredHighways.size(); ++i) {
	      Highway highwayCopy = scoredHighways[i].highway;
	      highwayCopy.proba = parameters[i];
	      _evaluator.addHighway(highwayCopy);
	    }
	    auto initial_ll = _evaluator.computeLikelihood();
	    _evaluator.saveSnapshotPerFamilyLL();
	    for (auto highway : _highways) {
	      (void)(highway);
	      _evaluator.removeHighway();
	    }
	    for (std::size_t i = 0; i < parameters.dimensions(); ++i) {
	      auto params = parameters;
	      params[i] = 0.0;
	      for (unsigned int i = 0; i < scoredHighways.size(); ++i) {
	          Highway highwayCopy = scoredHighways[i].highway;
	          highwayCopy.proba = params[i];
	          _evaluator.addHighway(highwayCopy);
	        }
	      auto new_ll = _evaluator.computeLikelihood();
	      auto lldiff = initial_ll - new_ll;
	      // scoredHighways[i].scoreDiff = lldiff;
	      Logger::info << "LL diff from highway " << scoredHighways[i].highway << ": " << lldiff << std::endl;
	      std::string out = FileSystem::joinPaths(
	        outputDir,
	        std::string("fixed_transferll_") + std::string(scoredHighways[i].highway.src->label) +
	        std::string("_") + std::string(scoredHighways[i].highway.dest->label) + std::string("_") + std::to_string(parameters[i]));
	      _evaluator.savePerFamilyLikelihoodDiff(out, true);
	      for (auto highway : _highways) {
	        (void)(highway);
	        _evaluator.removeHighway();
	      }
	    }
	  }

private:
  const std::vector<Highway> &_highways;
  AleEvaluator &_evaluator;
};

static bool isHighwayCompatible(const Highway &highway,
                                const RecModelInfo &info,
                                const DatedTree &tree) {
  auto from = highway.src;
  auto to = highway.dest;
  switch (info.transferConstraint) {
  case TransferConstaint::NONE:
    if (to == from) {
      return false;
    }
    return true;
  case TransferConstaint::PARENTS:
    while (from) {
      if (to == from) {
        return false;
      }
      from = from->parent;
    }
    return true;
  case TransferConstaint::RELDATED:
    return tree.canTransferUnderRelDated(from->node_index, to->node_index);
  }
  assert(false);
  return false;
}

static double testHighwayFast(AleEvaluator &evaluator, const Highway &highway,
                              const std::string &directory,
                              double highwayProba) {
  auto highwayCopy = highway;
  highwayCopy.proba = highwayProba;
  evaluator.addHighway(highwayCopy);
  double ll = evaluator.computeLikelihood();
  auto out = FileSystem::joinPaths(directory, std::string("transferll_") +
                                                  highway.src->label + "_to_" +
                                                  highway.dest->label + ".txt");
  evaluator.savePerFamilyLikelihoodDiff(out);
  evaluator.removeHighway();
  return ll;
}

static Parameters optimizeSingleHighwayProba(AleEvaluator &evaluator,
                                             const Highway &highway,
                                             double startingHighwayProba) {
  std::vector<Highway> highways;
  highways.push_back(highway);
  Parameters startingHighwayProbas(1);
  startingHighwayProbas[0] = startingHighwayProba;
  HighwayProbasOptimizer function(evaluator, highways);
  OptimizationSettings settings;
  settings.verbose = evaluator.isVerbose();
  settings.strategy = RecOpt::LBFGSB;
  settings.factr = LBFGSBPrecision::MEDIUM;
  ParallelContext::barrier();
  auto bestParameters = DTLOptimizer::optimizeParameters(
      function, startingHighwayProbas, settings);
  // function.evaluatePrint(bestParameters, true, highwaysOutputDir);
  return bestParameters;
}

static Parameters optimizeHighwayProbas(AleEvaluator &evaluator,
                                        std::vector<ScoredHighway> &scoredHighways,
                                        const Parameters &startingHighwayProbas,
                                        bool optimize, bool thorough, bool individual_contribution, const std::string outputDir) {
  assert(scoredHighways.size() == startingHighwayProbas.dimensions());
  std::vector<Highway> highways;
	  for (auto &highway : scoredHighways) {
	    highways.push_back(highway.highway);
	  }
  HighwayProbasOptimizer function(evaluator, highways);
  if (optimize) {
    OptimizationSettings settings;
    settings.verbose = evaluator.isVerbose();
    settings.strategy = evaluator.getRecModelInfo().recOpt;
    settings.minAlpha = 0.001;
    settings.epsilon = 0.000001;
    settings.factr = LBFGSBPrecision::MEDIUM;
    if (thorough) {
      settings.individualParamOpt = true;
      settings.individualParamOptMinImprovement = 10000.0;
    }
    ParallelContext::barrier();
    auto bestParameters = DTLOptimizer::optimizeParameters(
        function, startingHighwayProbas, settings);
    if (individual_contribution) {
      function.printIndividualContribution(bestParameters, outputDir, scoredHighways);
    }
    return bestParameters;
  } else {
    // like testHighwayFast, but for several highways
    auto parameters = startingHighwayProbas;
    function.evaluate(parameters);
    return parameters;
  }
}

static bool compareHighwaysByProba(const ScoredHighway &a,
                                   const ScoredHighway &b) {
  return a.highway.proba < b.highway.proba;
}

void Highways::getSortedCandidatesFromList(
    AleOptimizer &optimizer, const std::vector<Highway> &highways,
    std::vector<ScoredHighway> &candidateHighways) {
  Logger::timed << "[Highway search] Sorting highways from the "
                << "user-defined file" << std::endl;
  auto &speciesTree = optimizer.getSpeciesTree();
  unsigned int minTransfers = 1;
  MovesBlackList blacklist;
  std::vector<TransferMove> transferMoves;
  SpeciesTransferSearch::getSortedTransferList(
      speciesTree, optimizer.getEvaluator(), minTransfers, blacklist,
      transferMoves);
  // add highways to the candidates sorted according to their transfer direction
  // frequencies in the reconciliations
  for (const auto &transferMove : transferMoves) {
    // src species (potential branch to regraft to in a species tree search)
    auto regraft = speciesTree.getNode(transferMove.regraft);
    // dest species (potential branch to be pruned in a species tree search)
    auto prune = speciesTree.getNode(transferMove.prune);
    Highway highway(regraft, prune);
    if (std::find(highways.begin(), highways.end(), highway) !=
        highways.end()) {
      candidateHighways.push_back(ScoredHighway(highway, 0.0));
    }
  }
  // add highways absent from the reconciliations to the end of the candidates
  for (const auto &highway : highways) {
    if (std::find(candidateHighways.begin(), candidateHighways.end(),
                  highway) == candidateHighways.end()) {
      candidateHighways.push_back(ScoredHighway(highway, 0.0));
    }
  }
  // now all items from highways are added to candidateHighways
  Logger::timed << "[Highway search] Candidate highways obtained: "
                << candidateHighways.size() << std::endl;
}

void Highways::getCandidateHighways(
    AleOptimizer &optimizer, std::vector<ScoredHighway> &candidateHighways,
    unsigned int maxCandidates) {
  Logger::timed << "[Highway search] Inferring highways from the predicted "
                << "transfer directions" << std::endl;
  auto &speciesTree = optimizer.getSpeciesTree();
  unsigned int minTransfers = 2;
  MovesBlackList blacklist;
  std::vector<TransferMove> transferMoves;
  SpeciesTransferSearch::getSortedTransferList(
      speciesTree, optimizer.getEvaluator(), minTransfers, blacklist,
      transferMoves);
  for (const auto &transferMove : transferMoves) {
    // src species (potential branch to regraft to in a species tree search)
    auto regraft = speciesTree.getNode(transferMove.regraft);
    // dest species (potential branch to be pruned in a species tree search)
    auto prune = speciesTree.getNode(transferMove.prune);
    Highway highway(regraft, prune);
    unsigned int distance = 0;
    auto lca = speciesTree.getTree().getLCA(prune, regraft);
    while (prune != lca) {
      distance += 1;
      prune = prune->parent;
    }
    while (regraft != lca) {
      distance += 1;
      regraft = regraft->parent;
    }
    if (distance >= 5 && prune->left != nullptr) {
      candidateHighways.push_back(ScoredHighway(highway, 0.0));
    } else {
      Logger::timed << "Rejecting (speciesDist) candidate: "
                    << highway.src->label << "->" << highway.dest->label
                    << ", dist=" << distance << std::endl;
    }
    if (candidateHighways.size() >= maxCandidates) {
      break;
    }
  }
  Logger::timed << "[Highway search] After keeping at most " << maxCandidates
                << " the most frequent transfer directions: "
                << candidateHighways.size() << " highways kept" << std::endl;
}

void Highways::setFixedHighways(AleOptimizer &optimizer, std::vector<Highway> &highways, std::vector<ScoredHighway> &fixed_highways, const std::string outputDir) {
    auto fixedPath = FileSystem::joinPaths(outputDir,
                                        "fixed_tests");
    FileSystem::mkdir(fixedPath, true);
	  auto &evaluator = optimizer.getEvaluator();
	  Logger::timed << "Adding all fixed highways"
	                << std::endl;
	  Parameters startingProbabilities;
	  for (auto &highway : highways) {
	    startingProbabilities.addValue(highway.proba);
	    fixed_highways.push_back(ScoredHighway(highway));
	  }
	  auto parameters = optimizeHighwayProbas(evaluator, fixed_highways, startingProbabilities,
	                                 true, false, true, fixedPath);
	  Logger::info << parameters << std::endl;
	  for (unsigned int i = 0; i < highways.size(); ++i) {
	    fixed_highways[i].highway.proba = parameters[i];
	    evaluator.addHighway(fixed_highways[i].highway);
    }
}

void Highways::filterCandidateHighways(
    AleOptimizer &optimizer,
    const std::vector<ScoredHighway> &candidateHighways,
    std::vector<ScoredHighway> &filteredHighways, unsigned int maxCandidates,
    bool individual_test) {
  auto &speciesTree = optimizer.getSpeciesTree();
  auto &evaluator = optimizer.getEvaluator();
  double proba1 = 0.01;  // small hardcoded highway proba
  double proba2 = 0.1;   // higher hardcoded highway proba
  double minDiff = 0.01; // min ll increase to keep the candidate
  auto sampleSize =
      evaluator.getInputTreesNumber(); // input data size for BIC calculation
  auto testPath = FileSystem::joinPaths(optimizer.getHighwaysOutputDir(),
                                        "candidate_tests");
  FileSystem::mkdir(testPath, true);
  ParallelContext::barrier();
  Logger::timed << "[Highway search] Filtering candidate highways that "
                << "increase LL by more than " << minDiff << std::endl;
  double initialLL = evaluator.computeLikelihood();
  evaluator.saveSnapshotPerFamilyLL();
  Logger::timed << "initial ll=" << initialLL << std::endl;
  for (const auto &candidate : candidateHighways) {
    auto highway = candidate.highway;
    if (std::find(filteredHighways.begin(), filteredHighways.end(), candidate) != filteredHighways.end()) { continue; }
    // reject the highway if it is incompatible with the transfer constraint
    if (!isHighwayCompatible(highway, optimizer.getRecModelInfo(),
                             speciesTree.getDatedTree())) {
      Logger::timed << "Rejecting (incompatible) candidate: "
                    << highway.src->label << "->" << highway.dest->label
                    << std::endl;
      continue;
    }
    auto proba = proba1;
    Logger::timed << "Testing candidate: " << highway.src->label << "->"
                  << highway.dest->label << " with highway proba p=" << proba
                  << std::endl;
    // add the highway to test;
    // compute the new total LL after adding the highway and save the new
    // per-family LLs to a file; remove the highway
    double withHighwayLL = testHighwayFast(evaluator, highway, testPath, proba);
    double llDiff = withHighwayLL - initialLL;
    if (llDiff < minDiff) {
      proba = proba2;
      Logger::timed << "  No improvement with the small highway proba! Trying "
                    << "again with p=" << proba << std::endl;
      // ditto, but with higher highway proba
      withHighwayLL = testHighwayFast(evaluator, highway, testPath, proba);
      llDiff = withHighwayLL - initialLL;
    }
    // reject the highway if adding it hasn't increased the LL significantly
    if (llDiff >= minDiff) {
      // optimize the highway proba
      auto bestParameters =
          optimizeSingleHighwayProba(evaluator, highway, proba);
      highway.proba = bestParameters[0];
      withHighwayLL = bestParameters.getScore();
      llDiff = withHighwayLL - initialLL;
      // if BIC < 0, accept the highway and add it to the recmodel for a while
      if (2.0 * llDiff > log(sampleSize)) {
        Logger::timed << "  Accepting the candidate: ";
        filteredHighways.push_back(ScoredHighway(highway, llDiff));
        evaluator.addHighway(highway);
        initialLL = bestParameters.getScore();
        evaluator.saveSnapshotPerFamilyLL();
      } else {
        Logger::timed << "  Rejecting (BIC) the candidate: ";
      }
    } else {
      Logger::timed << "  Rejecting (noImprov) the candidate: ";
    }
    Logger::info << "llDiff=" << llDiff << ", proba=" << highway.proba
                 << std::endl;
  }
  // remove all the temporarily added highways from the recmodel
  for (const auto &candidate : filteredHighways) {
    (void)(candidate);
    evaluator.removeHighway();
  }
  // keep only the highways resulting in higher LL increase
  std::sort(filteredHighways.rbegin(), filteredHighways.rend());
  filteredHighways.resize(
      std::min(filteredHighways.size(), size_t(maxCandidates)));
  Logger::timed << "[Highway search] After keeping at most " << maxCandidates
                << " accepted candidates: " << filteredHighways.size()
                << " highways kept" << std::endl;
}

void Highways::optimizeAllHighways(AleOptimizer &optimizer,
                                   std::vector<ScoredHighway> &highways,
                                   bool thorough,
                                   const std::string outputDir) {
  auto &evaluator = optimizer.getEvaluator();
  double minProba =
      0.000001; // min highway proba after optimization to keep the candidate
  Logger::timed << "[Highway search] Trying to add all candidate highways "
                << "simultaneously" << std::endl;
  // jointly optimize highway probas of the filtered highways
  Parameters startingProbas;
  for (const auto &candidate : highways) {
    startingProbas.addValue(candidate.highway.proba);
  }
  auto bestParameters = optimizeHighwayProbas(evaluator, highways,
                                              startingProbas, true, thorough, false, outputDir);
  Logger::timed << "[Highway search] After highway proba opt, probas and ll:\n"
                << bestParameters << std::endl;
  // keep only the highways with optimized proba no less than minProba and
  // permanently add these highways to the recmodel
  for (unsigned int i = 0; i < highways.size(); ++i) {
    if (bestParameters[i] >= minProba) {
      auto accepted = highways[i];
      accepted.highway.proba = bestParameters[i];
      evaluator.addHighway(accepted.highway);
    }
  }
  std::sort(highways.rbegin(), highways.rend(),
            compareHighwaysByProba);
  Logger::timed << "[Highway search] After dropping highways with proba < "
                << minProba << ": " << highways.size()
                << " highways accepted" << std::endl;
}
