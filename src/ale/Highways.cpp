#include "Highways.hpp"
#include "AleOptimizer.hpp"
#include "IO/HighwayCandidateParser.hpp"
#include "util/enums.hpp"

#include <IO/FileSystem.hpp>
#include <IO/Logger.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optimizers/DTLOptimizer.hpp>
#include <search/SpeciesTransferSearch.hpp>
#include <string>
#include <vector>

// const double MIN_PH = 0.00000001;
// const double MAX_PH = 0.8;

class HighwayFunction : public FunctionToOptimize {
public:
  HighwayFunction(AleEvaluator &evaluator,
                  const std::vector<Highway *> &highways,
                  bool print,
                  const std::string &highwaysOutputDir = ""
                  )
      : _highways(highways), _evaluator(evaluator), _print(print),  _highwaysOutputDir(highwaysOutputDir) {}

  virtual double evaluate(Parameters &parameters) {
    double v = evaluatePrint(parameters, _print, _highwaysOutputDir);
    // Logger::timed << "Evaluate transfer " << std::setprecision(17) <<
    // parameters << std::endl;
    return v;
  }

  virtual double evaluatePrint(Parameters &parameters, bool print,
                               const std::string outputDir = "") {
    assert(parameters.dimensions() == _highways.size());
    // parameters.constrain(MIN_PH, MAX_PH);
    for (unsigned int i = 0; i < _highways.size(); ++i) {
      Highway highwayCopy = *_highways[i];
      highwayCopy.proba = parameters[i];
      _evaluator.addHighway(highwayCopy);
    }
    auto ll = _evaluator.computeLikelihood();
    if (print) {
      assert(outputDir.size());
      std::string out = FileSystem::joinPaths(
          outputDir,
          std::string("transferll_") + std::to_string(parameters[0]) + std::string("_") + std::string(_highways[0]->src->label) +
              std::string("_") + std::string(_highways[0]->dest->label));
      _evaluator.savePerFamilyLikelihoodDiff(out);
    }
    for (auto highway : _highways) {
      (void)(highway);
      _evaluator.removeHighway();
    }
    parameters.setScore(ll);
    return ll;
  }

private:
  const std::vector<Highway *> &_highways;
  AleEvaluator &_evaluator;
  bool _print;
  const std::string _highwaysOutputDir;
};

class HighwayFunctionSingle : public FunctionToOptimize {
public:
  HighwayFunctionSingle(AleEvaluator &evaluator,
                  Highway &highway)
      : _highway(highway), _evaluator(evaluator) {
        _evaluator.addHighway(highway);
        _evaluator.computeLikelihood();
      }
  ~HighwayFunctionSingle() {
    _evaluator.removeHighway();
  }

  virtual double evaluate(Parameters &parameters) {
    _highway.proba = parameters[0];
    auto ll = _evaluator.computeHighwayTerm(_highway);
    parameters.setScore(ll);
    return ll;
  }

  double evaluateFull(Parameters &parameters) {
    _evaluator.removeHighway();
    _highway.proba = parameters[0];
    _evaluator.addHighway(_highway);
    auto ll = _evaluator.computeLikelihood();
    parameters.setScore(ll);
    return ll;
  }
private:
  Highway &_highway;
  AleEvaluator &_evaluator;
};
static Parameters testHighwayFast(AleEvaluator &evaluator,
                                  const Highway &highway,
                                  const std::string &highwaysOutputDir,
                                  double startingProbability = 0.01) {
  std::vector<Highway *> highways;
  auto copy = highway;
  highways.push_back(&copy);
  HighwayFunction f(evaluator, highways, false, highwaysOutputDir);
  Parameters parameters(1);
  parameters[0] = startingProbability;
  f.evaluatePrint(parameters, true, highwaysOutputDir);
  return parameters;
}

static Parameters testHighways(AleEvaluator &evaluator,
                               std::vector<ScoredHighway> &scoredHighways,
                               const Parameters &startingProbabilities,
                               bool optimize, bool thorough, bool individual_contribution) {
  assert(scoredHighways.size() == startingProbabilities.dimensions());
  std::vector<Highway *> highways;
  for (auto &highway : scoredHighways) {
    highways.push_back(&highway.highway);
  }
  HighwayFunction f(evaluator, highways, false);
  if (optimize) {
    OptimizationSettings settings;
    settings.strategy = evaluator.getRecModelInfo().recOpt;
    settings.minAlpha = 0.001;
    settings.epsilon = 0.000001;
    settings.strategy = RecOpt::LBFGSB;
    settings.factr = LBFGSBPrecision::MEDIUM;
    // settings.verbose = true;
    if (thorough) {
      settings.individualParamOpt = true;
      settings.individualParamOptMinImprovement = 10000.0;
    }
    auto res =
        DTLOptimizer::optimizeParameters(f, startingProbabilities, settings);
    if (individual_contribution) {
      auto initial_ll = res.getScore();
      for (std::size_t i = 0; i < res.dimensions(); ++i) {
        auto parameters = res;
        parameters[i] = 0.0;
        auto new_ll = f.evaluate(parameters);
        auto lldiff = initial_ll - new_ll;
        scoredHighways[i].scoreDiff = lldiff;
        Logger::info << "LL diff from highway " << scoredHighways[i].highway << ": " << lldiff << std::endl;
      }
    }
    return res;
  } else {
    auto parameters = startingProbabilities;
    f.evaluate(parameters);
    return parameters;
  }
}

static Parameters optimizeSingleHighway(AleEvaluator &evaluator,
                                        Highway &highway,
                                        const std::string &highwaysOutputDir,
                                        double startingProbability) {
  // std::vector<Highway *> highways;
  // auto copy = highway;
  // highways.push_back(&copy);
  HighwayFunctionSingle f(evaluator, highway);
  Parameters startingProbabilities(1);
  startingProbabilities[0] = startingProbability;
  OptimizationSettings settings;
  settings.strategy = RecOpt::LBFGSB;
  settings.minAlpha = 0.001;
  settings.epsilon = 0.000001;
  // settings.verbose = true;
  settings.factr = LBFGSBPrecision::MEDIUM;
  auto res =
      DTLOptimizer::optimizeParameters(f, startingProbabilities, settings);
  // res.constrain(MIN_PH, MAX_PH);
  f.evaluateFull(res);
  return res;
}

static bool isHighwayCompatible(Highway &highway, const RecModelInfo &info,
                                const DatedTree &tree) {
  auto from = highway.src;
  auto to = highway.dest;
  switch (info.transferConstraint) {
  case TransferConstaint::NONE:
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

void Highways::getCandidateHighways(AleOptimizer &optimizer,
                                    std::vector<ScoredHighway> &scoredHighways,
                                    unsigned int maxCandidates) {
  auto &speciesTree = optimizer.getSpeciesTree();
  unsigned int minTransfers = 2;
  MovesBlackList blacklist;
  std::vector<TransferMove> transferMoves;
  SpeciesTransferSearch::getSortedTransferList(
      speciesTree, optimizer.getEvaluator(), minTransfers, blacklist,
      transferMoves);

  for (const auto &transferMove : transferMoves) {
    auto prune = speciesTree.getNode(transferMove.prune);
    auto regraft = speciesTree.getNode(transferMove.regraft);
    Highway highway(regraft, prune);
    auto lca = speciesTree.getTree().getLCA(prune, regraft);
    unsigned int distance = 0;
    while (prune != lca) {
      distance += 1;
      prune = prune->parent;
    }
    while (regraft != lca) {
      distance += 1;
      regraft = regraft->parent;
    }
    if (distance >= 5 && prune->left != nullptr) {
      scoredHighways.push_back(ScoredHighway(highway, 0.0));
    } else {
      Logger::timed << "Rejecting (dist) candidate: " << highway.src->label
                    << "->" << highway.dest->label << " d = " << distance
                    << std::endl;
    }
    if (scoredHighways.size() >= maxCandidates) {
      break;
    }
  }
}

void Highways::setFixedHighways(AleOptimizer &optimizer, std::vector<Highway> &highways, std::vector<ScoredHighway> &fixed_highways) {
  auto &evaluator = optimizer.getEvaluator();
  Logger::timed << "Adding all fixed highways"
                << std::endl;
  Parameters startingProbabilities;
  for (auto &highway : highways) {
    startingProbabilities.addValue(highway.proba);
    fixed_highways.push_back(ScoredHighway(highway));
  }
  auto parameters = testHighways(evaluator, fixed_highways, startingProbabilities,
                                 true, false, true);
  Logger::info << parameters << std::endl;
  for (unsigned int i = 0; i < highways.size(); ++i) {
    fixed_highways[i].highway.proba = parameters[i];
    evaluator.addHighway(fixed_highways[i].highway);
  }
}

std::vector<ScoredHighway>
Highways::getSortedCandidatesFromList(AleOptimizer &optimizer,
                                      std::vector<Highway> &candidateHighways) {
  auto &speciesTree = optimizer.getSpeciesTree();
  unsigned int minTransfers = 1;
  MovesBlackList blacklist;
  std::vector<ScoredHighway> scoredHighways;
  std::vector<TransferMove> transferMoves;
  SpeciesTransferSearch::getSortedTransferList(
      speciesTree, optimizer.getEvaluator(), minTransfers, blacklist,
      transferMoves);

  for (const auto &transferMove : transferMoves) {
    auto prune = speciesTree.getNode(transferMove.prune);
    auto regraft = speciesTree.getNode(transferMove.regraft);
    Highway highway(regraft, prune);
    if (std::find(candidateHighways.begin(), candidateHighways.end(),
                  highway) != candidateHighways.end()) {
      scoredHighways.push_back(ScoredHighway(highway, 0.0));
    }
  }
  for (const auto &highway : candidateHighways) {
    if (std::find(scoredHighways.begin(), scoredHighways.end(), highway) ==
        scoredHighways.end()) {
      scoredHighways.push_back(ScoredHighway(highway, 0.0));
    }
  }
  return scoredHighways;
}

void Highways::filterCandidateHighwaysFast(
    AleOptimizer &optimizer, const std::vector<ScoredHighway> &highways,
    std::vector<ScoredHighway> &filteredHighways, size_t sample_size, bool individual_test) {
  auto &evaluator = optimizer.getEvaluator();
  auto &speciesTree = optimizer.getSpeciesTree();
  Logger::timed << "Filering " << highways.size() << " candidate highways"
                << std::endl;
  double initialLL = evaluator.computeLikelihood();
  Logger::timed << "initial ll=" << initialLL << std::endl;
  evaluator.saveSnapshotPerFamilyLL();
  for (const auto &scoredHighway : highways) {
    if (std::find(filteredHighways.begin(), filteredHighways.end(), scoredHighway) != filteredHighways.end()) { continue; }
    double proba = 0.01;
    auto highway = scoredHighway.highway;
    if (!isHighwayCompatible(highway, optimizer.getRecModelInfo(),
                             speciesTree.getDatedTree())) {
      Logger::info << "Incompatible highway " << highway.src->label << "->"
                   << highway.dest->label << std::endl;
      continue;
    }
    Logger::timed << "Testing candidate: " << highway.src->label << "->"
                  << highway.dest->label << " with p = " << proba << std::endl;
    auto parameters = optimizeSingleHighway(evaluator, highway, optimizer.getHighwaysOutputDir(), 0.1);
    auto llDiff = parameters.getScore() - initialLL;
    if (individual_test || (2 * llDiff > log(sample_size))) {
      Logger::timed << "Accepting candidate: ";
      highway.proba = parameters[0];
      filteredHighways.push_back(ScoredHighway(highway, -llDiff));
      if (!individual_test) {
        evaluator.addHighway(highway);
        initialLL = parameters.getScore();
        evaluator.saveSnapshotPerFamilyLL();
      }
    } else {
      Logger::timed << "Rejecting (BIC) candidate: ";
    }
    Logger::info << highway.src->label << "->" << highway.dest->label
                 << " ll diff = " << llDiff << " best proba = " << highway.proba
                 << std::endl;
  }
  if (!individual_test) {
    for (auto &highway : filteredHighways) {
      (void)(highway);
      evaluator.removeHighway();
    }
  }
  std::sort(filteredHighways.begin(), filteredHighways.end());
}

void Highways::optimizeAllHighways(
    AleOptimizer &optimizer,
    std::vector<ScoredHighway> &highways,
    bool thorough) {
  auto &evaluator = optimizer.getEvaluator();
  Logger::timed << "Trying to add all candidate highways simultaneously"
                << std::endl;
  Parameters startingProbabilities;
  for (const auto candidate : highways) {
    startingProbabilities.addValue(candidate.highway.proba);
  }
  auto parameters = testHighways(evaluator, highways, startingProbabilities,
                                 true, thorough, true);
  Logger::info << parameters << std::endl;
  for (unsigned int i = 0; i < highways.size(); ++i) {
    highways[i].highway.proba = parameters[i];
    evaluator.addHighway(highways[i].highway);
  }
  std::sort(highways.rbegin(), highways.rend(),
            cmpHighwayByProbability);
}
