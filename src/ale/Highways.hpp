#pragma once

#include <vector>

#include "AleOptimizer.hpp"

class Highways {
public:
  /**
   *  Infer potential highway candidates from the undated reconciliations
   */
  static void getCandidateHighways(AleOptimizer &optimizer,
                                   std::vector<ScoredHighway> &highways,
                                   unsigned int maxCandidates);

  static std::vector<ScoredHighway>
  getSortedCandidatesFromList(AleOptimizer &optimizer,
                              std::vector<Highway> &highways);

  static void setFixedHighways(AleOptimizer &optimizer,
                               std::vector<Highway> &highways,
                               std::vector<ScoredHighway> &fixedHighways);
  /**
   *  Filter the highway candidates by testing them with a small hardcoded
   * probability
   */
  static void filterCandidateHighwaysFast(
      AleOptimizer &optimizer, const std::vector<ScoredHighway> &highways,
      std::vector<ScoredHighway> &filteredHighways, size_t sample_size, bool individual_test = false);

  static void
  optimizeAllHighways(AleOptimizer &optimizer,
                      std::vector<ScoredHighway> &highways,
                      bool thorough);
};
