// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/bertrand_oligopoly/bertrand_oligopoly.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace bertrand_oligopoly {
namespace {

namespace testing = open_spiel::testing;

void BasicBertrandOligopolyTests() {
  testing::LoadGameTest("bertrand_oligopoly");
  testing::RandomSimTest(*LoadGame("bertrand_oligopoly"), 100);
}

void LegalActionsValidAtEveryState() {
  GameParameters params;
  
  std::shared_ptr<const Game> game = LoadGameAsTurnBased("bertrand_oligopoly", params);
  testing::RandomSimTest(*game, /*num_sims=*/10);
}

void BertrandOligopolyWithLimitedTurns() {
  GameParameters params;
  params["num_turns"] = GameParameter(3);
  testing::RandomSimTest(*LoadGame("bertrand_oligopoly", params), 10);
}

void EgocentricViewOfSymmetricActions() {
  GameParameters params;
  params["egocentric"] = GameParameter(true);
  params["players"] = GameParameter(2);
  std::shared_ptr<const Game> game = LoadGame("bertrand_oligopoly", params);

  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // two action sequences each played by one player.
  std::vector<Action> seq1{3, 2, 0};
  std::vector<Action> seq2{0, 1, 2};

  // Accumulate info state histories form the perspective of `seq1` when playing
  // as one of the two players.
  std::vector<std::vector<std::vector<float>>> info_state_histories(
      game->NumPlayers());
  for (int as_player = 0; as_player < game->NumPlayers(); as_player++) {
    for (int t = 0; t < game->MaxGameLength() - 1; t++) {
      std::vector<Action> joint_actions(game->NumPlayers(), -1);
      joint_actions[as_player] = seq1[t];
      joint_actions[(as_player + 1) % game->NumPlayers()] = seq2[t];
      state->ApplyActions(std::move(joint_actions));
      auto info_state = state->InformationStateTensor(as_player);
      info_state_histories[as_player].push_back(std::move(info_state));
    }
    state = game->NewInitialState();
  }

  // Verify that the observations remain identical regardless of which player
  // `seq1` was executed for.
  SPIEL_CHECK_EQ(info_state_histories.size(), game->NumPlayers());
  SPIEL_CHECK_EQ(info_state_histories[0], info_state_histories[1]);
}

}  // namespace
}  // namespace bertrand_oligopoly
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::bertrand_oligopoly::BasicBertrandOligopolyTests();
  //open_spiel::bertrand_oligopoly::LegalActionsValidAtEveryState();
  //open_spiel::bertrand_oligopoly::BertrandOligopolyWithLimitedTurns();
  //open_spiel::bertrand_oligopoly::EgocentricViewOfSymmetricActions();
}
