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

#ifndef OPEN_SPIEL_GAMES_BERTRAND_OLIGOPOLY_H_
#define OPEN_SPIEL_GAMES_BERTRAND_OLIGOPOLY_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"

// Goofspiel, or the Game of Pure Strategy, is a bidding card game where players
// are trying to obtain the most points. In, Goofspiel(N,K), each player has bid
// cards numbered 1..N and a point card deck containing cards numbered 1..N is
// shuffled and set face-down. There are K turns. Each turn, the top point card
// is revealed, and players simultaneously play a bid card; the point card is
// given to the highest bidder or discarded if the bids are equal. For more
// detail, see: https://en.wikipedia.org/wiki/Goofspiel
//
// This implementation of Goofspiel is slightly more general than the standard
// game. First, more than 2 players can play it. Second, the deck can take on
// pre-determined orders rather than randomly determined. Third, there is an
// option to enable the imperfect information variant described in Sec 3.1.4
// of http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, where only
// the sequences of wins / losses is revealed (not the players' hands). Fourth,
// players can play for only K turns (if not specified, K=N by default).
//
// The returns_type parameter determines how returns (utilities) are defined:
//   - win_loss distributed 1 point divided by number of winners (i.e. players
//     with highest points), and similarly to -1 among losers
//   - point_difference means each player gets utility as number of points
//     collected minus the average over players.
//   - total_points means each player's return is equal to the number of points
//     they collected.
//
// Parameters:
//   "imp_info"      bool     Enable the imperfect info variant (default: false)
//   "egocentric"   bool     Enable the egocentric info variant (default: false)
//   "num_cards"     int      The highest bid card, and point card (default: 13)
//   "num_turns"     int       The number of turns to play (default: -1, play
//                            for the same number of rounds as there are cards)
//   "players"       int      number of players (default: 2)
//   "points_order"  string   "random" (default), "descending", or "ascending"
//   "returns_type"  string   "win_loss" (default), "point_difference", or
//                            "total_points".

namespace open_spiel {
namespace bertrand_oligopoly {



inline constexpr int kDefaultNumPlayers = 2;
inline constexpr int kDefaultNumOptions = 15; //options of price the agents may set
inline constexpr int kDefaultNumTurns = 100;
inline constexpr double kDefaultIntervalSize = 0.1; //extension of interval of reasonable price as a fraction of the size of the interval between the nash price and the monopoly price
inline constexpr int kDefaultMarginalCost = 1; //unit cost
inline constexpr double kDefaultHorizontalDifferentiation = 0.25; //index of how interchangeable the two items are. bounded on (0, 1]
inline constexpr int kDefaultOutsideGood = 0; //don't change this one
inline constexpr const char* kDefaultReturnsType = "total_points";
inline constexpr const bool kDefaultImpInfo = false;
inline constexpr const bool kDefaultEgocentric = false;

enum class ReturnsType {
  kWinLoss,
  kPointDifference,
  kTotalPoints,
};

inline constexpr const int kInvalidCard = -1;

class BertrandOligopolyObserver;

class BertrandOligopolyState : public SimMoveState {
 public:
  explicit BertrandOligopolyState(std::shared_ptr<const Game> game, int num_options,
                          int num_turns, double interval_size, int marginal_cost,
                          double horizontal_differentiation, 
                          int outside_good, bool impinfo,
                          bool egocentric, ReturnsType returns_type);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  //std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions(Player player) const override;

 protected:
  void DoApplyAction(Action action_id) override;
  void DoApplyActions(const std::vector<Action>& actions) override;

 private:
  friend class BertrandOligopolyObserver;
  // Increments the count and increments the player mod num_players_.
  void NextPlayer(int* count, Player* player) const;

  int num_options_;
  int num_turns_;
  double interval_size_;
  int marginal_cost_;
  double horizontal_differentiation_;
  std::vector<double> vertical_differentiation_;
  int outside_good_;  
  ReturnsType returns_type_;
  bool impinfo_;
  bool egocentric_;

  //derived attributes
  double monopoly_price_;
  double nash_price_;
  std::pair<double, double> interval_;
  double step_size_;
  std::vector<double> net_profit_;


  Player current_player_;
  std::set<int> winners_;
  int current_turn_;
  std::vector<double> points_;
  std::vector<Player> win_sequence_;  // Which player won, kInvalidPlayer if tie
  std::vector<std::vector<Action>> actions_history_;
};

class BertrandOligopolyGame : public Game {
 public:
  explicit BertrandOligopolyGame(const GameParameters& params);

  int NumDistinctActions() const override { return num_options_; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return num_turns_; }
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  int NumOptions() const { return num_options_; }
  int NumRounds() const { return num_turns_; }
  int NumTurns() const { return num_turns_; }
  int NUmOptions() const { return num_options_; }
  ReturnsType GetReturnsType() const { return returns_type_; }
  bool IsImpInfo() const { return impinfo_; }
  //int MaxPointSlots() const { return (NumCards() * (NumCards() + 1)) / 2 + 1; } //should not be necessary

  // Used to implement the old observation API.
  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;
  std::shared_ptr<Observer> public_observer_;
  std::shared_ptr<Observer> private_observer_;
  int MaxChanceNodesInHistory() const override { return 0; }

 private:
  int num_options_;
  int num_turns_;
  double interval_size_;
  int marginal_cost_;
  double horizontal_differentiation_;
  int outside_good_; 
  int num_players_;  // Number of players
  ReturnsType returns_type_;
  bool impinfo_;
  bool egocentric_;

  //derived attributes
  double monopoly_price_;
  double nash_price_;
  std::pair<double, double> interval_;
  double step_size_;
};

}  // namespace goofspiel
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GOOFSPIEL_H_
