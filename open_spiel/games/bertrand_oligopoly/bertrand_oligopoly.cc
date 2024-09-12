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

#include <algorithm>
#include <memory>
#include <ostream>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace bertrand_oligopoly {
namespace {

const GameType kGameType{
    /*short_name=*/"bertrand_oligopoly",
    /*long_name=*/"Bertrand_oligopoly",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"imp_info", GameParameter(kDefaultImpInfo)},
        {"egocentric", GameParameter(kDefaultEgocentric)},
        {"num_options", GameParameter(kDefaultNumOptions)},
        {"interval_size", GameParameter(kDefaultIntervalSize)},
        {"marginal_cost", GameParameter(kDefaultMarginalCost)},
        {"horizontal_differentiation", GameParameter(kDefaultHorizontalDifferentiation)},
        {"outside_good", GameParameter(kDefaultOutsideGood)},
        {"num_turns", GameParameter(kDefaultNumTurns)},
        {"players", GameParameter(kDefaultNumPlayers)},
        {"returns_type",
         GameParameter(static_cast<std::string>(kDefaultReturnsType))},
    },
    /*default_loadable=*/true,
    /*provides_factored_observation_string=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BertrandOligopolyGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

ReturnsType ParseReturnsType(const std::string& returns_type_str) {
  if (returns_type_str == "win_loss") {
    return ReturnsType::kWinLoss;
  } else if (returns_type_str == "point_difference") {
    return ReturnsType::kPointDifference;
  } else if (returns_type_str == "total_points") {
    return ReturnsType::kTotalPoints;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized returns_type parameter: ",
                                 returns_type_str));
  }
}

}  // namespace

class BertrandOligopolyObserver : public Observer {
 public:
  explicit BertrandOligopolyObserver(IIGObservationType iig_obs_type, bool egocentric)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type),
        egocentric_(egocentric) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const BertrandOligopolyState& state =
        open_spiel::down_cast<const BertrandOligopolyState&>(observed_state);
    const BertrandOligopolyGame& game =
        open_spiel::down_cast<const BertrandOligopolyGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());

    // List all predicates.
    const bool imp_info = game.IsImpInfo();
    const bool pub_info = iig_obs_type_.public_info;
    const bool perf_rec = iig_obs_type_.perfect_recall;
    const bool priv_one =
        iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer;

    // Conditionally write each field.
    if (pub_info) WritePointsTotal(game, state, player, allocator);
    if (imp_info && pub_info) WriteWinSequence(game, state, player, allocator);
    if (imp_info && perf_rec && priv_one)
      WritePlayerActionSequence(game, state, player, allocator);
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const BertrandOligopolyState& state =
        open_spiel::down_cast<const BertrandOligopolyState&>(observed_state);
    const BertrandOligopolyGame& game =
        open_spiel::down_cast<const BertrandOligopolyGame&>(*state.GetGame());
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game.NumPlayers());
    std::string result;

    // List all predicates.
    const bool imp_info = game.IsImpInfo();
    const bool pub_info = iig_obs_type_.public_info;
    const bool perf_rec = iig_obs_type_.perfect_recall;
    const bool priv_one =
        iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer;

    // Conditionally write each field.
    // This is done in a backwards-compatible way.
    if (imp_info && priv_one && perf_rec) {  // InformationState
      StringActionSequence(game, state, player, &result);
      StringWinSequence(state, &result);
      StringPoints(game, state, &result);
      StringIsTerminal(state, &result);
      return result;
    }
    if (imp_info && priv_one && !perf_rec) {  // Observation
      StringPoints(game, state, &result);
      StringWinSequence(state, &result);
      return result;
    }

    // Remaining public observation requests.
    if (pub_info) {
      StringWinSequence(state, &result);
      StringPoints(game, state, &result);
    }
    return result;
  }

 private:
// Point totals: vector of point totals
  // Writes this public information from the perspective
  // of the requesting player.
  void WritePointsTotal(const BertrandOligopolyGame& game, const BertrandOligopolyState& state,
                        int player, Allocator* allocator) const {
    auto out = allocator->Get("point_totals",
                              {game.NumPlayers()});
    Player p = player;
    for (int n = 0; n < game.NumPlayers(); state.NextPlayer(&n, &p)) {
      out.at(n) = state.points_[p];
    }
  }

  // Sequence of who won each trick.
  void WriteWinSequence(const BertrandOligopolyGame& game, const BertrandOligopolyState& state,
                        int player, Allocator* allocator) const {
    auto out =
        allocator->Get("win_sequence", {game.NumRounds(), game.NumPlayers()});
    for (int i = 0; i < state.win_sequence_.size(); ++i) {
      if (state.win_sequence_[i] != kInvalidPlayer) {
        int one_hot = state.win_sequence_[i];
        if (egocentric_) {
          // Positive, relative distance to the winner.
          one_hot = ((game.NumPlayers() + state.win_sequence_[i] - player) %
                     game.NumPlayers());
        }
        out.at(i, one_hot) = 1.0;
      }
    }
  }

  // The observing player's action sequence.
  void WritePlayerActionSequence(const BertrandOligopolyGame& game,
                                 const BertrandOligopolyState& state, int player,
                                 Allocator* allocator) const {
    auto out = allocator->Get("player_action_sequence",
                              {game.NumRounds(), game.NumOptions()});
    for (int round = 0; round < state.actions_history_.size(); ++round) {
      out.at(round, state.actions_history_[round][player]) = 1.0;
    }
  }

  void StringActionSequence(const BertrandOligopolyGame& game,
                            const BertrandOligopolyState& state, int player,
                            std::string* result) const {
    // Also show the player's sequence. We need this to ensure perfect
    // recall because two betting sequences can lead to the same hand and
    // outcomes if the opponent chooses differently.
    absl::StrAppend(result, "P", player, " action sequence: ");
    for (int i = 0; i < state.actions_history_.size(); ++i) {
      absl::StrAppend(result, state.actions_history_[i][player], " ");
    }
    absl::StrAppend(result, "\n");
  }
  void StringWinSequence(const BertrandOligopolyState& state,
                         std::string* result) const {
    absl::StrAppend(result, "Win sequence: ");
    for (int i = 0; i < state.win_sequence_.size(); ++i) {
      absl::StrAppend(result, state.win_sequence_[i], " ");
    }
    absl::StrAppend(result, "\n");
  }
  void StringPoints(const BertrandOligopolyGame& game, const BertrandOligopolyState& state,
                    std::string* result) const {
    absl::StrAppend(result, "Points: ");
    for (auto p = Player{0}; p < game.NumPlayers(); ++p) {
      absl::StrAppend(result, state.points_[p], " ");
    }
    absl::StrAppend(result, "\n");
  }
  void StringIsTerminal(const BertrandOligopolyState& state,
                        std::string* result) const {
    absl::StrAppend(result, "Terminal?: ", state.IsTerminal(), "\n");
  }

  IIGObservationType iig_obs_type_;
  const bool egocentric_;
};

BertrandOligopolyState::BertrandOligopolyState(std::shared_ptr<const Game> game, int num_options,
                               int num_turns, double interval_size, int marginal_cost,
                               double horizontal_differentiation, int outside_good,
                               bool impinfo, bool egocentric,
                               ReturnsType returns_type)
    : SimMoveState(game),
      num_options_(num_options),
      num_turns_(num_turns),
      interval_size_(interval_size),
      marginal_cost_(marginal_cost),
      horizontal_differentiation_(horizontal_differentiation),
      outside_good_(outside_good),
      returns_type_(returns_type),
      impinfo_(impinfo),
      egocentric_(egocentric),
      current_player_(kSimultaneousPlayerId),
      winners_({}),
      current_turn_(0),
      win_sequence_({}),
      actions_history_({}) {

  SPIEL_CHECK_GT(num_options_, 1); //prevent divide by zero error when calculating step size

  // Points.
  points_.resize(num_players_);
  std::fill(points_.begin(), points_.end(), 0);
  //vertical differentiation
  vertical_differentiation_.resize(num_players_);
  std::fill(vertical_differentiation_.begin(), vertical_differentiation_.end(), 2.0); //no vertical differentiation
  //net profit (for print statements)
  net_profit_.resize(num_players_);

  nash_price_ = 1.47292; //FIXME: THIS IS AN APPROXIMATION, I DO NOT KNOW HOW TO CALCULATE THE EXACT VALUE GENERICALLY
  monopoly_price_ = 1.92498; //FIXME: THIS IS AN APPROXIMATION, I DO NOT KNOW HOW TO CALCULATE THE EXACT VALUE GENERICALLY
  interval_.first = nash_price_ - interval_size_ * (monopoly_price_ - nash_price_);
  interval_.second = monopoly_price_ + interval_size_ * (monopoly_price_ - nash_price_);
  step_size_ = (interval_.second - interval_.first) / (num_options_ - 1); //subtract 1 so interval.first is action 0 and interval.second is action (num_options - 1)
}

int BertrandOligopolyState::CurrentPlayer() const { return current_player_; }


void BertrandOligopolyState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  //DealPointCard(action_id); //remove this line
  current_player_ = kSimultaneousPlayerId;
}

void BertrandOligopolyState::DoApplyActions(const std::vector<Action>& actions) {
  // Check the actions are valid.
  SPIEL_CHECK_EQ(actions.size(), num_players_);
  for (auto p = Player{0}; p < num_players_; ++p) {
    const int action = actions[p];
    SPIEL_CHECK_GE(action, 0);
    SPIEL_CHECK_LT(action, num_options_);
  }

  // Find the lowest price
  int min_price = INT_MAX;
  int num_min_prices = 0;
  int min_actor = -1;

  for (int p = 0; p < actions.size(); ++p) {
    if (actions[p] < min_price) {
      min_price = actions[p];
      num_min_prices = 1;
      min_actor = p;
    } else if (actions[p] == min_price) {
      num_min_prices++;
    }
  }

  if (num_min_prices == 1) {
    // Winner takes the point card.
    //points_[min_actor] += CurrentPointValue(); //relevant to winner take all version
    win_sequence_.push_back(min_actor);
  } else {
    // Tied among several players: discarded.
    win_sequence_.push_back(kInvalidPlayer);
  }

  //do payouts for each actor

  double price;
  double demand = 0;
  double demand_denominator = exp(outside_good_ / horizontal_differentiation_);

  for (int p = 0; p < actions.size(); ++p) {
    price = (actions[p] * step_size_) + interval_.first;
    demand_denominator += exp((vertical_differentiation_[p] - price) / horizontal_differentiation_);
  }
  for (int p = 0; p < actions.size(); ++p) {
    price = (actions[p] * step_size_) + interval_.first;
    net_profit_[p] = ((price - marginal_cost_) * exp((vertical_differentiation_[p] - price) / horizontal_differentiation_)) / demand_denominator;
    points_[p] += net_profit_[p];
  }

  // Add these actions to the history.
  actions_history_.push_back(actions);

  // Next player's turn.
  current_turn_++;

   if (current_turn_ == num_turns_) {
    // Game over - determine winner.
    int max_points = -1;
    for (auto p = Player{0}; p < num_players_; ++p) {
      if (points_[p] > max_points) {
        winners_.clear();
        max_points = points_[p];
        winners_.insert(p);
      } else if (points_[p] == max_points) {
        winners_.insert(p);
      }
    }
    current_player_ = kTerminalPlayerId;
  }
}

//This function needs to exist and I'm not sure why. it SHOULD never be called

/*std::vector<std::pair<Action, double>> BertrandOligopolyState::ChanceOutcomes() const { 
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  return outcomes;
}*/

std::vector<Action> BertrandOligopolyState::LegalActions(Player player) const {
  if (CurrentPlayer() == kTerminalPlayerId) return std::vector<Action>();
  if (player == kSimultaneousPlayerId) return LegalFlatJointActions();
  if (player == kChancePlayerId) return LegalChanceOutcomes();
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::vector<Action> movelist;
  for (int option = 0; option < num_options_; ++option) {
    movelist.push_back(option);
  }
  return movelist;
}

std::string BertrandOligopolyState::ActionToString(Player player,
                                           Action action_id) const {
  if (player == kSimultaneousPlayerId)
    return FlatJointActionToString(action_id);
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, num_options_);
  return absl::StrCat("[P", player, "]: ", (action_id + 1));
  
}

std::string BertrandOligopolyState::ToString() const {
  std::string points_line = "Points: ";
  std::string result = "";

  for (auto p = Player{0}; p < num_players_; ++p) {
    absl::StrAppend(&points_line, points_[p]);
    absl::StrAppend(&points_line, " ");
    absl::StrAppend(&result, "P");
    absl::StrAppend(&result, p);
    absl::StrAppend(&result, " profit: ");
    absl::StrAppend(&result, net_profit_[p]);
    absl::StrAppend(&result, "\n");
  }

  // In imperfect information, the full state depends on both betting sequences
  if (impinfo_) {
    for (auto p = Player{0}; p < num_players_; ++p) {
      absl::StrAppend(&result, "P", p, " actions: ");
      for (int i = 0; i < actions_history_.size(); ++i) {
        absl::StrAppend(&result, actions_history_[i][p]);
        absl::StrAppend(&result, " ");
      }
      absl::StrAppend(&result, "\n");
    }
  }

  
  absl::StrAppend(&result, "\n");

  return result + points_line + "\n";
}

bool BertrandOligopolyState::IsTerminal() const {
  return current_player_ == kTerminalPlayerId;
}

std::vector<double> BertrandOligopolyState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  if (returns_type_ == ReturnsType::kWinLoss) {
    if (winners_.size() == num_players_) {
      // All players have same number of points? This is a draw.
      return std::vector<double>(num_players_, 0.0);
    } else {
      int num_winners = winners_.size();
      int num_losers = num_players_ - num_winners;
      std::vector<double> returns(num_players_, (-1.0 / num_losers));
      for (const auto& winner : winners_) {
        returns[winner] = 1.0 / num_winners;
      }
      return returns;
    }
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    std::vector<double> returns(num_players_, 0);
    double sum = 0;
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] = points_[p];
      sum += points_[p];
    }
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] -= sum / num_players_;
    }
    return returns;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
    std::vector<double> returns(num_players_, 0);
    for (Player p = 0; p < num_players_; ++p) {
      returns[p] = points_[p];
    }
    return returns;
  } else {
    SpielFatalError(absl::StrCat("Unrecognized returns type: ", returns_type_));
  }
}

std::string BertrandOligopolyState::InformationStateString(Player player) const {
  const BertrandOligopolyGame& game =
      open_spiel::down_cast<const BertrandOligopolyGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string BertrandOligopolyState::ObservationString(Player player) const {
  const BertrandOligopolyGame& game =
      open_spiel::down_cast<const BertrandOligopolyGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void BertrandOligopolyState::NextPlayer(int* count, Player* player) const {
  *count += 1;
  *player = (*player + 1) % num_players_;
}

void BertrandOligopolyState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const BertrandOligopolyGame& game =
      open_spiel::down_cast<const BertrandOligopolyGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void BertrandOligopolyState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const BertrandOligopolyGame& game =
      open_spiel::down_cast<const BertrandOligopolyGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> BertrandOligopolyState::Clone() const {
  return std::unique_ptr<State>(new BertrandOligopolyState(*this));
}

BertrandOligopolyGame::BertrandOligopolyGame(const GameParameters& params)
    : Game(kGameType, params),
      num_options_(ParameterValue<int>("num_options")),
      num_turns_(ParameterValue<int>("num_turns")),
      interval_size_(ParameterValue<double>("interval_size")),
      marginal_cost_(ParameterValue<int>("marginal_cost")),
      horizontal_differentiation_(ParameterValue<double>("horizontal_differentiation")),
      outside_good_(ParameterValue<int>("outside_good")),
      num_players_(ParameterValue<int>("players")),
      returns_type_(
          ParseReturnsType(ParameterValue<std::string>("returns_type"))),
      impinfo_(ParameterValue<bool>("imp_info")),
      egocentric_(ParameterValue<bool>("egocentric")) {
  // Override the zero-sum utility in the game type if total point scoring.
  if (returns_type_ == ReturnsType::kTotalPoints) {
    game_type_.utility = GameType::Utility::kGeneralSum;
  }
  // Maybe override the perfect information in the game type.
  if (impinfo_) {
    game_type_.information = GameType::Information::kImperfectInformation;
  }

  //FIXME: ? derive derived attributes
  nash_price_ = 1.47292; //FIXME: THIS IS AN APPROXIMATION, I DO NOT KNOW HOW TO CALCULATE THE EXACT VALUE GENERICALLY
  monopoly_price_ = 1.92498; //FIXME: THIS IS AN APPROXIMATION, I DO NOT KNOW HOW TO CALCULATE THE EXACT VALUE GENERICALLY
  interval_.first = nash_price_ - interval_size_ * (monopoly_price_ - nash_price_);
  interval_.second = monopoly_price_ + interval_size_ * (monopoly_price_ - nash_price_);
  step_size_ = (interval_.second - interval_.first) / (num_options_ - 1); //subtract 1 so interval.first is action 0 and interval.second is action (num_options - 1)

  const GameParameters obs_params = {
      {"egocentric", GameParameter(egocentric_)}};
  default_observer_ = MakeObserver(kDefaultObsType, obs_params);
  info_state_observer_ = MakeObserver(kInfoStateObsType, obs_params);
  private_observer_ = MakeObserver(
      IIGObservationType{/*public_info*/false,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kSinglePlayer},
      obs_params);
  public_observer_ =
      MakeObserver(IIGObservationType{/*public_info*/true,
                                      /*perfect_recall*/false,
                                      /*private_info*/PrivateInfoType::kNone},
                   obs_params);
}

std::unique_ptr<State> BertrandOligopolyGame::NewInitialState() const {
  return std::make_unique<BertrandOligopolyState>(shared_from_this(), num_options_,
                                          num_turns_, interval_size_, marginal_cost_, 
                                          horizontal_differentiation_, outside_good_,
                                          impinfo_, egocentric_, returns_type_);
}

int BertrandOligopolyGame::MaxChanceOutcomes() const {
  return 0;
}

/* I do not know when these functions are used / if I need them to work for this game. they are copied from goofspiel */

std::vector<int> BertrandOligopolyGame::InformationStateTensorShape() const { 
  return {num_players_ * (num_options_)};
}

std::vector<int> BertrandOligopolyGame::ObservationTensorShape() const {
  return {num_players_ * (num_options_)};
}

double BertrandOligopolyGame::MinUtility() const {
  if (returns_type_ == ReturnsType::kWinLoss) {
    return -1;
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    // 0 - (1 + 2 + ... + N) / n
    return 0;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
      if(interval_.first - marginal_cost_ < 0) { //negative utility is possible! actual lower bound will be a bit above.
        return (interval_.first - marginal_cost_) * num_turns_;
      } else
      return 0;
  } else {
    SpielFatalError("Unrecognized returns type.");
  }
}

double BertrandOligopolyGame::MaxUtility() const {
  if (returns_type_ == ReturnsType::kWinLoss) {
    return 1;
  } else if (returns_type_ == ReturnsType::kPointDifference) {
    return (monopoly_price_ - marginal_cost_) * num_turns_;
  } else if (returns_type_ == ReturnsType::kTotalPoints) {
    return (monopoly_price_ - marginal_cost_) * num_turns_;
  } else {
    SpielFatalError("Unrecognized returns type.");
  }
}


absl::optional<double> BertrandOligopolyGame::UtilitySum() const {
  if (returns_type_ == ReturnsType::kTotalPoints)
    return absl::nullopt;
  else
    return 0;
}

std::shared_ptr<Observer> BertrandOligopolyGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  // Allows for `egocentric` overrides if observer variant is needed.
  bool egocentric = egocentric_;
  const auto& it = params.find("egocentric");
  if (it != params.end()) {
    egocentric = it->second.value<bool>();
  }
  return std::make_shared<BertrandOligopolyObserver>(
      iig_obs_type.value_or(kDefaultObsType), egocentric);
}

}  // namespace bertrand_oligopoly
}  // namespace open_spiel
