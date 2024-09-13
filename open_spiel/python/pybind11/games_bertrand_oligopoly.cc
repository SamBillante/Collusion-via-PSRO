#include "open_spiel/python/pybind11/games_bertrand_oligopoly.h"

#include "open_spiel/games/bertrand_oligopoly/bertrand_oligopoly.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

namespace py = ::pybind11;
using open_spiel::Game;
using open_spiel::State;
using open_spiel::bertrand_oligopoly::BertrandOligopolyState;
//using open_spiel::backgammon::CheckerMove;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(BertrandOligopolyState);

void open_spiel::init_pyspiel_games_bertrand_oligopoly(py::module& m) {
  /*py::classh<BertrandOligopolyState, State>(m, "BertrandOligoployState")
      .def("augment_with_hit_info", &BackgammonState::AugmentWithHitInfo)
      .def("board", &BackgammonState::board)
      .def("checker_moves_to_spiel_move",
           &BackgammonState::CheckerMovesToSpielMove)
      .def("spiel_move_to_checker_moves",
           &BackgammonState::SpielMoveToCheckerMoves)
      .def("translate_action", &BackgammonState::TranslateAction)
      // Pickle support
      .def(py::pickle(
          [](const BackgammonState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BackgammonState*>(
                game_and_state.second.release());
          }));*/
}