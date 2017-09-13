"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players
import timeit
from competition_agent import CustomPlayer, reflect_board, similar_boards

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = sample_players.RandomPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    # def testSymmetrie(self):
    #     game = self.game.forecast_move((5,4))
    #     vertical = game_agent.reflect_board_vertical(game)
    #     horizontal = game_agent.reflect_board_horizontal(game)
    #     both = game_agent.reflect_board_horizontal(vertical)
    #     nothing = self.game.forecast_move((0,0))
    #
    #     self.assertTrue(game_agent.compare_games(game, vertical))
    #     self.assertTrue(game_agent.compare_games(game, horizontal))
    #     self.assertTrue(game_agent.compare_games(game, both))
    #     self.assertFalse(game_agent.compare_games(game, nothing))
    #
    # def testRemoveEqualMoves(self):
    #     legal_moves = self.game.get_legal_moves()
    #     forecasts = [self.game.forecast_move(move) for move in legal_moves]
    #     legal_moves_hashed = set(map(game_agent.hash_board, forecasts))
    #     print(len(legal_moves_hashed))
    #     legal_moves_unique = game_agent.remove_equal_moves(self.game, legal_moves)
    #     print('{} vs {} unique moves.'.format(len(legal_moves), len(legal_moves_unique)))
    #     print(legal_moves)
    #     print(legal_moves_unique)

    def testReflection(self):
        game = self.game.forecast_move((5,4)).forecast_move((3,2))
        print(game.to_string())

        # Vertical
        reflection = reflect_board(game, True, False)
        print(reflection.to_string())
        self.assertFalse(reflection.hash() == game.hash())
        self.assertTrue(reflect_board(reflection, True, False).hash() == game.hash())

        # Horizontal
        reflection = reflect_board(game, False, True)
        print(reflection.to_string())
        self.assertFalse(reflection.hash() == game.hash())
        self.assertTrue(reflect_board(reflection, False, True).hash() == game.hash())

        # Both
        reflection = reflect_board(game, True, True)
        print(reflection.to_string())
        self.assertFalse(reflection.hash() == game.hash())
        self.assertTrue(reflect_board(reflection, True, True).hash() == game.hash())

    def testSimilarBoards(self):
        self.assertTrue(len(similar_boards(self.game)) == 1)
        self.game.apply_move((3,3))
        self.assertTrue(len(similar_boards(self.game)) == 1)
        self.game.apply_move((4,4))
        self.assertTrue(len(similar_boards(self.game)) == 4)

if __name__ == '__main__':
    unittest.main()
