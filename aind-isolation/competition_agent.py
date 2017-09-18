"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def reflect_board(game, vertical, horizontal):
    """Returns a reflected copy of the game along the vertical and/or horizontal
    axis.

    Parameters
    ----------
    game: isolation.Board
    vertical: boolean
        Determines if the game is reflected vertically.
    horizontal: boolean
        Determines if the game is reflected horizontally.

    Returns
    -------
    isolation.Board
        A reflected copy of the given game.
    """
    dim = game.height
    size = game.height * game.width

    # Reflect fields.
    board = [game._board_state[idx:idx+dim] for idx in range(0, size, dim)]
    if vertical:
        board = [col[::-1] for col in board]
    if horizontal:
        board.reverse()
    reflection = game.copy()
    reflection._board_state = [elem for col in board for elem in col] + game._board_state[size:]

    # Reflect player 1 position.
    position = game.get_player_location(game._player_1)
    if position is not None:
        row = game.height - position[0] - 1 if vertical else position[0]
        col = game.width - position[1] - 1 if horizontal else position[1]
        idx = row + col * game.height
        reflection._board_state[-1] = idx
    # Reflect player 2 position.
    position = game.get_player_location(game._player_2)
    if position is not None:
        row = game.height - position[0] - 1 if vertical else position[0]
        col = game.width - position[1] - 1 if horizontal else position[1]
        idx = row + col * game.height
        reflection._board_state[-2] = idx
    return reflection

def similar_boards(game):
    """Finds all game states that are equal to game if reflected along one or both
    axis.

    Parameters
    ----------
    game: isolation.Board
        The game to find similar games for.

    Returns
    -------
    [isolation.Board]
        A list of games similar to the given game, including a copy of the given
        game. Duplicates have been removed.
    """
    boards = [game.copy()]
    reflection = reflect_board(game, True, False)
    if game.hash() != reflection.hash():
        boards.append(reflection)
    reflection = reflect_board(game, False, True)
    if game.hash() != reflection.hash():
        boards.append(reflection)
    if len(boards) == 3:
        boards.append(reflect_board(game, True, True))
    return boards

def reflect_idx(idx, width, height, vertical, horizontal):
    if idx is None:
        return None
    row = idx % height
    col = idx // height
    row = height - row - 1 if horizontal else row
    col = width - col - 1 if vertical else col
    return row + col * height

def compare_games(lhs, rhs):
    if lhs is None or rhs is None:
        return False

    vertical = True
    horizontal = True
    both = True

    width = lhs.width
    height = lhs.height

    # Compare player 1 positions.
    vertical = vertical or (rhs._board_state[-1] == reflect_idx(lhs._board_state[-1], width, height, True, False))
    horizontal = horizontal or (rhs._board_state[-1] == reflect_idx(lhs._board_state[-1], width, height, False, True))
    both = both or (rhs._board_state[-1] == reflect_idx(lhs._board_state[-1], width, height, True, True))
    if not vertical and not horizontal and not both:
        return False

    # Compare player 2 positions.
    vertical = vertical or (rhs._board_state[-2] == reflect_idx(lhs._board_state[-2], width, height, True, False))
    horizontal = horizontal or (rhs._board_state[-2] == reflect_idx(lhs._board_state[-2], width, height, False, True))
    both = both or (rhs._board_state[-2] == reflect_idx(lhs._board_state[-2], width, height, True, True))
    if not vertical and not horizontal and not both:
        return False

    # Compare fields.
    for idx in range(0, len(lhs._board_state) - 3):
        idx_v = reflect_idx(idx, width, height, True, False)
        vertical = vertical and rhs._board_state[idx] == lhs._board_state[idx_v]
        idx_h = reflect_idx(idx, width, height, False, True)
        horizontal = horizontal and rhs._board_state[idx] == lhs._board_state[idx_h]
        idx_b = reflect_idx(idx, width, height, True, True)
        both = both and rhs._board_state[idx] == lhs._board_state[idx_b]
        if not vertical and not horizontal and not both:
            return False
    return vertical | horizontal | both

def remove_equal_moves(game, moves):
    forecasts = [game.forecast_move(move) for move in moves]
    for idx_1 in range(1, len(moves)):
        for idx_2 in range(0, idx_1):
            if compare_games(forecasts[idx_1], forecasts[idx_2]):
                moves[idx_1] = None
    return [move for move in moves if move is not None]

def num_moves_made(game):
    num_fields = len(game._board_state) - 3
    num_open_fields = len([1 for field in game._board_state[0:-3] if field == game.BLANK])
    return num_fields - num_open_fields

def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    delta_moves = own_moves - 2 * opp_moves;
    return delta_moves

    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    delta_center = 0
    delta_sided = 0
    if own_location is not None and opp_location is not None:
        center = (game.height / 2, game.width / 2)
        own_center_dist = distance(own_location, center)
        opp_center_dist = distance(opp_location, center)
        delta_center = opp_center_dist - own_center_dist

        # Check if we are both on the same vertical side of center.
        if (own_location[0] < center[0]) == (opp_location[0] < center[0]):
            delta_sided += abs(center[0] - opp_location[0]) - abs(center[0] - own_location[0])
        # Check if we are both on the same horizontal side of center.
        if (own_location[1] < center[1]) == (opp_location[1] < center[1]):
            delta_sided += abs(center[1] - opp_location[1]) - abs(center[1] - own_location[1])

    return delta_moves / 16 + delta_sided / 9 + delta_center / 9


class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.max_depth_searched = 0

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # Keep track of the maximum depth searched.
        self.max_depth_searched = 0
        # Set timer function.
        self.time_left = time_left
        # Keep track of the current best move.
        legal_moves = game.get_legal_moves()
        legal_moves = remove_equal_moves(game, legal_moves)
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]
        search_depth = 1
        while True:
            try:
                # Keep track of boards that we already solved.
                self.transposition_table = {}
                # Iteratively search the tree, increasing the depth after every completed search.
                best_move = self.alphabeta(game, legal_moves, search_depth)
                self.max_depth_searched = max(self.max_depth_searched, search_depth)
                search_depth += 1
            except SearchTimeout:
                return best_move
                # print('Max depth searched {}'.format(str(search_depth)))
        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, legal_moves, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Get all legal moves for current game state.
        legal_moves = game.get_legal_moves()
        if num_moves_made(game) <= 1:
            legal_moves = remove_equal_moves(game, legal_moves)
        # Keep track of current best move and best utility.
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]
        best_utility = float('-inf')
        # Check for timeout and raise an exception to be caught by get_move() in case we timed out.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout();
        # Iteratively search the tree, increasing the depth after every completed search.
        for move in legal_moves:
            # Check for timeout and raise an exception to be caught by get_move() in case we timed out.
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout();
            # Forecast and calculate utility.
            forecast = game.forecast_move(move)
            # Calculate utility.
            utility = self._min_value(forecast, depth - 1, alpha, beta)
            # Update best found utility.
            if utility > best_utility:
                best_utility = utility
                best_move = move
            # Update upper bound.
            alpha = max(alpha, best_utility)
        return best_move

    def _min_value(self, game, depth, alpha, beta):
        # Check if this board is known.
        # game_hash = game.hash()
        # if game_hash in self.transposition_table:
        #     return self.transposition_table[game_hash]
        # Find legal moves.
        legal_moves = game.get_legal_moves()
        # If there are no moves left we reached the end of the game and return the utility.
        if len(legal_moves) == 0:
            return game.utility(self)
        # Check for timeout.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Return heuristic score.
        if depth == 0:
            return self.score(game, self)
        # Iterate over all possible moves and calculate the utility recursively.
        best_utility = float('inf')
        for move in legal_moves:
            # Calculate utility for this move and compare it with current best utility.
            forecast = game.forecast_move(move)
            # Calculate utility.
            best_utility = min(best_utility, self._max_value(forecast, depth - 1, alpha, beta))
            # Cancel if we find a value that's smaller than beta.
            if best_utility <= alpha:
                # self.transposition_table[game_hash] = best_utility
                return best_utility
            # Update alpha.
            beta = min(beta, best_utility)
        # self.transposition_table[game_hash] = best_utility
        return best_utility

    def _max_value(self, game, depth, alpha, beta):
        # Check if this board is known.
        # game_hash = game.hash()
        # if game_hash in self.transposition_table:
        #     return self.transposition_table[game_hash]
        # Find legal moves.
        legal_moves = game.get_legal_moves()
        # If there are no moves left we reached the end of the game and return the utility.
        if len(legal_moves) == 0:
            return game.utility(self)
        # Check for timeout.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Return heuristic score.
        if depth == 0:
            return self.score(game, self)
        # Iterate over all possible moves and calculate the utility recursively.
        best_utility = float('-inf')
        for move in legal_moves:
            # Calculate utility for this move and compare it with current best utility.
            forecast = game.forecast_move(move)
            # Calculate utility.
            best_utility = max(best_utility, self._min_value(forecast, depth - 1, alpha, beta))
            # Cancel if we find a value that's smaller than beta.
            if best_utility >= beta:
                # self.transposition_table[game_hash] = best_utility
                return best_utility
            # Update alpha.
            alpha = max(alpha, best_utility)
        # self.transposition_table[game_hash] = best_utility
        return best_utility
