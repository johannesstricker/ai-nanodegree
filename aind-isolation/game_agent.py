"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def reflect_board_vertical(game):
    reflection = game.copy()

    if reflection._board_state[-1] is not None:
        row = reflection._board_state[-1] % game.height
        col = reflection._board_state[-1] // game.height
        col = game.width - col - 1
        idx = row + col * game.height
        reflection._board_state[-1] = idx

    if reflection._board_state[-2] is not None:
        row = reflection._board_state[-2] % game.height
        col = reflection._board_state[-2] // game.height
        col = game.width - col - 1
        idx = row + col * game.height
        reflection._board_state[-2] = idx

    for r in range(game.height):
        for c in range(game.width):
            idx = r + c * game.height
            newIdx = r + (game.width - c - 1) * game.height
            reflection._board_state[newIdx] = game._board_state[idx]
    return reflection

def reflect_board_horizontal(game):
    reflection = game.copy()

    if reflection._board_state[-1] is not None:
        row = reflection._board_state[-1] % game.height
        col = reflection._board_state[-1] // game.height
        row = game.height - row - 1
        idx = row + col * game.height
        reflection._board_state[-1] = idx

    if reflection._board_state[-2] is not None:
        row = reflection._board_state[-2] % game.height
        col = reflection._board_state[-2] // game.height
        row = game.height - row - 1
        idx = row + col * game.height
        reflection._board_state[-2] = idx

    for r in range(game.height):
        for c in range(game.width):
            idx = r + c * game.height
            newIdx = (game.height - r - 1) + c * game.height
            reflection._board_state[newIdx] = game._board_state[idx]
    return reflection

def reflect_idx(idx, width, height, vertical, horizontal):
    if idx is None:
        return None
    row = idx % height
    col = idx // height
    row = height - row - 1 if horizontal else row
    col = width - col - 1 if vertical else col
    return row + col * height

def hash_board_state(game):
    return hash(tuple(game._board_state))

def hash_board(game):
    horizontal = reflect_board_horizontal(game)
    vertical = reflect_board_vertical(game)
    both = reflect_board_vertical(horizontal)
    return min([hash_board_state(game), hash_board_state(vertical), hash_board_state(horizontal), hash_board_state(both)])

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
    # Check if this is the first move to make.
    if len(moves) == len(game._board_state) - 3:
        return list(map(lambda x: (x[0] + game.height/2, x[1] + game.width/2), set(map(lambda x: (abs(x[0] - game.height/2), abs(x[1]-game.width/2)), moves))))
    forecasts = [game.forecast_move(move) for move in moves]
    for idx_1 in range(1, len(moves)):
        for idx_2 in range(0, idx_1):
            if compare_games(forecasts[idx_1], forecasts[idx_2]):
                moves[idx_1] = None
    return [move for move in moves if move is not None]

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # Punish/reward losing/winning indefinitely.
    player_legal_moves = game.get_legal_moves(player)
    if len(player_legal_moves) == 0:
        return float('-inf')
    opponent_legal_moves = game.get_legal_moves(game.get_opponent(player))
    if len(opponent_legal_moves) == 0:
        return float('inf')
    # Take normalized error.
    open_fields = sum(game._board_state[0:-3])
    score = (len(player_legal_moves) / open_fields) - (len(opponent_legal_moves) / open_fields)
    return score * distanceToCenter(game, player)

def distanceToCenter(game, player):
    position = game.get_player_location(player)
    delta_row = abs(position[0] - game.height / 2)
    delta_col = abs(position[1] - game.width / 2)
    return delta_row + delta_col

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # Try to stay as close the the center as possible.
    return -(distanceToCenter(game,player) ** 2)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

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
    # Punish/reward losing/winning indefinitely.
    player_legal_moves = game.get_legal_moves(player)
    if len(player_legal_moves) == 0:
        return float('-inf')
    opponent_legal_moves = game.get_legal_moves(game.get_opponent(player))
    if len(opponent_legal_moves) == 0:
        return float('inf')
    # Take squared error.
    score = float(len(player_legal_moves) - len(opponent_legal_moves))
    if score < 0:
        return -(score ** 2)
    return score ** 2


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

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
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _minimax_helper(self, game, depth, choice_fn, next_choice_fn):
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return game.utility(self)

        if depth == 0 and self.time_left() >= self.TIMER_THRESHOLD:
            return self.score(game, self)

        utility = next_choice_fn(float('inf'), float('-inf'))
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                return utility
            forecast = game.forecast_move(move)
            utility = choice_fn(utility, self._minimax_helper(forecast, depth - 1, next_choice_fn, choice_fn))
        return utility

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

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
        # TODO: finish this function!
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]
        max_utility = float('-inf')
        for move in legal_moves:
            forecast = game.forecast_move(move)
            utility = self._minimax_helper(forecast, depth - 1, min, max)
            if utility > max_utility:
                best_move = move
                max_utility = utility
            if self.time_left() < self.TIMER_THRESHOLD:
                break

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
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
        # Set timer function.
        self.time_left = time_left
        # Keep track of the current best move.
        legal_moves = game.get_legal_moves()
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]
        search_depth = 1
        while True:
            try:
                # Iteratively search the tree, increasing the depth after every completed search.
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1
            except SearchTimeout:
                # print('Max depth searched {}'.format(str(search_depth)))
                return best_move
        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
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
                return best_utility
            # Update alpha.
            beta = min(beta, best_utility)
        return best_utility

    def _max_value(self, game, depth, alpha, beta):
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
                return best_utility
            # Update alpha.
            alpha = max(alpha, best_utility)
        return best_utility
