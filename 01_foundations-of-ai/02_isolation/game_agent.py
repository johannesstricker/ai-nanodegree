"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def distance(a, b):
    """Calculate the squared distance between two points a and b.

    Parameters
    ----------
    a : tuple(int, int)
        A location on the game board.
    b : tuple(int, int)
        A location on the game board.

    Returns
    -------
    float
        The squared distance between a and b.
    """
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Returns the difference between the number of available moves for player and
    his opponent.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Return the difference between the number of available moves.
    # The opponents moves are scaled by two, to punish him more when he
    # has zero moves left.
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    delta_moves = own_moves - 2 * opp_moves;
    return float(delta_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Returns the difference between the player's and his opponent's distance to the
    board's center.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Get player and opponent location.
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    if own_location is None or opp_location is None:
        return 0

    # Calculate the distance between the player's positions and the board's center.
    center = (game.height / 2, game.width / 2)
    own_center_dist = distance(own_location, center)
    opp_center_dist = distance(opp_location, center)
    return opp_center_dist - own_center_dist

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Returns a score describing how much either player has been pushed onto one
    side of the board.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Get player and opponent location.
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    if own_location is None or opp_location is None:
        return 0

    # Calculate the distance between the player's positions and the board's center.
    center = (game.height / 2, game.width / 2)
    value = 0
    # Check if we are both on the same vertical side of center.
    if (own_location[0] < center[0]) == (opp_location[0] < center[0]):
        # Add the difference between the players' distances to the board's center on the vertical axis.
        value += abs(center[0] - opp_location[0]) - abs(center[0] - own_location[0])
    # Check if we are both on the same horizontal side of center.
    if (own_location[1] < center[1]) == (opp_location[1] < center[1]):
        # Add the difference between the players' distances to the board's center on the horizontal axis.
        value += abs(center[1] - opp_location[1]) - abs(center[1] - own_location[1])
    return value

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
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def _minimax_helper(self, game, depth, choice_fn, next_choice_fn):
        """ A helper function to the minimax function. This is essentially the
        minvalue and maxvalue function combined.

        Parameters
        ----------
        game : 'isolation.Board'
            An instance of 'isolation.Board' encoding the current state of the game.

        depth: integer
            The current depth in the game tree.

        choice_fn: callable
            This should be 'min' if this is called as minvalue and 'max' otherwise.

        next_choice_fn: callable
            This should be 'max' if this is called as minvalue and 'min' otherwise.

        Returns
        -------
        float
            The value of this node in the game tree.
        """

        legal_moves = game.get_legal_moves()
        # If this is a leaf node, return the game's utility.
        if len(legal_moves) == 0:
            return game.utility(self)

        # If we reached the maximum depth or timed out, return the heuristic score.
        if depth == 0 and self.time_left() >= self.TIMER_THRESHOLD:
            return self.score(game, self)

        # Initialize the utility to an extremum.
        utility = next_choice_fn(float('inf'), float('-inf'))
        for move in legal_moves:
            # If we timed out, return current best utility.
            if self.time_left() < self.TIMER_THRESHOLD:
                return utility
            forecast = game.forecast_move(move)
            # Traverse down the tree to calculate utility for this node.
            # choice_fn and next_choice_fn should be swapped here.
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
        # Check if we timed out.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Initialize the current best move.
        legal_moves = game.get_legal_moves()
        best_move = (-1, -1) if len(legal_moves) == 0 else legal_moves[0]
        # Initialize the max utility to an extremum.
        max_utility = float('-inf')
        for move in legal_moves:
            forecast = game.forecast_move(move)
            # Calculate the utility by traversing down the tree.
            utility = self._minimax_helper(forecast, depth - 1, min, max)
            # Update best utility.
            if utility > max_utility:
                best_move = move
                max_utility = utility
            # Check if we timed out.
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

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
        # Initialize the search depth to level 1.
        search_depth = 1
        while True:
            # Use a try..catch.. block to catch the Timeout exceptions.
            try:
                # Iteratively search the tree, increasing the depth after every completed search (iterative deepening).
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1
            except SearchTimeout:
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
