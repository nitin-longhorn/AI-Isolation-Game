"""This file contains all the classes you must complete for this project.
You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.
You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """ This heuristic takes into account the following factors
    1. How much of the game has been played each time the score is being computed
    2. Who the active player is
    3. Deducting points for corner positions for the agent
    4. Adding points to the agent for opponent being in corner positions
    5. Deducting more points for agent occupying corner positions later in the game
    6. Adding more points for opponent occupying corner positions later in the game

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

    # Get the total number of Moves available for the agent
    my_legal_moves = len(game.get_legal_moves(player))
    if player == game._player_1:
        opponent_player = game._player_2
    else:
        opponent_player = game._player_1
    # Get the total number of Moves available for the Opponent
    opponent_legal_moves = len(game.get_legal_moves(opponent_player))

    # Initial Calculaton is based on the difference between the # of moves between the agent and the player
    calculated_score = my_legal_moves - opponent_legal_moves

    # Penalty determination depending on stage of game
    start_game_penalty = .1
    mid_game_penalty = .4
    near_end_game_penalty = .75
    end_game_penalty = .9

    total_available_space = game.width * game.height
    current_state_of_game = len(game.get_blank_spaces())
    game_level = current_state_of_game/total_available_space

    # Initial Penalty
    penalty = 0

    # Assigning Penalties depending on game stage
    if game_level <=.25:
        penalty = start_game_penalty

    elif game_level  > .25 and game_level <= .4:
        penalty = mid_game_penalty

    elif game_level > .4 and game_level <= .7:
        penalty = near_end_game_penalty

    elif game_level > .7:
        penalty = end_game_penalty

    # Getting current position of the agent and the opponent
    my_position = game.get_player_location(player)
    opponent_position = game.get_player_location(opponent_player)

    # Corner coordinates
    corners = [(0,0), (0,(game.width -1)), (game.height-1,0), ((game.height-1), (game.width-1))]

    #print ("Penalty is ", penalty)
    # Rewarding or Penalizing Scores for occupying corner positions
    if my_position in corners:
        calculated_score = calculated_score - (2*penalty * calculated_score)
    if opponent_position in corners:
        calculated_score = calculated_score + (2*penalty * calculated_score)

    #print ("Penalty is ", penalty)

    return float(calculated_score)


def custom_score_1(game, player):
    # Reward agent for being close to center and if opponent is further away from center

    # Identify player and opponent
    if player == game._player_1:
        opponent_player = game._player_2
    else:
        opponent_player = game._player_1

    # Agent Location = x1,y1
    x1,y1 = game.get_player_location(player)
    # Opponent Location = x3,y3
    x3,y3 = game.get_player_location(opponent_player)

    #Center Location = x2,y2
    x2,y2 = (game.height - 1)/2, (game.width -1)/2

    #How far is agent from center
    distance_1 = math.sqrt(((x1-x2)^2) - ((y1-y2)^2))
    distance_2 = math.sqrt(((x3-x2)^2) - ((y3-y2)^2))

    calculated_score = distance_2 - distance_1

    return float(calculated_score)


def custom_score_2(game, player):
    # Reward agent for being far away from opponent and for opponent being close to corner

    # Get the total number of Moves available for the agent
    if player == game._player_1:
        opponent_player = game._player_2
    else:
        opponent_player = game._player_1

    x1, y1 = game.get_player_location(player)
    x2, y2 = game.get_player_location(opponent_player)

    calculated_score = math.sqrt(((x1-x2)^2) - ((y1-y2)^ 2))

    # Corner coordinates
    corners = [(0, 0), (0, (game.width - 1)), (game.height - 1, 0), ((game.height - 1), (game.width - 1))]

    if (x2,y2) in corners:
        calculated_score = calculated_score * 2

    return float(calculated_score)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.
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
        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release
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

        # If there are no legal moves, return (-1,-1)
        if len(legal_moves) == 0:
            return (-1, -1)

        # Initialize best move to default None
        best_move = None

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            #Check if the call asks for iterative deepening
            if self.iterative == True:
                i = 0
                # Check if the call specifies for minimax or alphabeta
                if self.method == "minimax":
                    # Keep searching for best score until minimax or alphabeta times-out
                    while (i < float('inf')):
                        score, best_move = self.minimax(game, i)
                        i = i + 1
                elif self.method == "alphabeta":
                    while (i < float('inf')):
                        score, best_move = self.alphabeta(game, i)
                        i = i + 1
            else:
                # Do minimax or alphabeta just once
                if self.method == 'minimax':
                    score, best_move = self.minimax(game, self.search_depth)
                elif self.method =='alphabeta':
                    score, best_move = self.alphabeta(game, self.search_depth)
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass
        return best_move


    # Return the best move from the last completed search iteration


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()


        # At final depth, return the score computed by the evaluation function that is
        # written in the custom_score function
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # Get a list of initial moves available for the active player
        moves = game.get_legal_moves()

        # Check to see if there are any legal moves left for the active player;
        # if not return (-1, -1) for no legal moves
        if len(moves) == 0:
            return float("-inf"), (-1, -1)

        best_move = None
        # Propagating the evaluation score up the tree using recursion
        # Determine if the current node is maximizing or minimizing node
        if (maximizing_player == True):
            # For maximizing player, the node with the highest score should be taken
            # Assigning an unrealistic negative score to begin with
            max_score = float("-inf")

            for move in moves:
                #For every possible legal move analyse the impact of the move
                forecast_game = game.forecast_move(move)
                # Find out the new_score by recursing
                new_score, new_move = self.minimax(forecast_game, depth - 1, False)
                #print (new_score, max_score)
                # Find and return the move for the maximum score
                if new_score > max_score:
                    max_score = new_score
                    best_move = move
            return max_score, best_move

        else:
            # For minimizing player, the node with the lowest score should be taken
            # Assigning an unrealistic high scores to begin with
            min_score = float("inf")

            for move in moves:
                #For every possible legal move analyse the impact of the move
                forecast_game = game.forecast_move(move)
                # Find out the new_score by recursing
                new_score, new_move = self.minimax(forecast_game, depth - 1, True)
                #print (new_score, min_score)
                # Find and return the move for the min score
                if new_score < min_score:
                    min_score = new_score
                    best_move = move
            return min_score, best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # At final depth, return the score computed by the evaluation function that is
        # # written in the custom_score function
        if depth == 0:
            # (-1, -1) for no legal moves
            return self.score(game, self), (-1, -1)

        # Get a list of initial moves available for the active player
        moves = game.get_legal_moves()
        legal_moves = moves

        # Check to see if there are any legal moves left for the active player;
        # if not return (-1, -1) for no legal moves
        if len(moves) == 0:
            return game.utility(self), (-1, -1)

        max_move = None
        min_move = None
        # Propagating the evaluation score up the tree using recursion
        # Determine if the current node is maximizing or minimizing node
        if (maximizing_player == True):
            # For maximizing player, the node with the highest score should be taken
            # Assigning an unrealistic negative score to begin with
            max_score = float("-inf")
            for move in moves:
                #For every possible legal move analyse the impact of the move
                forecast_game = game.forecast_move(move)
                new_score, new_move = self.alphabeta(forecast_game, depth - 1, alpha, beta, False)
                if new_score > max_score:
                    max_score = new_score
                    max_move = move
                # Alpha Beta pruning - Compare max score to beta;
                # Compare min_score to alpha
                if max_score >= beta:
                    return max_score, max_move
                alpha = max(alpha, max_score)
            return max_score, max_move
        else:
            # For minimizing player, the node with the lowest score should be taken
            # Assigning an unrealistic high scores to begin with
            min_score = float("inf")
            for move in moves:
                # Find out the new_score by recursing
                forecast_game = game.forecast_move(move)
                new_score, new_move = self.alphabeta(forecast_game, depth - 1, alpha, beta, True)
                #print (new_score, min_score)
                # Find and return the move for the min score
                if new_score < min_score:
                    min_score = new_score
                    min_move = move
                # Alpha Beta pruning - Compare min score to alpha;
                # Compare min_score to beta
                if min_score <= alpha:
                    return min_score, min_move
                beta = min(beta, min_score)
            return min_score, min_move


# References -
#https://www.youtube.com/watch?v=xBXHtz4Gbdo
#https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/lecture-6-search-games-minimax-and-alpha-beta/
#https://github.com/mksmsrkn/#https://github.com/davidventuri/
#https://inst.eecs.berkeley.edu/~cs61a/su12/lec/notes/data.html
#https://tonypoer.io/2016/10/08/recursively-parsing-a-list-of-lists-into-a-game-tree-using-python/
#https://tonypoer.io/2016/10/28/implementing-minimax-and-alpha-beta-pruning-using-python/