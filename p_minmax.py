"""
Author: Katarzyna Nałęcz-Charkiewicz
"""

from board import Board
from player import Player


class MinMaxPlayer(Player):
    def __init__(self, name: str, depth_limit: int):
        super().__init__(name)
        self.depth_limit = depth_limit

    def make_move(self, board: Board, your_side: str):
        best_move = None
        best_value = float('-inf')

        for move in board.empty_indexes():
            new_board = board.clone()
            new_board.register_move(move)
            move_value = self.minimax(new_board, your_side, False, self.depth_limit - 1)

            if move_value > best_value or best_move is None:
                best_value = move_value
                best_move = move

        return best_move

    def _opponent(self, side: str) -> str:
        return 'x' if side == 'o' else 'o'

    def minimax(self, board: Board, ai_side: str, is_maximizing: bool, depth: int, alpha= float('-inf'), beta= float('inf')):
        # Sprawdzamy warunki terminalne
        winner = board.who_is_winner()
        if winner is not None or depth == 0 or not board.empty_indexes():
            return self.evaluate(board, ai_side)

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.empty_indexes():
                new_board = board.clone()
                new_board.register_move(move)
                eval = self.minimax(new_board, ai_side, False, depth - 1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    return beta
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.empty_indexes():
                new_board = board.clone()
                new_board.register_move(move)
                eval = self.minimax(new_board, ai_side, True, depth - 1, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    return alpha
            return min_eval

    def evaluate(self, board: Board, ai_side: str) -> float:
        winner = board.who_is_winner()

        if winner == ai_side:
            return 1
        elif winner == self._opponent(ai_side):
            return -1

        return 0

