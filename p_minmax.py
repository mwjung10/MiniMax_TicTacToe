"""
Author: Katarzyna Nałęcz-Charkiewicz
Functonality author: Monika Jung
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
            move_value = self.minimax(new_board, your_side, your_side, self.depth_limit)

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

    def _opponent(self, side: str) -> str:
        return 'x' if side == 'o' else 'o'

    def minimax(self, board: Board, side: str, ai_side: str, depth=10, alpha=float('-inf'), beta=float('inf')):
        if board.who_is_winner() or not board.empty_indexes() or depth == 0:
            return self.evaluate(board, side)

        moves = board.empty_indexes()

        if side == ai_side:  # AI's turn — maximize
            for move in moves:
                new_board = board.clone()
                new_board.register_move(move)
                alpha = max(alpha, self.minimax(new_board, ai_side, self._opponent(side), depth - 1, alpha, beta))
                if beta <= alpha:
                    return beta
            return alpha
        else:  # Opponent's turn — minimize
            for move in moves:
                new_board = board.clone()
                new_board.register_move(move)
                beta = min(beta, self.minimax(new_board, ai_side, self._opponent(side), depth - 1, alpha, beta))
                if beta <= alpha:
                    return alpha
            return beta

    def evaluate(self, board: Board, side: str):
        winner = board.who_is_winner()
        if winner == side:
            return 1
        elif winner == self._opponent(side):
            return -1
        else:
            return 0

