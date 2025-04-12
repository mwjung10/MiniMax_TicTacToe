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
        # TODO
        return None

    def minimax(self, board: Board, side: str, depth: int, alpha=float('-inf'), beta=float('inf')):
        if board.who_is_winner() or not board.empty_indexes() or depth == 0:
            return self.evaluate(board, side)

        moves = board.empty_indexes()

        if side == board.player_chars[0]:  # Circle's turn
            for move in moves:
                new_board = board.clone()
                new_board.register_move(move)
                alpha = max(alpha, self.minimax(new_board, board.player_chars[1], depth - 1, alpha, beta))
                if alpha >= beta:
                    return beta
            return alpha
        else:  # Cross's turn
            for move in moves:
                new_board = board.clone()
                new_board.register_move(move)
                beta = min(beta, self.minimax(new_board, board.player_chars[0], depth - 1, alpha, beta))
                if beta <= alpha:
                    return alpha
            return beta

    def evaluate(self, board: Board, side: str):
        # TODO
        return 0
