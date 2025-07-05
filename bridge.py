import chess
import numpy as np
import torch
from piece import pieces  # Your existing piece module
from move import Move

Piece = pieces()

class GameStateConverter:
    """Converts between your custom gamestate and standard chess representations"""
    
    def __init__(self):
        # Mapping from your piece system to standard chess
        self.piece_type_map = {
            Piece.pawn: chess.PAWN,
            Piece.knight: chess.KNIGHT,
            Piece.bishop: chess.BISHOP,
            Piece.rook: chess.ROOK,
            Piece.queen: chess.QUEEN,
            Piece.king: chess.KING
        }
        
        # Reverse mapping
        self.reverse_piece_map = {v: k for k, v in self.piece_type_map.items()}


    def gamestate_to_chess_board(self, game_state):
        board = chess.Board()  # Empty board
        board.clear()
        
        # Convert your board to standard format
        for row in range(8):
            for col in range(8):
                piece_value = game_state.board[row][col]
                
                if piece_value != 0:  # If there's a piece
                    # Extract piece type and color from your encoding
                    piece_type = self._extract_piece_type(piece_value)
                    color = self._extract_color(piece_value)
                    
                    if piece_type in self.piece_type_map:
                        chess_piece_type = self.piece_type_map[piece_type]
                        chess_color = chess.WHITE if color == Piece.white else chess.BLACK
                        
                        # Convert row,col to chess square (a1=0, h8=63)
                        square = row * 8 + col
                        
                        # Create and place piece
                        chess_piece = chess.Piece(chess_piece_type, chess_color)
                        board.set_piece_at(square, chess_piece)
        
        # Set game state properties
        board.turn = chess.WHITE if game_state.current_color == Piece.white else chess.BLACK
        
        return board
    
    
    def get_legal_moves_uci(self, game_state):
        legal_moves = []
        
        for move in game_state.final_allowed_moves:
            # Convert to UCI notation
            start_uci = self.coords_to_uci(move.start_square)
            target_uci = self.coords_to_uci(move.target_square)
            
            # Create UCI move string
            uci_move = start_uci + target_uci
            
            # Create chess.Move object from UCI
            chess_move = chess.Move.from_uci(uci_move)
            legal_moves.append(chess_move)
        
        return legal_moves

    def coords_to_uci(self, coords):
        """Convert (row, col) coordinates to UCI square notation"""
        row, col = coords
        
        # Convert column to file letter (0->a, 1->b, ..., 7->h)
        file = chr(ord('a') + col)
        
        # Convert row to rank number
        # Adjust this based on your coordinate system:
        rank = str(row + 1)  # if row 0 = rank 1
        # rank = str(8 - row)  # if row 0 = rank 8
        
        return file + rank
    
    def uci_to_coords(self,move_uci):
        start_square = move_uci[:2]
        target_square = move_uci[2:]

        move = Move()
        move.start_square = (ord([start_square[0]])-1,start_square[1])
        move.target_square = (ord([target_square[0]])-1,target_square[1])
        return move

    
    def _extract_piece_type(self, piece_value):
        """Extract piece type from your encoding"""
        # You'll need to implement this based on your piece encoding
        # This assumes you use bitwise operations like: piece_value & Piece.type_mask
        
        if Piece.is_type(piece_value, Piece.pawn):
            return Piece.pawn
        elif Piece.is_type(piece_value, Piece.knight):
            return Piece.knight
        elif Piece.is_type(piece_value, Piece.bishop):
            return Piece.bishop
        elif Piece.is_type(piece_value, Piece.rook):
            return Piece.rook
        elif Piece.is_type(piece_value, Piece.queen):
            return Piece.queen
        elif Piece.is_type(piece_value, Piece.king):
            return Piece.king
        
        return None
    
    def _extract_color(self, piece_value):
        """Extract color from your encoding"""
        # You'll need to implement this based on your piece encoding
        if Piece.is_color(piece_value, Piece.white, True):
            return Piece.white
        else:
            return Piece.black
    
# # Usage example
# def integrate_with_your_system():
#     """Example of how to integrate with your existing system"""
    
#     # 3. Use the trained model with your gamestate
#     def evaluate_your_position(game_state):
#         """Evaluate your custom gamestate"""
#         return model.evaluate_gamestate(game_state)
    
#     # 4. Integrate with your existing evaluation
#     def enhanced_evaluate(game_state):
#         """Combine neural network with your hand-crafted evaluation"""
        

#         # Neural network evaluation
#         nn_eval = model.evaluate_gamestate(game_state) * 1000  # Scale to centipawns
   
#         return nn_eval
    
#     return evaluate_your_position, enhanced_evaluate

# # What you need to add to your gamestate class
# class EnhancedGameState:
#     """Your gamestate with additional neural network support"""
    
#     def __init__(self):
#         # Your existing gamestate properties
#         self.board = None
#         self.current_color = None
#         self.white_positions = []
#         self.black_positions = []
        
#         # Add these for neural network compatibility
#         self.white_can_castle_kingside = True
#         self.white_can_castle_queenside = True
#         self.black_can_castle_kingside = True
#         self.black_can_castle_queenside = True
#         self.en_passant_square = None
#         self.halfmove_clock = 0
#         self.fullmove_number = 1
        
#         # Neural network converter
#         self.converter = GameStateConverter()
    
#     def to_neural_input(self):
#         """Convert this gamestate to neural network input"""
#         return self.converter.gamestate_to_tensor(self)
    
#     def evaluate_with_nn(self, neural_model):
#         """Evaluate this position using neural network"""
#         return neural_model.evaluate_gamestate(self)