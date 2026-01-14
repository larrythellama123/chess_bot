from piece import pieces
Piece = pieces()
class EnhancedEvaluation:
    def __init__(self):
        # Piece values (in centipawns)
        self.PAWN_VALUE = 100
        self.KNIGHT_VALUE = 300
        self.BISHOP_VALUE = 320
        self.ROOK_VALUE = 500
        self.QUEEN_VALUE = 900
        
        # Passed pawn bonuses by rank distance from promotion
        self.passed_pawn_bonuses = [0, 120, 80, 50, 30, 15, 15]
        
        # Isolated pawn penalties by count
        self.isolated_pawn_penalty_by_count = [0, -10, -25, -50, -75, -75, -75, -75, -75]
        
        # King pawn shield scores
        self.king_pawn_shield_scores = [4, 7, 4, 3, 6, 3]
        
        # Endgame material threshold
        self.endgame_material_start = self.ROOK_VALUE * 2 + self.BISHOP_VALUE + self.KNIGHT_VALUE
        
        # Piece-square tables
        self.piece_square_tables = self._initialize_piece_square_tables()

    def _initialize_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        
        # Pawn table (early game)
        pawn_table = [
            [0,   0,   0,   0,   0,   0,   0,   0],
            [50,  50,  50,  50,  50,  50,  50,  50],
            [10,  10,  20,  30,  30,  20,  10,  10],
            [5,   5,   10,  25,  25,  10,  5,   5],
            [0,   0,   0,   20,  20,  0,   0,   0],
            [5,   -5,  -10, 0,   0,   -10, -5,  5],
            [5,   10,  10,  -20, -20, 10,  10,  5],
            [0,   0,   0,   0,   0,   0,   0,   0]
        ]
        
        # Pawn table (endgame) - encourage advancement
        pawn_endgame_table = [
            [0,   0,   0,   0,   0,   0,   0,   0],
            [80,  80,  80,  80,  80,  80,  80,  80],
            [50,  50,  50,  50,  50,  50,  50,  50],
            [30,  30,  30,  30,  30,  30,  30,  30],
            [20,  20,  20,  20,  20,  20,  20,  20],
            [10,  10,  10,  10,  10,  10,  10,  10],
            [10,  10,  10,  10,  10,  10,  10,  10],
            [0,   0,   0,   0,   0,   0,   0,   0]
        ]
        
        # Knight table
        knight_table = [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20, 0,   0,   0,   0,   -20, -40],
            [-30, 0,   10,  15,  15,  10,  0,   -30],
            [-30, 5,   15,  20,  20,  15,  5,   -30],
            [-30, 0,   15,  20,  20,  15,  0,   -30],
            [-30, 5,   10,  15,  15,  10,  5,   -30],
            [-40, -20, 0,   5,   5,   0,   -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ]
        
        # Bishop table
        bishop_table = [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10, 0,   0,   0,   0,   0,   0,   -10],
            [-10, 0,   5,   10,  10,  5,   0,   -10],
            [-10, 5,   5,   10,  10,  5,   5,   -10],
            [-10, 0,   10,  10,  10,  10,  0,   -10],
            [-10, 10,  10,  10,  10,  10,  10,  -10],
            [-10, 5,   0,   0,   0,   0,   5,   -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ]
        
        # Rook table
        rook_table = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5,  10, 10, 10, 10, 10, 10, 5],
            [-5, 0,  0,  0,  0,  0,  0,  -5],
            [-5, 0,  0,  0,  0,  0,  0,  -5],
            [-5, 0,  0,  0,  0,  0,  0,  -5],
            [-5, 0,  0,  0,  0,  0,  0,  -5],
            [-5, 0,  0,  0,  0,  0,  0,  -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ]
        
        # Queen table
        queen_table = [
            [-20, -10, -10, -5,  -5,  -10, -10, -20],
            [-10, 0,   0,   0,   0,   0,   0,   -10],
            [-10, 0,   5,   5,   5,   5,   0,   -10],
            [-5,  0,   5,   5,   5,   5,   0,   -5],
            [0,   0,   5,   5,   5,   5,   0,   -5],
            [-10, 5,   5,   5,   5,   5,   0,   -10],
            [-10, 0,   5,   0,   0,   0,   0,   -10],
            [-20, -10, -10, -5,  -5,  -10, -10, -20]
        ]
        
        # King table (early game) - encourage castling
        king_early_table = [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [20,  20,  0,   0,   0,   0,   20,  20],
            [20,  30,  10,  0,   0,   10,  30,  20]
        ]
        
        # King table (endgame) - encourage centralization
        king_endgame_table = [
            [-50, -40, -30, -20, -20, -30, -40, -50],
            [-30, -20, -10, 0,   0,   -10, -20, -30],
            [-30, -10, 20,  30,  30,  20,  -10, -30],
            [-30, -10, 30,  40,  40,  30,  -10, -30],
            [-30, -10, 30,  40,  40,  30,  -10, -30],
            [-30, -10, 20,  30,  30,  20,  -10, -30],
            [-30, -30, 0,   0,   0,   0,   -30, -30],
            [-50, -30, -30, -30, -30, -30, -30, -50]
        ]
        
        return {
            'pawn': pawn_table,
            'pawn_endgame': pawn_endgame_table,
            'knight': knight_table,
            'bishop': bishop_table,
            'rook': rook_table,
            'queen': queen_table,
            'king_early': king_early_table,
            'king_endgame': king_endgame_table
        }

    def evaluate(self, game_state):
        """
        Main evaluation function - returns score from current player's perspective
        """
        # Reset evaluation data
        white_eval = EvaluationData()
        black_eval = EvaluationData()
        
        # Get material information
        white_material = self.get_material_info(game_state, Piece.white)
        black_material = self.get_material_info(game_state, Piece.black)
        
        # Material evaluation
        white_eval.material_score = white_material.material_score
        black_eval.material_score = black_material.material_score
        
        # Positional evaluation
        white_eval.piece_square_score = self.evaluate_piece_square_tables(game_state, True, black_material.endgame_t)
        black_eval.piece_square_score = self.evaluate_piece_square_tables(game_state, False, white_material.endgame_t)

        
        # Mop-up evaluation (push enemy king to edge in winning endgames)
        white_eval.mop_up_score = self.mop_up_eval(game_state, True, white_material, black_material)
        black_eval.mop_up_score = self.mop_up_eval(game_state, False, black_material, white_material)
        
        # Pawn structure evaluation
        white_eval.pawn_score = self.evaluate_pawns(game_state, Piece.white)
        black_eval.pawn_score = self.evaluate_pawns(game_state, Piece.black)
        
        # King safety evaluation
        white_eval.pawn_shield_score = self.king_pawn_shield(game_state, Piece.white, black_material)
        black_eval.pawn_shield_score = self.king_pawn_shield(game_state, Piece.black, white_material)

        white_eval.king_safety = self.evaluate_king_safety(game_state)
        black_eval.king_safety = self.evaluate_king_safety(game_state)
        
        # Calculate final evaluation
        perspective = 1 if game_state.current_color == Piece.white else -1
        eval_score = white_eval.sum() - black_eval.sum()
        
        return eval_score * perspective

    def get_material_info(self, game_state, color):
        """Calculate material information for a given color"""
        num_pawns = 0
        num_knights = 0
        num_bishops = 0
        num_rooks = 0
        num_queens = 0
        
        positions = game_state.white_positions if color == Piece.white else game_state.black_positions
        
        for row, col in positions:
            piece = game_state.board[row][col]
            if Piece.is_type(piece, Piece.pawn):
                num_pawns += 1
            elif Piece.is_type(piece, Piece.knight):
                num_knights += 1
            elif Piece.is_type(piece, Piece.bishop):
                num_bishops += 1
            elif Piece.is_type(piece, Piece.rook):
                num_rooks += 1
            elif Piece.is_type(piece, Piece.queen):
                num_queens += 1
        
        return MaterialInfo(num_pawns, num_knights, num_bishops, num_queens, num_rooks)

    def evaluate_piece_square_tables(self, game_state, is_white, endgame_t):
        """Evaluate pieces based on their positions using piece-square tables"""
        value = 0
        positions = game_state.white_positions if is_white else game_state.black_positions
        
        for row, col in positions:
            piece = game_state.board[row][col]
            
            if Piece.is_type(piece, Piece.pawn):
                # Interpolate between early and endgame pawn tables
                early_value = self.read_piece_square_table('pawn', row, col, is_white)
                late_value = self.read_piece_square_table('pawn_endgame', row, col, is_white)
                value += int(early_value * (1 - endgame_t) + late_value * endgame_t)
            elif Piece.is_type(piece, Piece.knight):
                value += self.read_piece_square_table('knight', row, col, is_white)
            elif Piece.is_type(piece, Piece.bishop):
                value += self.read_piece_square_table('bishop', row, col, is_white)
            elif Piece.is_type(piece, Piece.rook):
                value += self.read_piece_square_table('rook', row, col, is_white)
            elif Piece.is_type(piece, Piece.queen):
                value += self.read_piece_square_table('queen', row, col, is_white)
            elif Piece.is_type(piece, Piece.king):
                # Interpolate between early and endgame king tables
                early_value = self.read_piece_square_table('king_early', row, col, is_white)
                late_value = self.read_piece_square_table('king_endgame', row, col, is_white)
                value += int(early_value * (1 - endgame_t) + late_value * endgame_t)
        
        return value

    def read_piece_square_table(self, piece_type, row, col, is_white):
        """Read value from piece-square table, flipping for black pieces"""
        table = self.piece_square_tables[piece_type]
        actual_row = row if is_white else 7 - row
        return table[actual_row][col]

    def mop_up_eval(self, game_state, is_white, my_material, enemy_material):
        """Encourage king activity and enemy king edge-pushing in winning endgames"""
        if my_material.material_score > enemy_material.material_score + self.PAWN_VALUE * 2 and enemy_material.endgame_t > 0:
            mop_up_score = 0
            
            # Find kings
            my_king_pos = None
            enemy_king_pos = None
            
            my_positions = game_state.white_positions if is_white else game_state.black_positions
            enemy_positions = game_state.black_positions if is_white else game_state.white_positions
            
            for row, col in my_positions:
                if Piece.is_type(game_state.board[row][col], Piece.king):
                    my_king_pos = (row, col)
                    break
            
            for row, col in enemy_positions:
                if Piece.is_type(game_state.board[row][col], Piece.king):
                    enemy_king_pos = (row, col)
                    break
            
            if my_king_pos and enemy_king_pos:
                # Encourage moving king closer to opponent king
                king_distance = abs(my_king_pos[0] - enemy_king_pos[0]) + abs(my_king_pos[1] - enemy_king_pos[1])
                mop_up_score += (14 - king_distance) * 4
                
                # Encourage pushing opponent king to edge of board
                enemy_king_row, enemy_king_col = enemy_king_pos
                center_distance = max(abs(3.5 - enemy_king_row), abs(3.5 - enemy_king_col))
                mop_up_score += int(center_distance * 10)
                
                return int(mop_up_score * enemy_material.endgame_t)
        
        return 0

    def evaluate_pawns(self, game_state, color):
        """Evaluate pawn structure (passed pawns, isolated pawns)"""
        bonus = 0
        num_isolated_pawns = 0
        
        positions = game_state.white_positions if color == Piece.white else game_state.black_positions
        enemy_positions = game_state.black_positions if color == Piece.white else game_state.white_positions
        
        # Get pawn positions by file
        pawn_files = {}
        enemy_pawn_files = {}
        
        for row, col in positions:
            if Piece.is_type(game_state.board[row][col], Piece.pawn):
                if col not in pawn_files:
                    pawn_files[col] = []
                pawn_files[col].append(row)
        
        for row, col in enemy_positions:
            if Piece.is_type(game_state.board[row][col], Piece.pawn):
                if col not in enemy_pawn_files:
                    enemy_pawn_files[col] = []
                enemy_pawn_files[col].append(row)
        
        # Evaluate each pawn
        for file_col, pawn_rows in pawn_files.items():
            for pawn_row in pawn_rows:
                # Check for passed pawn
                is_passed = True
                is_white = color == Piece.white
                
                # Check files that can block this pawn
                for check_file in [file_col - 1, file_col, file_col + 1]:
                    if check_file in enemy_pawn_files:
                        for enemy_pawn_row in enemy_pawn_files[check_file]:
                            if is_white and enemy_pawn_row < pawn_row:
                                is_passed = False
                                break
                            elif not is_white and enemy_pawn_row > pawn_row:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                
                if is_passed:
                    num_squares_from_promotion = (7 - pawn_row) if not is_white else pawn_row
                    if num_squares_from_promotion < len(self.passed_pawn_bonuses):
                        bonus += self.passed_pawn_bonuses[num_squares_from_promotion]
                
                # Check for isolated pawn
                has_adjacent_pawns = False
                for adjacent_file in [file_col - 1, file_col + 1]:
                    if adjacent_file in pawn_files:
                        has_adjacent_pawns = True
                        break
                
                if not has_adjacent_pawns:
                    num_isolated_pawns += 1
        
        # Apply isolated pawn penalty
        if num_isolated_pawns < len(self.isolated_pawn_penalty_by_count):
            bonus += self.isolated_pawn_penalty_by_count[num_isolated_pawns]
        
        return bonus

    def king_pawn_shield(self, game_state, color, enemy_material):
        """Evaluate king safety based on pawn shield"""
        if enemy_material.endgame_t >= 1:
            return 0
        
        penalty = 0
        is_white = color == Piece.white
        
        # Find king position
        king_pos = None
        positions = game_state.white_positions if is_white else game_state.black_positions
        
        for row, col in positions:
            if Piece.is_type(game_state.board[row][col], Piece.king):
                king_pos = (row, col)
                break
        
        if not king_pos:
            return 0
        
        king_row, king_col = king_pos
        
        # Check if king is on castled position (files a-c or f-h)
        if king_col <= 2 or king_col >= 5:
            # Check pawn shield
            if is_white:
                shield_squares = [(king_row - 1, king_col - 1), (king_row - 1, king_col), (king_row - 1, king_col + 1)]
            else:
                shield_squares = [(king_row + 1, king_col - 1), (king_row + 1, king_col), (king_row + 1, king_col + 1)]
            
            for i, (shield_row, shield_col) in enumerate(shield_squares):
                if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                    piece = game_state.board[shield_row][shield_col]
                    expected_pawn = color|Piece.pawn                    
                    if piece != expected_pawn:
                        if i < len(self.king_pawn_shield_scores):
                            penalty += self.king_pawn_shield_scores[i]
            
            penalty *= penalty  # Square the penalty
        else:
            # King in center - penalty for not castling
            penalty += 40
        
        # Apply endgame weight (pawn shield less important in endgame)
        pawn_shield_weight = 1 - enemy_material.endgame_t
        
        return int(-penalty * pawn_shield_weight)
    
    def evaluate_king_safety(self,game_state):
        """Evaluate king safety based on pawn shield and piece proximity"""
        points = 0

        # Find kings
        white_king_pos = None
        black_king_pos = None

        for row, col in game_state.white_positions:
            if Piece.is_type(game_state.board[row][col], Piece.king):
                white_king_pos = (row, col)
                break

        for row, col in game_state.black_positions:
            if Piece.is_type(game_state.board[row][col], Piece.king):
                black_king_pos = (row, col)
                break

        if white_king_pos:
            points += self.evaluate_single_king_safety(white_king_pos, Piece.white,game_state)

        if black_king_pos:
            points += self.evaluate_single_king_safety(black_king_pos, Piece.black,game_state)

        return points

    def evaluate_single_king_safety(self, king_pos, color, game_state):
        """Evaluate safety for a single king"""
        king_row, king_col = king_pos
        safety_score = 0

        # Check pawn shield
        if color == Piece.white:
            # Check pawns in front of king
            shield_positions = [(king_row-1, king_col-1), (king_row-1, king_col), (king_row-1, king_col+1)]
            for shield_row, shield_col in shield_positions:
                if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                    piece = game_state.board[shield_row][shield_col]
                    if Piece.is_type(piece, Piece.pawn) and Piece.is_color(piece, Piece.white, True):
                        safety_score += 0.5
        else:
            # Check pawns in front of black king
            shield_positions = [(king_row+1, king_col-1), (king_row+1, king_col), (king_row+1, king_col+1)]
            for shield_row, shield_col in shield_positions:
                if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                    piece = game_state.board[shield_row][shield_col]
                    if Piece.is_type(piece, Piece.pawn) and Piece.is_color(piece, Piece.black, True):
                        safety_score += 0.5

        # Penalty for king in center during middle game
        if 1 <= king_row <= 5 or 1 <= king_col <= 5:
            safety_score -= 15.0

        return safety_score



class EvaluationData:
    """Container for evaluation components"""
    def __init__(self):
        self.material_score = 0
        self.mop_up_score = 0
        self.piece_square_score = 0
        self.pawn_score = 0
        self.pawn_shield_score = 0
        self.king_safety = 0
    
    def sum(self):
        return (self.material_score + self.mop_up_score + 
                self.piece_square_score + self.pawn_score + 
                self.pawn_shield_score + self.king_safety)


class MaterialInfo:
    def __init__(self, num_pawns, num_knights, num_bishops, num_queens, num_rooks):
        self.num_pawns = num_pawns
        self.num_knights = num_knights
        self.num_bishops = num_bishops
        self.num_queens = num_queens
        self.num_rooks = num_rooks
        
        # Calculate material score
        self.material_score = (num_pawns * 100 + num_knights * 300 + 
                              num_bishops * 320 + num_rooks * 500 + 
                              num_queens * 900)
        
        # Calculate endgame transition (0 = opening, 1 = endgame)
        queen_endgame_weight = 45
        rook_endgame_weight = 20
        bishop_endgame_weight = 10
        knight_endgame_weight = 10
        
        endgame_start_weight = (2 * rook_endgame_weight + 2 * bishop_endgame_weight + 
                               2 * knight_endgame_weight + queen_endgame_weight)
        
        endgame_weight_sum = (num_queens * queen_endgame_weight + 
                             num_rooks * rook_endgame_weight + 
                             num_bishops * bishop_endgame_weight + 
                             num_knights * knight_endgame_weight)
        
        self.endgame_t = 1 - min(1, endgame_weight_sum / endgame_start_weight)



