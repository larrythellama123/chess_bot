#keep track of attacked squares
#keep track of the pinned pieces
#keep track of the checked paths
#store 2 attack sqaure paths
#pinned paths is also remade on each turn



#self.attack_squares is a list generate on each new turn containing the list of the legal moves from one side

#to check if a piece can end a pin
    #check if the piece is not one of the pieces involved in another pinned_path
    #and its not the king

    #so when filtering moves, check if start piece is from a pinned_path

#to check whether a move can end a check
    #check if the move breaks the path of check and its not from a pinned piece
    # check if the king can move anywhere that is not on the attack square of the opposite side    


#generate all moves - returns the start squares and target squares
#generate legal moves - will filter the illegal stuff out 
    #if there are checks or pinned pieces
    #this will check if pinned ones move will result in a discovered check
    #check if a move ends the check
    #if no move can end the check, its checkmate


#generate the attack squares of the opp side every round

from piece import pieces
from move import Move
from square import squares
from evaluation import EnhancedEvaluation
import copy
Square = squares()
Piece = pieces()
class GameState:

    def __init__(self):
        self.fen_string =  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.board = [[0 for j in range(8)] for i in range(8)]
        #experimental
        self.final_allowed_moves = [] 
        self.total_moves = {}
        self.attack_squares = {}
        self.temp_attack_squares = {}
        self.castle_moves = [] 
        #will take in the sqaures involved in the pinning
        self.pinned_piece_paths=[]
        self.checked_path = []
        self.captures_moves_only = []
        

        self.black_positions = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7)]
        self.white_positions = [(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]

        self.translate_fen_strings()
        # self.distance_to_edge = {'n':0,'s':0,'e':0,'w':0, 'ne':0, 'nw':0, 'se':0, 'sw':0}
        self.distance_to_edge_white = {}
        self.distance_to_edge_black = {}

        self.knight_offsets = {'te':(2,1),'tw':(2,-1),'be':(-2,1),'bw':(-2,-1),'htw':(1,2),'hte':(1,-2),'hbe':(-1,2),'hbw':(-1,-2)}
        self.knight_positions = {'LWK':(7,1),'RWK':(7,6),'LBK':(0,1),'RBK':(0,6)}
        self.knight_moves_dict = {}

        self.defended_squares = []
        self.attacker_defended_squares = []

        
        self.current_color = Piece.white
        self.attack_color = Piece.black
        self.human_player = Piece.white
        self.AI_player = Piece.black

        self.WrookRMove = False
        self.WrookLMove = False
        self.BrookLMove = False
        self.BrookRMove =False
        self.WkingMove = False
        self.BkingMove = False

        self.castle_flags ={'WrookRMove':False,'WrookLMove':False,'BrookLMove':False,'BrookRMove':False,'WkingMove':False,'BkingMove':False}

        self.points = 0
        self.best_move = None
        self.changed_pawn = None
        self.visited = []
        self.initial_depth = 0
        #n,s,e,w,ne,nw,se,sw
        #    N
        # W     E
        #    S
        self.direction_offset = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

        #####for minmax#####
        self.Pawn = 1
        self.Knight = 3
        self.Bishop = 3
        self.Rook = 5
        self.Queen = 9
        self.points = 0


    def translate_fen_strings(self):
        col = 0
        row = 0 
        for i in range(len(self.fen_string)):
            if self.fen_string[i] == '8':
                for column in range(8):
                    self.board[row][column] = 0
                continue
            if self.fen_string[i] == '/':
                row += 1
                col = 0
                continue 
            if self.fen_string[i].lower() == 'r':
                piece_type = Piece.rook
            if self.fen_string[i].lower() == 'n':
                piece_type = Piece.knight
            if self.fen_string[i].lower() == 'b':
                piece_type = Piece.bishop
            if self.fen_string[i].lower() == 'q':
                piece_type = Piece.queen
            if self.fen_string[i].lower() == 'k':
                piece_type = Piece.king
            if self.fen_string[i].lower() == 'p':
                piece_type = Piece.pawn

            if self.fen_string[i].isupper():
                piece_color = Piece.white
            else:
                piece_color = Piece.black
            self.board[row][col] = piece_color|piece_type
            col += 1

        return self.board

    #find the num of squares to the edge of the board in every directionm then during move generatioin, just 
    #input the start quare in and the distances to the edgss can all be found 
    
    #gets the distance to the edges of the board
    def get_distance_to_edge_for_black(self,start_square_row, start_square_col):
        self.distance_to_edge_black[(start_square_row,start_square_col)] = {'n':0,'s':0,'e':0,'w':0, 'ne':0, 'nw':0, 'se':0, 'sw':0}
        self.distance_to_edge_black[(start_square_row,start_square_col)]['n'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['s']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['s'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['n']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['w'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['e']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['e'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['w']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['sw'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['ne']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['se'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['nw']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['nw'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['se']
        self.distance_to_edge_black[(start_square_row,start_square_col)]['ne'] = self.distance_to_edge_white[(start_square_row,start_square_col)]['sw']

    def get_distance_to_edge_for_white(self, start_square_row,start_square_col):
            self.distance_to_edge_white[(start_square_row,start_square_col)] = {'n':0,'s':0,'e':0,'w':0, 'ne':0, 'nw':0, 'se':0, 'sw':0}

            for direction_index in list(self.distance_to_edge_white[(start_square_row,start_square_col)].keys()):
                square_row = start_square_row
                square_col = start_square_col
                count = 0
                if direction_index == 'n':
                    while square_row>0:
                        square_row -= 1
                        count += 1
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['n'] = count

                if direction_index == 's':
                    while square_row<7:
                        square_row += 1
                        count += 1
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['s'] = count

                if direction_index == 'e':
                    while square_col<7:
                        square_col += 1
                        count += 1
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['e'] = count 

                if direction_index == 'w':
                    while square_col>0:
                        square_col -= 1
                        count += 1
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['w'] = count 

                if direction_index == 'ne':
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['ne'] = min(self.distance_to_edge_white[(start_square_row,start_square_col)]['n'],self.distance_to_edge_white[(start_square_row,start_square_col)]['e'])

                if direction_index == 'nw':
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['nw'] = min(self.distance_to_edge_white[(start_square_row,start_square_col)]['n'],self.distance_to_edge_white[(start_square_row,start_square_col)]['w'])

                if direction_index == 'se':
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['se'] = min(self.distance_to_edge_white[(start_square_row,start_square_col)]['s'],self.distance_to_edge_white[(start_square_row,start_square_col)]['e'])

                if direction_index == 'sw':
                    self.distance_to_edge_white[(start_square_row,start_square_col)]['sw'] = min(self.distance_to_edge_white[(start_square_row,start_square_col)]['s'],self.distance_to_edge_white[(start_square_row,start_square_col)]['w'])

    def change_current_color(self):
        if self.current_color == Piece.black:
            self.current_color = Piece.white
        else:
            self.current_color = Piece.black



    def initial_setup(self):
        for row in range(8):
            for col in range(8):
                self.get_distance_to_edge_for_white(row,col)
                self.get_distance_to_edge_for_black(row,col)



    #start of each round, clear the old keys, then iterate through the squares list to regenerate attack squares of the opp
    #then do move generation for current player
    def start_new_round(self,captures_only=False):
        self.total_moves = {}
        self.attack_squares = {}
        self.pinned_piece_paths = []
        self.checked_path = []
        self.defended_squares = []
        if self.current_color == Piece.white:
            self.attack_color = Piece.black
            self.current_color = Piece.black
            for position in self.black_positions:
                start_square_row, start_square_col = position
                self.move_generate(start_square_row,start_square_col, self.attack_squares)
            self.attacker_defended_squares = copy.copy(self.defended_squares)
            self.current_color = Piece.white
            for position in self.white_positions:
                # print("positions:",position)
                start_square_row, start_square_col = position
                self.move_generate(start_square_row,start_square_col, self.total_moves, is_current_player=True)

            # if(self.no_moves):
            #     # print("COOKED")
        else:
            self.attack_color = Piece.white
            self.current_color = Piece.white
            for position in self.white_positions:
                start_square_row, start_square_col = position
                self.move_generate(start_square_row,start_square_col, self.attack_squares)
            self.attacker_defended_squares = copy.copy(self.defended_squares)
            self.current_color = Piece.black
            for position in self.black_positions:
                start_square_row, start_square_col = position
                self.move_generate(start_square_row,start_square_col, self.total_moves,is_current_player=True)



    def start_new_round_minmax(self):
        self.total_moves = {}
        self.attack_squares = {}
        self.pinned_piece_paths = []
        self.checked_path = []
        self.defended_squares = []
        list_to_generate_moves_after_attackers_completed = []
        if self.current_color == Piece.white:
            self.attack_color = Piece.black
            self.current_color = Piece.black
            for row in range(8):
                for col in range(8):
                    if self.board[row][col] !=0:
                        if Piece.is_color(self.board[row][col],Piece.black,True): 
                            self.current_color = Piece.black
                            self.move_generate(row,col, self.attack_squares)
                        else:
                            list_to_generate_moves_after_attackers_completed.append((row,col))
                            # self.current_color = Piece.white
                            # self.move_generate(row,col, self.total_moves, is_current_player=True)
            self.attacker_defended_squares = copy.copy(self.defended_squares)                
            self.current_color = Piece.white
            for square in list_to_generate_moves_after_attackers_completed:
                row,col = square
                self.move_generate(row,col, self.total_moves, is_current_player=True)    
        else:
            self.attack_color = Piece.white
            self.current_color = Piece.white
            for row in range(8):
                for col in range(8):
                    if self.board[row][col] !=0:
                        if Piece.is_color(self.board[row][col],Piece.white,True): 
                            self.current_color = Piece.white
                            self.move_generate(row,col, self.total_moves, is_current_player=True)
                        else:
                            list_to_generate_moves_after_attackers_completed.append((row,col))
                            # self.current_color = Piece.black
                            # self.move_generate(row,col, self.total_moves, is_current_player=True)
            self.attacker_defended_squares = copy.copy(self.defended_squares)
            self.current_color = Piece.black
            for square in list_to_generate_moves_after_attackers_completed:
                row,col = square
                self.move_generate(row,col, self.total_moves, is_current_player=True)    



    def move_generate(self,start_square_row, start_square_col, square_dict, is_current_player=False, captures_only=False):  

        color = Piece.get_piece_color(self.board[start_square_row][start_square_col])  
        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.bishop):
            #generate attack sqaures for the config of opp pieces first            

            self.generate_sliding_moves(start_square_row, start_square_col,Piece.bishop,color, square_dict)
        
        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.rook):
            self.generate_sliding_moves(start_square_row, start_square_col,Piece.rook,color, square_dict)

        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.queen):
            self.generate_sliding_moves(start_square_row, start_square_col,Piece.queen,color,  square_dict)

        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.king):
            self.generate_king_moves(start_square_row, start_square_col,  square_dict)

        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.pawn):
            self.generate_pawn_moves(start_square_row, start_square_col, square_dict)

        if Piece.is_type(self.board[start_square_row][start_square_col],Piece.knight):
            self.generate_knight_moves(start_square_row,start_square_col,self.knight_moves_dict)
            self.add_knight_squares(square_dict, start_square_row, start_square_col)


        if is_current_player:
            self.build_castle_moves()






    def generate_sliding_moves(self,start_square_row, start_square_col,piece_type,piece_color,square_dict):
        distance_to_edge = {}
        if piece_color == Piece.black:
            distance_to_edge  = self.distance_to_edge_black
        else:
            distance_to_edge = self.distance_to_edge_white

        limit = list(distance_to_edge[(start_square_row,start_square_col)].keys())
        if piece_type == Piece.rook:
            limit = limit[:4]
        if piece_type == Piece.bishop:
            limit = limit[4:8]


        # break_flag  = False
        square_dict[(start_square_row,start_square_col)] = []
        for distance_index in limit:
            # if break_flag == True:
            #     break_flag = False
            #     continue
            moves = []
            pinned_piece = False
            # print(f"distance index: {distance_index}")
            # print(f"length:{distance_to_edge[(start_square_row,start_square_col)][distance_index]}")
            # print(f"start_sq_row:{start_square_row}, start_sq_col:{start_square_col}")
            square_row = start_square_row
            square_col = start_square_col
            is_extended_already = False

            for i in range(distance_to_edge[(start_square_row,start_square_col)][distance_index]):
               
                #iterate respective direction
                if piece_color == Piece.black:
                    if distance_index == 'n':
                        square_row += 1

                    if distance_index == 's':
                        square_row -= 1

                    if distance_index == 'e':
                        square_col -= 1

                    if distance_index == 'w':
                        square_col += 1

                    if distance_index == 'ne':
                        square_row += 1
                        square_col -= 1

                    if distance_index == 'se':
                        square_row -= 1
                        square_col -= 1

                    if distance_index == 'nw':
                        square_row += 1
                        square_col += 1

                    if distance_index == 'sw':
                        square_row -= 1
                        square_col += 1
                else:

                    if distance_index == 'n':
                        square_row -= 1

                    if distance_index == 's':
                        square_row += 1

                    if distance_index == 'e':
                        square_col += 1

                    if distance_index == 'w':
                        square_col -= 1

                    if distance_index == 'ne':
                        square_row-= 1
                        square_col += 1

                    if distance_index == 'se':
                        square_row += 1
                        square_col += 1

                    if distance_index == 'nw':
                        square_row -= 1
                        square_col -= 1

                    if distance_index == 'sw':
                        square_row += 1
                        square_col -= 1

                if square_col > 7 or square_row < 0 or  square_row >7 or square_col < 0:
                    square_dict[(start_square_row,start_square_col)].extend(moves)
                    is_extended_already = True
                    # for move in square_dict[(start_square_row,start_square_col)]:
                    #     print("move start",move.start_square)
                    #     print("move target", move.target_square)
                    # break_flag = True    
                    break

                if (square_row,square_col) != (start_square_row,start_square_col):
                    if Piece.is_color(self.board[square_row][square_col],piece_color,True):
                        #only add if pinned piece path is not already added(see below)
                        if not pinned_piece:
                            square_dict[(start_square_row,start_square_col)].extend(moves)
                            is_extended_already = True
                            self.defended_squares.append((square_row,square_col))
                        break

                move = Move()
                move.start_square = (start_square_row, start_square_col)
                move.target_square = (square_row, square_col)
                moves.append(move)

                if (square_row,square_col) != (start_square_row,start_square_col):
                    if Piece.is_color(self.board[square_row][square_col],piece_color,False):
                        if Piece.is_type(self.board[square_row][square_col],Piece.king):
                            if pinned_piece:
                                #if true there is another of piece of opoosing color blocking the opposing king
                                #add the full path (to the king) to the pinned paths 
                                # self.pinned_piece_paths.extend(square_dict[(start_square_row,start_square_col)])
                                self.pinned_piece_paths.extend(moves)
                            else:
                                self.checked_path.extend(moves)
                        else:
                            #there is an opposing color in this direction
                            if is_extended_already:
                                break
                            pinned_piece = True
                            self.captures_moves_only.append(move)
                            #save the attack squares here
                            #do not break, continue checking to see if it is pinned
                            square_dict[(start_square_row,start_square_col)].extend(moves)
                            is_extended_already = True
                            

            if not is_extended_already:
                square_dict[(start_square_row,start_square_col)].extend(moves)


    def move_pawn_two_spaces(self,square_row,square_col,square_dict, move_spaces):
        for space in range(1,abs(move_spaces)+1):
            move = Move()
            move.start_square = (square_row, square_col)
            if move_spaces > 0:
                if self.board[square_row+space][square_col] == 0:
                    move.target_square = (square_row+space, square_col)
                else: 
                    break
            else:
                if self.board[square_row-space][square_col] == 0:
                    move.target_square = (square_row-space, square_col) 
                else:
                    break  
            if self.current_color != self.attack_color:
                square_dict[(square_row,square_col)].append(move)



    def generate_pawn_moves(self,square_row, square_col, square_dict):
        #check if the pawn has been moved before to see if it can move spaces
        start_square_row  = square_row
        start_square_col  = square_col
        square_dict[(start_square_row,start_square_col)] = []
        
        if self.current_color == Piece.white:
            if (square_row,square_col) in Square.white_pawn_original_location:
                self.move_pawn_two_spaces(square_row,square_col,square_dict,-2)
            else:
                move = Move()
                move.start_square = (square_row, square_col)
                square_row -= 1 
                if self.board[square_row][square_col]==0:
                    move.target_square = (square_row, square_col)
                    if self.current_color != self.attack_color:
                        square_dict[(start_square_row,start_square_col)].append(move)
                square_row += 1


            #check if there is an opp piece to the NE
            move = Move()
            move.start_square = (square_row, square_col)
            square_row -= 1
            square_col += 1
            if (square_row<8 and square_row > -1 and square_col<8 and square_col >-1):
                if self.current_color == self.attack_color or Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],False):
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                    if Piece.is_type(self.board[square_row][square_col],Piece.king):
                        print("YEP COOKED")
                        self.checked_path.append(move)
                    self.captures_moves_only.append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],True):
                    self.defended_squares.append((square_row,square_col))


            square_row += 1
            square_col -= 1

            #check if there is an opp piece to the NW
            move = Move()
            move.start_square = (square_row, square_col)
            square_row -= 1
            square_col -= 1
            if (square_row<8 and square_row > -1 and square_col<8 and square_col >-1):
                if self.current_color == self.attack_color or Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],False):
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                    if Piece.is_type(self.board[square_row][square_col],Piece.king):
                        self.checked_path.append(move)
                    self.captures_moves_only.append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],True):
                    self.defended_squares.append((square_row,square_col))

        else:
            if (square_row,square_col) in Square.black_pawn_original_location:
                self.move_pawn_two_spaces(square_row,square_col,square_dict,2)
            else:
                move = Move()
                move.start_square = (square_row, square_col)
                square_row += 1 
                if self.board[square_row][square_col]==0:
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                square_row -= 1

            #check if there is an opp piece to the NE
            move = Move()
            move.start_square = (square_row, square_col)
            square_row += 1
            square_col += 1
            if (square_row<8 and square_row > -1 and square_col<8 and square_col >-1):
                if self.current_color == self.attack_color:
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],False):
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                    if Piece.is_type(self.board[square_row][square_col],Piece.king):
                        self.checked_path.append(move)
                    self.captures_moves_only.append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],True):
                    self.defended_squares.append((square_row,square_col))
            square_row -= 1
            square_col -= 1

            #check if there is an opp piece to the NW
            move = Move()
            move.start_square = (square_row, square_col)
            square_row += 1
            square_col -= 1
            if (square_row<8 and square_row > -1 and square_col<8 and square_col >-1):
                if self.current_color == self.attack_color:
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],False):
                    move.target_square = (square_row, square_col)
                    square_dict[(start_square_row,start_square_col)].append(move)
                    if Piece.is_type(self.board[square_row][square_col],Piece.king):
                        self.checked_path.append(move)
                    self.captures_moves_only.append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[start_square_row][start_square_col],True):
                    self.defended_squares.append((square_row,square_col))
            
        
                
        #check if it can move one piece forward
        

    def add_knight_squares(self,square_dict, square_row, square_col):
        start_square = (square_row,square_col)

        square_dict[(square_row,square_col)] = []
        if start_square in self.knight_moves_dict:
            for target_square in self.knight_moves_dict[start_square]:
                move = Move()
                move.start_square = start_square
                row, col = target_square
                if Piece.is_color(self.board[row][col],self.board[square_row][square_col],False) or self.board[row][col] == 0 : 
                    move.target_square = target_square
                    self.captures_moves_only.append(move)
                    if Piece.is_type(self.board[row][col],Piece.king):
                        self.checked_path.append(move)
                        #continue here to avoid the issue of accidentally eating the king
                        continue
                    square_dict[(square_row,square_col)].append(move)
                elif Piece.is_color(self.board[square_row][square_col],self.board[square_row][square_col],True):
                    self.defended_squares.append((row,col))

        

    #generate this at the start since these are fixed 
    def generate_knight_moves(self,knight_row, knight_col,knight_moves_dict):
        initial_knight_row = knight_row
        initial_knight_col = knight_col
        if (initial_knight_row) not in knight_moves_dict:
            knight_moves_dict[(initial_knight_row,initial_knight_col)] = []
            for offset in self.knight_offsets:
                knight_row = initial_knight_row
                knight_col = initial_knight_col

                row_movement , col_movement = self.knight_offsets[offset]
                knight_row += row_movement
                knight_col += col_movement
                if knight_row > 7 or knight_row <0 or knight_col > 7 or knight_col < 0:
                    continue

                if(knight_row,knight_col) in knight_moves_dict[(initial_knight_row,initial_knight_col)]:
                    continue
                # add the new position
                knight_moves_dict[(initial_knight_row,initial_knight_col)].append((knight_row,knight_col))
                
                if(knight_row,knight_col) in knight_moves_dict:
                    #if position has already been explored then do not recurse
                    continue
                self.generate_knight_moves(knight_row,knight_col,knight_moves_dict)

    
    def generate_king_moves(self,king_row,king_col, square_dict):
        
        initial_king_row = king_row
        initial_king_col = king_col
        if (initial_king_row,initial_king_col) not in square_dict.keys():
            square_dict[(initial_king_row,initial_king_col)] = []
        
        for direction in self.direction_offset:
            king_row = initial_king_row
            king_col = initial_king_col
            move = Move()
            move.start_square = (initial_king_row,initial_king_col) 
            row_movement,col_movement = direction
            king_row+=row_movement
            king_col+=col_movement
            move.target_square  = (king_row,king_col)
            if king_row > 7 or king_row <0 or king_col > 7 or king_col < 0:
                continue
            if Piece.is_color(self.board[king_row][king_col], self.board[initial_king_row][initial_king_col], True):
                self.defended_squares.append((king_row,king_col))
                continue
            if Piece.is_color(self.board[king_row][king_col], self.board[initial_king_row][initial_king_col], False) and self.board[king_row][king_col]!=0:
                self.captures_moves_only.append(move)
            square_dict[(initial_king_row,initial_king_col)].append(move)
            
    
    def create_list_attack_squares(self):
        attack_squares = []
        for val in self.temp_attack_squares.values():
            attack_squares.extend(val)
        attack_squares = set(attack_squares)
        return attack_squares
    
    def create_list_of_total_moves(self):
        total_moves = []
        for val in self.total_moves.values():
            total_moves.extend(val)
        total_moves= set(total_moves)
        return total_moves
    
    def check_if_pawn_can_change(self,row,col):
        if row==7 or row==0 and Piece.is_type(self.board[row][col],Piece.pawn):
            return True
    
    def change_to_knight_or_queen(self,knight_flag,queen_flag):
        if knight_flag==True:
            self.changed_pawn = Piece.knight|self.AI_player
        elif queen_flag==True:
            self.changed_pawn = Piece.queen|self.AI_player


    def change_pawn_queen_and_append_back_move(self,row,col,final_allowed_moves,move):
        self.board[row][col] = self.current_color|Piece.queen
        final_allowed_moves.append(move)

    def change_pawn_knight(self,row,col):
        self.board[row][col] = self.current_color|Piece.knight

        
    def filter_illegal_moves(self):
        self.final_allowed_moves = []
        #if under check, get all the legal moves to be made
        #if one of those moves intercepts the path of check 
        attack_squares = self.create_list_attack_squares()
        if self.current_color == Piece.white:
            self.black_positions.clear()
            #filter list of new attack squares agaist the old ones
            self.filter(self.attack_squares, self.black_positions, attack_squares)
            self.temp_attack_squares = self.attack_squares
            #redo the list of attack squares for the current player to use 
            attack_squares = self.create_list_attack_squares()
            self.white_positions.clear()
            self.filter(self.total_moves, self.white_positions, attack_squares, is_current_player=True)
            self.under_check()
                                
        else:
            self.white_positions.clear()
            self.filter(self.attack_squares, self.white_positions, attack_squares)
            self.temp_attack_squares = self.attack_squares
            #redo the list of attack squares for the current player to use 
            attack_squares = self.create_list_attack_squares()
            self.black_positions.clear()
            self.filter(self.total_moves, self.black_positions, attack_squares, is_current_player=True)
            self.under_check()


    def filter_illegal_moves_minmax(self):
        self.final_allowed_moves = []
        #if under check, get all the legal moves to be made
        #if one of those moves intercepts the path of check 
        attack_squares = self.create_list_attack_squares()
        if self.current_color == Piece.white:
            #filter list of new attack squares agaist the old ones
            self.filter_minmax(self.attack_squares, attack_squares)
            self.temp_attack_squares = self.attack_squares
            #redo the list of attack squares for the current player to use 
            attack_squares = self.create_list_attack_squares()
            self.filter_minmax(self.total_moves, attack_squares, is_current_player=True)
            self.under_check()
                                
        else:
            self.filter_minmax(self.attack_squares, attack_squares)
            self.temp_attack_squares = self.attack_squares
            #redo the list of attack squares for the current player to use 
            attack_squares = self.create_list_attack_squares()
            self.filter_minmax(self.total_moves, attack_squares, is_current_player=True)
            self.under_check()
            

    def remove_castle_moves(self, move,total_moves):
        if move.start_square == (0,4) and move.target_square == (0,6):
            if move in self.final_allowed_moves:
                self.final_allowed_moves.remove(move)
        if move.start_square == (0,4) and move.target_square == (0,2):
            if move in self.final_allowed_moves:
                self.final_allowed_moves.remove(move)
        if move.start_square == (7,4) and move.target_square == (7,2):
            if move in self.final_allowed_moves:
                self.final_allowed_moves.remove(move)
        if move.start_square == (7,4) and move.target_square == (7,6):
            if move in self.final_allowed_moves:
                self.final_allowed_moves.remove(move)

    def under_check(self):
        total_moves = self.create_list_of_total_moves()
        #pick from the list of final allowed moves
        # temp_final_allowed_moves = []
  
        if self.checked_path:       
            for checked_move in self.checked_path:
                for move in total_moves:
                    self.remove_castle_moves(move,total_moves)
                    if move.target_square == checked_move.target_square or move.target_square == checked_move.start_square:
                        self.final_allowed_moves.append(move)

        else:
            self.final_allowed_moves.extend(total_moves)


        
    def filter(self, square_dict, positions, attack_squares, is_current_player=False):

        remove_start_squares = []
        for start_square in square_dict:
            positions.append(start_square)
            #filter king moves that would put it under check
            row,col = start_square
            remove_list = []
            if Piece.is_type(self.board[row][col], Piece.king) and Piece.is_color(self.board[row][col],self.current_color):
                #should not rmeove while iterating
                for move in square_dict[start_square]:
                    for square in attack_squares:
                       
                        if move.target_square == square.target_square:
                            print("king moves to put it in check",move.start_square,move.target_square)
                            s_row, s_col = square.start_square
                            print(self.board[s_row][s_col], "this is the piece cuasing issues ",(s_row,s_col))
                            remove_list.append(move)
                            break

                    for square in self.attacker_defended_squares:
                        if move.target_square == square:
                            print("king moves to put it in check",move.start_square,move.target_square)
                            s_row, s_col = square
                            print(self.board[s_row][s_col], "this is the piece cuasing issues ",(s_row,s_col))
                            if move not in remove_list:
                                remove_list.append(move)
                            break

                for move in remove_list:
                    square_dict[start_square].remove(move)

                if is_current_player:
                    self.final_allowed_moves.extend(square_dict[start_square])

            

            #implement 2 methods, check if the piece can end the life of the opp piece pinning it 
            #but first check if more than one piece has it pinned 
            pinners = 0
            pinner_square = (0,0)
            for pinned_piece in self.pinned_piece_paths:
                if start_square == pinned_piece.target_square:
                    pinner_square = pinned_piece.start_square
                    pinners+=1 #

            if pinners == 1:
                remove = True
                for move in square_dict[start_square]:
                    if move.target_square == pinner_square:
                        remove = False
                        break
                if remove:
                    remove_start_squares.append(start_square)

                
            if pinners == 2:
                remove_start_squares.append(start_square)

        for start_square in remove_start_squares:
            del square_dict[start_square]

            
    def filter_minmax(self, square_dict, attack_squares, is_current_player=False):

        remove_start_squares = []
        for start_square in square_dict:
            #filter king moves that would put it under check
            row,col = start_square
            remove_list = []
            if Piece.is_type(self.board[row][col], Piece.king) and Piece.is_color(self.board[row][col],self.current_color):
                #should not rmeove while iterating
                for move in square_dict[start_square]:
                    for square in attack_squares:
                       
                        if move.target_square == square.target_square:
                            print("king moves to put it in check",move.start_square,move.target_square)
                            s_row, s_col = square.start_square
                            print(self.board[s_row][s_col], "this is the piece cuasing issues ",(s_row,s_col))
                            remove_list.append(move)
                            break

                    for square in self.attacker_defended_squares:
                        if move.target_square == square:
                            print("king moves to put it in check",move.start_square,move.target_square)
                            s_row, s_col = square
                            print(self.board[s_row][s_col], "this is the piece cuasing issues ",(s_row,s_col))
                            if move not in remove_list:
                                remove_list.append(move)
                            break

                for move in remove_list:
                    square_dict[start_square].remove(move)

                if is_current_player:
                    self.final_allowed_moves.extend(square_dict[start_square])

            

            #implement 2 methods, check if the piece can end the life of the opp piece pinning it 
            #but first check if more than one piece has it pinned 
            pinners = 0
            pinner_square = (0,0)
            for pinned_piece in self.pinned_piece_paths:
                if start_square == pinned_piece.target_square:
                    pinner_square = pinned_piece.start_square
                    pinners+=1 #

            if pinners == 1:
                remove = True
                for move in square_dict[start_square]:
                    if move.target_square == pinner_square:
                        remove = False
                        break
                if remove:
                    remove_start_squares.append(start_square)

                
            if pinners == 2:
                remove_start_squares.append(start_square)

        for start_square in remove_start_squares:
            del square_dict[start_square]


    def resetting_back_changed_castling_flags(self,recent_changed_castle_flags):
        for key in recent_changed_castle_flags:
            if recent_changed_castle_flags[key]!=self.castle_flags[key]:
                self.castle_flags[key] = not self.castle_flags[key]
                recent_changed_castle_flags[key] = self.castle_flags[key]


    # def check_if_rook_king_moved(self,square_row,square_col,recent_changed_castle_flags):

    #     if (square_row,square_col) == (7,7):
    #         self.WrookRMove = True

    #     if (square_row,square_col) == (7,0):
    #         self.WrookLMove = True

    #     if (square_row,square_col) == (0,0):
    #         self.BrookLMove = True

    #     if (square_row,square_col) == (0,7):
    #         self.BrookRMove = True

    #     if (square_row,square_col) == (7,4):
    #         self.WkingMove = True

    #     if (square_row,square_col) == (0,4):
    #         self.BkingMove = True

    def check_if_rook_king_moved(self,square_row,square_col):

        if (square_row,square_col) == (7,7):
            self.castle_flags['WRookRMove'] = True

        if (square_row,square_col) == (7,0):
            self.castle_flags['WRookLMove'] = True

        if (square_row,square_col) == (0,0):
            self.castle_flags['BRookLMove'] = True

        if (square_row,square_col) == (0,7):
            self.castle_flags['BRookRMove'] = True

        if (square_row,square_col) == (7,4):
            self.castle_flags['WkingMove'] = True

        if (square_row,square_col) == (0,4):
            self.castle_flags['BkingMove'] = True

    def build_castle_moves(self):
        if self.current_color == Piece.black:
            if self.board[0][5]==0 and self.board[0][6] == 0 and Piece.is_type(self.board[0][4],Piece.king) and Piece.is_type(self.board[0][7],Piece.rook) :
                if not self.castle_flags['BrookRMove'] and not self.castle_flags['BkingMove']:
                    self.king_side_white_castling = True
                    king_move = Move()
                    king_move.start_square = (0,4)
                    king_move.target_square = (0,6)
                    if king_move.start_square not in self.total_moves.keys():
                        self.total_moves[king_move.start_square] = []
                    self.total_moves[king_move.start_square].append(king_move)
                
            if self.board[0][1] == 0 and self.board[0][2]==0 and self.board[0][3]==0 and Piece.is_type(self.board[0][4],Piece.king) and Piece.is_type(self.board[0][0],Piece.rook):
                if not self.castle_flags['BrookLMove'] and not self.castle_flags['BkingMove']:
                    self.queen_side_white_castling = True
                    king_move = Move()
                    king_move.start_square = (0,4)
                    king_move.target_square = (0,2)
                    if king_move.start_square not in self.total_moves.keys():
                        self.total_moves[king_move.start_square] = []
                    self.total_moves[king_move.start_square].append(king_move)
        else:
            if self.board[7][5]==0 and self.board[7][6] == 0 and Piece.is_type(self.board[7][4],Piece.king) and Piece.is_type(self.board[7][7],Piece.rook):
                if not self.castle_flags['WrookRMove'] and not self.castle_flags['WkingMove']:
                    self.king_side_black_castling = True
                    king_move = Move()
                    king_move.start_square = (7,4)
                    king_move.target_square = (7,6)
                    if king_move.start_square not in self.total_moves.keys():
                        self.total_moves[king_move.start_square] = []
                    self.total_moves[king_move.start_square].append(king_move)

                
            if self.board[7][1] == 0 and self.board[7][2]==0 and self.board[7][3]==0 and Piece.is_type(self.board[7][4],Piece.king) and Piece.is_type(self.board[7][0],Piece.rook):
                if not self.castle_flags['WrookLMove'] and not self.castle_flags['WkingMove']:
                    self.queen_side_black_castling = True
                    king_move = Move()
                    king_move.start_square = (7,4)
                    king_move.target_square = (7,2)
                    if king_move.start_square not in self.total_moves.keys():
                        self.total_moves[king_move.start_square] = []
                    self.total_moves[king_move.start_square].append(king_move)



    def no_moves(self):
        if(self.final_allowed_moves == []):
            return True
        return False


########################MINMAX SECTION################################################

    #use pluses for in favour of white, as the AI is supposed to be black
    #use minuses for in favour of black
    def check_indiv_pawns(self,row,col):
        if (row,col) in self.attack_squares:
            self.points += 2
            #worse for isolated pawns
            if (row,col) not in self.defended_squares:
                self.points += 4 


        
    def minmax(self, depth, is_maximising,alpha, beta):
        # print(self.current_color,"current",depth)
        self.start_new_round()
        self.filter_illegal_moves()
        # print(self.current_color,"currenty",depth)

        if depth == 0 or self.no_moves():
            if self.no_moves() and self.human_player == self.current_color:
                #return the largest -ve amount to acheive this outcome
                return -100
            elif self.no_moves() and self.AI_player == self.current_color:
                return 100
            return self.enhanced_evaluate()

        final_allowed_moves = copy.copy(self.final_allowed_moves)
        recent_changed_castle_flags = copy.copy(self.castle_flags)
        pawn_to_knight = False
        pawn_to_queen = False
        if is_maximising:
            best_score = float("-inf")
            for move in final_allowed_moves:
                if move in self.visited:
                    continue
                self.visited.append(move)
                was_removed = False
                target_row, target_col = move.target_square
                start_row, start_col = move.start_square
                piece = self.board[start_row][start_col]
                target_piece = self.board[target_row][target_col]
                print("is there a piece on target square",self.board[target_row][target_col])
                print("then what is the piece on the start",self.board[start_row][start_col])
                self.board[start_row][start_col] = 0
                self.board[target_row][target_col] =piece

                #check if pawn has reached the end and change it to a queen first
                #append the move back to the end of the list to check for knight after
                if self.check_if_pawn_can_change(target_row,target_col): 
                    self.change_pawn_queen_and_append_back_move(target_row,target_col,final_allowed_moves,move)
                    pawn_to_queen = True
                    pawn_to_knight = False
                elif pawn_to_queen and self.check_if_pawn_can_change(target_row,target_col):
                    self.change_pawn_knight()
                    pawn_to_knight = True
                    pawn_to_queen = False
                self.check_if_rook_king_moved(start_row,start_col)
                if self.current_color == Piece.black:
                    self.black_positions.remove((start_row,start_col))
                    self.black_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.white_positions:
                        self.white_positions.remove((target_row,target_col)) 
                        was_removed = True
                else:
                    print("prob is here",start_row,start_col,self.board[start_row][start_col],depth)
                    print("arget is here",target_row,target_col,self.board[target_row][target_col],depth)
                    print(self.white_positions)
                    self.white_positions.remove((start_row,start_col))
                    self.white_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.black_positions:
                        self.black_positions.remove((target_row,target_col))
                        was_removed = True
                print("bef reset",self.white_positions)
                self.change_current_color()
                score = self.minmax(depth-1, False,alpha,beta)
                self.change_current_color()
                
                self.resetting_back_changed_castling_flags(recent_changed_castle_flags)
                if self.current_color == Piece.black:
                    self.black_positions.remove((target_row,target_col))
                    self.black_positions.append((start_row,start_col))
                    if was_removed:
                        self.white_positions.append((target_row,target_col))          
                else:
                    self.white_positions.remove((target_row,target_col))
                    self.white_positions.append((start_row,start_col))
                    if was_removed:
                        self.black_positions.append((target_row,target_col))

                print("reset white pos",self.white_positions)
                self.board[start_row][start_col] = piece
                self.board[target_row][target_col] = target_piece

                if score>best_score:
                    if depth==self.initial_depth:
                        if self.check_if_pawn_can_change(target_row,target_col):
                            self.change_to_knight_or_queen(pawn_to_knight,pawn_to_queen)
                        self.best_move = move
                    best_score = score
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break 

        else:

            best_score = float("inf")
            for move in final_allowed_moves:
                if move in self.visited:
                    continue
                self.visited.append(move)
                was_removed = False
                target_row, target_col = move.target_square
                start_row, start_col = move.start_square
                piece  = self.board[start_row][start_col]
                target_piece = self.board[target_row][target_col]
                self.board[start_row][start_col] = 0
                self.board[target_row][target_col] = piece

                if self.check_if_pawn_can_change(target_row,target_col): 
                    self.change_pawn_queen_and_append_back_move(target_row,target_col,final_allowed_moves,move)
                    pawn_to_queen = True
                    pawn_to_knight = False
                elif pawn_to_queen and self.check_if_pawn_can_change(target_row,target_col):
                    self.change_pawn_knight()
                    pawn_to_knight = True
                    pawn_to_queen = False
                self.check_if_rook_king_moved(start_row,start_col)
                # reverse these changes to the positions
                if self.current_color == Piece.black:
                    print((start_row,start_col),"fked uup one",self.board[start_row][start_col])
                    print((target_row,target_col),"fked uup 2",self.board[target_row][target_col])
                    print(recent_changed_castle_flags, 'yeye', self.castle_flags, 'depth',depth)
                    self.black_positions.remove((start_row,start_col))
                    self.black_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.white_positions:
                        self.white_positions.remove((target_row,target_col))
                        was_removed = True 
                else:                   
                    self.white_positions.remove((start_row,start_col))
                    self.white_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.black_positions:
                        self.black_positions.remove((target_row,target_col))
                        was_removed = True


                self.change_current_color()
                score = self.minmax(depth-1,True,alpha,beta)
                self.change_current_color() 

                self.resetting_back_changed_castling_flags(recent_changed_castle_flags)
                if self.current_color == Piece.black:
                    self.black_positions.remove((target_row,target_col))
                    self.black_positions.append((start_row,start_col))
                    if was_removed:
                        self.white_positions.append((target_row,target_col)) 
                    
                else:
                    self.white_positions.remove((target_row,target_col))
                    self.white_positions.append((start_row,start_col))
                    if was_removed:
                        self.black_positions.append((target_row,target_col))
                self.board[start_row][start_col] = piece
                self.board[target_row][target_col] = target_piece
    
                
                if score<best_score:
                    if depth==self.initial_depth:
                        if self.check_if_pawn_can_change(target_row,target_col):
                            self.change_to_knight_or_queen(pawn_to_knight,pawn_to_queen)
                        self.best_move = move
                    best_score = score
                beta = min(beta, best_score)
                if beta <= alpha:
                    break 

        return best_score


    # def evaluate(self):
    #     #check the total value of pieces

    #     points = 0
    #     for white in self.white_positions:
    #         row,col = white
    #         if Piece.is_type(self.board[row][col], Piece.pawn):
    #             points += self.Pawn
    #         if Piece.is_type(self.board[row][col], Piece.knight):
    #             points += self.Knight
    #         if Piece.is_type(self.board[row][col], Piece.bishop):
    #             points += self.Bishop

    #         if Piece.is_type(self.board[row][col], Piece.queen):
    #             points += self.Queen

    #         if Piece.is_type(self.board[row][col], Piece.rook):
    #             points += self.Rook

    #     for black in self.black_positions:
    #         row,col = black
    #         if Piece.is_type(self.board[row][col], Piece.pawn):
    #             points -= self.Pawn
    #         if Piece.is_type(self.board[row][col], Piece.knight):
    #             points -= self.Knight
    #         if Piece.is_type(self.board[row][col], Piece.bishop):
    #             points -= self.Bishop

    #         if Piece.is_type(self.board[row][col], Piece.queen):
    #             points -= self.Queen

    #         if Piece.is_type(self.board[row][col], Piece.rook):
    #             points -= self.Rook

    #     return points

    def evaluate(self):
        """Enhanced evaluation function combining multiple factors"""
        points = 0
        
        # Material evaluation (your existing code)
        points += self.evaluate_material()
        
        # King safety
        # points += self.evaluate_king_safety()
        
        # Pawn structure
        # points += self.evaluate_pawn_structure()
        
        # Piece mobility and activity
        # points += self.evaluate_piece_activity()
        
        # Control of center squares
        points += self.evaluate_center_control()
        
        return points

    def evaluate_material(self):
        """Basic material count (your existing logic)"""
        points = 0
        for white in self.white_positions:
            row, col = white
            if Piece.is_type(self.board[row][col], Piece.pawn):
                points += self.Pawn
            elif Piece.is_type(self.board[row][col], Piece.knight):
                points += self.Knight
            elif Piece.is_type(self.board[row][col], Piece.bishop):
                points += self.Bishop
            elif Piece.is_type(self.board[row][col], Piece.queen):
                points += self.Queen
            elif Piece.is_type(self.board[row][col], Piece.rook):
                points += self.Rook

        for black in self.black_positions:
            row, col = black
            if Piece.is_type(self.board[row][col], Piece.pawn):
                points -= self.Pawn
            elif Piece.is_type(self.board[row][col], Piece.knight):
                points -= self.Knight
            elif Piece.is_type(self.board[row][col], Piece.bishop):
                points -= self.Bishop
            elif Piece.is_type(self.board[row][col], Piece.queen):
                points -= self.Queen
            elif Piece.is_type(self.board[row][col], Piece.rook):
                points -= self.Rook
        
        return points
    
    


    def evaluate_king_safety(self):
        """Evaluate king safety based on pawn shield and piece proximity"""
        points = 0
        
        # Find kings
        white_king_pos = None
        black_king_pos = None
        
        for row, col in self.white_positions:
            if Piece.is_type(self.board[row][col], Piece.king):
                white_king_pos = (row, col)
                break
        
        for row, col in self.black_positions:
            if Piece.is_type(self.board[row][col], Piece.king):
                black_king_pos = (row, col)
                break
        
        if white_king_pos:
            points += self.evaluate_single_king_safety(white_king_pos, Piece.white)
        
        if black_king_pos:
            points -= self.evaluate_single_king_safety(black_king_pos, Piece.black)
        
        return points

    def evaluate_single_king_safety(self, king_pos, color):
        """Evaluate safety for a single king"""
        king_row, king_col = king_pos
        safety_score = 0
        
        # Check pawn shield
        if color == Piece.white:
            # Check pawns in front of king
            shield_positions = [(king_row-1, king_col-1), (king_row-1, king_col), (king_row-1, king_col+1)]
            for shield_row, shield_col in shield_positions:
                if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                    piece = self.board[shield_row][shield_col]
                    if Piece.is_type(piece, Piece.pawn) and Piece.is_color(piece, Piece.white, True):
                        safety_score += 0.5
        else:
            # Check pawns in front of black king
            shield_positions = [(king_row+1, king_col-1), (king_row+1, king_col), (king_row+1, king_col+1)]
            for shield_row, shield_col in shield_positions:
                if 0 <= shield_row < 8 and 0 <= shield_col < 8:
                    piece = self.board[shield_row][shield_col]
                    if Piece.is_type(piece, Piece.pawn) and Piece.is_color(piece, Piece.black, True):
                        safety_score += 0.5
        
        # Penalty for king in center during middle game
        if 2 <= king_row <= 5 and 2 <= king_col <= 5:
            safety_score -= 1.0
        
        return safety_score

    def evaluate_pawn_structure(self):
        """Evaluate pawn structure (doubled, isolated, passed pawns)"""
        points = 0
        
        # Analyze white pawns
        white_pawn_files = {}
        for row, col in self.white_positions:
            if Piece.is_type(self.board[row][col], Piece.pawn):
                if col not in white_pawn_files:
                    white_pawn_files[col] = []
                white_pawn_files[col].append(row)
        
        # Analyze black pawns
        black_pawn_files = {}
        for row, col in self.black_positions:
            if Piece.is_type(self.board[row][col], Piece.pawn):
                if col not in black_pawn_files:
                    black_pawn_files[col] = []
                black_pawn_files[col].append(row)
        
        # Check for doubled pawns (penalty)
        for file_pawns in white_pawn_files.values():
            if len(file_pawns) > 1:
                points -= 0.5 * (len(file_pawns) - 1)
        
        for file_pawns in black_pawn_files.values():
            if len(file_pawns) > 1:
                points += 0.5 * (len(file_pawns) - 1)
        
        # Check for isolated pawns (penalty)
        for file_col, file_pawns in white_pawn_files.items():
            has_adjacent_pawns = False
            for adjacent_file in [file_col - 1, file_col + 1]:
                if adjacent_file in white_pawn_files:
                    has_adjacent_pawns = True
                    break
            if not has_adjacent_pawns:
                points -= 0.5 * len(file_pawns)
        
        for file_col, file_pawns in black_pawn_files.items():
            has_adjacent_pawns = False
            for adjacent_file in [file_col - 1, file_col + 1]:
                if adjacent_file in black_pawn_files:
                    has_adjacent_pawns = True
                    break
            if not has_adjacent_pawns:
                points += 0.5 * len(file_pawns)
        
        # Check for passed pawns (bonus)
        for file_col, file_pawns in white_pawn_files.items():
            for pawn_row in file_pawns:
                is_passed = True
                # Check if any black pawns can stop this pawn
                for check_file in [file_col - 1, file_col, file_col + 1]:
                    if check_file in black_pawn_files:
                        for black_pawn_row in black_pawn_files[check_file]:
                            if black_pawn_row < pawn_row:  # Black pawn ahead of white pawn
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    # Bonus increases as pawn advances
                    advancement_bonus = (7 - pawn_row) * 0.2
                    points += 1.0 + advancement_bonus
        
        for file_col, file_pawns in black_pawn_files.items():
            for pawn_row in file_pawns:
                is_passed = True
                # Check if any white pawns can stop this pawn
                for check_file in [file_col - 1, file_col, file_col + 1]:
                    if check_file in white_pawn_files:
                        for white_pawn_row in white_pawn_files[check_file]:
                            if white_pawn_row > pawn_row:  # White pawn ahead of black pawn
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    # Bonus increases as pawn advances
                    advancement_bonus = pawn_row * 0.2
                    points -= 1.0 + advancement_bonus
        
        return points

    def evaluate_piece_activity(self):
        """Evaluate piece mobility and activity"""
        points = 0
        
        # Simple mobility evaluation based on number of legal moves
        # You can use your existing move generation for this
        
        # Generate moves for current position and count them
        current_color_backup = self.current_color
        
        # Count white piece mobility
        self.current_color = Piece.white
        white_moves = 0
        for row, col in self.white_positions:
            temp_dict = {}
            self.move_generate(row, col, temp_dict)
            if (row, col) in temp_dict:
                white_moves += len(temp_dict[(row, col)])
        
        # Count black piece mobility
        self.current_color = Piece.black
        black_moves = 0
        for row, col in self.black_positions:
            temp_dict = {}
            self.move_generate(row, col, temp_dict)
            if (row, col) in temp_dict:
                black_moves += len(temp_dict[(row, col)])
        
        # Restore current color
        self.current_color = current_color_backup
        
        # Mobility bonus (scaled down)
        points += (white_moves - black_moves) * 0.05
        
        return points

    def evaluate_center_control(self):
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        extended_center = [(2, 2), (2, 3), (2, 4), (2, 5), 
                        (3, 2), (3, 5), (4, 2), (4, 5),
                        (5, 2), (5, 3), (5, 4), (5, 5)]
        
        points = 0
        
        # Points for occupying center
        for row, col in center_squares:
            piece = self.board[row][col]
            if piece != 0:
                if Piece.is_color(piece, Piece.white, True):
                    points += 0.3
                else:
                    points -= 0.3
        
        # Points for occupying extended center
        for row, col in extended_center:
            piece = self.board[row][col]
            if piece != 0:
                if Piece.is_color(piece, Piece.white, True):
                    points += 0.1
                else:
                    points -= 0.1
        
        return points
    

    def enhanced_evaluate(self):
        """Replace your existing evaluate method with this"""
        if not hasattr(self, 'evaluator'):
            self.evaluator = EnhancedEvaluation()
        
        return self.evaluator.evaluate(self)
        

    def self.

    def check_for_captures(self, alpha, beta):
        score = self.enhanced_evaluate()
        if beta >= score:
            return beta
        alpha = max(alpha, score)
        self.start_new_round()
        self.total_moves = self.captures_moves_only
        self.filter_illegal_moves()
        final_allowed_moves = copy.copy(self.final_allowed_moves)
        for move in final_allowed_moves:
            if move in self.visited:
                    continue
                self.visited.append(move)
                was_removed = False
                target_row, target_col = move.target_square
                start_row, start_col = move.start_square
                piece  = self.board[start_row][start_col]
                target_piece = self.board[target_row][target_col]
                self.board[start_row][start_col] = 0
                self.board[target_row][target_col] = piece

                if self.check_if_pawn_can_change(target_row,target_col): 
                    self.change_pawn_queen_and_append_back_move(target_row,target_col,final_allowed_moves,move)
                    pawn_to_queen = True
                    pawn_to_knight = False
                elif pawn_to_queen and self.check_if_pawn_can_change(target_row,target_col):
                    self.change_pawn_knight()
                    pawn_to_knight = True
                    pawn_to_queen = False
                self.check_if_rook_king_moved(start_row,start_col)
                # reverse these changes to the positions
                if self.current_color == Piece.black:
                    self.black_positions.remove((start_row,start_col))
                    self.black_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.white_positions:
                        self.white_positions.remove((target_row,target_col))
                        was_removed = True 
                else:                   
                    self.white_positions.remove((start_row,start_col))
                    self.white_positions.append((target_row,target_col))
                    if (target_row,target_col) in self.black_positions:
                        self.black_positions.remove((target_row,target_col))
                        was_removed = True


                self.change_current_color()
                score = self.minmax(depth-1,True,alpha,beta)
                self.change_current_color() 

                self.resetting_back_changed_castling_flags(recent_changed_castle_flags)
                if self.current_color == Piece.black:
                    self.black_positions.remove((target_row,target_col))
                    self.black_positions.append((start_row,start_col))
                    if was_removed:
                        self.white_positions.append((target_row,target_col)) 
                    
                else:
                    self.white_positions.remove((target_row,target_col))
                    self.white_positions.append((start_row,start_col))
                    if was_removed:
                        self.black_positions.append((target_row,target_col))
                self.board[start_row][start_col] = piece
                self.board[target_row][target_col] = target_piece
    
                
                if score<best_score:
                    if depth==self.initial_depth:
                        if self.check_if_pawn_can_change(target_row,target_col):
                            self.change_to_knight_or_queen(pawn_to_knight,pawn_to_queen)
                        self.best_move = move
                    best_score = score
                beta = min(beta, best_score)
                if beta <= alpha:
                    break 

        return best_score
        

            
            
            
            
