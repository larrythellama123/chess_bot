import pygame as p
from chess_engine import GameState
import math
from piece import pieces
from move import Move
import copy

Piece = pieces()

WIDTH = 800
HEIGHT = 800
ROWS = 8
COLS = 8
SQUARE_SIZE = WIDTH//ROWS
win = p.display.set_mode((WIDTH,HEIGHT))

p.display.set_caption(('Chess'))
run  = True
FPS =60
IMAGES={}
ENLARGED_IMAGES = {}

WHITE = (255, 255, 255)
BLACK = (100,100,100)

gameState = GameState()
# Move_  = Move()
board  = gameState.translate_fen_strings()
pic_board = [[0 for i in range(8)] for i in range(8)]
Dragging = False

is_checkmate = False
selected_square= (0,0)
gameState.initial_setup()


def load_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load('images/'+ piece +'.png'), (SQUARE_SIZE, SQUARE_SIZE))

def load_enlarged_images():
    pieces = ['wK','wQ']
    for piece in pieces:
        ENLARGED_IMAGES[piece] = p.transform.scale(p.image.load('images/'+ piece +'.png'), (2*SQUARE_SIZE, 2*SQUARE_SIZE))



def draw_board(win):
    colors = [WHITE,BLACK]
    for row in range(ROWS):
        for column in range(COLS):
            color = colors[((row + column) % 2)]
            p.draw.rect(win, color, p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pic_board[row][column] = p.Rect(column * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)


def drawPieces(win, board):
    for row in range(ROWS):
        for column in range(COLS):
            piece = board[row][column]
            if piece == 0:
                continue
            # if selected_square == (row,column) and Dragging:
            #     continue
            if Piece.is_type(piece, Piece.king):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wK'], (column * SQUARE_SIZE, row * SQUARE_SIZE ))

                else:
                    win.blit(IMAGES['bK'], (column * SQUARE_SIZE, row * SQUARE_SIZE))

            if Piece.is_type(piece, Piece.queen):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wQ'],  (column * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    win.blit(IMAGES['bQ'],  (column * SQUARE_SIZE, row * SQUARE_SIZE))
                

            if Piece.is_type(piece, Piece.bishop):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wB'], (column * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    win.blit(IMAGES['bB'], (column * SQUARE_SIZE, row * SQUARE_SIZE))

            if Piece.is_type(piece, Piece.pawn):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wp'], (column * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    win.blit(IMAGES['bp'],  (column * SQUARE_SIZE, row * SQUARE_SIZE))

            if Piece.is_type(piece, Piece.knight):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wN'], (column * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    win.blit(IMAGES['bN'], (column * SQUARE_SIZE, row * SQUARE_SIZE))

            if Piece.is_type(piece, Piece.rook):
                if Piece.is_color(piece, Piece.white):
                    win.blit(IMAGES['wR'], (column * SQUARE_SIZE, row * SQUARE_SIZE))
                else:
                    win.blit(IMAGES['bR'], (column * SQUARE_SIZE, row * SQUARE_SIZE))


def draw_moving_image(win,Dragged_Piece,mouse_pos):
    piece = Dragged_Piece 
    piece_img = None
    if Piece.is_type(piece, Piece.king):
        if Piece.is_color(piece, Piece.white):
            piece_img = IMAGES['wK']
        else:
            piece_img = IMAGES['bK']
    if Piece.is_type(piece, Piece.queen):
        if Piece.is_color(piece, Piece.white):
            piece_img = IMAGES['wQ']
        else:
            piece_img = IMAGES['bQ']
        

    if Piece.is_type(piece, Piece.bishop):
        if Piece.is_color(piece, Piece.white):
            piece_img = IMAGES['wB']
        else:
            piece_img = IMAGES['bB']

    if Piece.is_type(piece, Piece.pawn):
        if Piece.is_color(piece, Piece.white):
           piece_img = IMAGES['wp']
        else:
            piece_img = IMAGES['bp']

    if Piece.is_type(piece, Piece.knight):
        if Piece.is_color(piece, Piece.white):
            piece_img = IMAGES['wN']
        else:
            piece_img = IMAGES['bN']

    if Piece.is_type(piece, Piece.rook):
        if Piece.is_color(piece, Piece.white):
            piece_img = IMAGES['wR']
        else:
            piece_img = IMAGES['bR']

    if piece_img:
        piece_rect = piece_img.get_rect(center=mouse_pos)
        win.blit(piece_img, piece_rect)


def get_clicked_square():
    x, y = p.mouse.get_pos()
    column = math.floor(x/SQUARE_SIZE)
    row  = math.floor(y/SQUARE_SIZE)
    # print(row, column)
    # print(board[row][column])
    if board[row][column] == 0:
        return None, None
    return column, row

def drop_square():
    x, y = p.mouse.get_pos()
    column = math.floor(x/SQUARE_SIZE)
    row  = math.floor(y/SQUARE_SIZE)
    # if board[row][column] == 0:
    return column,row
    # return None,None



def highlight_square(win, column, row):
    highlight = p.Surface((SQUARE_SIZE, SQUARE_SIZE))
    highlight.set_alpha(200)  # Transparency value (0-255)
    highlight.fill((255, 255, 0))  # Highlight color (yellow)
    win.blit(highlight, (column * SQUARE_SIZE, row * SQUARE_SIZE))



def check_legal_moves(initial_row,initial_col, row, col):
    for move in gameState.final_allowed_moves:
        # Move.print_move(move)
        if (initial_row,initial_col) == move.start_square and (row,col) == move.target_square:
            return True
    return False


def check_piece_usable(clicked_row, clicked_col):
    for move in gameState.final_allowed_moves:
        # Move.print_move(move)
        if (clicked_row,clicked_col) == move.start_square:
            return True
    return False
        

def AI_move(black_positions,white_positions):
    print( gameState.best_move.start_square, gameState.best_move.target_square,"y")
    initial_row,initial_col = gameState.best_move.start_square
    new_row,new_column = gameState.best_move.target_square
    moving_piece = board[initial_row][initial_col]
    board[new_row][new_column] = moving_piece
    board[initial_row][initial_col] = 0

    if gameState.AI_player == Piece.black:
        black_positions.remove((initial_row,initial_col))
        black_positions.append((new_row,new_column))
        if (new_row, new_column) in white_positions:
            white_positions.remove((new_row,new_column)) 
        
    else:
        white_positions.remove((initial_row,initial_col))
        white_positions.append((new_row,new_column))
        if (new_row, new_column) in black_positions:
            black_positions.remove((new_row,new_column))
    
    if Piece.is_type(moving_piece,Piece.king):
        if (initial_row,initial_col) == (7,4):
            if(new_row, new_column) == (7,2):
                rook_piece = board[7][0]
                board[7][3] = rook_piece
                board[7][0] = 0
            if (new_row,new_column) == (7,6):
                rook_piece = board[7][7]
                board[7][5] = rook_piece
                board[7][7] = 0

        if (initial_row,initial_col) == (0,4):
            if(new_row, new_column) == (0,2):
                rook_piece = board[0][0]
                board[0][3] = rook_piece
                board[0][0] = 0
            if (new_row,new_column) == (0,6):
                rook_piece = board[0][7]
                board[0][5] = rook_piece
                board[0][7] = 0

    gameState.black_positions = black_positions
    gameState.white_positions = white_positions
            

def decide_queen_or_knight(selected_row, selected_col):
    if pawn_change:
        if queen_rect.collidepoint(event.pos):  
            board[selected_row][selected_col] = Piece.queen
        elif knight_rect.collidepoint(event.pos):
            board[selected_row][selected_col] = Piece.knight

# white_queen_rect = ENLARGED_IMAGES['wQ'].get_rect()

# white_knight_rect = ENLARGED_IMAGES['wK'].get_rect()

# black_queen_rect = ENLARGED_IMAGES['bQ'].get_rect()
# black_knight_rect = ENLARGED_IMAGES['bK'].get_rect()

clock = p.time.Clock()
is_highlight = False
column = 0
row = 0
load_images()
load_enlarged_images()
Dragged_Piece = None
initial_row, initial_col = 0,0
selected_row, selected_col = 0,0
selected_square
print(board)
gameState.start_new_round()
pawn_change = False


while run:

    clock.tick(FPS)
    draw_board(win)
    drawPieces(win, board)
    if is_highlight:
        highlight_square(win,column,row)
    
    if Dragging and Dragged_Piece:
        draw_moving_image(win,Dragged_Piece,event.pos)

    if pawn_change:
        print(121212)
        if gameState.current_color == gameState.human_player:
            if gameState.current_color == Piece.white:
                queen_rect = ENLARGED_IMAGES['wQ'].get_rect()
                knight_rect = ENLARGED_IMAGES['wK'].get_rect()
            else:
                queen_rect = ENLARGED_IMAGES['bQ'].get_rect()
                knight_rect = ENLARGED_IMAGES['bK'].get_rect()
            queen_rect.center = (0,200)
            knight_rect.center = (600,400)
            win.blit(ENLARGED_IMAGES['wQ'], queen_rect)
            win.blit(ENLARGED_IMAGES['wK'], knight_rect)

    for event in p.event.get():
        if event.type == p.QUIT:
            run = False

        if event.type == p.MOUSEBUTTONDOWN:
            if pawn_change:
                if gameState.current_color == gameState.human_player and gameState.current_color == Piece.white:
                    if queen_rect.collidepoint(event.pos):  
                        board[selected_row][selected_col] = Piece.queen
                    elif knight_rect.collidepoint(event.pos):
                        board[selected_row][selected_col] = Piece.knight
                    else:
                        continue
            #game is over
            if is_checkmate:
                continue
            # print(f"new moves:{gameState.total_moves}")
            print("NEW MOVE")
            gameState.start_new_round()
            gameState.filter_illegal_moves()

            #check if checkmated
            if gameState.final_allowed_moves == []:
                print("CHECKMATED")
                is_checkmate = True
            if is_checkmate:continue


            # if gameState.current_color == gameState.AI_player:
            #     # for move in gameState.final_allowed_moves:
            #     #     target_row, target_col = move.target_square
            #     #     start_row, start_col = move.start_square
            #     #     piece = board[start_row][start_col]
            #     #     target_piece = board[target_row][target_col]
            #     #     board[start_row][start_col] = 0
            #     #     board[target_row][target_col] =piece
            #     black_positions_save = copy.copy(gameState.black_positions)
            #     white_positions_save = copy.copy(gameState.white_positions)
            #     gameState.initial_depth = 3
            #     temp_GS = copy.deepcopy(gameState)
            #     print(temp_GS.current_color,"BLACK")
            #     temp_GS.minmax(3,False,float('-inf'),float('inf'))
            #     gameState.best_move = temp_GS.best_move 
            #     AI_move(black_positions_save,white_positions_save)
            #     gameState.change_current_color()
            #     continue


            # print("final allowed moves:")
            # for move in gameState.final_allowed_moves:
            #     print("start_sq:",move.start_square)
            #     print("target_sq:",move.target_square)

            clicked_column,clicked_row  = get_clicked_square()
            
            if clicked_column!= None and clicked_row!=None:
                #save the clicked square
                if check_piece_usable(clicked_row,clicked_column):
                    Dragging = True
                    Dragged_Piece = board[clicked_row][clicked_column]
                    board[clicked_row][clicked_column] = 0
                    initial_row = clicked_row
                    initial_col = clicked_column
                    row = clicked_row
                    column = clicked_column
                    is_highlight = True

        if event.type == p.MOUSEBUTTONUP:
            if not Dragging:
                continue
            new_column, new_row = drop_square()
            if new_column !=None and new_row != None:
                is_legal = check_legal_moves(initial_row,initial_col,new_row,new_column)
                #if the move is not legal and its not the same as the original square
                # if not is_legal and (new_row, new_column)!=(initial_row,initial_col):
                #     continue  
                is_highlight = False

                if is_legal:
                    board[new_row][new_column] = Dragged_Piece
                    if gameState.current_color == Piece.black:
                        gameState.black_positions.remove((initial_row,initial_col))
                        gameState.black_positions.append((new_row,new_column))
                        if (new_row, new_column) in gameState.white_positions:
                            gameState.white_positions.remove((new_row,new_column)) 
                        
                    else:
                        gameState.white_positions.remove((initial_row,initial_col))
                        gameState.white_positions.append((new_row,new_column))
                        if (new_row, new_column) in gameState.black_positions:
                            gameState.black_positions.remove((new_row,new_column))

                    if Piece.is_type(Dragged_Piece,Piece.king):
                        if (initial_row,initial_col) == (7,4):
                            if(new_row, new_column) == (7,2):
                                rook_piece = board[7][0]
                                board[7][3] = rook_piece
                                board[7][0] = 0
                            if (new_row,new_column) == (7,6):
                                rook_piece = board[7][7]
                                board[7][5] = rook_piece
                                board[7][7] = 0

                        if (initial_row,initial_col) == (0,4):
                            if(new_row, new_column) == (0,2):
                                rook_piece = board[0][0]
                                board[0][3] = rook_piece
                                board[0][0] = 0
                            if (new_row,new_column) == (0,6):
                                rook_piece = board[0][7]
                                board[0][5] = rook_piece
                                board[0][7] = 0
                    #change pawn to queen or knight            
                    if new_row==7 or new_row==0:
                        if Piece.is_type(board[new_row][new_column],Piece.pawn):
                            board[new_row][new_column] = Piece.queen
                            pawn_change  = True
                            selected_row,selected_col = new_row,new_column
                            


                else:
                    board[initial_row][initial_col] = Dragged_Piece

                selected_square = None
                Dragging = False
                Dragged_Piece = None
                
                #check if rook or king is moved if a legit move is made 
                #will only change turn when a move is made to a different square
                if (new_row, new_column)!=(initial_row,initial_col) and is_legal:
                    gameState.check_if_rook_moved(initial_row,initial_col)
                    if not pawn_change:
                        gameState.change_current_color()

            else:
                is_highlight = False
                board[initial_row][initial_col] = Dragged_Piece
                selected_square = None
                Dragging = False
                Dragged_Piece = None

                
  


                
            

    p.display.update()
p.quit()

# if __name__ == "__main__":
#     main()

