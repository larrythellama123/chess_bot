THERE ARE LIKE A thousand things wrong with just moving the pieces

after the first moves the queens are not able to move

other sliding pieces are unable to move



///////////////////////////////////////////////////////////////

eval issues:
king is primarily the one moving
Black pieces pick illogical decisions

issues with minmax:
the black king positions sometimes seems to be removed from the list even though it was the best move
black sliding pieces are able to pass thru other pieces
if the black king is in check, it seems as if the white king is in check as well


issues with move gen:
pawns dont become other pieces when they reach the end 


fix func and style of all code later

to increase the speed of minmax: cache already evaluated positions

king can eat something that is putting it in check, even though eating it puts in it check as well