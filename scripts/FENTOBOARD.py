fen = "rnbqkbnr/p1pppp1p/1P6/5Pp1/8/8/1PPPP1PP/RNBQKBNR"
move = "f5g6"
board = [[]]
board_row = 0
alphabet = "abcdefgh"
for i in fen:
    if i == "/":
        board_row += 1
        board.append([])
        continue
    if i.isnumeric():
        for j in range(int(i)):
            board[board_row].append('*')
        continue
    board[board_row].append(i)

for i in board:
    print(i)

if board[7-(int(move[1])-1)][alphabet.index(move[0])].lower() == "k":
    if abs(alphabet.index(move[0]) - alphabet.index(move[2])) == 2:
        print("castling")
        if alphabet.index(move[0]) - alphabet.index(move[2]) == 2:
            #MOVE 0 kingx -1

            # self.narwhal.MoveChess(dl[1], dl[2], dl[3])
            # self.move_buffer[0], self.move_buffer[1], self.move_buffer[2] = dl[4], dl[5], dl[6]
            # self.waitMove = True
            # self.time_delay = time.time()
            pass
        elif alphabet.index(move[0]) - alphabet.index(move[2]) == -2:
            pass
            #MOVE 7 kingx +1

elif board[7-(int(move[1])-1)][alphabet.index(move[0])].lower() == "p":
    if board[7-(int(move[3])-1)][alphabet.index(move[2])].lower() == "*":
        print("enpassant")
        print(((int(move[1])))*8 + (alphabet.index(move[2])))
        print(((int(move[1])))*8 + (alphabet.index(move[0])))
        print(((int(move[3]))) * 8 + (alphabet.index(move[2])))
        board[7-(int(move[1])-1)][alphabet.index(move[2])] = "69" #REMOVE
        #BUFFER = MOVE

for i in board:
    print(i)
