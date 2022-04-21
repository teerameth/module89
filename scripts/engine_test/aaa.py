import numpy as np
import chess
import time
fen_set = np.load("fen_set.npy")
move_list_set = np.load("move_list_set.npy", allow_pickle=True)
fen_set_new = []
move_list_set_new = []
for i in range(len(move_list_set)):
    print(i)
    L = move_list_set[i]
    L_new = []
    for j in range(len(L)):
        fen = L[j]
        print(fen)
        # fen.replace(chr(10), '')
        # fen.replace(chr(39), '')
        l = [ord(fen[g]) for g in range(len(fen))]
        if 39 in l: l.remove(39)
        string = ""
        for ll in range(len(l)): string += chr(l[ll])
        L_new.append(string)
        board = chess.Board(string)
    move_list_set_new.append(L_new)
np.save("move_list_set", move_list_set_new)