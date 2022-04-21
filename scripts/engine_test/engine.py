import chess.engine
import numpy as np
from tqdm import tqdm

engine = chess.engine.SimpleEngine.popen_uci("stopfish")

fen_set = np.load("fen_set.npy")
move_list_set = np.load("move_list_set.npy", allow_pickle=True)

counter = {1:0, 3:0, 5:0, 7:0, 9:0, 11:0, 13:0}
for move_list in move_list_set:
    if len(move_list) not in counter.keys(): counter[len(move_list)] = 0
    counter[len(move_list)] += 1
print(counter)

score_counter = {1:0, 3:0, 5:0, 7:0, 9:0, 11:0, 13:0}
for i in tqdm(range(len(fen_set))):
    move_list = move_list_set[i]
    # print(len(move_list))
    fen = fen_set[i]
    # print(fen)
    my_color = fen[fen.find(" ")+1]
    board = chess.Board(fen)
    failed = False
    for step in range(len(move_list)):
        if step%2==1: # My turn
            result = engine.play(board, chess.engine.Limit(time=3))
            board.push(result.move)
            # print(board)
            if board.fen() != move_list[step]: failed = True
        else: # Puzzle turn
            board = chess.Board(move_list[step])
        if failed: break
    if len(move_list) not in score_counter.keys(): score_counter[len(move_list)] = 0
    if not failed: score_counter[len(move_list)] += 1

    # while not board.is_game_over():
    #     result = engine.play(board,print(counter) chess.engine.Limit(time=3))
    #     board.push(result.move)
    #     print(board)
    #     if board.fen() == move_list_set[i][1]:
    #         print("AAA")
    # print("Game over")
engine.quit()
print(counter)
print(score_counter)