import numpy as np
from queue import PriorityQueue

def create_random_matrix(size_puzzle):
    while True:
        matrix = np.arange(size_puzzle * size_puzzle)
        np.random.shuffle(matrix)
        matrix = matrix.reshape((size_puzzle, size_puzzle))
        if isSolvable(matrix):
            return matrix

def isSolvable(puzzle):
    puzzle_1d = puzzle.flatten()
    parity = 0
    gridWidth = int(len(puzzle_1d) ** 0.5)
    row = 0
    blankRow = 0

    for i in range(len(puzzle_1d)):
        if i % gridWidth == 0:
            row += 1
        if puzzle_1d[i] == 0:
            blankRow = row
            continue
        for j in range(i + 1, len(puzzle_1d)):
            if puzzle_1d[i] > puzzle_1d[j] and puzzle_1d[j] != 0:
                parity += 1

    if gridWidth % 2 == 0:
        if blankRow % 2 == 0:
            return parity % 2 == 0
        else:
            return parity % 2 != 0
    else:
        return parity % 2 == 0

def target_state(size_puzzle):
    numbers = np.arange(1, size_puzzle * size_puzzle + 1)
    target = numbers.reshape((size_puzzle, size_puzzle))
    target[size_puzzle - 1, size_puzzle - 1] = 0

    return target

# Hàm tính khoảng cách Manhattan giữa hai điểm trên ma trận
def manhattan_distance(start, goal):
    distance = 0
    size = len(start)
    for i in range(size):
        for j in range(size):
            if start[i][j] != 0:  # Không tính khoảng cách nếu là ô trống
                # Tìm vị trí của start[i][j] trong goal
                x, y = np.where(goal == start[i][j])
                distance += abs(i - x) + abs(j - y)
    return distance

# Hàm trả về các hướng di chuyển có thể từ một trạng thái
def get_moves(matrix):
    moves = []
    blank_position = np.where(matrix == 0)
    row, col = blank_position[0][0], blank_position[1][0]
    size = len(matrix)

    if row > 0:
        up_move = np.copy(matrix)
        up_move[row][col], up_move[row - 1][col] = up_move[row - 1][col], up_move[row][col]
        moves.append(up_move)
    if row < size - 1:
        down_move = np.copy(matrix)
        down_move[row][col], down_move[row + 1][col] = down_move[row + 1][col], down_move[row][col]
        moves.append(down_move)
    if col > 0:
        left_move = np.copy(matrix)
        left_move[row][col], left_move[row][col - 1] = left_move[row][col - 1], left_move[row][col]
        moves.append(left_move)
    if col < size - 1:
        right_move = np.copy(matrix)
        right_move[row][col], right_move[row][col + 1] = right_move[row][col + 1], right_move[row][col]
        moves.append(right_move)

    return moves

def a_star(start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start.tobytes()))  # Store the bytes representation of the start state
    came_from = {}
    cost_so_far = {start.tobytes(): 0}  # Store the bytes representation of the start state
    came_from[start.tobytes()] = None

    while not frontier.empty():
        _, current_state_bytes = frontier.get()
        current_state = np.frombuffer(current_state_bytes, dtype=start.dtype).reshape(start.shape)

        if np.array_equal(current_state, goal):
            break

        for next_state in get_moves(current_state):
            new_cost = cost_so_far[current_state_bytes] + 1
            next_state_bytes = next_state.tobytes()
            if next_state_bytes not in cost_so_far or new_cost < cost_so_far[next_state_bytes]:
                cost_so_far[next_state_bytes] = new_cost
                priority = new_cost + manhattan_distance(next_state, goal)
                frontier.put((priority, next_state_bytes))
                came_from[next_state_bytes] = current_state_bytes

    path = []
    current_state_bytes = goal.tobytes()
    while current_state_bytes is not None:
        current_state = np.frombuffer(current_state_bytes, dtype=goal.dtype).reshape(goal.shape)
        path.append(current_state)
        current_state_bytes = came_from[current_state_bytes]
    path.reverse()

    return path

# Hàm in ra các bước giải
def print_solution(path):
    print("Các bước giải trò chơi:")
    for i, move in enumerate(path):
        print(f"Bước {i}:")
        print(move)
        print()

# Hàm in ra số bước giải trò chơi
def solution_step(path):
    return len(path) - 1 # Trừ đi 1 vì trạng thái ban đầu cũng được tính


# Hàm chính
def main():
    size_puzzle = int(input("Nhập cỡ ô chữ: "))
    start_matrix = create_random_matrix(size_puzzle)
    target_matrix = target_state(size_puzzle)

    # print("Trạng thái ban đầu:")
    # print(start_matrix)
    # print("Trạng thái mục tiêu:")
    # print(target_matrix)

    path = a_star(start_matrix, target_matrix)
    # print_solution(path)
    print(solution_step(path))

# Chạy chương trình
if __name__ == "__main__":
    main()
