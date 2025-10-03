"""
Programming Assignment 1 — Sudoku Solver (A* + DFS)

READ ME FIRST
-------------
Complete the TODOs below. Do NOT change function names or signatures.
Your solver must:
  1) Load puzzles (from predefined list or "Sudoku_Puzzles.txt")
  2) Validate moves (row, col, 3x3 box)
  3) Generate successors by filling ONE empty cell with all valid digits
  4) Check the goal condition (no zeros; all rows/cols/boxes valid)
  5) Implement A* with a heuristic based on "Most Constrained Variable" (MCV)
  6) Implement DFS (uninformed) and compare to A*

Optional (extra credit in write-up):
  - Least Constraining Value (LCV) ordering for values
  - BFS for comparison on easy puzzles
  - Collect stats: time, nodes expanded, max frontier
"""

import copy
from operator import sub, truediv
from pdb import run
import random
import heapq
import time
import os
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Set

# SubBoxesRows = Enum('SubBoxRows', [('Top', 1),('Middle',2),('Bottom',3)])

# SubBoxesCols = Enum('SubBoxCols', [('Left', 1), ('Middle',2), ('Right',3)])

# Get workspace directory
WORKSPACE_DIR = os.getcwd()  # Current working directory
# Alternative: WORKSPACE_DIR = Path.cwd()  # Using pathlib
# Alternative: WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))  # Script directory

Grid = List[List[int]]

# ========================
# Step 1: Load Sudoku Puzzles
# ========================

def load_sudoku() -> Grid:
    puzzles = {
        "puzzle1": [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ],
        "puzzle2": [
            ],
    }
    """
    Returns a randomly chosen Sudoku puzzle from a predefined list.

    TODO:
      - Create a Python list named `puzzles` containing multiple 9x9 grids.
      - Represent each grid as a list of lists of ints, where 0 means empty.
      - Return a random choice from that list.

    Example row format for one puzzle (9 lines of 9 digits):
      [5, 3, 0, 0, 7, 0, 0, 0, 0]
    """
    '''
    # TODO: build and return a random puzzle from a predefined list
    raise NotImplementedError("load_sudoku: return a random predefined puzzle")
    '''


def load_sudoku_from_file(filename: str = "Sudoku_Puzzles.txt") -> Grid:
  finalGrid = Grid
  try:
    # Use workspace directory to construct full path
    filepath = os.path.join(WORKSPACE_DIR, filename)
    with open(filepath, 'r') as file:
      puzzles = file.read()
      puzzles = puzzles.split('\n')
      puzzleChoices = []
      currentPuzzle = []
      for puzzleRow in puzzles:
        if (puzzleRow == ''):
          puzzleChoices.append(currentPuzzle)
          currentPuzzle = []
        else:
          currentPuzzle.append(puzzleRow)
      gridChoices = []
      tempGrid = []

      for puzzle in puzzleChoices:
        for puzzleRow in puzzle:
          
          tempGrid.append([int(char) for char in puzzleRow if char.isdigit()])
        gridChoices.append(tempGrid)
        tempGrid = []
      
      return random.choice(gridChoices)
      # puzzles = [[int(num) for num in puzzle] for puzzle in puzzles]
      # return random.choice(puzzles)
  except FileNotFoundError:
    raise FileNotFoundError("load_sudoku_from_file: file not found")
  except ValueError:
    raise ValueError("error reading from file...")
  except TypeError:
    raise TypeError("error reading from file...")
  except Exception as e:
    raise Exception("unknown error reading from file...")
    
    """
    Loads Sudoku puzzles from a text file and returns a random one.

    File format requirements:
      - Each puzzle = 9 lines, each line has 9 digits (0-9). '0' means empty.
      - Puzzles separated by a blank line.
      - Lines starting with '#' are comments and should be ignored.

    TODO:
      - Open the file and read all lines.
      - Parse lines into 9-line blocks, skipping comments and blank separators.
      - Convert each block into a 9x9 integer grid; validate shapes.
      - Accumulate all parsed puzzles in a list.
      - Return random.choice(puzzles).

    Raise:
      - FileNotFoundError if the file is missing.
      - ValueError if no valid puzzles are found or format is incorrect.
    """
    # TODO: implement file loading as described
    raise NotImplementedError("load_sudoku_from_file: parse file and return a random puzzle")


# ========================
# Step 2: Validity Check Functions
# ========================

def in_row(grid: Grid, row: int, num: int) -> bool:
    return num in grid[row]

def in_col(grid: Grid, col: int, num: int) -> bool:
    # """
    # Returns True iff `num` already appears in column `col`.

    # TODO:
    #   - Scan the given column and check if num is present.
    # """
    # # TODO
    # raise NotImplementedError("in_col: check column membership")
    in_col = False
    for row in grid:
      in_col = (in_col or num == row[col])
    return in_col


def in_box(grid: Grid, row: int, col: int, num: int) -> bool:
  # subBoxRangeTup = (
  #   ( (row // 3) * 3 , ((row // 3) * 3) + 2 ),
  #   ( (col // 3) * 3 , ((col // 3) * 3) + 2 )
  # )
  subBoxRowFloor= (row//3) * 3
  subBoxColFloor= (col//3) * 3

  rowCounter = subBoxRowFloor
  while rowCounter <= subBoxRowFloor + 2: #check all rows in the range
    colCounter = subBoxColFloor
    while colCounter <= subBoxColFloor + 2:
      if grid[rowCounter][colCounter] == num:
        return True
      colCounter+=1
    rowCounter+=1
  return False    
  
    # """
    # Returns True iff `num` already appears in the 3x3 subgrid containing (row, col).

    # TODO:
    #   - Compute top-left corner of the 3x3 box: (row//3)*3, (col//3)*3
    #   - Check the 9 cells of that box for `num`.
    # """
    # # TODO
    # raise NotImplementedError("in_box: check 3x3 box membership")


def is_valid(grid: Grid, row: int, col: int, num: int) -> bool:
    """
    Returns True iff placing `num` at (row, col) is valid and the cell is currently empty.

    TODO:
      - If grid[row][col] != 0, return False (cell already filled).
      - Return False if `num` is in the same row, column, or 3x3 box.
      - Otherwise return True.
    """
    if grid[row][col] != 0 or in_col(grid,col,num) or in_row(grid,row,num) or in_box(grid,row,col,num):
      return False
    return True
    

def legal_values(grid: Grid, row: int, col: int) -> List[int]:
    """
    Returns a list of all valid digits (1..9) that can be placed at (row, col).

    TODO:
      - If the cell is not empty, return [].
      - Test 1..9 with is_valid and collect valid ones.
    """
    possibleVals = []
    if grid[row][col] != 0:
      return possibleVals
    for i in range(1,10):
      if is_valid(grid, row, col, i):
        possibleVals.append(i)
    return possibleVals
    
# ========================
# Step 3: Successor Function
# ========================

def find_first_empty(grid: Grid) -> Optional[Tuple[int, int]]:
    """
    Returns the (row, col) of the first empty cell in row-major order, or None if full.

    TODO:
      - Scan grid rows 0..8 and cols 0..8; return the first cell with value 0.
      - If none, return None.
    """
    for row in range(len(grid)):
      for col in range(len(grid[row])):
        if grid[row][col] == 0:
          return (row,col)
    return None
    

def find_most_constrained_cell(grid: Grid) -> Optional[Tuple[int, int, List[int]]]:
    """
    Returns (row, col, domain) for the empty cell with the fewest legal values (MCV),
    or None if the grid is full.

    TODO:
      - For each empty cell, compute its legal_values (domain).
      - Track the smallest domain size and remember that position and domain.
      - If multiple cells tie, any is fine (document your tie-breaking).
      - If no empty cells, return None.
    """
    smallestDomain = []
    smallestDomainSize = 1000
    smallestDomainPos = (10,10)
    for row in range(len(grid)):
      for col in range(len(grid[row])):
        if grid[row][col] == 0: #if empty
          currDomain = legal_values(grid,row,col)
          if (len(currDomain) < smallestDomainSize):
            smallestDomain = currDomain
            smallestDomainSize = len(currDomain)
            smallestDomainPos = (row,col)
    if smallestDomain != []:
       return (smallestDomainPos[0],smallestDomainPos[1],smallestDomain)  
    return None    

def get_successors(grid: Grid, use_mcv: bool = True, use_lcv: bool = False) -> List[Grid]:
  successors = []
  #lcv is the 
  if use_mcv:
    result = find_most_constrained_cell(grid)
    if result == None:
      return successors
    (row, col, domain) = result
    candidates = domain
  
  else:
    empty_cell_pos = find_first_empty(grid)
    if empty_cell_pos == None:
      return successors
    row, col = empty_cell_pos
    candidates = legal_values(grid, row, col)
    
  if use_lcv:
    raise NotImplementedError("get_successors: order candidates using LCV")
    # maybe implemnt later...

  for value in candidates: #for each value in the possible candidates create states with the values and add to succ
    new_grid = copy.deepcopy(grid)
    new_grid[row][col] = value
    successors.append(new_grid)
  return successors
  
  
  """
  Generates next states by filling ONE empty cell with each valid value.

  Parameters:
    - use_mcv: If True, choose the Most Constrained Variable (fewest legal values).
                If False, choose the first empty cell in row-major order.
    - use_lcv: If True, order candidate values using Least Constraining Value (optional).

  TODO:
    - Choose (r, c) either by MCV or first-empty.
    - Compute candidate values via legal_values(grid, r, c).
    - If use_lcv is True: order candidates by how few constraints they impose
      on neighbors (row, column, box). (Lower impact first.)
    - For each candidate value v:
        * Copy the grid
        * Set new_grid[r][c] = v
        * Append to the successor list
    - Return the list of new grids.
  """
  # TODO
  raise NotImplementedError("get_successors: generate valid child states")


# ========================
# Step 4: Goal Check
# ========================

def has_duplicates(values: List[int]) -> bool:
    """
    Returns True if `values` contains duplicates among NON-ZERO digits.

    TODO:
      - Filter out zeros.
      - Compare length to length of set.
    """
    runningList = []
    for val in values:
      if val != 0:
        if val in runningList:
          return True
        runningList.append(val)
    return False

def is_goal(grid: Grid) -> bool:
  #check if zeros
  subBoxRow = [[],[],[]]
  cols = [[], [], [], [], [], [], [], [], []]
  rowCount = 1
  for row in grid:
    if has_duplicates(row):
      return False
    for col in range(row):
      if row[col] == 0:
        return False
      cols[col] = row[col]
      subBoxRow[col//3]
    if rowCount+1 % 3 == 1: #if the next row is a change for subBox
      for subBox in subBoxRow:
        if has_duplicates(subBox):
          return False
      subBoxRow = [[],[],[]] #reset subbox row
    rowCount += 1
  
  for col in cols:
    if has_duplicates(col):
      return False
      
  return True
  """
  Returns True iff:
    - There are NO zeros in the grid, and
    - Every row, column, and 3x3 subgrid contains digits 1..9 with no duplicates.

  TODO:
    - Check every row (use has_duplicates).
    - Check every column (build a list and use has_duplicates).
    - Check each 3x3 box (build a list and use has_duplicates).
    - Ensure no zeros remain.
  """




# ========================
# Step 5: A* Search
# ========================

def heuristic(grid: Grid) -> int:
  zeroCount = 0
  for row in grid:
    for item in row:
      if item == 0 : zeroCount+=1
  return zeroCount

  """
  Heuristic for A*.
  Baseline option: number of empty cells (lower is better).

  TODO:
    - Count and return how many cells are 0.
    - (Optional) Propose/document an alternative admissible heuristic in your write-up.

  # TODO
  raise NotImplementedError("heuristic: estimate remaining work")
  """

def grid_to_key(grid: Grid):
    return tuple(tuple(row) for row in grid)
    """
    Converts the grid to an immutable, hashable key for visited/frontier sets.

    TODO:
      - Return a tuple of tuples representing the grid.
    """
    # TODO
    raise NotImplementedError("grid_to_key: create a hashable state key")


def a_star_sudoku(start: Grid,
                  use_mcv: bool = True,
                  use_lcv: bool = False,
                  time_limit: Optional[float] = None):
  
    nodesExpanded = 0
    maxFrontier = 0
    
    frontier = []
    tie_breaker = 0
    
    g_start = 0
    h_start = heuristic(start)
    f_start = g_start + h_start
    
    heapq.heappush(frontier, (f_start, g_start, tie_breaker, start))
    
    visited = set()
    visited.add(grid_to_key(start))
    
    startTime = time.time()
    
    while frontier:
      #update values...
      maxFrontier = max(maxFrontier, len(frontier))
      #current node is popped from heapq, which has lowest f value...
      f_curr, g_curr, _ , grid_curr = heapq.heappop(frontier)
      tie_breaker += 1
      
      #check goal is the current grid
      if is_goal(grid_curr):
        return (grid_curr, 
                {"success": True, 
                 "nodes_expanded": nodesExpanded,
                 "max_frontier": maxFrontier,
                 "time": time.time() - startTime}
                )
      #not goal, so lets expand more...
      nodesExpanded += 1
      succesors = get_successors(grid_curr, use_mcv, use_lcv)
      
      for state in succesors:
        state_key = grid_to_key(state)
        if state_key in visited:
          continue
        else:
          visited.add(state_key)
        
        #calc scores for this state
        state_g = g_curr + 1
        state_h = heuristic(state_key)
        state_f = state_g + state_h
        
        # add state with scores to frontier
        heapq.heappush(frontier, (state_f, state_g, tie_breaker, state))
    #if we get here we failed
    return None, {
      "success": False,
      "nodes_expanded": nodesExpanded,
      "max_frontier": maxFrontier,
      "time": time.time() - startTime}
      
        
    """
    Solves Sudoku using A* search.
    Returns: (solution_grid or None, stats_dict)

    Required behavior:
      - Use a priority queue (heapq) of tuples: (f, g, tie_breaker, grid)
      - f = g + h, where g is steps/assignments and h is heuristic(grid)
      - Pop the lowest f; if goal -> return solution and stats
      - Expand using get_successors with MCV/LCV flags
      - Maintain a visited set using grid_to_key to avoid re-expansion
      - Track stats: nodes_expanded, max_frontier, elapsed_time
      - Respect time_limit if provided (stop and return with success=False)

    TODO:
      - Implement the A* loop described above.
      - Return a stats dict like:
        {
          "success": bool,
          "nodes_expanded": int,
          "max_frontier": int,
          "time": float
        }
    """
    # TODO
    raise NotImplementedError("a_star_sudoku: implement A* search")


# ========================
# Step 6: Depth-First Search (DFS)
# ========================

def solve_dfs(start: Grid,
              use_mcv: bool = True,
              use_lcv: bool = True,
              depth_limit: Optional[int] = None,
              time_limit: Optional[float] = None):
    """
    Solves Sudoku using uninformed Depth-First Search (stack-based or recursion).
    Returns: (solution_grid or None, stats_dict)

    Required behavior:
      - Use a stack (LIFO) or recursion for DFS.
      - Enhance with MCV (variable choice) and optional LCV (value ordering).
      - Maintain a visited set keyed by grid_to_key to avoid duplicate expansions.
      - Track stats: nodes_expanded, max_frontier (stack max size), elapsed_time.
      - Respect depth_limit and/or time_limit; if hit, return success=False with reason.

    TODO:
      - Implement DFS using get_successors with chosen flags.
      - Stop and return solution as soon as is_goal(grid) is True.
    """
    # TODO
    raise NotImplementedError("solve_dfs: implement DFS search")


# ========================
# (Optional) Step 7: Breadth-First Search (BFS)
# ========================

def solve_bfs(start: Grid):
    """
    OPTIONAL: Solve Sudoku using BFS (queue).
    WARNING: BFS can be very memory-intensive; only try on easy puzzles.

    TODO (optional):
      - Implement BFS with a queue (FIFO).
      - Keep visited states and track stats similar to A* / DFS.
    """
    # OPTIONAL
    raise NotImplementedError("solve_bfs (optional): implement if you choose")


# ========================
# Utilities & Demo
# ========================

def print_sudoku(grid: Grid) -> None:
    """
    Pretty-prints a 9x9 Sudoku grid.

    TODO:
      - Print rows with '.' for zeros.
      - Add separators every 3 rows and 3 columns to improve readability.
    """
    # TODO
    raise NotImplementedError("print_sudoku: format and print the board")


if __name__ == "__main__":
  
    """
    Demo harness for your solver.

    TODO:
      - Uncomment ONE of the loaders below.
      - Print the loaded puzzle.
      - Run A* and DFS; print their stats and (if solved) the solution.
      - For your experiments, time multiple puzzles and compare:
          * time (seconds)
          * nodes expanded
          * max frontier size
      - Save or screenshot before/after boards for your write-up.
    """

    # Choose ONE loader:
    # puzzle = load_sudoku()
    puzzle = load_sudoku_from_file("Sudoku_Puzzles.txt")
    print(puzzle)

    print(in_box(puzzle, 0, 1, 5))

    # print("Loaded Sudoku:")
    # print_sudoku(puzzle)

    # A* solve:
    # a_solution, a_stats = a_star_sudoku(puzzle, use_mcv=True, use_lcv=False, time_limit=10)
    # print("A* stats:", a_stats)
    # if a_solution:
    #     print("Solved (A*):")
    #     print_sudoku(a_solution)

    # DFS solve:
    # d_solution, d_stats = solve_dfs(puzzle, use_mcv=True, use_lcv=True, time_limit=10)
    # print("DFS stats:", d_stats)
    # if d_solution:
    #     print("Solved (DFS):")
    #     print_sudoku(d_solution)

    # (Optional) BFS:
    # b_solution, b_stats = solve_bfs(puzzle)
    # print("BFS stats:", b_stats)
    # if b_solution:
    #     print("Solved (BFS):")
    #     print_sudoku(b_solution)
