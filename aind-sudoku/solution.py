import pprint
pp = pprint.PrettyPrinter(depth=6)

assignments = []

# Set constants.
LETTERS = 'ABCDEFGHI'
DIGITS  = '123456789'
SIZE = len(LETTERS)

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a + b for a in A for b in B]

# Calculate units.
BOXES = cross(LETTERS,DIGITS)
SQUARES = [ cross(LETTERS[i:i+3],DIGITS[j:j+3]) for i in range(0, len(LETTERS), 3) for j in range(0, len(DIGITS), 3) ]
DIAGONALS = [[letter + digit for letter, digit in zip(LETTERS, DIGITS)], [letter + digit for letter, digit in zip(LETTERS[::-1], DIGITS)]]
UNITS = [cross(r,DIGITS) for r in LETTERS] + [cross(LETTERS,c) for c in DIGITS] + SQUARES + DIAGONALS

# Calculate peers.
ROW_PEERS = {box: cross(box[0], DIGITS.replace(box[1], '')) for box in BOXES}
COL_PEERS = {box: cross(LETTERS.replace(box[0], ''), box[1]) for box in BOXES}
SQUARE_PEERS = {box: list(filter(lambda x: x != box, square)) for square in SQUARES for box in square}
FIRST_DIAGONAL_PEERS = {box: list(filter(lambda x: x != box, DIAGONALS[0])) for box in DIAGONALS[0]}
SECOND_DIAGONAL_PEERS = {box: list(filter(lambda x: x != box, DIAGONALS[1])) for box in DIAGONALS[1]}
DIAGONAL_PEERS = {box: FIRST_DIAGONAL_PEERS.setdefault(box, []) + SECOND_DIAGONAL_PEERS.setdefault(box, []) for box in BOXES}
PEERS = {box: list(set(ROW_PEERS[box] + COL_PEERS[box] + SQUARE_PEERS[box] + DIAGONAL_PEERS[box])) for box in BOXES}

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    for unit in UNITS:
        # Identify naked twins.
        boxes = {box:values[box] for box in unit if len(values[box]) == 2}
        nakedTwins = set(value for box,value in boxes.items() if list(boxes.values()).count(value) == 2)

        # Remove naked twins from other boxes in the same unit.
        for box in unit:
            if values[box] not in nakedTwins:
                val = ''.join([x for x in values[box] if x not in ''.join(list(nakedTwins))])
                assign_value(values, box, val)
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    return {BOXES[idx]: grid[idx].replace('.', DIGITS) for idx in range(len(grid))}

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    if values == False:
        print('No solution to the sudoku could be found.')
        return

    boxwidth = max(list(map(len, values.values())))
    colCount = rowCount = len(DIGITS) + 1
    rowSeperator = '-+-{}\n'.format(('-' * boxwidth + '-+-') * colCount)

    # Display column header.
    output = rowSeperator
    output += ' | ' + ' | '.join([c.center(boxwidth) for c in ' ' + DIGITS]) + ' | \n'
    output += rowSeperator

    for r in LETTERS:
        # Display row header.
        output += ' | ' + r.center(boxwidth) + ' | '

        # Display boxes.
        for c in DIGITS:
            box = r+c
            output += values[box].center(boxwidth)
            output += ' | '

        output += '\n' + rowSeperator
    print(output)

def eliminate(values):
    """
    For every solved box removes it's digits from the boxes' peers.
    Args:
        values(dict): The sudoku in dictionary form. Will be modified in place.
    Returns:
        The reduced sudoku in dictionary form.
    """
    # Find all boxes that have been solved.
    solvedValues = {k:v for k,v in values.items() if len(v) == 1}
    for box, val in solvedValues.items():
        # Remove the box solution for all it's peers.
        for peer in PEERS[box]:
            values = assign_value(values, peer, values[peer].replace(val, ''))
    return values

def only_choice(values):
    """
    For every digit that is only possible in one of a unit's boxes, the box will be set to that digit.
    Args:
        values(dict): The sudoku in dictionary form. Will be modified in place.
    Returns:
        The reduced sudoku in dictionary form.
    """
    for unit in UNITS:
        for digit in DIGITS:
            # Consider all boxes within this unit that has the digit as a possible solution.
            choices = [box for box in unit if digit in values[box]]
            # If there's only one box that can contain this value, assign it.
            if len(choices) == 1:
                values = assign_value(values, choices[0], digit)
    return values

def reduce_puzzle(values):
    """
    Repeatedly apply elimination, only_choice and naked_twins until there is no more progress.
    Args:
        values(dict): The sudoku in dictionary form. Will be modified in place.
    Returns:
        The reduced sudoku in dictionary form.
    """
    progress = True
    while progress:
        # Count the number of currently solved boxes.
        solved_boxes_before = len([val for val in values.values() if len(val) == 1])
        # Apply elimination, naked_twins and only_choice.
        values = eliminate(values)
        values = naked_twins(values)
        values = only_choice(values)
        # If no boxes have been solved this iteration then abort.
        solved_boxes_after = len([val for val in values.values() if len(val) == 1])
        progress = solved_boxes_before != solved_boxes_after
        # Check if we removed an invalid field and end up with no possible solution.
        if list(values.values()).count('') > 0:
            return False
    return values

def search(values):
    """
    Reduces the sudoku as far as possible and then uses depth first search to find
    a solution among the remaining possibilities.
    Args:
        values(dict): The sudoku in dictionary form.
    Returns:
        The reduced sudoku in dictionary form.
    """
    values = reduce_puzzle(values)
    # If we ended up with an invalid solution, return false.
    if values is False:
        return False
    # If the sudoku has been solved, return the solution.
    if all(len(values[box]) == 1 for box in BOXES):
        return values
    # Find the box with the least possible solutions.
    boxesNotSolved = filter(lambda t: len(t[1]) > 1, values.items())
    box,n = min(boxesNotSolved, key=lambda t: len(t[1]))
    # And try to solve each of them.
    for digit in values[box]:
        valuesCopy = values.copy()
        assign_value(valuesCopy, box, digit)
        solution = search(valuesCopy)
        if solution is not False:
            return solution
    return False

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    # Apply search until the sudoku has been solved.
    while len([v for v in values.values() if len(v) > 1]) > 0:
        values = search(values)
        if not values:
            return False
    return values

def assertCorrectSolution(solution):
    """
    Asserts that a solution contains no duplicates within units.
    Args:
        solution(dict): The solved sudoku in dictionary form.
    """
    assert(solution != False)
    for unit in UNITS:
        # Check for duplicates within the unit.
        for box in unit:
            duplicates = [peer for peer in unit if box != peer and solution[peer] == solution[box]]
            assert(len(duplicates) == 0)

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    # Solve the sudoku.
    solution = solve(diag_sudoku_grid)
    # Display the solution.
    display(solution)
    # Assert that the solution is correct.
    assertCorrectSolution(solution)

    # Try to visualize the solution with pygame.
    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
