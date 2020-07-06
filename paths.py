import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import time

# To be used to keep track of the maze
# that was the hardest with respect
# to these metrics
largestMaximalNodes = 0
deepestFringe = 0
deepestFringeTime = 0

blocked = 0
clear = 1
onFire = 2
path = 3
labels = "bcfp"

class maze:

    def __init__(self, dim, p, q=None, difficulty=None, fire=False):

        global blocked, clear, onFire, path
        # Create 2D array to represent graph
        test = np.random.randint(10, size=(dim, dim))

        # Calculating constants to use in path randomization
        numerator, denominator = maze.calculateRatio(p)

        # Traversing 2D array to populate it based on the probability parameter
        for i in range(0, dim):
            for j in range(0, dim):
                if random.randint(1, denominator) <= numerator:
                    test[i, j] = blocked
                else:
                    test[i, j] = clear

        if fire == True and q != None:
            # Pick a spot for the fire to start
            fireLocation = random.randint(1, (dim**2)-1)

            test[fireLocation//dim, fireLocation%dim] = onFire
            test[0,0] = clear
            test[dim-1, dim-1] = clear
            #Could be future parameter
            self.maze = test
            self.qRatio = maze.calculateRatio(q)
            self.q = q
            self.dim = dim
            return
        else:
            test[0,0] = clear
            test[dim-1, dim-1] = clear
            self.q = None
            self.maze = test
            self.fireMap = None
            self.dim = dim
            return

    def calculateRatio(probability):
        # Calculating constants to use in path randomization
        numDigits = len(str(probability).strip("0").replace(".", ""))
        numerator = int(probability*(10**numDigits))
        denominator = 10**numDigits
        return(numerator, denominator)

    def updateFire(self):
        global blocked, clear, onFire, path

        if self.q == 0:
            return None
        # 1 - (1-q)^k
        # Traverse maze and search for spaces that could ignite
        newFires = []
        for i in range(0, self.dim):
            for j in range(0, self.dim):

                if self.maze[i, j] != clear and self.maze[i, j] != path:
                    continue

                count = 0

                upperI = upperJ = lowerI = lowerJ = False

                if i == 0 : lowerI = True
                if j == 0 : lowerJ = True
                if i == self.dim - 1 : upperI = True
                if j == self.dim - 1 : upperJ = True

                if not lowerI:
                    if self.maze[i-1, j] == onFire : count += 1
                if not lowerJ:
                    if self.maze[i, j-1] == onFire : count += 1
                if not upperI:
                    if self.maze[i+1, j] == onFire : count += 1
                if not upperJ:
                    if self.maze[i, j+1] == onFire : count += 1

                if count == 0:
                    continue

                numerator, denominator = maze.calculateRatio(1 - (1-self.q)**count)
                if random.randint(0, denominator) <= numerator:
                    newFires.append((i, j))

        for fire in newFires:
            self.maze[fire[0], fire[1]] = onFire

    def display(self, title=None):
        fig, ax = plt.subplots()
        global labels, blocked, clear, onFire, path
        ax.matshow(self.maze, cmap=plt.cm.Blues)

        for i in range(self.dim):
            for j in range(self.dim):
                c = self.maze[j,i]
                ax.text(i, j, labels[c], va='center', ha='center')
        ax.set_xlabel("b = blocked,\nc = clear\n f = fire\np = path\n")
        # plt.show()
        plt.savefig(title)

class starNode():

    def __init__(self, coor, pred):
        self.g = 0
        self.h = 0
        self.cost = 0
        self.coordinates = coor
        self.predecessor = pred

def aStar(maze, mode = None):

    # Data structures / variables
    valid = []
    closed = []
    shortestPath = []
    target = (maze.dim-1, maze.dim-1)
    global clear, blocked, onFire, path


    # Initial spot at top left corner
    valid.append(starNode((0,0), None))

    # Run until we find the target node or run out of valid nodes to visit
    while valid != []:

        # Grab node with smallest cost estimate
        present = min(valid, key=lambda temp: temp.cost)

        valid.remove(present)
        closed.append(present.coordinates)

        if maze.q != None:
            maze.updateFire()

        if maze.maze[present.coordinates[0], present.coordinates[1]] == onFire:
            return "burned"

        if present.coordinates == target:
            # We have arrived at the target node, find path
            shortestPath.append(present.coordinates)
            while present != None:
                if present.coordinates == (0, 0):
                    shortestPath.append((0, 0))
                else:
                    shortestPath.append(present.predecessor.coordinates)
                present = present.predecessor

            # Fill in path on the graph
            shortestPath.reverse()
            for coordinate in shortestPath:
                maze.maze[coordinate[0], coordinate[1]] = path

            return "valid solution"


        # Searching for possible new nodes
        i = present.coordinates[0]
        j = present.coordinates[1]

        lowerI = lowerJ = upperI = upperJ = False

        if i == 0 : lowerI = True
        if j == 0 : lowerJ = True
        if i == maze.dim - 1 : upperI = True
        if j == maze.dim - 1 : upperJ = True

        possible_spots = []
        if not lowerI:
            if maze.maze[i-1, j] == clear : possible_spots.append(starNode((i-1, j), present))
        if not lowerJ:
            if maze.maze[i, j-1] == clear : possible_spots.append(starNode((i, j-1), present))
        if not upperI:
            if maze.maze[i+1, j] == clear : possible_spots.append(starNode((i+1, j), present))
        if not upperJ:
            if maze.maze[i, j+1] == clear : possible_spots.append(starNode((i, j+1), present))

        # Process new nodes
        if mode == "euclidean":
            for node in possible_spots:
                if node.coordinates in closed:
                    continue
                else:
                    node.g = present.g + 1
                    node.h = math.sqrt(((node.coordinates[0] - (maze.dim - 1))**2 + (node.coordinates[1] - (maze.dim - 1)) ** 2))
                    node.cost = node.g + node.h

                    exists = next((x for x in valid if x.coordinates == node.coordinates), None)
                    if exists:
                        if exists.cost > node.cost:
                            valid.remove(exists)
                            valid.append(node)
                    else:
                        valid.append(node)

        elif mode == "manhattan":
            for node in possible_spots:
                if node.coordinates in closed:
                    continue
                else:
                    node.g = present.g + 1
                    node.h = abs(node.coordinates[0] - (maze.dim - 1)) + abs(node.coordinates[1] - (maze.dim - 1))
                    node.cost = node.g + node.h

                    exists = next((x for x in valid if x.coordinates == node.coordinates), None)
                    if exists:
                        if exists.cost > node.cost:
                            valid.remove(exists)
                            valid.append(node)
                    else:
                        valid.append(node)
        else:
            print("Please call aStar with either euclidean of manhattan as the second parameter")
            return "Please call aStar with either euclidean of manhattan as the second parameter"

    return "No path found"

# Local search algorithm

def bestfirstsearch(maze):
    global clear, blocked, onFire, path

    queue = collections.deque()
    queue.append((0,0))

    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    visited = dict([(coordinate, 0) for coordinate in coordinates])
    visited[(0,0)] = 1
    prev = dict([(coordinate, (-1, -1)) for coordinate in coordinates])

    #maze.display()
    valid = False

    while queue:
        currentNode = queue.popleft()
        i = currentNode[0]
        j = currentNode[1]
        if maze.q != None : maze.updateFire()
        # [TOP, BOTTOM, LEFT, RIGHT]
        neighbors = [(i, j-1), (i, j+1), (i-1,j), (i+1,j)]
        # Sorts the list based on how close the adjacent values are in comparison to the goal node.
        neighbors.sort(key=lambda adj: adj[0]+adj[1], reverse=True)
        for adj in neighbors:
            if adj[0] < maze.dim and adj[0] > -1 and adj[1] < maze.dim and adj[1] > -1:
                if visited[adj] != 1 and maze.maze[adj[0],adj[1]] == clear:
                    queue.append(adj)
                    visited[adj] = 1
                    prev[adj] = currentNode
                    if adj == (maze.dim-1,maze.dim-1):
                        valid = True
    if not valid:
        return "No solution found"

    #print("valid solution")

    bfsPath = []
    current = (maze.dim-1,maze.dim-1)
    while current != (-1,-1):
        bfsPath.append(current)
        current = prev[current]
    bfsPath.reverse()
    for r in bfsPath:
        maze.maze[r[0],r[1]] = path

    return "done"

# BFS

def BFS(maze):
    global clear, blocked, onFire, path

    # Data structures / variables
    queue = collections.deque()
    queue.append((0,0))

    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    visited = dict([(coordinate, 0) for coordinate in coordinates])
    visited[(0,0)] = 1
    prev = dict([(coordinate, (-1, -1)) for coordinate in coordinates])

    valid = False

    # Run until there are no more elements in the queue
    while queue:
        currentNode = queue.popleft()
        i = currentNode[0]
        j = currentNode[1]
        if maze.q != None : maze.updateFire()
        # [TOP, BOTTOM, LEFT, RIGHT]
        neighbors = [(i, j-1), (i, j+1), (i-1,j), (i+1,j)]
        for adj in neighbors:
            # Pushes an element to the stack if a move is legal.
            # Does nothing if an adjacent/neighboring node is already visited or
            # if the adjacent/neigboring node is out of bounds
            if adj[0] < maze.dim and adj[0] > -1 and adj[1] < maze.dim and adj[1] > -1:
                if visited[adj] != 1 and maze.maze[adj[0],adj[1]] == clear:
                    queue.append(adj)
                    visited[adj] = 1
                    prev[adj] = currentNode
                    if adj == (maze.dim-1,maze.dim-1):
                        valid = True
    if not valid:
        return "No solution found"

    # Traverses through the shortest path found via BFS and marks the traversed nodes of the matrix as "path"
    bfsPath = []
    current = (maze.dim-1,maze.dim-1)
    while current != (-1,-1):
        bfsPath.append(current)
        current = prev[current]
    bfsPath.reverse()
    for r in bfsPath:
        maze.maze[r[0],r[1]] = path

    return "valid solution"

# BI-DIRECTIONAL BFS

def bidirectionalBFS(maze):
    global clear, blocked, onFire, path

    # Queue data structure initialization used for both start and finish nodes
    queueFront = collections.deque()
    queueBack = collections.deque()
    queueFront.append((0,0))
    queueBack.append((maze.dim-1,maze.dim-1))

    # Variables
    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    visited = dict([(coordinate, 0) for coordinate in coordinates])
    visited[(0,0)] = 1
    prev = dict([(coordinate, (-1, -1)) for coordinate in coordinates])

    valid = False

    intersectingCoord = (-1, -1)

    mode = 0
    lastNodeFront = (-1,-1)
    lastNodeBack = (-1,-1)

    # Run bi-directional BFS until either an intersecting vertex is found or there are no more elements in both queues
    while queueFront and queueBack:
        # Switches between front and back, depending on which of the two sides move for each step
        if mode == 0:
            currentNode = queueFront.popleft()
        else:
            currentNode = queueBack.popleft()
        i = currentNode[0]
        j = currentNode[1]

        # Spread fire
        if maze.q != None : maze.updateFire()

        # [TOP, BOTTOM, LEFT, RIGHT]
        neighbors = [(i, j-1), (i, j+1), (i-1,j), (i+1,j)]
        for adj in neighbors:
            # This checks whether or not the front node is in the other queue and vice-versa. This determines the intersecting vertex, if found.
            if mode == 0:
                if adj in queueBack:
                    intersectingCoord = adj
                    lastNodeFront = adj
                    lastNodeBack = currentNode
            else:
                if adj in queueFront:
                    intersectingCoord = adj
                    lastNodeFront = currentNode
                    lastNodeBack = adj

            # Pushes an element to the stack if a move is legal.
            # Does nothing if an adjacent/neighboring node is already visited or
            # if the adjacent/neigboring node is out of bounds
            if adj[0] < maze.dim and adj[0] > -1 and adj[1] < maze.dim and adj[1] > -1:
                if visited[adj] != 1 and maze.maze[adj[0],adj[1]] == clear:
                    if mode == 0:
                        queueFront.append(adj)
                    else:
                        queueBack.append(adj)
                    visited[adj] = 1
                    prev[adj] = currentNode
        # If an intersecting vertex is found, break out of the loop and consider the solution to be valid.
        if intersectingCoord != (-1,-1):
            valid = True
            break
        if mode == 0:
            mode = 1
        else:
            mode = 0
    if not valid:
        return "No solution found"

    # Initialize path-defining variables
    bibfsPath = []
    currentFront = lastNodeFront
    currentBack = lastNodeBack

    # Marks nodes in the shortest path found as "path"
    mode = 0
    while currentFront != (-1,-1) or currentBack != (-1,-1):
        if mode == 0:
            bibfsPath.append(currentFront)
            try:
                currentFront = prev[currentFront]
            except:
                break
            mode = 1
        else:
            bibfsPath.append(currentBack)
            try:
                currentBack = prev[currentBack]
            except:
                break
            mode = 0
    for r in bibfsPath:
        maze.maze[r[0],r[1]] = path

    return "valid solution"

def DFS(maze):
    global clear, blocked, onFire, path, deepestFringe

    # Data Structures
    # Main stack
    # To help find mazes that make deep dfs fringes
    deepest = False
    stack = collections.deque()
    # Root node
    stack.append((0, 0))
    # dim*x + y = num

    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    visited = dict([(coordinate, 0) for coordinate in coordinates])
    pred = dict([(coordinate, (-1, -1)) for coordinate in coordinates])
    history = dict([(coordinate, -1) for coordinate in coordinates])
    valid = True
    while valid:

        currentNode = ''

        # Fire spreading function
        if maze.q != None : maze.updateFire()

        try:
            currentNode = stack.pop()
        except Exception as e:
            valid = False
            break
        # Skipping node if already visited or marking as visited
        if visited[currentNode] == 0:
            visited[currentNode] = 1
        else:
            continue

        i = currentNode[0]
        j = currentNode[1]

        # Check to see if traveler is on fire
        if maze.maze[i, j] == onFire:
            return "burned"

        maze.maze[i, j] = path
        # Check to see if we reached the finish line
        if i == maze.dim-1 and j == maze.dim-1:
            break

        upperI = upperJ = lowerI = lowerJ = False

        if i == 0 : lowerI = True
        if j == 0 : lowerJ = True
        if i == maze.dim - 1 : upperI = True
        if j == maze.dim - 1 : upperJ = True

        check = 0
        if not lowerI:
            if maze.maze[i-1, j] == clear and history[(i-1, j)] != 0:
                stack.append((i-1, j))
        if not lowerJ:
            if maze.maze[i, j-1] == clear and history[(i, j-1)] != 0:
                stack.append((i, j-1))
        if not upperI:
            if maze.maze[i+1, j] == clear and history[(i+1, j)] != 0:
                stack.append((i+1, j))
        if not upperJ:
            if maze.maze[i, j+1] == clear and history[(i, j+1)] != 0:
                stack.append((i, j+1))
        if deepestFringe < len(stack):
            deepestFringe = len(stack)
            deepest = True

    if not valid:
        return "no solution found"

    current = (maze.dim-1, maze.dim-1)
    # covering over fires that went through the path after the person went that way

    if deepest:
        maze.display("deepestFringe.png")
        return "valid solution--deepest"
    return "valid solution"

################Testing####################
classtest = maze(7, 0.1, q=.03, fire=True)

dfspng = maze(7, 0.1, q=.03, fire=True)
DFS(dfspng)
dfspng.display("DFS.png")

a1png = maze(7, 0.1, q=.03, fire=True)
aStar(a1png, mode='euclidean')
a1png.display("aStarEuclidean.png")

a2png = maze(7, 0.1, q=.03, fire=True)
aStar(a2png, mode='manhattan')
a2png.display("aStarManhattan.png")

bfspng = maze(7, 0.1, q=.03, fire=True)
BFS(bfspng)
bfspng.display("BFS.png")

bipng = maze(7, 0.1, q=.03, fire=True)
bidirectionalBFS(bipng)
bipng.display("bidirectionalBFS.png")

# Google Doc: https://docs.google.com/document/d/1WX8ar1ThimxyDiIjv-F0ElOq6JlK7nA6qdcDmx7Ehbc/edit?usp=sharing



f = open("MazeAlogorithmAnalysis.txt", "w+")
# Test each function with many parameters
qScores = [.03, .09, .27, .70]
pScores = [0.0, .05, .1, .25, .5, .75]
dims = [4, 6, 10, 20]

dfsTimes = []
bfsTimes = []
aStarEuclideanTimes = []
aStarManhattanTimes = []
biBFSTimes = []

for dim in dims:
    for pScore in pScores:
        print("dim: "+str(dim)+"p: "+str(pScore))
        i = 0
        dfsTime = 0
        dfsTime1 = 0
        bfsTime = 0
        bfsTime1 = 0
        aStarEuclideanTime = 0
        aStarEuclideanTime1 = 0
        aStarManhattanTime = 0
        aStarManhattanTime1 = 0
        biBFSTime = 0
        biBFSTime1 = 0

        x = 0
        while x < 5:
            x += 1

            d = dim
            p = pScore
            start = time.time()
            maze1 = maze(dim, pScore)
            temp = DFS(maze1)
            stop = time.time()
            if temp == "valid solution":
                dfsTime += stop - start
                dfsTime1 += 1
            elif temp == "valid solution--deepest":
                dfsTime += stop - start
                dfsTime1 += 1
                deepestFringeTime = dfsTime

            start = time.time()
            maze1 = maze(d, p)
            temp = BFS(maze1)
            stop = time.time()
            if temp == "valid solution":
                bfsTime += stop - start
                bfsTime1 += 1

            start = time.time()
            maze1 = maze(dim, pScore)
            temp = aStar(maze1, mode="euclidean")
            stop = time.time()
            if temp == "valid solution":
                aStarEuclideanTime += stop - start
                aStarEuclideanTime1 += 1

            start = time.time()
            maze1 = maze(dim, pScore)
            temp = aStar(maze1, mode="manhattan")
            stop = time.time()
            if temp == "valid solution":
                aStarManhattanTime += stop - start
                aStarManhattanTime1 += 1
                if aStarManhattanTime > largestMaximalNodes:
                    largestMaximalNodes = aStarManhattanTime
                    maze1.display("largestMaximalNodes.png")

            start = time.time()
            maze1 = maze(dim, pScore)
            temp = bidirectionalBFS(maze1)
            stop = time.time()
            if temp == "valid solution":
                biBFSTime += stop - start
                biBFSTime1 += 1

        try:
            dfsTimea = dfsTime/dfsTime1
            dfsTimes.append(dfsTimea)
        except Exception as e:
            dfsTimea = "No paths found"

        try:
            bfsTimea = bfsTime/bfsTime1
            bfsTimes.append(bfsTimea)
        except Exception as e:
            bfsTimea ="No paths found"

        try:
            aStarEuclideanTimea = aStarEuclideanTime/aStarEuclideanTime1
            aStarEuclideanTimes.append(aStarEuclideanTimea)
        except Exception as e:
            aStarEuclideanTimea = "No paths found"

        try:
            aStarManhattanTimea = aStarManhattanTime/aStarManhattanTime1
            aStarManhattanTimes.append(aStarManhattanTimea)
        except Exception as e:
            aStarManhattanTime = "No paths found"

        try:
            biBFSTimea = biBFSTime/biBFSTime1
            biBFSTimes.append(biBFSTimea)
        except Exception as e:
            biBFSTimea = "No paths found"


        print("Dimension: "+str(dim)+" p="+str(pScore)+" in seconds.")
        f.write("Dimension: "+str(dim)+" p="+str(pScore)+" in seconds.")
        print("\nDFS: "+str(dfsTimea)+"\nBFS: "+str(bfsTimea)+"\naStar with euclidean heuristic: "+
            str(aStarEuclideanTimea)+"\naStar with manhattan heuristic: "+str(aStarManhattanTimea)+"\nBidirectional BFS: "+str(biBFSTimea))
        f.write("\nDFS: "+str(dfsTimea)+"\nBFS: "+str(bfsTimea)+"\naStar with euclidean heuristic: "+
            str(aStarEuclideanTimea)+"\naStar with manhattan heuristic: "+str(aStarManhattanTimea)+"\nBidirectional BFS: "+str(biBFSTimea))

for q in qScores:
    for dim in dims:
        for pScore in pScores:
            print("q: "+str(q)+"dim: "+str(dim)+"p: "+str(pScore))
            i = 0
            dfsTime = 0
            dfsTime1 = 0
            bfsTime = 0
            bfsTime1 = 0
            aStarEuclideanTime = 0
            aStarEuclideanTime1 = 0
            aStarManhattanTime = 0
            aStarManhattanTime1 = 0
            biBFSTime = 0
            biBFSTime1 = 0

            x = 0
            while x < 5:
                x += 1

                start = time.time()
                maze1 = maze(dim, pScore, q=q, fire=True)
                temp = DFS(maze1)
                stop = time.time()
                if temp == "valid solution":
                    dfsTime += stop - start
                    dfsTime1 += 1
                elif temp == "valid solution--deepest":
                    dfsTime += stop - start
                    dfsTime1 += 1
                    deepestFringeTime = dfsTime

                start = time.time()
                maze1 = maze(dim, pScore, q=q, fire=True)
                temp = BFS(maze1)
                stop = time.time()
                if temp == "valid solution":
                    bfsTime += stop - start
                    bfsTime1 += 1

                start = time.time()
                maze1 = maze(dim, pScore, q=q, fire=True)
                temp = aStar(maze1, mode="euclidean")
                stop = time.time()
                if temp == "valid solution":
                    aStarEuclideanTime += stop - start
                    aStarEuclideanTime1 += 1

                start = time.time()
                maze1 = maze(dim, pScore, q=q, fire=True)
                temp = aStar(maze1, mode="manhattan")
                stop = time.time()
                if temp == "valid solution":
                    aStarManhattanTime += stop - start
                    aStarManhattanTime1 += 1
                    if aStarManhattanTime > largestMaximalNodes:
                        largestMaximalNodes = aStarManhattanTime
                        maze1.display("largestMaximalNodes.png")

                start = time.time()
                maze1 = maze(dim, pScore, q=q, fire=True)
                temp = bidirectionalBFS(maze1)
                stop = time.time()
                if temp == "valid solution":
                    biBFSTime += stop - start
                    biBFSTime1 += 1

                try:
                    dfsTimea = dfsTime/dfsTime1
                    dfsTimes.append(dfsTimea)
                except Exception as e:
                    dfsTimea = "No paths found"

                try:
                    bfsTimea = bfsTime/bfsTime1
                    bfsTimes.append(bfsTimea)
                except Exception as e:
                    bfsTimea ="No paths found"

                try:
                    aStarEuclideanTimea = aStarEuclideanTime/aStarEuclideanTime1
                    aStarEuclideanTimes.append(aStarEuclideanTimea)
                except Exception as e:
                    aStarEuclideanTimea = "No paths found"

                try:
                    aStarManhattanTimea = aStarManhattanTime/aStarManhattanTime1
                    aStarManhattanTimes.append(aStarManhattanTimea)
                except Exception as e:
                    aStarManhattanTimea = "No paths found"

                try:
                    biBFSTimea = biBFSTime/biBFSTime1
                    biBFSTimes.append(biBFSTimea)
                except Exception as e:
                    biBFSTimea = "No paths found"

            print("Dimension: "+str(dim)+" p="+str(pScore)+"q="+str(q)+" in seconds.")
            f.write("\nDimension: "+str(dim)+" p="+str(pScore)+"q="+str(q)+" in seconds.")
            print("DFS: "+str(dfsTimea)+"\nBFS: "+str(bfsTimea)+"\naStar with euclidean heuristic: "+
                str(aStarEuclideanTimea)+"\naStar with manhattan heuristic: "+str(aStarManhattanTimea)+"\nBidirectional BFS: "+str(biBFSTimea))
            f.write("\nDFS: "+str(dfsTimea)+"\nBFS: "+str(bfsTimea)+"\naStar with euclidean heuristic: "+
                str(aStarEuclideanTimea)+"\naStar with manhattan heuristic: "+str(aStarManhattanTimea)+"\nBidirectional BFS: "+str(biBFSTimea))
print("=============================================\nOverall times for each algorithm over all parameters")
f.write("\n=============================================\nOverall times for each algorithm over all parameters")

print("DFS: "+str(sum(dfsTimes)/len(dfsTimes)))
f.write("\nDFS: "+str(sum(dfsTimes)/len(dfsTimes)))
print("BFS: "+str(sum(bfsTimes)/len(bfsTimes)))
f.write("\nBFS: "+str(sum(bfsTimes)/len(bfsTimes)))
print("aStar with euclidean heuristic: "+str(sum(aStarEuclideanTimes)/len(aStarEuclideanTimes)))
f.write("\naStar with euclidean heuristic: "+str(sum(aStarEuclideanTimes)/len(aStarEuclideanTimes)))
print("aStar with manhattan heuristic: "+str(sum(aStarManhattanTimes)/len(aStarManhattanTimes)))
f.write("\naStar with manhattan heuristic: "+str(sum(aStarManhattanTimes)/len(aStarManhattanTimes)))
print("bidirectional BFS:  "+str(sum(biBFSTimes)/len(biBFSTimes)))
f.write("\nbidirectional BFS:  "+str(sum(biBFSTimes)/len(biBFSTimes)))
print("Maze that generates largest dfs fringe time: "+str(deepestFringeTime))
f.write("Maze that generates largest dfs fringe time: "+str(deepestFringeTime))
print("Maze that expands the maximal nodes time to complete: "+str(largestMaximalNodes))

f.close()
