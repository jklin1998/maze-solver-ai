import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import collections
from time import sleep

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
            print("New fire: ")


    def display(self):
        fig, ax = plt.subplots()
        global labels, blocked, clear, onFire, path
        ax.matshow(self.maze, cmap=plt.cm.Blues)

        for i in range(self.dim):
            for j in range(self.dim):
                c = self.maze[j,i]
                ax.text(i, j, labels[c], va='center', ha='center')
        ax.set_xlabel("b = blocked,\nc = clear\n f = fire\np = path\n")
        plt.show()

        #print(self.maze)

# DFS
dfsVisited = dict()
dfsPrev = dict()

def dfsHelper(maze, node):
    print(node)
    if node == (maze.dim-1, maze.dim-1):
        return True
    global dfsVisited, dfsPrev
    i = node[0]
    j = node[1]
    if dfsVisited[node] != 1:
        dfsVisited[node] = 1
        neighbors = [(i, j-1), (i, j+1), (i-1,j), (i+1,j)]
        print(neighbors)
        for adj in neighbors:
            if maze.dim > adj[0] > -1 and maze.dim > adj[1] > -1:
                if dfsVisited[adj] != 1 and maze.maze[adj[0],adj[1]] == clear:
                    dfsPrev[adj] = node
                    dfsHelper(maze, adj)
    else:
        return

def DFS(maze):
    global clear, blocked, onFire, path, dfsVisited, dfsPrev

    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    dfsVisited = dict([(coordinate, 0) for coordinate in coordinates])
    dfsPrev = dict([(coordinate, (-1, -1)) for coordinate in coordinates])
    valid = False
    maze.display()
    valid = dfsHelper(maze, (0,0))
    #if not valid:
        #return "No solution found"
    print("valid solution")

    dfsPath = []
    current = (maze.dim-1,maze.dim-1)
    while current != (-1,-1):
        dfsPath.append(current)
        current = dfsPrev[current]
    dfsPath.reverse()
    for r in dfsPath:
        maze.maze[r[0],r[1]] = path

    return "done"

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

# BI-DIRECTIONAL BFS

def bidirectionalBFS(maze):
    global clear, blocked, onFire, path

    queueFront = collections.deque()
    queueBack = collections.deque()
    queueFront.append((0,0))
    queueBack.append((maze.dim-1,maze.dim-1))

    xCord = [x for x in range(0, maze.dim)]
    yCord = [y for y in range(0, maze.dim)]
    coordinates = [(x, y) for x in xCord for y in yCord]
    visited = dict([(coordinate, 0) for coordinate in coordinates])
    visited[(0,0)] = 1
    prev = dict([(coordinate, (-1, -1)) for coordinate in coordinates])

    maze.display()
    valid = False

    intersectingCoord = (-1, -1)

    mode = 0
    lastNodeFront = (-1,-1)
    lastNodeBack = (-1,-1)

    while queueFront and queueBack:
        if mode == 0:
            currentNode = queueFront.popleft()
        else:
            currentNode = queueBack.popleft()
        i = currentNode[0]
        j = currentNode[1]
        if maze.q != None : maze.updateFire()
        # [TOP, BOTTOM, LEFT, RIGHT]
        neighbors = [(i, j-1), (i, j+1), (i-1,j), (i+1,j)]
        for adj in neighbors:
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
            if adj[0] < maze.dim and adj[0] > -1 and adj[1] < maze.dim and adj[1] > -1:
                if visited[adj] != 1 and maze.maze[adj[0],adj[1]] == clear:
                    if mode == 0:
                        queueFront.append(adj)
                    else:
                        queueBack.append(adj)
                    visited[adj] = 1
                    prev[adj] = currentNode
                    #print(prev[adj])
        if intersectingCoord != (-1,-1):
            valid = True
            break
        if mode == 0:
            mode = 1
        else:
            mode = 0
    if not valid:
        return "No solution found"

    #print("valid solution")

    bibfsPath = []
    currentFront = lastNodeFront
    currentBack = lastNodeBack

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

    return "done"

################Testing####################
#classtest = maze(7, 0)

#classtest.display()

print()
#classtest.display()
classtest = maze(35, 0.5)
while(bestfirstsearch(classtest) == 'No solution found'):
    classtest = maze(35, 0.5)
classtest.display()
stack = collections.deque()
