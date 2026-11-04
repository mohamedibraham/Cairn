
X_GAP = 230
Y_GAP = 90


def layered_layout(node_ids: list, edges: list, *, x_gap: int = X_GAP, y_gap: int = Y_GAP) -> dict:
    
    index = {n: i for i, n in enumerate(node_ids)}
    forward = {n: [] for n in node_ids}
    for s, t in edges:
        if s in index and t in index and s != t and t not in forward[s]:
            forward[s].append(t)

    
    state, back = {}, set()
    for root in node_ids:
        if root in state:
            continue
        stack = [(root, iter(forward[root]))]
        state[root] = 1
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if state.get(child) == 1:
                    back.add((node, child))
                elif child not in state:
                    state[child] = 1
                    stack.append((child, iter(forward[child])))
                    advanced = True
                    break
            if not advanced:
                state[node] = 2
                stack.pop()
    dag = {n: [t for t in forward[n] if (n, t) not in back] for n in node_ids}

    # longest-path layering (Kahn)
    indegree = {n: 0 for n in node_ids}
    for n in node_ids:
        for t in dag[n]:
            indegree[t] += 1
    layer = {n: 0 for n in node_ids}
    ready = [n for n in node_ids if indegree[n] == 0]
    while ready:
        n = ready.pop(0)
        for t in dag[n]:
            layer[t] = max(layer[t], layer[n] + 1)
            indegree[t] -= 1
            if indegree[t] == 0:
                ready.append(t)

    columns = {}
    for n in node_ids:
        columns.setdefault(layer[n], []).append(n)
    parents = {n: [] for n in node_ids}
    for n in node_ids:
        for t in dag[n]:
            parents[t].append(n)

    
    position = {n: i for col in columns.values() for i, n in enumerate(col)}
    for _ in range(3):
        for depth in sorted(columns):
            if depth == 0:
                continue
            columns[depth].sort(key=lambda n: (
                sum(position[p] for p in parents[n]) / len(parents[n]) if parents[n] else position[n], index[n]))
            for i, n in enumerate(columns[depth]):
                position[n] = i

    coords = {}
    for depth, col in columns.items():
        for i, n in enumerate(col):
            coords[n] = (depth * x_gap, round((i - (len(col) - 1) / 2) * y_gap))
    return coords
