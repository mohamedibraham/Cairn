from __future__ import annotations


def layout_diagram_content(content: dict, direction: str = "TD") -> dict:
    nodes: list[dict] = content.get("nodes", [])
    edges: list[dict] = content.get("edges", [])
    if not nodes:
        return content

    ids = [n["id"] for n in nodes]
    id_set = set(ids)


    level: dict[str, int] = {nid: 0 for nid in ids}
    for _ in range(len(nodes)):
        changed = False
        for e in edges:
            s, t = e.get("source", ""), e.get("target", "")
            if s in id_set and t in id_set:
                nxt = min(level[s] + 1, len(nodes) - 1)
                if nxt > level[t]:
                    level[t] = nxt
                    changed = True
        if not changed:
            break

    
    rows: dict[int, list[str]] = {}
    for nid in ids:
        lvl = level[nid]
        rows.setdefault(lvl, []).append(nid)

    # Assign x, y
    node_map = {n["id"]: n for n in nodes}
    for lvl, row_ids in rows.items():
        for i, nid in enumerate(row_ids):
            offset = i - (len(row_ids) - 1) / 2
            if direction == "LR":
                node_map[nid]["x"] = lvl * 250
                node_map[nid]["y"] = round(200 + offset * 90)
            else:  # TD (default)
                node_map[nid]["x"] = round(300 + offset * 230)
                node_map[nid]["y"] = lvl * 110

    return content


def apply_layout_if_needed(content: dict) -> dict:
    
    if content.get("type") != "diagram":
        return content
    nodes = content.get("nodes", [])
    
    all_zero = all(n.get("x", 0) == 0 and n.get("y", 0) == 0 for n in nodes)
    if all_zero or not nodes:
        return layout_diagram_content(content)
    return content
