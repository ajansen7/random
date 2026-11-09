"""
Rip-up and reroute for connections boxed in by other traces.

    python3 ripup.py --rip [radius_mm]   # remove other nets' tracks around each stuck spot
    python3 ripup.py --route             # route the stuck connections first, then re-route
                                         # whatever was ripped up
Run as separate processes (KiCad 7's Python bindings break after Remove()).
The stuck connections are recorded in ripup_state.json between the two steps.
"""
import json
import math
import os
import sys

import numpy as np
import pcbnew

import fixup
from fixup import mm, nm, LAYERS

STATE = os.path.join(fixup.HERE, "ripup_state.json")


def stuck_spots(board, grid):
    """[(net, x, y, layer_or_None)] for every stuck connection end."""
    spots = []
    for net, items in fixup.unconnected(board):
        if net == "GND":
            continue
        for x, y, d in items:
            ls = fixup.item_layers(d)
            spots.append((net, x, y, LAYERS[ls[0]] if len(ls) == 1 else None))
    labels, main, counts = fixup.gnd_components(board, grid)
    gnd = board.FindNet("GND").GetNetCode()
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNetCode() != gnd:
                continue
            c = p.GetPosition()
            j, i = grid.ij(mm(c.x), mm(c.y))
            for L in LAYERS:
                if p.IsOnLayer(L) and labels[L][j, i] >= 0 and labels[L][j, i] != main:
                    spots.append(("GND", mm(c.x), mm(c.y), L))
    return spots


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def rip(radius):
    board = pcbnew.LoadBoard(fixup.PCB)
    for z in board.Zones():
        if not z.GetIsRuleArea():
            z.SetLocalClearance(nm(0.2)); z.SetMinThickness(nm(0.15))
            z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
            z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    grid = fixup.Grid(board)
    spots = stuck_spots(board, grid)
    stuck_nets = sorted({s[0] for s in spots})
    victims = []
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK":
            continue
        for net, x, y, L in spots:
            if t.GetNetname() == net or t.GetNetname() == "GND":   # both layers: makes room for a via too
                continue
            d = seg_dist(x, y, mm(t.GetStart().x), mm(t.GetStart().y), mm(t.GetEnd().x), mm(t.GetEnd().y))
            if d < radius:
                victims.append(t)
                break
    ripped = sorted({t.GetNetname() for t in victims})
    print(f"stuck: {len(spots)} spots on {stuck_nets}; ripping {len(victims)} segments of {ripped}")
    json.dump({"stuck_nets": stuck_nets, "ripped": ripped}, open(STATE, "w"))
    for t in victims:
        board.Remove(t)
    board.Save(fixup.PCB)


def route():
    st = json.load(open(STATE))
    board = pcbnew.LoadBoard(fixup.PCB)
    for z in board.Zones():
        if not z.GetIsRuleArea():
            z.SetLocalClearance(nm(0.2)); z.SetMinThickness(nm(0.15))
            z.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
            z.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    pairs = fixup.unconnected(board)
    first = [p for p in pairs if p[0] in st["stuck_nets"] and p[0] != "GND"]
    print("1) stuck signals")
    fixup.route_signals(board, fixup.Grid(board), first)
    if "GND" in st["stuck_nets"]:
        print("2) stuck GND pieces")
        fixup.join_components(board, fixup.Grid(board))
    for rnd in range(3):
        pairs = fixup.unconnected(board)
        rest = [p for p in pairs if p[0] != "GND"]
        print(f"3.{rnd + 1}) re-route ripped/other signals: {len(rest)} open")
        if not rest:
            break
        if not fixup.route_signals(board, fixup.Grid(board), rest):
            break
    pairs = fixup.unconnected(board)
    board.Save(fixup.PCB)
    print(f"done: {len(pairs)} unconnected: {sorted(p[0] for p in pairs)}")


if __name__ == "__main__":
    if "--rip" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        rip(float(args[0]) if args else 1.0)
    elif "--route" in sys.argv:
        route()
