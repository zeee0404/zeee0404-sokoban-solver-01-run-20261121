// Sokoban push-optimal solver, written for make.fun sokoban-solver-01-run-20261121.
// Method: cut the dead squares of the hallway row to split the level into rooms,
// solve each room push-optimally with A* over pushes, then stitch the moves.
// A state is the boxes plus the lowest square of the player's reachable region
// (the player's exact square inside the region costs nothing), so the search
// stays small. Pruning: dead squares (a box that can never reach a goal) and
// frozen 2x2 blocks. The heuristic is the cheapest box-to-goal matching, which
// never overestimates, so the first solution A* pops is push-optimal.
// All budgets are counts of expanded states, never the clock, so two runs agree.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
] as const;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For one goal, the fewest pushes a box needs from every square to reach it,
 * ignoring the other boxes (a reverse push search: a box on y can be pushed to
 * x when the player can stand on y - d). -1 means unreachable.
 */
function pushDistances(w: number, wall: Uint8Array, goal: number): Int32Array {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue: number[] = [goal];
  dist[goal] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k][1] * w + DIRS[k][0];
      const y = x - d;
      const y2 = y - d;
      if (y < 0 || y >= wall.length || y2 < 0 || y2 >= wall.length) continue;
      if (!wall[y] && !wall[y2] && dist[y] < 0) {
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
  }
  return dist;
}

/** Cheapest assignment of boxes to goals by push distance; Infinity when no box can reach its goal. */
function matchingHeuristic(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  const used = new Array<boolean>(dists.length).fill(false);
  let best = Infinity;
  const rec = (i: number, sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let j = 0; j < dists.length; j++) {
      if (used[j]) continue;
      const d = dists[j][boxes[i]];
      if (d < 0) continue;
      used[j] = true;
      rec(i + 1, sum + d);
      used[j] = false;
    }
  };
  rec(0, 0);
  return best;
}

/**
 * Flood fill from `from` over open floor (not wall, not occupied); returns the
 * lowest square reached. `seen`/`stack` are scratch arrays, `stamp` a fresh tag.
 */
function reachLow(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DIRS[k][1] * w + DIRS[k][0];
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the flood fill from `from` reaches any cell of the given row. */
function reachesRow(
  w: number,
  wall: Uint8Array,
  boxes: number[],
  from: number,
  row: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): boolean {
  const occupied = new Uint8Array(wall.length);
  for (const b of boxes) occupied[b] = 1;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (Math.floor(p / w) === row) return true;
    for (let k = 0; k < 4; k++) {
      const q = p + DIRS[k][1] * w + DIRS[k][0];
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return false;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** Binary heap of [f, h, counter, state] ordered by f, then h, then insertion order. */
function heapPush(heap: number[][], e: number[]): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (less(heap[i], heap[p])) {
      const tmp = heap[i];
      heap[i] = heap[p];
      heap[p] = tmp;
      i = p;
    } else break;
  }
}
function less(a: number[], b: number[]): boolean {
  return a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
}
function heapPop(heap: number[][]): number[] {
  const top = heap[0];
  const last = heap.pop() as number[];
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && less(heap[l], heap[m])) m = l;
      if (r < heap.length && less(heap[r], heap[m])) m = r;
      if (m === i) break;
      const tmp = heap[i];
      heap[i] = heap[m];
      heap[m] = tmp;
      i = m;
    }
  }
  return top;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DIRS[k][1] * w + DIRS[k][0];
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = DIRS[k][2];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

interface RoomResult {
  pushes: number;
  moves: string;
  finalPlayer: number;
}

/**
 * Push-optimal (A*) solution of one sub-level, or null when the state budget
 * runs out. `greedy` switches the priority to the heuristic alone, for a
 * suboptimal-but-complete fallback. `needExit` requires the player to still
 * reach the hallway row after the last push (so the next room stays reachable);
 * the final room of a level does not need it.
 */
function searchRoom(sub: Level, hallwayRow: number, maxStates: number, greedy: boolean, needExit: boolean, pruneFrozen = true): RoomResult | null {
  const { w, wall, goals, boxes, player } = sub;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = goals.map((g) => pushDistances(w, wall, g));
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < goals.length; j++) {
      if (dists[j][i] >= 0) {
        live[i] = 1;
        break;
      }
    }
  }
  for (const b of boxes) if (!live[b]) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b] === 1);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  // A second scratch flood-fill array: the successor's region-low must be
  // computed without overwriting the current state's region marks, or later
  // pushes in the same expansion would fail the reachability test.
  const seenLow = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let stampLow = 0;
  const keyOf = (bs: number[], low: number): string => `${bs.join(',')}|${low}`;

  const sBoxes: number[][] = [boxes.slice().sort((a, b) => a - b)];
  const sPlayer: number[] = [player];
  const sParent: number[] = [-1];
  const sPushFrom: number[] = [-1];
  const sPushTo: number[] = [-1];
  const sG: number[] = [0];
  const sLow: number[] = [0];
  for (const b of sBoxes[0]) occupied[b] = 1;
  sLow[0] = reachLow(w, wall, occupied, player, seen, ++stamp, stack);
  for (const b of sBoxes[0]) occupied[b] = 0;

  const h0 = matchingHeuristic(sBoxes[0], dists);
  if (h0 === Infinity) return null;
  const best = new Map<string, number>();
  best.set(keyOf(sBoxes[0], sLow[0]), 0);
  const heap: number[][] = [];
  let counter = 0;
  heapPush(heap, [h0, h0, counter++, 0]);
  let expansions = 0;
  let found = -1;
  while (heap.length > 0) {
    const top = heapPop(heap);
    const idx = top[3];
    const g = sG[idx];
    if (best.get(keyOf(sBoxes[idx], sLow[idx])) !== g) continue;
    if (solved(sBoxes[idx])) {
      if (!needExit || reachesRow(w, wall, sBoxes[idx], sPlayer[idx], hallwayRow, seen, ++stamp, stack)) {
        found = idx;
        break;
      }
      continue;
    }
    if (expansions >= maxStates) break;
    expansions++;
    const curBoxes = sBoxes[idx];
    for (const b of curBoxes) occupied[b] = 1;
    reachLow(w, wall, occupied, sPlayer[idx], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < curBoxes.length; bi++) {
      const b = curBoxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k][1] * w + DIRS[k][0];
        const t = b + d;
        if (t < 0 || t >= size || wall[t] || occupied[t] || !live[t]) continue;
        const behind = b - d;
        if (behind < 0 || behind >= size || seen[behind] !== here) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        // A 2x2 block of walls and boxes is frozen: no box inside it can move
        // in any direction, so a box pushed into one off its goal is a dead
        // end. The prune can be switched off for the last-resort retry below.
        if (pruneFrozen && frozenBlock(w, wall, occupied, goalSet, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = curBoxes.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        const low = reachLow(w, wall, occupied, b, seenLow, ++stampLow, stack);
        const key = keyOf(next, low);
        const ng = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const nh = matchingHeuristic(next, dists);
        if (nh === Infinity) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        best.set(key, ng);
        const ni = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(idx);
        sPushFrom.push(b);
        sPushTo.push(t);
        sG.push(ng);
        sLow.push(low);
        heapPush(heap, [greedy ? nh : ng + nh, nh, counter++, ni]);
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of curBoxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let pl = player;
  let cur = boxes.slice().sort((a, b) => a - b);
  for (const s of path) {
    const next = sBoxes[s];
    const from = cur.find((b) => !next.includes(b)) as number;
    const to = next.find((b) => !cur.includes(b)) as number;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d) as readonly [number, number, string, string];
    for (const b of cur) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, pl, from - d);
    for (const b of cur) occupied[b] = 0;
    moves += dir[3];
    pl = from;
    cur = next;
  }
  return { pushes: path.length, moves, finalPlayer: pl };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor, and each part left that holds a
 * box is a room. Null when a box starts in the hallway, a goal is cut off, or
 * some part holds more or fewer boxes than goals.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const dists = goals.map((g) => pushDistances(w, wall, g));
  const live = new Uint8Array(wall.length);
  for (let i = 0; i < wall.length; i++) {
    for (let j = 0; j < goals.length; j++) {
      if (dists[j][i] >= 0) {
        live[i] = 1;
        break;
      }
    }
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DIRS[k][1] * w + DIRS[k][0];
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level: room by room, each room push-optimally, moves stitched together. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const hallwayRow = Math.floor(level.player / level.w);
  const rooms = roomsOf(level);
  if (rooms === null) {
    if (level.boxes.length === 0) return '';
    const r = searchRoom(level, hallwayRow, 100_000, false, false);
    return r ? r.moves : '';
  }
  if (rooms.length === 0) return '';
  let player = level.player;
  let moves = '';
  const done: boolean[] = new Array<boolean>(rooms.length).fill(false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      const other = rooms[oj];
      for (const b of done[oj] ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const needExit = ri < rooms.length - 1;
    let solved = searchRoom(sub, hallwayRow, 250_000, false, needExit);
    if (solved === null) solved = searchRoom(sub, hallwayRow, 500_000, true, needExit);
    // Last resort for an unfamiliar hidden room: full A* with a larger budget
    // and the frozen-block prune switched off, in case the prune ever rejects
    // a room the heuristic alone would still solve.
    if (solved === null) solved = searchRoom(sub, hallwayRow, 1_000_000, false, needExit, false);
    if (solved === null) return '';
    moves += solved.moves;
    player = solved.finalPlayer;
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
