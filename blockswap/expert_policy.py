from __future__ import annotations
import enum, time, numpy as np
try:
    from blockswap.blockswap_env import BlockSwapEnv
except ModuleNotFoundError:
    from blockswap_env import BlockSwapEnv


# ───────── tunables ─────────
MAX_DELTA   = 0.08          # env's internal 8 cm scale
XY_STEP     = 0.06
Z_FAST      = 0.05
Z_SLOW      = 0.015
XY_TOL      = 0.010         # 1 cm – used for hovering
GRASP_XY_TOL = 0.025        # 2.5 cm – OK for grasp
DESCEND_START_TOL = 0.05    # begin descent when within 4 cm
Z_TOL       = 0.003
EARLY_Z_GRAB = 0.120      # close when TCP is within 2 cm vertically
DESCEND_TIMEOUT = 100       # 300 ticks ≈ 6 s
HOVER_DZ    = 0.12
GRASP_DZ    = 0.010
EXTRA_PUSH  = 0.03          # push block down a bit
HOLD_CLOSE  = 60            # keep grip closed this long
HOLD_OPEN   = 20
DEBUG_EVERY = 25            # print freestyle debug


class Phase(enum.Enum):
    START_EP, NEXT_TASK, MOVE_TO_HOVER, DESCEND_TO_GRASP, CLOSE_GRIP, \
    LIFT_WITH_OBJ, MOVE_TO_DST_HOVER, DESCEND_TO_PLACE, OPEN_GRIP, \
    LIFT_AFTER_PLACE, DONE = range(11)


class BlockSwapExpertSM:
    def __init__(self, env: BlockSwapEnv):
        self.env = env
        cfg = env.initial_config
        self.tasks = [("red",  cfg["empty_slot"]),
                      ("blue", cfg["red_slot"]),
                      ("red",  cfg["blue_slot"])]
        self.idx = -1
        self.state = Phase.START_EP
        self.timer = self.push = self.descend_ticks = 0
        self.col = self.dst = None
        self.step_cnt = 0
        self.SAFE_Z = env.cylinder_height + 0.25

    # helpers -----------------------------------------------------------
    def ee(self):            return self.env.data.site("ee_site").xpos.copy()
    def blk(self,c):         return self.env.data.qpos[9:12] if c=="red" else self.env.data.qpos[16:19]
    def slot(self,s):        return self.env.slot_positions[s]
    def fingers_closed(self):return np.all(self.env.data.qpos[7:9] < 0.01)
    def _bounded(self, err, step): return np.sign(err)*min(abs(err), step)
    def _send(self, dxyz, grip):
        act = np.clip(np.array(dxyz)/MAX_DELTA, -1, 1).astype(np.float32)
        self.env.step(np.append(act, grip)); self.env.render(); time.sleep(0.02)

    # FSM plumbing ------------------------------------------------------
    def _switch(self, new):
        print(f"[PHASE] → {new.name}{'' if not self.col else f'({self.col}→{self.dst})'}")
        self.state, self.step_cnt = new, 0
        if new is Phase.DESCEND_TO_GRASP:
            self.descend_ticks = 0

    def run(self):
        while self.state is not Phase.DONE:
            getattr(self, f"_tick_{self.state.name.lower()}")(); self.step_cnt += 1

    # ───── phase ticks ─────────────────────────────────────────────────
    def _tick_start_ep(self): self._switch(Phase.NEXT_TASK)

    def _tick_next_task(self):
        self.idx += 1
        if self.idx == len(self.tasks): return self._switch(Phase.DONE)
        self.col, self.dst = self.tasks[self.idx]; self._switch(Phase.MOVE_TO_HOVER)

    # ---- PICK : move above block ----
    def _tick_move_to_hover(self):
        blk, cur = self.blk(self.col), self.ee()
        if cur[2] < self.SAFE_Z - Z_TOL:                         # 1) rise
            self._send([0,0,self._bounded(self.SAFE_Z-cur[2], Z_FAST)], -1); return

        dx, dy = blk[0]-cur[0], blk[1]-cur[1]                    # 2) XY coarse
        if self.step_cnt % DEBUG_EVERY == 0:
            print(f"  ↳ hover XY err = {dx:.3f},{dy:.3f}")
        if abs(dx) > DESCEND_START_TOL or abs(dy) > DESCEND_START_TOL:
            self._send([self._bounded(dx, XY_STEP),
                        self._bounded(dy, XY_STEP), 0], -1); return

        self._switch(Phase.DESCEND_TO_GRASP)

    # ---- PICK : descend & grab ----
    def _tick_descend_to_grasp(self):
        blk, cur = self.blk(self.col), self.ee()
        dx, dy = blk[0]-cur[0], blk[1]-cur[1]
        dz = (blk[2] + GRASP_DZ) - cur[2]
        self._send([self._bounded(dx, XY_STEP/4),
                    self._bounded(dy, XY_STEP/4),
                    self._bounded(dz, Z_SLOW)], -1)
        self.descend_ticks += 1

        aligned_xy = abs(dx) < GRASP_XY_TOL and abs(dy) < GRASP_XY_TOL
        aligned_z  = cur[2] <= blk[2] + EARLY_Z_GRAB
        aligned    = aligned_xy and aligned_z
        if aligned or self.descend_ticks > DESCEND_TIMEOUT:
            self.push, self.timer = int(EXTRA_PUSH/Z_SLOW)+1, HOLD_CLOSE
            self._switch(Phase.CLOSE_GRIP)

    def _tick_close_grip(self):
        dz = -Z_SLOW if self.push else 0
        self.push = max(0, self.push-1)
        self._send([0,0,dz], +1);  self.timer -= 1
        if self.fingers_closed() or self.timer <= 0:
            self._switch(Phase.LIFT_WITH_OBJ)

    def _tick_lift_with_obj(self):
        cur_z = self.ee()[2]
        if cur_z < self.SAFE_Z - Z_TOL:
            self._send([0,0,self._bounded(self.SAFE_Z-cur_z, Z_FAST)], +1); return
        self._switch(Phase.MOVE_TO_DST_HOVER)

    # ---- PLACE : move above destination ----
    def _tick_move_to_dst_hover(self):
        dst, cur = self.slot(self.dst), self.ee()
        if cur[2] < self.SAFE_Z - Z_TOL:
            self._send([0,0,self._bounded(self.SAFE_Z-cur[2], Z_FAST)], +1); return

        dx, dy = dst[0]-cur[0], dst[1]-cur[1]
        if self.step_cnt % DEBUG_EVERY == 0:
            print(f"  ↳ place XY err = {dx:.3f},{dy:.3f}")
        if abs(dx) > DESCEND_START_TOL or abs(dy) > DESCEND_START_TOL:
            self._send([self._bounded(dx, XY_STEP),
                        self._bounded(dy, XY_STEP), 0], +1); return

        self._switch(Phase.DESCEND_TO_PLACE)

    def _tick_descend_to_place(self):
        tgt = self.slot(self.dst)[2] + GRASP_DZ
        dz  = tgt - self.ee()[2]
        self._send([0,0,self._bounded(dz, Z_SLOW)], +1)
        if abs(dz) < Z_TOL:
            self.timer = HOLD_OPEN; self._switch(Phase.OPEN_GRIP)

    def _tick_open_grip(self):
        self._send([0,0,0], -1); self.timer -= 1
        if self.timer <= 0: self._switch(Phase.LIFT_AFTER_PLACE)

    def _tick_lift_after_place(self):
        cur_z = self.ee()[2]
        if cur_z < self.SAFE_Z - Z_TOL:
            self._send([0,0,self._bounded(self.SAFE_Z-cur_z, Z_FAST)], -1); return
        self._switch(Phase.NEXT_TASK)

    def _tick_done(self):
        self._send([0,0,0], -1); print("Task sequence finished!"); time.sleep(1); exit()


def main():
    env = BlockSwapEnv(render_mode="human"); env.reset(seed=0)
    BlockSwapExpertSM(env).run()


if __name__ == "__main__":
    main()
