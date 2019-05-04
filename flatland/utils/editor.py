import numpy as np
from numpy import array
import time
from collections import deque
from matplotlib import pyplot as plt
import threading
# from contextlib import redirect_stdout
# import os
# import sys

# import io
# from PIL import Image
# from ipywidgets import IntSlider, link, VBox

from flatland.envs.rail_env import RailEnv, random_rail_generator
# from flatland.core.transitions import RailEnvTransitions
from flatland.core.env_observation_builder import TreeObsForRailEnv
import flatland.utils.rendertools as rt
from examples.play_model import Player
from flatland.envs.env_utils import mirror


class View(object):
    def __init__(self, editor):
        self.editor = editor
        self.oRT = rt.RenderTool(self.editor.env)
        plt.figure(figsize=(10, 10))
        self.oRT.renderEnv(spacing=False, arrows=False, sRailColor="gray", show=False)
        img = self.oRT.getImage()
        plt.clf()
        import jpy_canvas
        self.wid_img = jpy_canvas.Canvas(img)


class JupEditor(object):
    def __init__(self, env, wid_img):
        print("Correct editor")
        self.env = env
        self.wid_img = wid_img

        self.qEvents = deque()

        self.regen_size = 10

        # TODO: These are currently estimated values
        self.yxBase = array([6, 21])  # pixel offset
        self.nPixCell = 700 / self.env.rail.width  # 35

        self.rcHistory = []
        self.iTransLast = -1
        self.gRCTrans = array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # NESW in RC

        self.debug = False
        self.debug_move = False
        self.wid_output = None
        self.drawMode = "Draw"
        self.env_filename = "temp.npy"
        self.set_env(env)
        self.iAgent = None
        self.player = None
        self.thread = None

    def set_env(self, env):
        self.env = env
        self.yxBase = array([6, 21])  # pixel offset
        self.nPixCell = 700 / self.env.rail.width  # 35
        self.oRT = rt.RenderTool(env)

    def setDebug(self, dEvent):
        self.debug = dEvent["new"]
        self.log("Set Debug:", self.debug)

    def setDebugMove(self, dEvent):
        self.debug_move = dEvent["new"]
        self.log("Set DebugMove:", self.debug)

    def setOutput(self, wid_output):
        self.wid_output = wid_output

    def setDrawMode(self, dEvent):
        self.drawMode = dEvent["new"]

    def on_click(self, wid, event):
        x = event['canvasX']
        y = event['canvasY']
        rcCell = ((array([y, x]) - self.yxBase) / self.nPixCell).astype(int)

        if self.drawMode == "Origin":
            self.iAgent = self.env.add_agent(rcCell, rcCell, None)
            self.drawMode = "Destination"
            self.player = None  # will need to start a new player

        elif self.drawMode == "Destination" and self.iAgent is not None:
            self.env.agents_target[self.iAgent] = rcCell
            self.drawMode = "Origin"
        
        # self.log("agent", self.drawMode, self.iAgent, rcCell)

        if self.debug:
            self.log("debug:", event)
            binTrans = self.env.rail.get_transitions(rcCell)
            sbinTrans = format(binTrans, "#018b")[2:]
            self.log("cell ", rcCell, "Transitions: ", binTrans, sbinTrans,
                [sbinTrans[i:i+4] for i in range(0, len(sbinTrans), 4)])

        self.redraw()

    def event_handler(self, wid, event):
        """Mouse motion event handler for drawing.
        """
        x = event['canvasX']
        y = event['canvasY']
        env = self.env
        qEvents = self.qEvents
        rcHistory = self.rcHistory
        bRedrawn = False
        writableData = None

        if self.debug and (event["buttons"] > 0 or self.debug_move):
            self.log("debug:", len(qEvents), len(rcHistory), event)

        assert wid == self.wid_img, "wid not same as wid_img"

        # If the mouse is held down, enqueue an event in our own queue
        if event["buttons"] > 0:
            qEvents.append((time.time(), x, y))
        
        if len(qEvents) > 0:
            tNow = time.time()
            if tNow - qEvents[0][0] > 0.1:   # wait before trying to draw
                height, width = wid.data.shape[:2]
                writableData = np.copy(self.wid_img.data)  # writable copy of image - wid_img.data is somehow readonly
                
                with self.wid_img.hold_sync():
                    while len(qEvents) > 0:
                        t, x, y = qEvents.popleft()  # get events from our queue

                        # Draw a black square
                        if x > 10 and x < width and y > 10 and y < height:
                            writableData[y-2:y+2, x-2:x+2, :] = 0
                        
                        # Translate and scale from x,y to integer row,col (note order change)
                        rcCell = ((array([y, x]) - self.yxBase) / self.nPixCell).astype(int)

                        if len(rcHistory) > 1:
                            rcLast = rcHistory[-1]
                            if not np.array_equal(rcLast, rcCell):  # only save at transition
                                # print(y, x, rcCell)
                                rcHistory.append(rcCell)
                        else:
                            rcHistory.append(rcCell)

        elif len(rcHistory) >= 3:
            # If we have already touched 3 cells
            # We have a transition into a cell, and out of it.
            
            if self.drawMode == "Draw":
                bTransition = True
            elif self.drawMode == "Erase":
                bTransition = False

            while len(rcHistory) >= 3:
                rc3Cells = array(rcHistory[:3])  # the 3 cells
                rcMiddle = rc3Cells[1]  # the middle cell which we will update

                # Save the original state of the cell
                oTransrcMiddle = self.env.rail.get_transitions(rcMiddle)
                sTransrcMiddle = self.env.rail.cell_repr(rcMiddle)

                # get the 2 row, col deltas between the 3 cells, eg [-1,0] = North
                rc2Trans = np.diff(rc3Cells, axis=0)
                
                # get the direction index for the 2 transitions
                liTrans = []
                for rcTrans in rc2Trans:
                    # gRCTrans - rcTrans gives an array of vector differences between our rcTrans
                    # and the 4 directions stored in gRCTrans.
                    # Where the vector difference is zero, we have a match...
                    # np.all detects where the whole row,col vector is zero.
                    # argwhere gives the index of the zero vector, ie the direction index
                    iTrans = np.argwhere(np.all(self.gRCTrans - rcTrans == 0, axis=1))
                    if len(iTrans) > 0:
                        iTrans = iTrans[0][0]
                        liTrans.append(iTrans)

                # check that we have two transitions
                if len(liTrans) == 2:
                    # Set the transition
                    env.rail.set_transition((*rcMiddle, liTrans[0]), liTrans[1], bTransition)

                    # Also set the reverse transition
                    # use the reversed outbound transition for inbound
                    # and the reversed inbound transition for outbound
                    env.rail.set_transition((*rcMiddle, mirror(liTrans[1])), mirror(liTrans[0]), bTransition)

                    bValid = env.rail.is_cell_valid(rcMiddle)
                    if not bValid:
                        # Reset cell transition values
                        env.rail.grid[tuple(rcMiddle)] = oTransrcMiddle

                self.log(rcMiddle, "Orig:", sTransrcMiddle, "Mod:", self.env.rail.cell_repr(rcMiddle))
                rcHistory.pop(0)  # remove the last-but-one

            # If final cell empty, insert deadend:
            if len(rcHistory) == 2 and (self.env.rail.get_transitions(rcHistory[1]) == 0):
                rc2Cells = array(rcHistory[:2])  # the 2 cells
                rcFinal = rc2Cells[1]  # the final cell which we will update

                # get the row, col delta between the 2 cells, eg [-1,0] = North
                rc2Trans = np.diff(rc2Cells, axis=0)
                
                # get the direction index for the 2 transitions
                liTrans = []
                for rcTrans in rc2Trans:
                    iTrans = np.argwhere(np.all(self.gRCTrans - rcTrans == 0, axis=1))
                    if len(iTrans) > 0:
                        iTrans = iTrans[0][0]
                        liTrans.append(iTrans)

                # check that we have one transition
                if len(liTrans) == 1:
                    # Set the transition as a deadend
                    env.rail.set_transition((*rcFinal, liTrans[0]), mirror(liTrans[0]), bTransition)

                    bValid = env.rail.is_cell_valid(rcMiddle)
                    if not bValid:
                        # Reset cell transition values
                        env.rail.grid[tuple(rcMiddle)] = oTransrcMiddle

                self.log(rcMiddle, "Orig:", sTransrcMiddle, "Mod:", self.env.rail.cell_repr(rcMiddle))
                rcHistory.pop(0)  # remove the last-but-one

            self.redraw()
            bRedrawn = True

        # only redraw with the dots/squares if necessary
        if not bRedrawn and writableData is not None:
            # This updates the image in the browser to be the new edited version
            self.wid_img.data = writableData
    
    def redraw(self, hide_stdout=True, update=True):

        # if hide_stdout:
        #     stdout_dest = os.devnull
        # else:
        #     stdout_dest = sys.stdout

        # TODO: bit of a hack - can we suppress the console messages from MPL at source?
        # with redirect_stdout(stdout_dest):
        with self.wid_output:
            plt.figure(figsize=(10, 10))
            self.oRT.renderEnv(spacing=False, arrows=False, sRailColor="gray", show=False)
            img = self.oRT.getImage()
            plt.clf()
            plt.close()
        
            if update:
                self.wid_img.data = img
            return img

    def redraw_event(self, event):
        img = self.redraw()
        self.wid_img.data = img
    
    def clear(self, event):
        self.env.rail.grid[:, :] = 0
        self.env.number_of_agents = 0
        self.env.agents_position = []
        self.env.agents_direction = []
        self.env.agents_handles = []
        self.env.agents_target = []
        self.player = None

        self.redraw_event(event)

    def setFilename(self, filename):
        self.log("filename = ", filename, type(filename))
        self.env_filename = filename

    def setFilename_event(self, event):
        self.setFilename(event["new"])

    def load(self, event):
        self.env.rail.load_transition_map(self.env_filename, override_gridsize=True)
        self.fix_env()
        self.set_env(self.env)
        self.wid_img.data = self.redraw()
    
    def save(self, event):
        self.log("save to ", self.env_filename)
        self.env.rail.save_transition_map(self.env_filename)

    def regenerate_event(self, event):
        self.env = RailEnv(width=self.regen_size,
              height=self.regen_size,
              rail_generator=random_rail_generator(cell_type_relative_proportion=[1, 1] + [0.5] * 6),
              number_of_agents=self.env.number_of_agents,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))
        self.env.reset(regen_rail=True)
        self.set_env(self.env)
        self.player = Player(self.env)
        self.redraw()
        
    def setRegenSize_event(self, event):
        self.regen_size = event["new"]

    def step_event(self, event=None):
        if self.player is None:
            self.player = Player(self.env)
            self.env.reset(regen_rail=False, replace_agents=False)
        self.player.step()
        self.redraw()

    def start_run_event(self, event=None):
        if self.thread is None:
            self.thread = threading.Thread(target=self.bg_updater, args=())
            self.thread.start()
        else:
            self.log("thread already present")

    def bg_updater(self):
        try:
            for i in range(20):
                # self.log("step ", i)
                self.step_event()
                time.sleep(0.2)
        finally:
            self.thread = None

    def fix_env(self):
        self.env.width = self.env.rail.width
        self.env.height = self.env.rail.height

    def log(self, *args, **kwargs):

        if self.wid_output:
            with self.wid_output:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

