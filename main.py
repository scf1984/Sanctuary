from world import World
from tkinter import Canvas, Tk, mainloop


class Universe(object):
    worlds = []
    pass


class WorldRunner(object):
    def __init__(self, **kwargs):
        self.world = World(**kwargs)
        self.canvas = kwargs.get('canvas')
        if self.canvas is not None:
            self.canvas.pack()
            y = int(100 / 2)
            self.canvas.create_line(0, y, 100, y, fill="#476042")

    def run(self):
        import time
        loop_tick = 0.25
        prev_time = time.time()
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            self.world.process_events()
            self.world.update(dt)
            if self.canvas is not None:
                self.world.render(self.canvas)

            sleep_time = max(loop_tick - (time.time() - prev_time), 0)
            print(str(sleep_time))
            time.sleep(sleep_time)


if __name__ == '__main__':
    master = Tk()
    master.update()
    WorldRunner(world_map=[], canvas=Canvas(master, width=500, height=500)).run()
