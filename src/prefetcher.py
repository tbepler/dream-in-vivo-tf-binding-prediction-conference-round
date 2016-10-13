from contextlib import contextmanager
from threading import Thread, Event
import multiprocessing as mp
from queue import Queue, Empty, Full
import sys

def run(stream, Q, stop):
    for x in stream:
        while not stop.is_set():
            try:
                Q.put(x, timeout=1)
                break
            except Full:
                pass
        if stop.is_set():
            break
    if not stop.is_set():
        Q.put('DONE')

class Prefetcher(object):
    def __init__(self, stream, n, thread=True):
        self.stream = stream
        self.n = n
        self.thread = thread

    def copy(self):
        stream = self.stream.copy()
        return Prefetcher(stream, self.n, thread=self.thread)

    @property
    def randomize(self):
        return self.stream.randomize

    @randomize.setter
    def randomize(self, x):
        self.stream.randomize = x

    @property
    def infinite(self):
        return self.stream.infinite

    @infinite.setter
    def infinite(self, x):
        self.stream.infinite = x

    @property
    def dtype(self):
        return self.stream.dtype

    @infinite.setter
    def dtype(self, x):
        self.stream.dtype = x

    @property
    def weights(self):
        return self.stream.weights

    @infinite.setter
    def weights(self, x):
        self.stream.weights = x

    @property
    def Y(self):
        return self.stream.Y

    @property
    def Y_M(self):
        return self.stream.Y_M

    def __len__(self):
        return len(self.stream)

   # def chunks(self, size):
   #     return Prefetcher(self.stream.chunks(size), self.n//size + 1, self.thread)

    def __iter__(self):
        if self.thread:
            Q = Queue(self.n)
            stop = Event()
            p = Thread(target=run, args=(self.stream, Q, stop), daemon=True)
        else:
            Q = mp.Queue(self.n)
            stop = mp.Event()
            p = mp.Process(target=run, args=(self.stream, Q, stop), daemon=True)
        p.start()
        try:
            while True:
                try:
                    x = Q.get(timeout=1)
                except Empty:
                    #print('Waiting for get')
                    #sys.stdout.flush()
                    if not p.is_alive():
                        raise Exception("Thread died unexpectedly.")
                    continue
                if x == 'DONE':
                    break
                yield x
        finally:
            stop.set()
            it = 0
            try:
                while p.is_alive():
                    #print('Waiting for process to exit')
                    #sys.stdout.flush()
                    p.join(timeout=1.0)
                    it += 1
                    if it > 10: #waited too long, thread is unresponsive so just go on, send terminate if not thread
                        if not self.thread:
                            p.terminate()
                        break
            except AssertionError:
                #well this is awkward - finalizer was called from something other than main thread?
                pass



