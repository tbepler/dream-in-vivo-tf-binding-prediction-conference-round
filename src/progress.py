import sys
import time

class Progress(object):
    def __init__(self, out=sys.stdout, delta=1.0, bar_len=40):
        self.out = out
        self.delta = delta
        self.bar_len = bar_len
        self.clear_code = ''

    def write(self, x):
        self.out.write(self.clear_code)
        self.out.write(x)
        self.out.flush()
        self.clear_code = ''

    def format_time(self, time):
        hours, rem = divmod(time, 3600)
        mins, secs = divmod(rem, 60)
        time_str = '{:0>2}:{:0>2}:{:0>2}'.format(int(hours), int(mins), int(secs))
        return time_str

    def progress_monitor(self):
        stack = {}
        def progress(p, label):
            if label in stack:
                state = stack[label]
            else:
                state = State()
                stack[label] = state
            if p == 0:
                state.tstart = time.time()
                state.t = state.tstart
                state.n = 0
            tcur = time.time()
            state.n += 1
            if tcur - state.t >= self.delta:
                state.t = tcur
                #write a hash first to mark the start of line, but clear it
                self.out.write('#\r\033[K') 
                self.out.write(self.clear_code)
                n = int(p*self.bar_len)
                bar = ''.join(['#']*n + [' ']*(self.bar_len-n))
                if p == 0 or p == 1:
                    eta = 0
                else:
                    eta = (tcur-state.tstart)/p*(1-p)
                eta_str = 'eta '+self.format_time(eta)
                time_per_iter = (tcur-state.tstart)/state.n
                avg_per_iter = 'avg per iter '+self.format_time(time_per_iter)
                line = '# {} [{}] {:7.2%}, {}, {}'.format(label, bar, p, eta_str, avg_per_iter)
                self.out.write(line)
                self.out.write('\n')
                self.out.flush()
                self.clear_code = '\033[1F\033[K'
        return progress

    def flush(self):
        self.out.flush()

class State(object):
    def __init__(self):
        self.tstart = 0
        self.t = 0
        self.n = 0

