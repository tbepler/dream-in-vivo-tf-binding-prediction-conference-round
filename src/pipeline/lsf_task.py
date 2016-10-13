import os
import pickle
import luigi
import subprocess
import re

import src.pipeline.exec_thunk

class LSFTask(luigi.Task):
    lsf = luigi.BoolParameter(significant=False, default=False)
    q = luigi.Parameter(significant=False, default='')
    n = luigi.Parameter(significant=False, default=None)
    resources = luigi.Parameter(significant=False, default=None)
    runlimit = luigi.Parameter(significant=False, default=None)
    machine = luigi.Parameter(significant=False, default=None)
    jobid = luigi.Parameter(significant=False, default=None)
    errfile = luigi.Parameter(significant=False, default=None)
    outfile = luigi.Parameter(significant=False, default=None)
    cachedir = luigi.Parameter(default='.lsf_task_cache')

    def set_lsf_options(self):
        pass

    def run(self):
        self.set_lsf_options()
        thunk = self.thunk()
        if thunk is None:
            return
        if self.lsf:
            self.__run_lsf(thunk)
        else:
            thunk()

    def __serialize_thunk(self, thunk):
        if not os.path.exists(self.cachedir):
            os.makedirs(self.cachedir)
        thunk_path = os.path.join(self.cachedir, self.task_id)
        with open(thunk_path, 'wb') as f:
            pickle.dump(thunk, f)
        return thunk_path

    def __lsf_flags(self):
        tokens = ['-K', '-q', self.q]
        if self.n is not None:
            tokens.extend(['-n', self.n])
        if self.runlimit is not None:
            tokens.extend(['-W', self.runlimit])
        if self.machine is not None:
            tokens.extend(['-m', self.machine])
        if self.resources is not None:
            template = '{}={}'*len(self.resources)
            template = ','.join(['{}={}']*len(self.resources))
            flat = [x for y in self.resources for x in y]
            resource_str = template.format(*flat)
            tokens.extend(['-R', 'rusage['+resource_str+']'])
        if self.jobid is not None:
            tokens.extend(['-J', self.jobid])
        if self.errfile is not None:
            tokens.extend(['-eo', self.errfile])
        if self.outfile is not None:
            tokens.extend(['-oo', self.outfile])
        return tokens

    def __run_lsf(self, thunk):
        thunk_path = self.__serialize_thunk(thunk)
        runner_path = src.pipeline.exec_thunk.__file__
        if runner_path.endswith('pyc'):
            runner_path = runner_path[:-1]
        job_flags = ['python', runner_path, thunk_path]
        bsub_flags = self.__lsf_flags()
        args = ['bsub']+bsub_flags+job_flags
        try:
            ps = subprocess.Popen(args)
            ps.wait()
        except Exception as e:
            #kill the job
            ps.terminate()
            ps.wait()
            raise e
        if ps.returncode != 0:
            if self.errfile is not None:
                with open(self.errfile) as f:
                    message = f.read()
            else:
                message = ps.returncode
            raise Exception(message) 




