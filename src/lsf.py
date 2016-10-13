import subprocess
import sys
import os
import pickle
import gzip

class LSF(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cache_path = kwargs.get('cache_path', '.lsf-cache')

    def __lsf_flags(self, job, index=None):
        tokens = {}
        if hasattr(self, 'queue'):
            tokens['-q'] = self.queue
        if hasattr(self, 'ncpus'):
            tokens['-n'] = self.ncpus
        if hasattr(self, 'runlimit'):
            tokens['-W'] = self.runlimit
        if hasattr(self, 'machine'):
            tokens['-m'] = self.machine
        if hasattr(self, 'resources'):
            resource_str = ','.join(self.resources)
            tokens['-R'] = 'rusage['+resource_str+']'
        if hasattr(self, 'name') or hasattr(job, 'name'):
            prefix = self.name if hasattr(self, 'name') else ''
            suffix = job.name if hasattr(job, 'name') else ''
            if index is None:
                jobid = '.'.join([prefix, suffix])
            else:
                jobid = '.'.join([prefix, str(index), suffix])
            tokens['-J'] = jobid
        path_prefix = job.path if hasattr(job, 'path') else os.path.join(self.cache_path, jobid) + os.sep
        if not os.path.exists(os.path.dirname(path_prefix)):
            os.makedirs(os.path.dirname(path_prefix))
        if hasattr(self, 'errfile'):
            suffix = self.errfile
        else:
            suffix = 'lsf.err'
        tokens['-eo'] = path_prefix+suffix
        if hasattr(self, 'outfile'):
            suffix = self.outfile
        else:
            suffix = 'lsf.out'
        tokens['-oo'] = path_prefix+suffix
        return tokens

    def batch(self, jobs):
        index = 0
        for job in jobs:
            yield self.submit(job, index=index)
            index += 1

    def submit(self, job, index=None, setup=None):
        bsub_flags = self.__lsf_flags(job, index=index)
        runner_path = __file__
        if runner_path.endswith('pyc'):
            runner_path = runner_path[:-1]
            thunk_path = self.__serialize_job(job)
        job_path = self.__serialize_job(job, bsub_flags.get('-J', None))
        job_flags = ' '.join(['python', runner_path, job_path])
        if setup is None and hasattr(self, 'setup'):
            job_flags = self.setup + ' && ' + job_flags
        elif setup is not None:
            job_flags = setup + ' && ' + job_flags
        bsub_flags_ = []
        for k,v in bsub_flags.items():
            bsub_flags_.append(k)
            bsub_flags_.append(v)
        bsub_flags = bsub_flags_
        args = ['bsub']+bsub_flags+[job_flags]
        ## submit to bsub and capture output
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
        ## pull the jobid from bsub
        import re
        match = re.search('<(\d+)>', output.decode())
        ident = int(match.group(1))
        return ident

    def __serialize_job(self, thunk, jobid):
        if hasattr(thunk, 'path'):
            path = thunk.path
        else:
            path = self.cache_path + os.sep
        if jobid is None:
            jobid = hex(hash(thunk))
        path = ''.join([path, jobid, '.lsf.batch.pi'])
        with open(path, 'wb') as f:
            pickle.dump(thunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

if __name__=='__main__':
    import sys
    import os
    root_split = os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]
    sys.path.append('/'.join(root_split))
    sys.path.append('/'.join(root_split+['rnn']))
    sys.path.append('/'.join(root_split+['experiments']))
    import pickle
    with open(sys.argv[1], 'rb') as f:
        thunk = pickle.load(f)
    thunk()




