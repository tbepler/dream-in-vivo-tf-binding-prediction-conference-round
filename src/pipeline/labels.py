import luigi
import os
import subprocess

from src.pipeline.lsf_task import LSFTask
from src.pipeline.raw_data import RawData

class LabelsTask(luigi.Task):
    data_id = luigi.Parameter()
    dest_data_dir = luigi.Parameter(default='data/processed/labels')
    labels_suffix = luigi.Parameter(default='labels.pi.gz')

    def prefix(self):
        return os.path.join(self.dest_data_dir, self.data_id)

    def suffix(self):
        return self.labels_suffix

class MakeLabels(LabelsTask):
    partition = luigi.Parameter()
    split = luigi.IntParameter(default=-1)
    pad = luigi.IntParameter(default=0)
    balance_labels = luigi.BoolParameter(default=False)

    def requires(self):
        return BalanceLabels(data_id=self.data_id, partition=self.partition, split=self.split, pad=self.pad
                            , balance_labels=self.balance_labels, dest_data_dir=self.dest_data_dir)

    def output(self):
        return self.input()

    def run(self):
        pass

class BalanceLabelsThunk(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

    def __call__(self):
        with open(self.outfile, 'wb') as f:
            ps = subprocess.Popen(('scripts/preprocess/downsample', '-b', self.infile), stdout=subprocess.PIPE)
            subprocess.run('gzip', stdin=ps.stdout, stdout=f, check=True)

class BalanceLabels(LabelsTask, LSFTask):
    partition = luigi.Parameter()
    split = luigi.IntParameter()
    pad = luigi.IntParameter()
    balance_labels = luigi.BoolParameter()

    def set_lsf_options(self):
        name = os.path.basename(self.input().path)
        suffix = 'labels.pi.gz'
        name = name[:-len(suffix)]
        if self.q == '':
            self.q = 'short'
        self.runlimit = '2:00'
        self.jobid = 'BalanceLabels.{}'.format(name)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def requires(self):
        return SplitAndPadLabels(data_id=self.data_id, partition=self.partition, split=self.split, pad=self.pad
                                , dest_data_dir=self.dest_data_dir)

    def output(self):
        if not self.balance_labels:
            return self.input()
        path = self.input().path[:-len(self.suffix())-1]
        path = '.'.join([path, 'balanced', self.suffix()])
        return luigi.LocalTarget(path)

    def thunk(self):
        if not self.balance_labels:
            return None
        return BalanceLabelsThunk(self.input().path, self.output().path)

class SplitAndPadLabelsThunk(object):
    def __init__(self, infile, outfile, split, pad):
        self.infile = infile
        self.outfile = outfile
        self.split = split
        self.pad = pad

    def __call__(self):
        with open(self.outfile, 'wb') as f:
            args = ['scripts/preprocess/split_and_pad_labels', '-n', str(self.split), '-p', str(self.pad), self.infile]
            ps = subprocess.Popen(args, stdout=subprocess.PIPE)
            subprocess.run('gzip', stdin=ps.stdout, stdout=f, check=True)
                
    
class SplitAndPadLabels(LabelsTask, LSFTask):
    partition = luigi.Parameter()
    split = luigi.IntParameter()
    pad = luigi.IntParameter()

    def set_lsf_options(self):
        name = os.path.basename(self.output().path)
        suffix = 'labels.pi.gz'
        name = name[:-len(suffix)]
        if self.q == '':
            self.q = 'short'
        self.runlimit = '2:00'
        self.jobid = 'SplitAndPadLabels.{}'.format(name)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def requires(self):
        return PartitionLabels(data_id=self.data_id, partition=self.partition, dest_data_dir=self.dest_data_dir)

    def output(self):
        if self.split <= 0:
            return self.input()
        path = '.'.join([self.prefix(), self.partition, 'split{}.pad{}'.format(self.split, self.pad), self.suffix()])
        return luigi.LocalTarget(path)

    def thunk(self):
        if self.split <= 0:
            return None
        return SplitAndPadLabelsThunk(self.input().path, self.output().path, self.split, self.pad)

class PartitionLabelsThunk(object):
    def __init__(self, infile, outfiles):
        self.infile = infile
        self.train_path = outfiles[0]
        self.heldout_path = outfiles[1]

    def __call__(self):
        args = ['scripts/preprocess/partition_labels', '--held-out', self.heldout_path, '--train'
                , self.train_path, self.infile]
        ps = subprocess.run(args, check=True)

class PartitionLabels(LabelsTask, LSFTask):
    partition = luigi.Parameter(significant=False)

    def set_lsf_options(self):
        name = os.path.basename(self.inputs().path)
        suffix = 'labels.pi.gz'
        name = name[:-len(suffix)]
        if self.q == '':
            self.q = 'short'
        self.runlimit = '2:00'
        self.jobid = 'PartitionLabels.{}'.format(name)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def requires(self):
        return MergeLabels(data_id=self.data_id, dest_data_dir=self.dest_data_dir)

    def output(self):
        if self.partition == 'all':
            return self.input()
        elif self.partition == 'train':
            train_path = '.'.join([self.prefix(), 'train', 'merged', self.suffix()])
            return luigi.LocalTarget(train_path)
        elif self.partition == 'heldout':
            heldout_path = '.'.join([self.prefix(), 'heldout', 'merged', self.suffix()])
            return luigi.LocalTarget(heldout_path)
        else:
            raise Exception(self.partition)

    def thunk(self):
        if self.partition == 'all':
            return None
        train_path = '.'.join([self.prefix, 'train', 'merged', self.suffix()])
        heldout_path = '.'.join([self.prefix, 'heldout', 'merged', self.suffix()])
        return PartitionLabelsThunk(self.input().path, (train_path, heldout_path))

class MergeLabelsThunk(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

    def __call__(self):
        with open(self.outfile, 'wb') as f:
            ps = subprocess.Popen(('scripts/preprocess/merge_labels', self.infile), stdout=subprocess.PIPE)
            subprocess.run('gzip', stdin=ps.stdout, stdout=f, check=True)

class MergeLabels(LabelsTask, LSFTask):

    def set_lsf_options(self):
        if self.q == '':
            self.q = 'short'
        self.runlimit = '2:00'
        self.jobid = 'MergeLabels.{}'.format(self.data_id)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def requires(self):
        return RawData(type='labels', data_id=self.data_id)

    def output(self):
        outfile = '.'.join(self.prefix(), 'all', 'merged', self.suffix())
        return luigi.LocalTarget(self.outfile)

    def thunk(self):
        return MergeLabelsThunk(self.input().path, self.output().path)


