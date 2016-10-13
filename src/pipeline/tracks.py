import luigi
import os
import gzip

from src.pipeline.lsf_task import LSFTask
from src.pipeline.raw_data import RawData, GenomeFasta, ChromSizes, GeneAnnotations, DNaseSignal

class CombineTracksThunk(object):
    def __init__(self, genome, tracks, output):
        self.genome = genome
        self.tracks = tracks
        self.output = output

    def __call__(self):
        from src.preprocess.features import combine_tracks
        try:
            combine_tracks(self.genome, self.tracks, self.output)
        except Exception as e:
            if os.path.isfile(self.output):
                os.remove(self.outfile)
            raise e

class CombineTracks(LSFTask):
    data_id = luigi.Parameter()
    genome = luigi.Parameter(default='hg19')
    track_ids = luigi.Parameter()
    dest_data_dir = luigi.Parameter(default=os.path.join('data', 'processed', 'features'))

    def basename(self):
        return '.'.join([self.genome]+self.track_ids+[self.data_id, 'combined.tracks.h5'])

    def set_lsf_options(self):
        name = '.'.join([self.genome]+self.track_ids+[self.data_id])
        if self.q == '':
            self.q = 'short'
        self.runlimit = '6:00'
        self.jobid = 'CombineTracks.{}'.format(name)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def track_id_to_task(self, track_id):
        if track_id == 'DNAse':
            return DNaseSignal(data_id=self.data_id)
        elif track_id == 'NearestGene5p':
            return NearestGene5pTrack(data_id=self.data_id)
        else:
            raise Exception(track_id)

    def requires(self):
        return GenomeFasta(genome=self.genome), [self.track_id_to_task(track_id) for track_id in self.track_ids]

    def output(self):
        name = self.basename()
        path = os.path.join(self.dest_data_dir, name)
        return luigi.LocalTarget(path)

    def thunk(self):
        genome, tracks = self.input()
        genome = genome.path
        tracks = [track.path for track in tracks]
        output = self.output().path
        return CombineTracksThunk(genome, tracks, output)

class NearestGene5pThunk(object):
    def __init__(self, infiles, outfile, chrom_sizes_path, gene_annotations_path):
        self.infiles = infiles
        self.outfile = outfile
        self.chrom_sizes_path = chrom_sizes_path
        self.gene_annotations_path = gene_annotations_path

    def __call__(self):
        from src.preprocess.features import nearest_gene_5p_track
        try:
            with gzip.open(self.outfile, 'w') as f:
                for chrom, start, end, tpm in nearest_gene_5p_track(self.chrom_sizes_path
                                                                    , self.gene_annotations_path
                                                                    , self.infiles):
                    print(chrom, start, end, tpm, file=f)
        except Exception as e: #this enforces atomicity of the output file
            if os.path.isfile(self.outfile):
                os.remove(self.outfile)
            raise e

class NearestGene5pTrack(LSFTask):
    data_id = luigi.Parameter()
    dest_data_dir = luigi.Parameter(default=os.path.join('data', 'processed', 'RNAseq'))

    def set_lsf_options(self):
        if self.q == '':
            self.q = 'short'
        self.runlimit = '2:00'
        self.jobid = 'NearestGene5pTrack.{}'.format(self.data_id)
        self.errfile = 'logs/{}.err'.format(self.jobid)
        self.outfile = 'logs/{}.out'.format(self.jobid)

    def requires(self):
        return RawData(data_id=self.data_id, type='RNAseq'), ChromSizes(), GeneAnnotations()

    def output(self):
        basename = '.'.join(['NearestGene5p', self.data_id, 'bed.gz'])
        path = os.path.join(self.dest_data_dir, basename)
        return luigi.LocalTarget(path)

    def thunk(self):
        rna_seq, chrom_sizes, gene_annots = self.input()
        rna_seq = [obj.path for obj in rna_seq]
        chrom_sizes = chrom_sizes.path
        gene_annots = gene_annots.path
        return NearestGene5pThunk(rna_seq, self.output().path, chrom_sizes, gene_annots)

