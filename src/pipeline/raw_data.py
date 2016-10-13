import luigi

class RawData(luigi.Task):
    data_id = luigi.Parameter()
    type = luigi.Parameter()
    raw_data_dir = luigi.Parameter(default='data/raw')

    def output(self):
        if type == 'labels':
            dirpath = os.path.join(self.raw_data_dir, 'ChIPseq/labels')
            basename = '.'.join([self.data_id, 'labels.tsv.gz'])
            return luigi.LocalTarget(os.path.join(dirpath, basename))
        elif type == 'RNAseq':
            dirpath = os.path.join(self.raw_data_dir, 'RNAseq')
            rep1 = '.'.join(['gene_expression', self.data_id, 'biorep1.tsv'])
            rep2 = '.'.join(['gene_expression', self.data_id, 'biorep2.tsv'])
            rep1 = luigi.LocalTarget(os.path.join(dirpath, rep1))
            rep2 = luigi.LocalTarget(os.path.join(dirpath, rep2))
            return rep1, rep2

    def run(self):
        pass #TODO - download data if needed?

class DNaseSignal(luigi.Task):
    data_id = luigi.Parameter()
    raw_data_dir = luigi.Parameter(default='data/raw')

    def output(self):
        basename = '.'.join(['DNASE', self.data_id, 'fc.signal.bigwig'])
        return luigi.LocalTarget(os.path.join(self.raw_data_dir, 'essential_training_data', 'DNASE', basename))

    def run(self):
        pass #TODO download stuff

class GenomeFasta(luigi.Task):
    genome = luigi.Parameter(default='hg19')
    raw_data_dir = luigi.Parameter(default='data/raw')

    def output(self):
        basename = '.'.join([self.genome, 'genome.fa.gz'])
        return luigi.LocalTarget(os.path.join(self.raw_data_dir, 'annotations', basename))

    def run(self):
        pass #TODO download stuff

class ChromSizes(luigi.Task):
    genome = luigi.Parameter(default='hg19')
    raw_data_dir = luigi.Parameter(default='data/raw')

    def output(self):
        basename = '.'.join([self.genome, 'chrom.sizes'])
        return luigi.LocalTarget(os.path.join(self.raw_data_dir, 'annotations', basename))

    def run(self):
        pass #TODO download stuff

class GeneAnnotations(luigi.Task):
    raw_data_dir = luigi.Parameter(default='data/raw')

    def output(self):
        basename = 'gencode.v19.annotation.gff3.gz'
        return luigi.LocalTarget(os.path.join(self.raw_data_dir, 'annotations', basename))

    def run(self):
        pass #TODO download stuff

