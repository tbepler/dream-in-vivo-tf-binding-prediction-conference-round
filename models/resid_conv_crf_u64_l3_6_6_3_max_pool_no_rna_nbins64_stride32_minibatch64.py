
import glob
tracks = glob.glob('data/processed/features/hg19.DNAse.NearestGene5p.*')
fixed = None
labels = 'data/processed/labels/{}.train.all.merged.labels.pi.gz'

nbins = 64
stride = 32

def load_data(TF, config={}):
    import src.dataset as dataset
    ## load the labels and regions 
    labels_path = labels.format(TF)
    config['labels'] = {}
    config['labels']['path'] = labels_path
    regions, Y = dataset.load_regions(labels_path)
    if nbins > 0: 
        config['labels']['slice'] = {'nbins': nbins, 'stride': stride}
        regions, Y = dataset.slice_regions(regions, nbins, stride, Y=Y) ## this can easily cause explosive memory usage if stride is too low...
    ## load the tracks and fixed data for the regions
    config['tracks'] = tracks
    config['fixed'] = fixed
    X = dataset.load_features(tracks, fixed, regions)
    return (X, Y)

def load_model(dtype='float32'):
    from src.models.resid_conv_crf import ResidConvCRF
    predictor = ResidConvCRF(units=64, layers=[3, 6, 6, 3], pool='max', use_fixed=False
                              , fixed_l1=0, fixed_l2=0)
    from src.models.preprocess import DreamChallengePreprocessor2
    preprocess = DreamChallengePreprocessor2(use_rna_seq_track=False)
    from src.models.optimizer import RMSprop, Nesterov, SGD
    optimizer = SGD(nu=0.001, conditioner=RMSprop(), momentum=Nesterov(), minibatch_size=64
                    , convergence_sample=1000)
    from src.models.binding_model import BindingModel, Regularizer
    regularizer = Regularizer(l2=1e-5) #, l1=1e-7)
    model = BindingModel(predictor, optimizer=optimizer, preprocessor=preprocess, dtype=dtype)
                        #, class_weights='balanced')
    return model
