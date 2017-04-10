import bob.bio.gmm

algorithm = bob.bio.gmm.algorithm.GMMSegment(
    number_of_gaussians = 512,
    features_per_seg = 250,       # number of features per segment, the default value is 250, which means that if the window shift of mfcc is 10 ms, then the duration of the segment is 2.5 s
    seg_overlap = 50,             # number of overlapped segments
)