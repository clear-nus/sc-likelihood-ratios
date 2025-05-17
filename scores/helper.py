from .msp import MSP
from .maxlogit import MaxLogit
from .energy import Energy
from .mds import MDS
from .knn import KNN
from .rlog import RLog
from .delta_mds import DeltaMDS
from .delta_knn import DeltaKNN


def get_score_fn(args):
    
    if args.score == 'msp':
        return MSP(args)
    elif args.score == 'maxlogit':
        return MaxLogit(args)
    elif args.score == 'energy':
        return Energy(args)
    elif args.score == 'mds':
        return MDS(args)
    elif args.score == 'knn':
        return KNN(args)
    elif args.score == 'rlog':
        return RLog(args)
    elif args.score == 'delta-mds':
        return DeltaMDS(args)
    elif args.score == 'delta-knn':
        return DeltaKNN(args)
    else:
        raise ValueError(f"Unknown score type: {args.score}")
    