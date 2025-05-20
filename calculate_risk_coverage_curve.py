import argparse
import numpy as np
from sklearn import metrics
from data_utils.helper import get_cov_shift_dataset_names, get_im_split_names


def get_rc_curve_values(residuals, confidence):
    """
    Calculate each point on the RC curve based on residuals and SC scores.
    """
    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        coverage = cov / m
        risk = acc / (m-i)
        curve.append((coverage, risk))
    curve = np.asarray(curve)
    coverage, risk = curve[:, 0], curve[:, 1]
    return coverage[::-1], risk[::-1]


def get_rc_curve_values_optimal(residuals):
    """
    Calculate theoretical optimal RC curve based on residuals (i.e., an oracle selector).
    """
    curve = []
    m = len(residuals)
    residuals = np.sort(residuals)[::-1]
    cov = len(residuals)
    acc = sum(residuals)
    curve.append((cov/ m, acc / len(residuals)))
    for i in range(0, len(residuals)-1):
        cov = cov-1
        acc = acc-residuals[i]
        coverage = cov / m
        risk = acc / (m-i)
        curve.append((coverage, risk))
    curve = np.asarray(curve)
    coverage, risk = curve[:, 0], curve[:, 1]
    return coverage[::-1], risk[::-1]


def calc_aurc_coverage(coverage_array, sc_risk_array, alpha=1.0):
    
    # alpha is coverage level
    total_len = len(coverage_array)
    res_list = []
    end_idx = int(alpha * total_len)
    coverage_slice = coverage_array[-end_idx:]
    risk_slice = sc_risk_array[-end_idx:]
    AUC = metrics.auc(coverage_slice, risk_slice)
    return AUC


def return_aurc_naurc(residuals, scores, alpha=1.0):

    coverage, risk = get_rc_curve_values(residuals, scores)
    optimal_coverage, optimal_risk = get_rc_curve_values_optimal(residuals)
    
    aurc = calc_aurc_coverage(coverage, risk)
    optimal_aurc = calc_aurc_coverage(optimal_coverage, optimal_risk)
    naurc = (aurc - optimal_aurc) / (risk[-1] - optimal_aurc)
    
    return aurc, naurc


def process_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root-dir', default="/home/users/alvin/MCM/datasets", type=str, help='root dir of datasets')
    parser.add_argument('--model_type', default='dfn', choices=['dfn', 'eva'], type=str, help='model type')
    parser.add_argument('--score1', default='msp', type=str, 
                        choices=['msp', 'maxlogit', 'energy', 'mds', 'knn', 'rlog', 'delta-mds', 'delta-knn'], help='score options')
    parser.add_argument('--score2', default='none', type=str, 
                        choices=['msp', 'maxlogit', 'energy', 'mds', 'knn', 'rlog', 'delta-mds', 'delta-knn', 'none'], help='score options')
    parser.add_argument('--lam', type=float, default=3, help='coefficient of score2')
    parser.add_argument('--task', type=str, choices=['imagenet1k', 'imagenetv2', 'imagenet-sketch', 'imagenet-c-blur', 
                                                     'imagenet-c-noise', 'imagenet-c-digital',
                                                     'imagenet-c-weather', 'objectnet', 'imagenet-a', 'imagenet-r'])
    # parser.add_argument('--k', type=int, default=25, help="kth nearest neigbor distance to take") # only for knn-based scores
    args = parser.parse_args()
    if args.score2 == 'none':
        args.score2 = None
    return args


def main():
    args = process_args()
    
    # get imagenet scores and residuals (in-distribution)
    im_val_name = get_im_split_names(args)
    im_val_selector_scores1_path = f"selector_scores/{args.model_type}_{args.score1}_{im_val_name}.npy"
    in_scores1 = np.load(im_val_selector_scores1_path)
    
    if args.score2 is not None:
        im_val_selector_scores2_path = f"selector_scores/{args.model_type}_{args.score2}_{im_val_name}.npy"
        in_scores2 = np.load(im_val_selector_scores2_path)
        in_scores = in_scores1 + args.lam * in_scores2
    else:
        in_scores = in_scores1
    
    im_val_residuals_path = f"residuals/{args.model_type}_{im_val_name}.npy"
    in_residuals = np.load(im_val_residuals_path)
    name_dict = {}
    
    # get covariate shift scores and residuals 
    if args.task == 'imagenet1k':
        aurc, naurc = return_aurc_naurc(in_residuals, in_scores)
        name_dict['imagenet1k'] = (aurc, naurc)
    
    else:
        cov_shift_names = get_cov_shift_dataset_names(args)
        for name in cov_shift_names:
            
            # load selector scores
            cov_shift_selector_scores1_path = f"selector_scores/{args.model_type}_{args.score1}_{name}.npy"
            out_scores1 = np.load(cov_shift_selector_scores1_path)
            
            if args.score2 is not None:
                cov_shift_selector_scores2_path = f"selector_scores/{args.model_type}_{args.score2}_{name}.npy"
                out_scores2 = np.load(cov_shift_selector_scores2_path)
                out_scores = out_scores1 + args.lam * out_scores2
            else:
                out_scores = out_scores1
            
            # load residuals
            cov_shift_residuals_path = f"residuals/{args.model_type}_{name}.npy"
            out_residuals = np.load(cov_shift_residuals_path)
    
            # concatenate scores and residuals and compute risk coverage curve
            all_scores = np.concatenate((in_scores, out_scores), axis=0)
            all_residuals = np.concatenate((in_residuals, out_residuals), axis=0)
            aurc, naurc = return_aurc_naurc(all_residuals, all_scores)
            
            name_dict[name] = (aurc, naurc)
        
        # print aurc and naurc nicely stating model type, score1, score2, lambda
    for name, (aurc, naurc) in name_dict.items():
        print("---------------------------------------------------------")
        if args.score2 is not None:
            print(f"Model: {args.model_type}, Dataset: {name}, s1(x): {args.score1}, s2(x): {args.score2}, Lambda: {args.lam}")
        else:
            print(f"Model: {args.model_type}, Dataset: {name}, s(x): {args.score1}")
        print(f"AURC: {round(aurc * 100, 2):.3f}, NAURC: {naurc:.3f}")
    

if __name__ == "__main__":
    main()