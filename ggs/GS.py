
# from typing import List, Optional, Tuple
# import hydra
# import numpy as np
# from sklearn.neighbors import kneighbors_graph
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# from ggs.models.predictors import BaseCNN, ToyMLP
# from sklearn.model_selection import train_test_split
# from random import sample
# from petsc4py import PETSc
# from slepc4py import SLEPc
# from sporco.admm import bpdn
# from scipy.sparse.csgraph import laplacian
# from scipy.sparse import csr_matrix
# from omegaconf import DictConfig
# from omegaconf import OmegaConf
# import pyrootutils
# import torch
# from copy import deepcopy
# import logging
# import time
# import os
# from datetime import datetime
from tqdm import tqdm

# logging.basicConfig()
# logging.root.setLevel(logging.NOTSET)
# logger = logging.getLogger('Graph-based Smoothing')
# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

#ALPHABET = list("ARNDCQEGHILKMFPSTWYV")
# #ALPHABET=list("01")
ALPHABET=list("ABCDEFG")


# def run_predictor(seqs, batch_size, predictor):
#     batches = torch.split(seqs, batch_size, 0)
#     scores = []
#     for b in batches:
#         if b is None:
#             continue
#         results = predictor(b).detach()
#         scores.append(results)
#     return torch.concat(scores, dim=0)


# def get_neighbours_via_mutations(seq, num, single_level_only=False):
#     seq_list = list(seq)
#     seq_len = len(seq)
#     positions = sample(list(range(seq_len)), num)
#     substitutions = np.random.choice(ALPHABET, num)
#     neighbours = []
#     for pos, new_val in zip(positions, substitutions):
#         seq_new = seq_list.copy()
#         seq_new[pos] = new_val
#         neighbours.append(''.join(seq_new))
#     if single_level_only:
#         return neighbours
#     neighbours_of_neighbours = sum([get_neighbours_via_mutations(seq_neighb, num, single_level_only=True)
#                                     for seq_neighb in neighbours], [])
#     return neighbours_of_neighbours


# def solve_eigensystem(A, number_of_requested_eigenvectors, problem_type=SLEPc.EPS.ProblemType.HEP):
#     xr, xi = A.createVecs()

#     E = SLEPc.EPS().create()
#     E.setOperators(A, None)
#     E.setDimensions(number_of_requested_eigenvectors, PETSc.DECIDE)
#     E.setProblemType(problem_type)
#     E.setFromOptions()
#     E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

#     E.solve()
#     nconv = E.getConverged()

#     eigenvalues, eigenvectors = [], []
#     if nconv > 0:
#         for i in range(min(nconv, number_of_requested_eigenvectors)):
#             k = E.getEigenpair(i, xr, xi)
#             if k.imag == 0.0:
#                 eigenvalues.append(k.real)
#                 eigenvectors.append(xr.array.copy())
#     return eigenvalues, eigenvectors


# def soft_thr_matrices(x, y, gamma=0.25):
#     z_1 = np.maximum(x - gamma, y)
#     z_2 = np.maximum(0, np.minimum(x + gamma, y))
#     f_1 = 0.5 * np.power(z_1 - x, 2) + gamma * np.absolute(z_1 - y)
#     f_2 = 0.5 * np.power(z_2 - x, 2) + gamma * np.absolute(z_2 - y)
#     return np.where(f_1 <= f_2, z_1, z_2)





# def to_seq_tensor(seq):
#     seq_ints = [
#         ALPHABET.index(x) for x in seq
#     ]
#     return torch.tensor(seq_ints)


# def to_batch_tensor(seq_list, subset=None, device='cpu'):
#     if subset is not None:
#         seq_list = seq_list[:subset]
#     return torch.stack([to_seq_tensor(x) for x in seq_list]).to(device)


# @hydra.main(version_base="1.3", config_path="../configs", config_name="GS.yaml")
# def main(cfg: DictConfig) -> Optional[float]:

#     # Extract data path from predictor_dir
#     predictor_dir = cfg.experiment.predictor_dir
#     num_mutations = [
#         x for x in predictor_dir.split('/') if 'mutations' in x][0]
#     starting_range = [
#         x for x in predictor_dir.split('/') if 'percentile' in x][0]
#     if 'GFP' in predictor_dir:
#         task = 'GFP'
#     elif 'AAV' in predictor_dir:
#         task = 'AAV'
#     elif 'Diamond' in predictor_dir:
#         task = 'Diamond'
#     elif 'Diagonal' in predictor_dir:
#         task = 'Diagonal'
#     else:
#         raise ValueError(f'Task not found in predictor path: {predictor_dir}')
#     data_dir = os.path.join(
#         cfg.paths.data_dir, task, num_mutations, starting_range)
#     base_pool_path = os.path.join(data_dir, 'base_seqs.csv')
#     df_base = pd.read_csv(base_pool_path)
#     logger.info(f'Loaded base sequences {base_pool_path}')

#     # Load predictor
#     predictor_path = os.path.join(predictor_dir, cfg.ckpt_file)
#     cfg_path = os.path.join(predictor_dir, 'config.yaml')
#     with open(cfg_path, 'r') as fp:
#         ckpt_cfg = OmegaConf.load(fp.name)
#     predictor = BaseCNN(**ckpt_cfg.model.predictor)
#     #predictor=ToyMLP(**ckpt_cfg.model.predictor)
#     predictor_info = torch.load(predictor_path, map_location='cuda:0')
#     predictor.load_state_dict({k.replace('predictor.', ''): v for k, v in predictor_info['state_dict'].items()}, strict=True)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     predictor.to(device).eval()
#     logger.info(f'Loading base predictor {predictor_path}')

#     # Random walk
#     logger.info('Generating sequences by random walk from the base sequence pool..')
#     start_time = time.time()
#     init_seqs = df_base['sequence'].values
#     all_seqs_generated = list(init_seqs)
#     max_n_seqs = cfg.max_n_seqs
#     i_pointer = 0
#     with tqdm(total=max_n_seqs) as pbar:
#         while len(all_seqs_generated) < max_n_seqs:
#             next_seq = all_seqs_generated[i_pointer]
#             neighbs = get_neighbours_via_mutations(next_seq, num=cfg.random_traversal_neighborhood)
#             all_seqs_generated.extend(neighbs)
#             i_pointer += 1
#             pbar.update(len(neighbs))

#     all_seqs = list(sorted(set(all_seqs_generated)))
#     all_seqs_pt = to_batch_tensor(all_seqs, subset=None, device=device)
#     node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()

#     # preserving the explored upper tail of predictor's outputs
#     _, indices_all = train_test_split(
#         np.arange(len(all_seqs)),
#         test_size=cfg.subsample,
#         stratify=np.digitize(
#             node_scores_init,
#             bins=np.quantile(
#                 node_scores_init,
#                 q=np.arange(0, 1, 0.01))
#             )
#     )
#     elapsed_time = time.time() - start_time
#     logger.info(f'Finished generation in {elapsed_time:.2f} seconds')

#     all_seqs_list = [all_seqs[i] for i in indices_all]
#     # to access later the original list of strings, some of the following methods perform inplace operations
#     all_seqs_list_orig = deepcopy(all_seqs_list)
#     node_scores_init = node_scores_init[indices_all]

#     logger.info('Creating KNN graph..')
#     start_time = time.time()
#     ohe = OneHotEncoder()
#     all_seqs_list = ohe.fit_transform([list(seq) for seq in all_seqs_list])
#     knn_graph = kneighbors_graph(
#         all_seqs_list, n_neighbors=500, metric='l1', mode='distance',
#         include_self=True, n_jobs=20)
    
#     knn_graph = (knn_graph + knn_graph.T) / 2
#     knn_graph = csr_matrix((1 / knn_graph.data, knn_graph.indices, knn_graph.indptr))
#     elapsed_time = time.time() - start_time
#     logger.info(f'Finished kNN construction in {elapsed_time:.2f} seconds')

#     logger.info('Computing Laplacian..')
#     start_time = time.time()
#     laplacian_normed = laplacian(knn_graph, normed=True)
#     laplacian_normed_csr = laplacian_normed.tocsr()
#     p1 = laplacian_normed_csr.indptr
#     p2 = laplacian_normed_csr.indices
#     p3 = laplacian_normed_csr.data
#     petsc_laplacian_normed_mat = PETSc.Mat().createAIJ(size=laplacian_normed_csr.shape, csr=(p1, p2, p3))
#     elapsed_time = time.time() - start_time
#     logger.info(f'Finished Laplacian calculation in {elapsed_time:.2f} seconds')

#     logger.info('Computing eigenvectors..')
#     start_time = time.time()
#     eigenvalues, eigenvectors = solve_eigensystem(
#         petsc_laplacian_normed_mat,
#         number_of_requested_eigenvectors=cfg.num_eigenvalues)
#     elapsed_time = time.time() - start_time
#     logger.info(f'Finished eigenvalue calculation in {elapsed_time:.2f} seconds')

#     logger.info('De-noising scores of the base model..')
#     weak_labels_global_orig = np.array(node_scores_init).reshape(-1, 1)
#     weak_labels_global_min, weak_labels_global_max = weak_labels_global_orig.min(), weak_labels_global_orig.max()
#     scaled_ub = 1
#     weak_labels_global = (weak_labels_global_orig - weak_labels_global_min) / (
#                 weak_labels_global_max - weak_labels_global_min)
#     Y_opt, objectives = get_smoothed(eigenvalues, eigenvectors, weak_labels_global)

#     logger.info('Returning de-noised values to the original scale and storing results..')
#     bool_idx = Y_opt < scaled_ub
#     if cfg.rescaling == 'ratio':
#         new_99_perc = np.quantile(Y_opt, 0.99)
#         orig_99_perc = np.quantile(weak_labels_global_orig, 0.99)
#         ratio = orig_99_perc/new_99_perc
#         Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*ratio
#     elif cfg.rescaling == 'minmax':
#         Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*(weak_labels_global_max - weak_labels_global_min) + weak_labels_global_min
#     else:
#         raise NotImplementedError
#     df_smoothed = pd.DataFrame({'sequence': all_seqs_list_orig, 'target': Y_opt_scaled})
#     df_smoothed = df_smoothed[bool_idx]

#     now = datetime.now()
#     now = now.strftime("%Y-%m-%d_%H-%M-%S")
#     if cfg.results_file is None:
#         results_file = f'smoothed'
#     results_file = f'{cfg.results_file}-{now}'
#     results_path = os.path.join(
#         data_dir, results_file+'.csv')
#     logger.info(f'Writing results to {results_path}')
#     df_smoothed.to_csv(results_path, index=None)
#     #save objectives to file
#     objectives_path = os.path.join(
#         data_dir, results_file+'_objectives.csv')
#     logger.info(f'Writing objectives to {objectives_path}')
#     np.savetxt(objectives_path, objectives, delimiter=",")
#     cfg_write_path = os.path.join(
#         data_dir, results_file+'.yaml')
#     with open(cfg_write_path, 'w') as f:
#         OmegaConf.save(config=cfg, f=f)

# if __name__ == '__main__':
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#     main()



from typing import List, Optional, Tuple
import hydra
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from ggs.models.predictors import BaseCNN
from sklearn.model_selection import train_test_split
from random import sample
from petsc4py import PETSc
from slepc4py import SLEPc
from sporco.admm import bpdn
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pyrootutils
import torch
from copy import deepcopy
import logging
import time
import os
from datetime import datetime
from pykeops.torch import Vi, Vj

from ggs.data.utils.tokenize import Encoder




logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger('Graph-based Smoothing')
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

#ALPHABET=list("ABCDEFG")
#ALPHABET=list("ARNDCQEGHILKMFPSTWYV")
DNA_ALPHABET = list("ATCG")

use_cuda = torch.cuda.is_available()
def tensor(*x):
    if use_cuda:
        return torch.cuda.FloatTensor(*x)
    else:
        return torch.FloatTensor(*x)

def KNN_KeOps(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        #start = timer()

        # Encoding as KeOps LazyTensors:
        D = x_train.shape[1]
        X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
        X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array

        # Symbolic distance matrix:
        if metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1)
        elif metric == "manhattan":
            D_ij = ((X_i - X_j).abs()).sum(-1)
        elif metric == 'levenshtein':
            D_ij = (-((X_i-X_j).abs())).ifelse(0, 1).sum(-1)
        elif metric == "angular":
            D_ij = -(X_i | X_j)
        elif metric == "hyperbolic":
            D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
        else:
            raise NotImplementedError(f"The '{metric}' distance is not supported.")

        # K-NN query operator:
        KNN_fun = D_ij.Kmin_argKmin(K, dim=1)

        # N.B.: The "training" time here should be negligible.
        #elapsed = timer() - start

        def f(x_test):
            x_test = tensor(x_test)
            # start = timer()

            # Actual K-NN query:
            vals, indices  = KNN_fun(x_test, x_train)

            #elapsed = timer() - start
            vals = vals.cpu().numpy()
            indices = indices.cpu().numpy()
            return vals, indices

        return f

    return fit


def run_BERT_predictor(seqs, batch_size, predictor):
    scores = []
    for i in range(0, len(seqs[0]), batch_size):
        seqs1 = tuple(seqs[0][i:i+batch_size])
        seqs2 = tuple(seqs[1][i:i+batch_size])
        batch = [seqs1, seqs2]
        results = predictor(batch).detach()
        scores.append(results)
    return torch.cat(scores, dim=0)

def run_predictor(seqs, batch_size, predictor):
    batches = torch.split(seqs, batch_size, 0)
    scores = []
    for b in batches:
        if b is None:
            continue
        results = predictor(b).detach()
        scores.append(results)
    return torch.concat(scores, dim=0)


def get_neighbours_via_mutations(seq,task, num, single_level_only=False):
    seq_list = list(seq)
    seq_len = len(seq)
    positions = sample(list(range(seq_len)), num)
    substitutions = np.random.choice(ALPHABET, num) if task != 'integrase' else np.random.choice(DNA_ALPHABET, num)
    neighbours = []
    for pos, new_val in zip(positions, substitutions):
        seq_new = seq_list.copy()
        seq_new[pos] = new_val
        neighbours.append(''.join(seq_new))
    if single_level_only:
        return neighbours
    neighbours_of_neighbours = sum([get_neighbours_via_mutations(seq_neighb, task, num, single_level_only=True)
                                    for seq_neighb in neighbours], [])
    return neighbours_of_neighbours
   


def solve_eigensystem(A, number_of_requested_eigenvectors, problem_type=SLEPc.EPS.ProblemType.HEP):
    xr, xi = A.createVecs()

    E = SLEPc.EPS().create()
    E.setOperators(A, None)
    E.setDimensions(number_of_requested_eigenvectors, PETSc.DECIDE)
    E.setProblemType(problem_type)
    E.setFromOptions()
    E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

    E.solve()
    nconv = E.getConverged()

    eigenvalues, eigenvectors = [], []
    if nconv > 0:
        for i in range(min(nconv, number_of_requested_eigenvectors)):
            k = E.getEigenpair(i, xr, xi)
            if k.imag == 0.0:
                eigenvalues.append(k.real)
                eigenvectors.append(xr.array.copy())
    return eigenvalues, eigenvectors


def soft_thr_matrices(x, y, gamma=0.25):
    z_1 = np.maximum(x - gamma, y)
    z_2 = np.maximum(0, np.minimum(x + gamma, y))
    f_1 = 0.5 * np.power(z_1 - x, 2) + gamma * np.absolute(z_1 - y)
    f_2 = 0.5 * np.power(z_2 - x, 2) + gamma * np.absolute(z_2 - y)
    return np.where(f_1 <= f_2, z_1, z_2)


# def get_smoothed(eigenvalues, eigenvectors, weak_labels_global, iter_max = 1000):
#     # Init denoising
#     l1_weights = np.array([eig ** 0.5 for eig in eigenvalues])
#     l1_weights = np.expand_dims(l1_weights, axis=-1)

#     Y_init = weak_labels_global

#     # Construct random dictionary and random sparse coefficients
#     V_m = np.array(eigenvectors).T
#     Y_opt = Y_init.copy()
#     opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 5000,
#                              'RelStopTol': 1e-5, 'AutoRho': {'RsdlTarget': 1.0}, 'L1Weight': l1_weights})

#     def solve_for_label(j=0,lmbda=0.001,opt=opt):
#         Y_j = Y_opt[:, [j]]
#         b = bpdn.BPDN(V_m, Y_j, lmbda, opt)
#         A_j = b.solve()
#         return A_j

#     def get_current_A(Y_opt):
#         A_list = []
#         for j in range(Y_opt.shape[-1]):
#             A_list.append(solve_for_label(j))
#         return np.hstack(A_list)

#     # Optimization
#     Y_opt_prev = None
#     iter_i = 0
#     while np.any(Y_opt != Y_opt_prev) and iter_i < iter_max:
#         Y_opt_prev = deepcopy(Y_opt)
#         A = get_current_A(Y_opt)
#         F = V_m.dot(A)
#         Y_opt = soft_thr_matrices(F, Y_init)
#         iter_i += 1
#     return Y_opt
def get_smoothed(eigenvalues, eigenvectors, weak_labels_global, iter_max = 1000):
    # Init denoising
    l1_weights = np.array([eig ** 0.5 for eig in eigenvalues])
    l1_weights = np.expand_dims(l1_weights, axis=-1)

    Y_init = weak_labels_global

    # Construct random dictionary and random sparse coefficients
    V_m = np.array(eigenvectors).T
    Y_opt = Y_init.copy()
    opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 5000,
                             'RelStopTol': 1e-5, 'AutoRho': {'RsdlTarget': 1.0}, 'L1Weight': l1_weights})

    def solve_for_label(j=0,lmbda=0.001,opt=opt):
        Y_j = Y_opt[:, [j]]
        b = bpdn.BPDN(V_m, Y_j, lmbda, opt)
        A_j = b.solve()
        obj_val = b.getitstat().ObjFun[-1]
        return A_j, obj_val

    def get_current_A(Y_opt):
        A_list = []
        obj_val_total = 0
        for j in range(Y_opt.shape[-1]):
            A_j, obj_val = solve_for_label(j)
            A_list.append(A_j)
            obj_val_total += obj_val
        return np.hstack(A_list), obj_val_total

    # Optimization
    Y_opt_prev = None
    iter_i = 0

    #Construct B matrix - Sigma*U^T
    eigenvalues = np.array([e  if np.abs(e) > 1e-10 else 0 for e in eigenvalues]) # Machine precision errors
    sorted_indices = np.argsort(eigenvalues)
    B = np.diag(np.sqrt(np.array(eigenvalues)[sorted_indices])).dot(np.array(eigenvectors)[sorted_indices, :])

    
    #Objective is ||B*Y_opt||_{L_1} + 0.25*||Y_opt - Y_init||_{L_1}
    objective = lambda Y_opt: np.linalg.norm(B.dot(Y_opt), ord=1) + 0.25*np.linalg.norm(Y_opt - Y_init, ord=1)

    # Use TQDM bar to keep track of objective value
    objectives = [objective(Y_opt)]
    with tqdm(total=iter_max) as pbar:
        while  iter_i < iter_max:   
            Y_opt_prev = deepcopy(Y_opt)
            A, _ = get_current_A(Y_opt)
            
            F = V_m.dot(A)
            Y_opt = soft_thr_matrices(F, Y_init)
            objectives.append(objective(Y_opt))
            iter_i += 1
            # Update progress bar
            pbar.update(1)
            pbar.set_description(f"Objective value: {objective(Y_opt):.4f}")

    return Y_opt, objectives


def to_seq_tensor(seq):
    seq_ints = [
        ALPHABET.index(x) for x in seq
    ]
    return torch.tensor(seq_ints)


def to_batch_tensor(seq_list, task, subset=None, device='cpu'):
    if subset is not None:
        seq_list = seq_list[:subset]
    return torch.stack([to_seq_tensor(x) for x in seq_list]).to(device) if task != 'integrase' else torch.stack([x for x in seq_list]).to(device)


@hydra.main(version_base="1.3", config_path="../configs", config_name="GS.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # Extract data path from predictor_dir
    predictor_dir = cfg.experiment.predictor_dir
    num_mutations = [
        x for x in predictor_dir.split('/') if 'mutations' in x][0]
    starting_range = [
        x for x in predictor_dir.split('/') if 'percentile' in x][0]
    if 'GFP' in predictor_dir:
        task = 'GFP'
    elif 'AAV' in predictor_dir:
        task = 'AAV'
    elif 'recombination' in predictor_dir:
        task = 'integrase'
    elif 'Diagonal' in predictor_dir:
        task = 'Diagonal'
    else:
        raise ValueError(f'Task not found in predictor path: {predictor_dir}')
    data_dir = os.path.join(
        cfg.paths.data_dir, task, num_mutations, starting_range)
    base_pool_path = os.path.join(data_dir, 'base_seqs.csv')
    df_base = pd.read_csv(base_pool_path)
    logger.info(f'Loaded base sequences {base_pool_path}')

    # Load predictor
    predictor_path = os.path.join(predictor_dir, cfg.ckpt_file)
    cfg_path = os.path.join(predictor_dir, 'config.yaml')
    with open(cfg_path, 'r') as fp:
        ckpt_cfg = OmegaConf.load(fp.name)
    predictor = BaseCNN(**ckpt_cfg.model.predictor) if task != 'integrase' else RecombinationBERT(**ckpt_cfg.model.predictor)
    predictor_info = torch.load(predictor_path, map_location='cuda:0')
    predictor.load_state_dict({k.replace('predictor.', ''): v for k, v in predictor_info['state_dict'].items()}, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor.to(device).eval()
    logger.info(f'Loading base predictor {predictor_path}')

    # Random walk
    logger.info('Generating sequences by random walk from the base sequence pool..')
    start_time = time.time()
    #sequence_cols = cfg.sequence_column
    all_seqs_generated = df_base['sequence'].values.tolist()
    split_point = None
    # if len(sequence_cols) == 1:
    #     all_seqs_generated = list(init_seqs)
    #     split_point = None
    # elif len(sequence_cols) == 2:
    #     all_seqs_generated = list(init_seqs[:, 0] + init_seqs[:, 1])
    #     split_point = len(init_seqs[0, 0])
    # else:
    #     raise ValueError(f'Invalid number of sequence columns: {len(sequence_cols)}')
    max_n_seqs = cfg.max_n_seqs
    i_pointer = 0
    while len(all_seqs_generated) < max_n_seqs:
        next_seq = all_seqs_generated[i_pointer]
        neighbs = get_neighbours_via_mutations(next_seq, task, num=cfg.random_traversal_neighborhood)
        all_seqs_generated.extend(neighbs)
        i_pointer += 1
    if split_point is None:
        all_seqs = list(sorted(set(all_seqs_generated)))
        all_seqs_pt = to_batch_tensor(all_seqs, task, subset=None, device=device)
        node_scores_init = run_predictor(all_seqs_pt, batch_size=256, predictor=predictor).cpu().numpy()
    else:
        all_seqs = list(sorted(set(all_seqs_generated)))
        all_seqs_1 = [x[:split_point] for x in all_seqs]
        all_seqs_2 = [x[split_point:] for x in all_seqs]
        node_scores_init = run_BERT_predictor([all_seqs_1, all_seqs_2], batch_size=256, predictor=predictor).cpu().numpy()



    # preserving the explored upper tail of predictor's outputs
    _, indices_all = train_test_split(
        np.arange(len(all_seqs)),
        test_size=cfg.subsample,
        stratify=np.digitize(
            node_scores_init,
            bins=np.quantile(
                node_scores_init,
                q=np.arange(0, 1, 0.01))
            )
    )
    elapsed_time = time.time() - start_time
    logger.info(f'Finished generation in {elapsed_time:.2f} seconds')

    all_seqs_list = [all_seqs[i] for i in indices_all]
    
    # to access later the original list of strings, some of the following methods perform inplace operations
    all_seqs_list_orig = deepcopy(all_seqs_list)
    node_scores_init = node_scores_init[indices_all]
    encoder = Encoder(alphabet = ''.join(ALPHABET))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_seqs = tensor(encoder.encode(all_seqs_list).to(torch.float).to(device))
    logger.info('Creating KNN graph..')
    start_time = time.time()
    fit_KeOps = KNN_KeOps(K=501, metric='levenshtein')(all_seqs)
    vals, indices = fit_KeOps(all_seqs)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished kNN construction in {elapsed_time:.2f} seconds')
    vals = 1/vals[:, 1:]
    indices = indices[:, 1:]
    # turn vals, indices into a csr matrix
    knn_graph = csr_matrix((vals.flatten(), indices.flatten(), np.arange(0, len(vals.flatten()) + 1, len(vals[0])))) 
    knn_graph = (knn_graph + knn_graph.T) / 2
    #knn_graph = csr_matrix((1 / knn_graph.data, knn_graph.indices, knn_graph.indptr))
    #import pdb; pdb.set_trace()

    # logger.info('Creating KNN graph..')
    # start_time = time.time()
    # ohe = OneHotEncoder()
    # all_seqs_list = ohe.fit_transform([list(seq) for seq in all_seqs_list])

    # knn_graph = kneighbors_graph(
    #     all_seqs_list, n_neighbors=500, metric='l1', mode='distance',
    #     include_self=True, n_jobs=20)
    # import pdb; pdb.set_trace()
    # #import pdb; pdb.set_trace()

    # knn_graph = (knn_graph + knn_graph.T) / 2
    # knn_graph = csr_matrix((1 / knn_graph.data, knn_graph.indices, knn_graph.indptr))


    logger.info('Computing Laplacian..')
    start_time = time.time()
    laplacian_normed = laplacian(knn_graph, normed=True)
    laplacian_normed_csr = laplacian_normed.tocsr()
    p1 = laplacian_normed_csr.indptr
    p2 = laplacian_normed_csr.indices
    p3 = laplacian_normed_csr.data
    petsc_laplacian_normed_mat = PETSc.Mat().createAIJ(size=laplacian_normed_csr.shape, csr=(p1, p2, p3))
    elapsed_time = time.time() - start_time
    logger.info(f'Finished Laplacian calculation in {elapsed_time:.2f} seconds')

    logger.info('Computing eigenvectors..')
    start_time = time.time()
    eigenvalues, eigenvectors = solve_eigensystem(
        petsc_laplacian_normed_mat,
        number_of_requested_eigenvectors=cfg.num_eigenvalues)
    elapsed_time = time.time() - start_time
    logger.info(f'Finished eigenvalue calculation in {elapsed_time:.2f} seconds')

    logger.info('De-noising scores of the base model..')
    weak_labels_global_orig = np.array(node_scores_init).reshape(-1, 1)
    weak_labels_global_min, weak_labels_global_max = weak_labels_global_orig.min(), weak_labels_global_orig.max()
    scaled_ub = 1
    weak_labels_global = (weak_labels_global_orig - weak_labels_global_min) / (
                weak_labels_global_max - weak_labels_global_min)
    Y_opt, objectives = get_smoothed(eigenvalues, eigenvectors, weak_labels_global)

    logger.info('Returning de-noised values to the original scale and storing results..')
    bool_idx = Y_opt < scaled_ub
    if cfg.rescaling == 'ratio':
        new_99_perc = np.quantile(Y_opt, 0.99)
        orig_99_perc = np.quantile(weak_labels_global_orig, 0.99)
        ratio = orig_99_perc/new_99_perc
        Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*ratio
    elif cfg.rescaling == 'minmax':
        Y_opt_scaled = Y_opt.reshape((len(Y_opt),))*(weak_labels_global_max - weak_labels_global_min) + weak_labels_global_min
    else:
        raise NotImplementedError
    df_smoothed = pd.DataFrame({'sequence': all_seqs_list_orig, 'target': Y_opt_scaled})
    df_smoothed = df_smoothed[bool_idx]

    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.results_file is None:
        results_file = f'smoothed'
    results_file = f'{cfg.results_file}-{now}'
    results_path = os.path.join(
        data_dir, results_file+'.csv')
    logger.info(f'Writing results to {results_path}')
    df_smoothed.to_csv(results_path, index=None)
    cfg_write_path = os.path.join(
        data_dir, results_file+'.yaml')
        #save objectives to file
    objectives_path = os.path.join(
        data_dir, results_file+'_objectives.csv')
    logger.info(f'Writing objectives to {objectives_path}')
    np.savetxt(objectives_path, objectives, delimiter=",")
    with open(cfg_write_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()
