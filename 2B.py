import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import MDS
import random
from scipy.optimize import minimize
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_array

def load_data(file_path="/Users/rubenwikarshall/VSCode/Machine-Learning/2B/VoteWatch-EP-voting-data_2004-2022/EP6_RCVs_2022_06_13.xlsx",sample_n=0, sheet=0):
    data = pd.read_excel(file_path, sheet_name=sheet)# , nrows=200)
    data = data.drop([0,1], axis=0)
    if sample_n == 0:
        out = data
    else:
        sample = random.sample(range(len(data)), sample_n)
        out = [data.iloc[index][:] for index in sample]
        out = pd.DataFrame(out)
    return out

def process(data):
    # Get x
    n = data.shape[0]
    x = [[] for i in range(3)]
    for i in range(n):
        cur = data.iloc[i, 10:].values
        if list(cur).count(0) > len(cur) / 2:
            pass
        else:
            x[0].append(data.iloc[i, 5])
            x[1].append(data.iloc[i, 7])
            x[2].append(data.iloc[i, 10:].values)
    out = pd.DataFrame(x, index = ["Country", "EPG", "Votes"]).T
    return out
    # n = data.shape[0] For other files
    # x = [[] for i in range(3)]
    # for i in range(n):
    #     cur = data.iloc[i, 9:].values
    #     check = data.iloc[i, 7]
    #     if list(cur).count(0) > len(cur) / 2 or  pd.isna(check):
    #         pass
    #     else:
    #         x[0].append(data.iloc[i, 5])
    #         x[1].append(data.iloc[i, 7])
    #         x[2].append(data.iloc[i, 10:].values)
    # out = pd.DataFrame(x, index = ["Country", "EPG", "Votes"]).T
    # return out

def sim(x, y):
    # x, y lists of votes, values 0 to 6.
    cnt = 0
    norm = 0
    cur = 0
    matrix = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0.3, 0.3, 0.3, 0],
              [0, 0, 1, 0.5, 0.5, 0.5, 0],
              [0, 0.3, 0.5, 1, 0.8, 0.8, 0],
              [0, 0.3, 0.5, 0.8, 1, 0.8, 0],
              [0, 0.3, 0.5, 0.8, 0.8, 1, 0],
              [0, 0, 0, 0, 0, 0, 1],]
    for x_vote, y_vote in zip(x, y):
        cur = matrix[int(x_vote)][int(y_vote)]
        cnt += cur
        if x_vote == 0 or y_vote == 0:
            norm += 1
    out = cnt / (x.shape[0] - norm) #norm
    return out

def sim_matrix(x):
    # Creates the similarity matrix S = N*N
    n = x.shape[0]
    S = np.eye(n,n)
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = sim(x.iloc[i]["Votes"], x.iloc[j]["Votes"])
        print(f"Progress: {i} out of {n}")
    return S

def MDS_metric(S, target_dim=2):
    # Returns the target_dim dimension representation from the S similarity matrix.
    # 1: Eigen-decomposition S = ULU^T
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices for sorting in descending order
    eigenvalues = eigenvalues[sorted_indices]  # Reorder eigenvalues
    eigenvectors = eigenvectors[:, sorted_indices] 
    eigenvalues[eigenvalues < 0] = 0
    Lmbda = np.diag(eigenvalues)
    U = eigenvectors
    S_reconstructed = U @ Lmbda @ U.T 
    # 2: X = I_{kxn}L^{1/2}U^T X is the k embedding of S
    Id = np.eye(target_dim, S.shape[0])
    X = Id @ np.sqrt(Lmbda) @ U.T 
    return X

def plot_low_dim(x, low_dim_rep, choice="EPG"):
    # Assign different colors to the different categories
    categories = x[choice].unique()
    categories.sort()
    cat_map = {category: i for i, category in enumerate(categories)}
    data_colors = [cat_map[cat] for cat in x.iloc[:][choice]]
    plt.figure(8)
    plt.scatter(low_dim_rep[0][:], low_dim_rep[1][:], c=data_colors, cmap="tab20",vmin=0, vmax=len(categories)-1)
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i / (len(categories) - 1)), markersize=10, label=cat) for i, cat in enumerate(categories)]

    plt.legend(handles=legend_labels, loc="upper left", title=f"{choice}")
    plt.show()

def loss_function(X_flat, D_original):
    X = X_flat.reshape(2, D_original.shape[0])
    D_recon = compute_dist(D_original, X)
    loss = 0
    for j in range(D_original.shape[0]):
        for i in range(j):
            cur = 1 / (D_original[i][j]) * (D_original[i][j] - D_recon[i][j]) ** 2
            loss += cur
    print("current loss: ", loss)
    return loss

def diagonal_constraint(X_flat, D_original):
    X = X_flat.reshape(2, D_original.shape[0])
    D_recon = compute_dist(D_original, X)
    return np.diag(D_recon)

def symmetry_constraint(X_flat, D_original):
    X = X_flat.reshape(2, D_original.shape[0])
    D_recon = compute_dist(D_original, X)
    return np.sum(np.abs(D_recon - D_recon.T)) 

def compute_dist(D, low_dim_rep):
    D_recon = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            D_recon[i][j] = np.sqrt((low_dim_rep[0][i] - low_dim_rep[0][j]) ** 2 + (low_dim_rep[1][i] - low_dim_rep[1][j]) ** 2)
    return D_recon

def wrapper(D_original, x_coord):
    x_coord_flat = x_coord.flatten()
    constraints = [
    {'type': 'eq', 'fun': diagonal_constraint, 'args': (D_original,)},  # Zero diagonal
    {'type': 'eq', 'fun': symmetry_constraint, 'args': (D_original,)}  # Symmetry
]
    result = minimize(loss_function, x_coord_flat,args=(D_original,), tol=1e-1, method='BFGS', options={'maxiter': 4, 'disp': True})#, constraints=constraints)#, method='SLSQP')
    print("Result: ", result.success)
    x_coord_optimized = result.x.reshape(x_coord.shape)
    return x_coord_optimized

def get_graph(D, p=5):
    G = np.eye(D.shape[0], D.shape[0])
    D = D
    for i in range(D.shape[0]):
        sorted_indices = np.argsort(D[i][:])
        # Get the p smallest values from the sorted indices
        smallest_indices = sorted_indices[:p + 1] # + 1 for it to grab itself too
        G[i, smallest_indices] = 1
    return G

def assignment_2_1_2(n_samples, choice):
    print("Loading Data...")
    print(f"Number of samples: {n_samples}")
    data = load_data(sample_n=n_samples)
    print("Data loaded")
    print("Processing data...")
    x = process(data)
    print(f"data processed, number of MEPs: {x.shape[0]}")
    print("Generating similarity matrix...")
    S = sim_matrix(x)
    print("Getting the low dimensional representation...")
    low_dim_rep = MDS_metric(S, 2)
    print("Creating plot...")
    print(f"Plotting: {choice[0]}")
    plot_low_dim(x, low_dim_rep, choice[0])
    print("Creating plot...")
    print(f"Plotting: {choice[1]}")
    plot_low_dim(x, low_dim_rep, choice[1])
    # print("Pre-implemented method...")
    # D = np.sqrt(1 - S)
    # method = MDS(n_components=2, dissimilarity='precomputed')
    # low_dim_rep_sklearn = method.fit_transform(D)
    # plot_low_dim(x, low_dim_rep_sklearn.T)

def assignment_2_1_3(n_samples, choice):
    intervals = [5000]
    print("Loading Data...")
    print(f"Number of samples: {n_samples}")
    data = load_data(sample_n=n_samples)
    data_segmented = [[] for i in range(len(intervals) + 1)]
    print("Data loaded")
    data_segmented[0] = data.drop(data.columns[intervals[0]:], axis=1)
    for i in range(1, len(intervals)):
        data_segmented[i] = data.drop(data.columns[13:intervals[0]], axis=1)
        data_segmented[i] = data_segmented[i].drop(data.columns[intervals[i]:], axis=1)
    data_segmented[-1] = data.drop(data.columns[13:intervals[-1]], axis=1)
    x_list = [[] for i in range(len(intervals) + 1)]
    for i in range(len(x_list)):
        print("Processing data...")
        x_list[i] = process(data_segmented[i])
    print(f"data processed, number of MEPs: {[x_list[j].shape[0] for j in range(len(x_list))]}")

    S_list = [[] for i in range(len(intervals) + 1)]
    for i in range(len(x_list)):
        S_list[i] = sim_matrix(x_list[i])
    
    low_dim_rep_list = [[] for i in range(len(intervals) + 1)]
    for i in range(len(x_list)):
        low_dim_rep_list[i] = MDS_metric(S_list[i], 2)
    
    print("Creating plots...")
    for i in range(len(x_list)):
        print(f"Plotting: {i, choice[0]}")
        plot_low_dim(x_list[i], low_dim_rep_list[i], choice[0])

def assignment_2_1_3_opt(n_samples, choice):
    print("Loading Data...")
    print(f"Number of samples: {n_samples}")
    data = load_data(sample_n=n_samples)
    print("Data loaded")
    print("Processing data...")
    x = process(data)
    print(f"data processed, number of MEPs: {x.shape[0]}")
    print("Generating similarity matrix...")
    S = sim_matrix(x)
    D = np.sqrt(1 - S)
    print("Getting the low dimensional representation...")
    low_dim_rep = MDS_metric(S, 2)
    print("Running optimizer")
    X_optimized = wrapper(D, low_dim_rep)

    print("Creating plot...")
    print(f"Plotting: {choice[0]}")
    plot_low_dim(x, X_optimized, choice[0])
    print("Creating plot 2...")
    print(f"Plotting: {choice[0]}")
    plot_low_dim(x, low_dim_rep, choice[0])
    print("Creating plot...")
    print(f"Plotting: {choice[1]}")
    plot_low_dim(x, X_optimized, choice[1])

def assignment_2_1_3_non_linear(n_samples, choice):
    print("Loading Data...")
    print(f"Number of samples: {n_samples}")
    data = load_data(sample_n=n_samples)
    print("Data loaded")
    print("Processing data...")
    x = process(data)
    print(f"data processed, number of MEPs: {x.shape[0]}")
    print("Generating similarity matrix...")
    S = sim_matrix(x)
    D = np.sqrt(1 - S)
    # Construct graph G
    p = 10
    G = get_graph(D, p)
    # d_ij = Floyd-Warshall algorithm
    G_in = csr_array(G)
    dist_matrix, predecessors = floyd_warshall(csgraph=G_in, directed=False, return_predecessors=True)
    # Use MDS on d_ij
    n = dist_matrix.shape[0]
    # Centering
    S_nonlinear = -0.5 * (dist_matrix - dist_matrix @ np.eye(n,n) @ np.eye(n,n).T / n - np.eye(n,n) @ np.eye(n,n).T @ dist_matrix / n + np.eye(n,n) @ np.eye(n,n).T @ dist_matrix @ np.eye(n,n) @ np.eye(n,n).T / (n ** 2))
    low_dim_nonlinear = MDS_metric(S_nonlinear, 2)
    low_dim_rep = MDS_metric(S, 2)
    print("Creating plot...")
    print(f"Plotting: {choice[0]}")
    plot_low_dim(x, low_dim_nonlinear, choice[0])
    print("Creating plot 2...")
    print(f"Plotting: {choice[0]}")
    plot_low_dim(x, low_dim_rep, choice[0])
    plot_low_dim(x, low_dim_nonlinear, choice[1])
                            
def main():
    assignment_2_1_2(0, ["EPG", "Country"])
    # assignment_2_1_3_opt(300, ["EPG", "Country"])
    # assignment_2_1_3_opt(0, ["EPG", "Country"])
    # assignment_2_1_3_non_linear(300, ["EPG", "Country"])
    # assignment_2_1_3(300, ["EPG", "Country"])

if __name__ == "__main__":
    main()