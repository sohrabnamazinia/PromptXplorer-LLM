# Install dependencies if needed:
#    pip install prefixspan networkx

import random
from prefixspan import PrefixSpan
import networkx as nx
import pandas as pd




def mine_sequential_rules(sequences, min_support=2, min_confidence=0.6):
    """
    sequences: list of lists of prompt-IDs
    min_support: int (absolute) or float (fraction of total sequences)
    min_confidence: float between 0 and 1
    Returns:
      - freq_patterns: list of (support_count, pattern_list)
      - rules: list of dicts {antecedent, consequent, support, confidence}
    """
    N = len(sequences)
    # convert fractional support to absolute
    if 0 < min_support < 1:
        min_sup_cnt = int(min_support * N)
    else:
        min_sup_cnt = int(min_support)

    ps = PrefixSpan(sequences)
    freq_patterns = ps.frequent(min_sup_cnt)
    supp_dict = {tuple(p): sup for sup, p in freq_patterns}

    rules = []
    for sup, pat in freq_patterns:
        if len(pat) < 2:
            continue
        pat = tuple(pat)
        for split in range(1, len(pat)):
            A = pat[:split]
            B = pat[split:]
            sup_A = supp_dict.get(A, 0)
            if sup_A == 0:
                continue
            conf = sup / sup_A
            if conf >= min_confidence:
                rules.append({
                    'antecedent': A,
                    'consequent': B,
                    'support': sup,
                    'confidence': conf
                })
    # sort by confidence then support desc
    rules.sort(key=lambda r: (r['confidence'], r['support']), reverse=True)
    return freq_patterns, rules

def build_transition_graph(rules, min_sup=1, min_conf=0.0):
    """
    Build a directed graph from length-2 rules.
    Only include edges with support >= min_sup and confidence >= min_conf.
    Node labels are prompt-IDs; edges carry 'support' & 'confidence' attrs.
    """
    G = nx.DiGraph()
    for r in rules:
        if len(r['antecedent']) == 1 and len(r['consequent']) == 1:
            u = r['antecedent'][0]
            v = r['consequent'][0]
            sup, conf = r['support'], r['confidence']
            if sup >= min_sup and conf >= min_conf:
                G.add_edge(u, v, support=sup, confidence=conf)
    return G

def random_walk(G, start=None, length=10):
    """
    Perform a random walk of up to `length` steps.
    At each node, choose the next neighbor proportional to
    (support * confidence) of outgoing edges.
    If `start` is None, pick a random node with outgoing edges.
    """
    if start is None:
        # pick any node with out_edges
        candidates = [n for n in G.nodes() if G.out_degree(n) > 0]
        if not candidates:
            return []
        current = random.choice(candidates)
    else:
        current = start
    path = [current]

    for _ in range(length - 1):
        out = list(G.out_edges(current, data=True))
        if not out:
            break
        # compute weights
        weights = []
        for _, v, attrs in out:
            # weight = support × confidence (you can change this)
            weights.append(attrs['support'] * attrs['confidence'])
        total = sum(weights)
        if total == 0:
            break
        # pick one edge
        r = random.uniform(0, total)
        upto = 0
        for (_, v, attrs), w in zip(out, weights):
            upto += w
            if upto >= r:
                path.append(v)
                current = v
                break
    return path

if __name__ == "__main__":
    df = pd.read_csv("result_lda/clusterized_data_10_10000.csv")

# Drop the first column (index or prompt_id)
    df = df.iloc[:, 1:]

# Replace NaNs and convert each row to a list of integers (ignore missing values)
    sequences = [
        [int(x) for x in row.dropna()] 
        for _, row in df.iterrows()
    ]
    # --- sample data ---
    # sequences = [
    #     [101, 202, 303, 404, 505],
    #     [202, 303, 606, 707, 808, 909],
    #     [101, 202, 505, 606, 101, 303],
    #     [303, 404, 505, 606, 707],
    #     [101, 808, 909, 202, 303, 404],
    #     [202, 303, 404, 505],
    #     [505, 606, 707, 808, 909],
    #     [101, 202, 303, 404, 505, 606],
    #     [202, 404, 606, 808],
    #     [101, 303, 505, 707, 909],
    #     [909, 808, 707, 606, 505, 404],
    #     [101, 202, 303, 404],
    #     [202, 303, 505, 808, 101],
    #     [303, 404, 606, 909],
    #     [101, 505, 808, 909, 202],
    #     [202, 303, 404, 505, 606, 707],
    #     [101, 202, 303, 808, 909],
    #     [303, 505, 707, 909],
    #     [404, 505, 606, 707, 808],
    #     [101, 202, 404, 606, 808, 909]
    # ]

    # mining thresholds
    MIN_SUPPORT = 0.1      # fraction of sequences → 0.4 * 7 ≈ 2.8 → effectively ≥ 2
    MIN_CONFIDENCE = 0.1   # 70%

    patterns, rules = mine_sequential_rules(
        sequences,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE
    )

    # build graph from only length-2 rules
    G = build_transition_graph(
        rules,
        min_sup=int(MIN_SUPPORT * len(sequences)),
        min_conf=MIN_CONFIDENCE
    )

    # Example: run several random walks
    for i in range(5):
        chain = random_walk(G, length=6)
        print(f"Chain {i+1}:", chain)

    # If you want to visualize:
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=800, font_size=10)
    edge_labels = {(u,v): f"{d['support']}/{d['confidence']:.2f}"
                    for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()