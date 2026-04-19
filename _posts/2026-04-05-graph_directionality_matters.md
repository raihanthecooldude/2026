---
layout: distill
title: The Role of Directionality in Graph Neural Networks
description: We investigate how graph directionality may influence GNN performance across homophilic and heterophilic benchmarks. By controlling for directionality, we observe performance changes that are not explained by homophily alone. Our results suggest that directionality may be an underexplored factor in graph learning.
date: 2026-04-05
future: true
htmlwidgets: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Bertran Miquel-Oliver
    affiliations:
      name: Universitat Politècnica de Catalunya, Barcelona Supercomputing Center
  - name: Manel Gil-Sorribes
    affiliations:
      name: Nostrum Biodiscovery
  - name: Alexis Molina
    affiliations:
      name: Nostrum Biodiscovery

# must be the exact same name as your blogpost
bibliography: 2026-04-15-graph_directionality_matters.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Results
    subsections:
      - name: Bidirectionalization transformation improves GNN performance
  - name: Discussion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
    .table-caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: left;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.5;
    font-weight: 400;
  }
  .caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: center;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
    line-height: 1.5;
  }
---

# **Introduction**

Graph Neural Networks (GNNs) have achieved strong performance on node classification benchmarks, particularly in **homophilic settings**, where neighboring nodes tend to share labels. However, a widely accepted narrative is that GNNs struggle in **heterophilic graphs**, where connected nodes do not necessarily share labels. This observation has motivated a search for improving GNN performance in heterophilic regimes through architectural modifications and alternative aggregation schemes <d-cite key="kipf2016semi"></d-cite>, <d-cite key="velivckovic2017graph"></d-cite>, <d-cite key="rossi2024edge"></d-cite>.

By definition, the key distinction between homophilic and heterophilic datasets lies in the degree of label similarity among neighboring nodes. However, an often overlooked aspect is that these two regimes also differ structurally: many heterophilic datasets are often used in **directed** or asymmetric form, whereas homophilic datasets are typically constructed as **undirected**.

Historically, widely used homophilic benchmarks such as Cora, CiteSeer, and PubMed were constructed as **undirected graphs**. In early work (e.g., <d-cite key="kipf2016semi"></d-cite>), citation edges were explicitly symmetrized to produce a **symmetric adjacency matrix**, enabling the use of spectral graph convolutions. This design choice enabled compatibility with models relying on symmetric normalization. As a consequence, modern libraries such as PyTorch Geometric represent these graphs as **bidirectional edge lists**, even when the underlying relationships (e.g., citations) are inherently directed.

Despite the development of models capable of handling asymmetric adjacency matrices, many benchmarks continue to rely on bidirectional versions of homophilic datasets. As shown in Table 1, most homophilic datasets exhibit a **bidirectionality ratio of 1.0**, indicating that every edge has a corresponding reverse edge. The main exception is OGBN-Arxiv, which retains its original directionality and exhibits a very low bidirectionality ratio (1.4%).

We compute the bidirectionality ratio for each dataset as the fraction of edges that have a corresponding reverse edge. Formally, for a directed graph $G = (V, E)$, the bidirectionality ratio is defined as:
$$
\text{Bidirectionality Ratio} = \frac{|\{(u, v) \in E : (v, u) \in E\}|}{|E|}
$$

<div class="table-caption">
<b>Table 1.</b> Bidirectional properties of graph datasets. We report the dataset type (homophilic vs heterophilic), whether the graph is effectively bidirectional, and the bidirectionality ratio (fraction of edges with a reverse counterpart).
</div>

| Dataset                 | Type         | Bidirectionality | Bidirectionality Ratio |
| ----------------------- | ------------ | ---------------- | ---------------------- |
| amazon-computers        | Homophilic   | True             | 1.0000                 |
| amazon-photo            | Homophilic   | True             | 1.0000                 |
| citeseer_full           | Homophilic   | True             | 1.0000                 |
| coauthor-cs             | Homophilic   | True             | 1.0000                 |
| coauthor-phy            | Homophilic   | True             | 1.0000                 |
| cora_ml                 | Homophilic   | True             | 1.0000                 |
| ogbn-arxiv              | Homophilic   | False            | 0.0145                 |
| pubmed                  | Homophilic   | True             | 1.0000                 |
| chameleon               | Heterophilic | False            | 0.2596                 |
| cornell                 | Heterophilic | False            | 0.1220                 |
| directed-roman-empire   | Heterophilic | False            | 0.3180                 |
| directed_amazon_ratings | Heterophilic | False            | 0.3571                 |
| squirrel                | Heterophilic | False            | 0.1713                 |
| texas                   | Heterophilic | False            | 0.1942                 |
| wisconsin               | Heterophilic | False            | 0.1964                 |

This persistent design choice can be attributed to several factors:

* Some datasets are naturally undirected (e.g., co-authorship and co-purchase networks) <d-cite key="shchur2018pitfalls"></d-cite>
* Others are commonly symmetrized in practice, despite having an underlying directed structure (e.g., citation graphs)
* Even recent works, such as <d-cite key="liang2025towards"></d-cite>, explicitly enforce bidirectionality to isolate specific phenomena (e.g., long-range dependencies).

In contrast, many heterophilic datasets (e.g., Cornell, Texas, Wisconsin, Chameleon, Squirrel) are typically used in their **original directed** or asymmetric form, as their underlying relationships (e.g., hyperlinks or interactions) are inherently directional <d-cite key="pei2020geom"></d-cite>. As reported in Table 1, these datasets exhibit significantly lower bidirectionality ratios, reflecting a high degree of asymmetry.

Importantly, several works have reported poor performance of GNNs on these heterophilic benchmarks and often attributed in part to label inconsistency, analyzing structural properties of the graphs without explicitly controlling for directionality <d-cite key="zhu2020beyond"></d-cite>, <d-cite key="ma2021homophily"></d-cite>. While recent studies such as <d-cite key="rossi2024edge"></d-cite> show that **directed GNNs (DirGNNs)** can improve performance on heterophilic datasets, a performance gap with respect to homophilic benchmarks still remains. However, these analyses do not disentangle the effects of heterophily from those of directionality, leaving open the question of which factor is the primary driver of GNN performance.

Taken together, these observations raise a fundamental question:

> Are GNNs failing because of **heterophily**, or because of **directionality**?

To answer this question, we perform a systematic study across widely used homophilic and heterophilic benchmarks. Specifically, we consider homophilic datasets such as Cora, CiteSeer, PubMed, Amazon, OGBN-Arxiv, and Coauthor, and heterophilic datasets including Cornell, Texas, Wisconsin, Chameleon, and Squirrel.

We explicitly control for directionality by applying bidirectionality transformation. Directed or asymmetric graphs are converted into fully bidirectional graphs by adding reverse edges:
$$
(u, v) \rightarrow (u, v), (v, u)
$$

After applying the transformation, we evaluate standard GNN architectures (GCN, GAT, SAGE) and their directed counterparts across multiple random seeds to ensure robustness.

---

# **Results**

We evaluate the performance of GNNs on the datasets in their original form, as well as after applying the bidirectionalization and directed graph construction transformations. This allows us to partially disentangle the effects of directionality from those of homophily.
For the experiments, we use the standard splits and the same hyperparameters for all models, as in <d-cite key="rossi2024edge"></d-cite>, to ensure a fair comparison across different settings. We report the average and standard deviation of performance across multiple random seeds to account for variability, together with statistical significance tests (Welch's t-test) to assess the significance of observed differences between models and graph transformations.

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/directed_baseline_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 1.</b> Performance of GNNs on directed datasets. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements. Statistical significance under Welch's t-test is indicated by asterisks (* p < 0.05, ** p < 0.01, *** p < 0.001).
</div>

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/bidirected_baseline_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 2.</b> Performance of GNNs on undirected or bidirectional datasets. Standard GNNs (GCN, GAT, SAGE) and directed GNNs (DirGCN, DirGAT, DirSAGE) perform comparably. Statistical significance under Welch's t-test is indicated by asterisks (* p < 0.05, ** p < 0.01, *** p < 0.001).
</div>

In Figure 1, we observe in directed datasets how directed GNNs (DirGCN, DirGAT, DirSAGE) consistently outperform standard GNNs (GCN, GAT, SAGE) across most datasets, with smaller differences observed in smaller datasets (e.g., Cornell, Texas, Wisconsin), which are known to exhibit higher variance. 

This suggests that standard GNNs may struggle to effectively propagate information when interactions are asymmetric, whereas directed variants may be better suited to leverage directional structure.

On the other hand, on bidirectional or undirected datasets (Figure 2), we see how directed GNNs have no significant difference in performance compared to standard GNNs, without losing performance.

Both results suggest that the presence of directionality in the underlying graph structure may play an important role in influencing GNN performance, and that standard GNNs may struggle to effectively leverage directional information, while directed GNNs are designed to handle it.

One possible explanation is that directionality restricts the number of available paths for information propagation. In bidirectional graphs, information can flow symmetrically between nodes, enabling multiple paths for information aggregation. In contrast, directed graphs limit this exchange, potentially reducing effective connectivity and increasing the likelihood of information bottlenecks. This may hinder the ability of standard GNNs to capture long-range dependencies.

## **Bidirectionalization transformation improves GNN performance**

To confirm our hypothesis that directionality may play an important role influencing GNN performance, we apply the bidirectionalization transformation to directed graphs and evaluate the performance of standard GNNs and directed GNNs on these transformed graphs.

In Figure 3, we observe a significant improvement in the performance of standard GNNs, which now perform with no significant difference to DirGNNs. These results suggest that the performance gap may not be solely explained by heterophily, but rather because of the presence of directionality in the underlying graph structure.

{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/directed_bidirected_gnn_vs_directed.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 3.</b> Performance of GNNs on bidirected heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) and directed GNNs (DirGCN, DirGAT, DirSAGE) perform comparably after applying the bidirectionality transformation. Statistical significance under Welch's t-test is indicated by asterisks (* p < 0.05, ** p < 0.01, *** p < 0.001).
</div>

This improvement may be explained by the increase in connectivity introduced by the bidirectionalization process. By adding reverse edges, the number of available paths between nodes increases, potentially alleviating structural bottlenecks and facilitating more effective message passing. This may allow standard GNNs to better aggregate information, even when label homophily remains unchanged. Indeed, in Table 2, we can see how the average degree of the graph increases significantly after applying the bidirectionalization transformation to the directed graphs, which may partially explain the observed performance improvement.

<div class="table-caption">
<b>Table 2.</b> Average degree of directed and bidirected graphs. The average degree is computed as the total number of edges divided by the number of nodes. We report the average degree for both the original directed graphs and the bidirectionalized versions, showing a significant increase in connectivity after applying the transformation in directed datasets, while remaining unchanged in bidirectional datasets. This increase in connectivity may contribute to the observed performance improvement of standard GNNs after bidirectionalization.
</div>

| Dataset                 | Type       |   N Nodes |   N Edges |   Average Degree Directed |   Average Degree Bidirected |
|:------------------------|:-----------|----------:|----------:|--------------------------:|----------------------------:|
| chameleon               | Directed   |      2277 |     36101 |                    15.855 |                      27.577 |
| cornell                 | Directed   |       183 |       298 |                     1.628 |                       3.044 |
| directed-roman-empire   | Directed   |     22662 |     39143 |                     1.958 |                       3.136 |
| directed_amazon_ratings | Directed   |     24492 |     93050 |                     4.625 |                       7.598 |
| ogbn-arxiv              | Directed   |    169343 |   1157799 |                     6.887 |                      13.674 |
| squirrel                | Directed   |      5201 |    217073 |                    41.737 |                      76.302 |
| texas                   | Directed   |       183 |       325 |                     1.776 |                       3.137 |
| wisconsin               | Directed   |       251 |       515 |                     2.052 |                       3.649 |
| amazon-computers        | Bidirected |     13752 |    491722 |                    35.756 |                      35.756 |
| amazon-photo            | Bidirected |      7650 |    238162 |                    31.132 |                      31.132 |
| citeseer_full           | Bidirected |      4230 |     10674 |                     2.523 |                       2.523 |
| coauthor-cs             | Bidirected |     18333 |    163788 |                     8.934 |                       8.934 |
| coauthor-phy            | Bidirected |     34493 |    495924 |                    14.378 |                      14.378 |
| cora_ml                 | Bidirected |      2995 |     16316 |                     5.448 |                       5.448 |
| pubmed                  | Bidirected |     19717 |     88648 |                     4.496 |                       4.496 |

This suggests that part of the observed improvement may be related not only to the removal of directionality, but also to the increase in connectivity, which facilitates information propagation.

However, it is important to note that DirGNNs with the original directed graphs still outperform standard GNNs with the bidirectionalized versions. This suggests that while the bidirectionalization transformation can improve the performance of standard GNNs, it may not fully capture the benefits of directed GNNs, which are specifically designed to leverage directional information. Moreover, it may also indicate that directionality, hence initial structure of the graph, may still play a role in the performance of GNNs, particularly in DirGNNs.

It is important to note that bidirectionalization transformation may also introduce additional edges which are altering graph structure and its properties, which may also contribute to the observed performance improvement. Therefore, further analysis is needed to disentangle the effects of directionality from those of structural changes introduced by the transformation.

## **Homophily changes after bidirectionalization transformation**

In order to understand whether the performance improvement can be attributed to changes in homophily, we compute the homophily of the original directed graphs and the bidirectionalized versions. We compute the homophily of each node applying the following formula:

$$
h_i = \frac{1}{\deg(i)} \sum_{j \in \mathcal{N}(i)} \mathbf{1}\left[y_i = y_j\right]
$$

Importantly, this improvement in performance occurs despite minimal changes in homophily. This suggests that the observed gains in GNN performance are unlikely to be driven by label similarity, and may instead be attributed to structural changes in the graph, such as increased connectivity or reduced asymmetry. 

This indicates that the low performance of GNNs on heterophilic datasets may not be solely explained by the degree of homophily, but could also be influenced by the directionality of the underlying graph structure.


{% include figure.liquid
  path="assets/img/2026-04-15-graph_directionality_matters/homophily_changes.png" class="img-fluid"
%}
<div class="caption" align="center">
  <b>Figure 4.</b> Homophily delta after applying the bidirectionalization transformation.
</div>

---

# **Discussion**

In this work, our results suggest that graph directionality may play an important role in GNN performance. In particular, we observe that applying a simple bidirectionalization transformation to directed graphs consistently improves the performance of standard GNNs, while reducing the performance gap between standard and directed GNNs.

Importantly, these improvements occur despite minimal changes in homophily, indicating that label similarity alone may not fully explain the observed behavior. Instead, these findings suggest that structural properties such as average degree, direction and connectivity may influence how effectively information is propagated in graph neural networks.

One possible interpretation is that standard GNN architectures, which rely on symmetric message-passing mechanisms, may not be well suited to handle asymmetric interactions. In directed graphs, restricted information flow may introduce additional bottlenecks, limiting the ability of these models to capture long-range dependencies. In contrast, directed GNNs are explicitly designed to account for this asymmetry, which may explain their improved performance in such settings.

Overall, our results point toward the need to more carefully consider directionality when evaluating and designing GNN models. Rather than attributing performance differences solely to homophily, it may be useful to examine how structural properties of the graph interact with model assumptions.

## **Takeaways**

- GNN performance may depend not only on homophily, but also on **graph directionality**.
- Bidirectional graphs may **facilitate information propagation**, improving performance even when homophily remains unchanged.
- Directed GNNs appear to better handle asymmetric structures, while maintaining performance on bidirectional graphs.

## **Future Directions**

Our findings suggest several directions for future research:

- **Model design.**  
  Developing GNN architectures that more effectively handle asymmetric message passing may improve performance on directed graphs.

- **Dataset design.**  
  More diverse benchmarks are needed to disentangle the effects of homophily and directionality. In particular, there is a lack of:
  - homophilic datasets with inherent directionality, and  
  - heterophilic datasets without directionality.  

  Addressing this gap would enable a more systematic evaluation of how these factors influence GNN performance. Moreover, will let to better train and evaluate models that can leverage directionality while maintaining performance in both homophilic and heterophilic regimes.

- **Analysis of structural effects.**  
  Further work is needed to better understand how directionality interacts with connectivity, path multiplicity, and potential bottlenecks in message passing.

Taken together, these observations suggest that directionality is a structural property that may deserve more explicit consideration in the study of graph neural networks.
These findings suggest that revisiting the role of directionality may help provide a more complete understanding of GNN behavior across different graph regimes.