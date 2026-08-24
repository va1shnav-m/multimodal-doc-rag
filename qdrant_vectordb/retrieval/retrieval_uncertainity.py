# """
# Retrieval Uncertainty Estimator

# Computes confidence/uncertainty of retrieval using
# multiple statistically grounded signals.

# Signals:
# ---------
# 1. Softmax Confidence
# 2. Margin Confidence
# 3. Entropy
# 4. Dense/BM25 Consensus (Rank Biased Overlap)
# 5. Reranker Stability

# Author:
# Production RAG
# """

# from scipy.special import softmax
# from scipy.stats import entropy
# import numpy as np
# from rbo import RankingSimilarity


# class RetrievalUncertaintyEstimator:

#     def __init__(self):
#         pass

#     ##########################################################

#     def softmax_confidence(self, scores):
#         """
#         Converts similarity scores into probabilities.

#         Returns
#         -------
#         float
#             Highest probability.
#         """

#         if len(scores) == 0:
#             return 0.0

#         probabilities = softmax(scores)

#         return float(np.max(probabilities))

#     ##########################################################

#     def margin_confidence(self, scores):
#         """
#         Confidence based on the dominance of the top result.

#         Computes the ratio of the highest softmax probability
#         to the combined probability of the top two results.
#         Returns a value in the range [0, 1].
#         """

#         if len(scores) < 2:
#             return 1.0

#         probabilities = softmax(scores)

#         probabilities = np.sort(probabilities)[::-1]

#         return float(
#             probabilities[0] /
#             (probabilities[0] + probabilities[1])
#         )

#     ##########################################################

#     def entropy_confidence(self, scores):
#         """
#         Lower entropy
#             =>
#         Higher confidence
#         """

#         if len(scores) == 0:
#             return 0.0

#         probabilities = softmax(scores)

#         e = entropy(probabilities)

#         max_entropy = np.log(len(probabilities))

#         normalized = 1 - (e / max_entropy)

#         return float(normalized)

#     ##########################################################

#     def consensus(
#         self,
#         dense_results,
#         bm25_results,
#         p=0.9
#     ):
#         """
#         Rank agreement between
#         Dense Retrieval and BM25.

#         Uses Rank Biased Overlap.
#         """

#         if len(dense_results) == 0 or len(bm25_results) == 0:
#             return 0.0

#         dense_ids = [
#             node.node.node_id
#             for node in dense_results
#         ]

#         bm25_ids = []

#         for chunk in bm25_results:

#             node_id = chunk.get(
#                 "node_id",
#                 chunk["text"]
#             )

#             bm25_ids.append(node_id)

#         similarity = RankingSimilarity(
#             dense_ids,
#             bm25_ids
#         )

#         return float(similarity.rbo(p=p))

#     ##########################################################

#     def reranker_stability(
#         self,
#         dense_results,
#         reranked_results,
#         p=0.9
#     ):
#         """
#         Measures how much
#         BGE changes ranking.
#         """

#         if len(dense_results) == 0 or len(reranked_results) == 0:
#             return 0.0

#         dense_ids = [
#             node.node.node_id
#             for node in dense_results
#         ]

#         reranked_ids = []

#         for doc, _ in reranked_results:

#             if "node_id" in doc:

#                 reranked_ids.append(
#                     doc["node_id"]
#                 )

#         similarity = RankingSimilarity(
#             dense_ids,
#             reranked_ids
#         )

#         return float(similarity.rbo(p=p))

#     ##########################################################

#     def overall(
#         self,
#         softmax_score,
#         margin_score,
#         entropy_score,
#         consensus_score,
#         stability_score
#     ):
#         """
#         Overall retrieval confidence.

#         Currently
#         arithmetic mean.

#         Can later be replaced by
#         LightGBM/XGBoost.
#         """

#         values = np.array([
#             softmax_score,
#             margin_score,
#             entropy_score,
#             consensus_score,
#             stability_score
#         ])

#         return float(np.mean(values))

#     ##########################################################

#     def compute(
#         self,
#         dense_results,
#         bm25_results,
#         reranked_results
#     ):

#         scores = [
#             node.score
#             for node in dense_results
#             if node.score is not None
#         ]

#         softmax_score = self.softmax_confidence(scores)

#         margin_score = self.margin_confidence(scores)

#         entropy_score = self.entropy_confidence(scores)

#         consensus_score = self.consensus(
#             dense_results,
#             bm25_results
#         )

#         stability_score = self.reranker_stability(
#             dense_results,
#             reranked_results
#         )

#         overall = self.overall(
#             softmax_score,
#             margin_score,
#             entropy_score,
#             consensus_score,
#             stability_score
#         )

#         return {

#             "overall": round(overall, 4),

#             "softmax": round(
#                 softmax_score,
#                 4
#             ),

#             "margin": round(
#                 margin_score,
#                 4
#             ),

#             "entropy": round(
#                 entropy_score,
#                 4
#             ),

#             "consensus": round(
#                 consensus_score,
#                 4
#             ),

#             "reranker_stability": round(
#                 stability_score,
#                 4
#             )
#         }