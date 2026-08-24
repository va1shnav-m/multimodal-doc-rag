from kneed import KneeLocator

class AdaptiveTopK:

    def select(self, dense_results):

        if len(dense_results) <= 5:
            return dense_results

        scores = [node.score for node in dense_results]

        max_score = max(scores)
        min_score = min(scores)

        normalized = [
            (s - min_score) / (max_score - min_score + 1e-9)
            for s in scores
        ]

        x = list(range(len(normalized)))

        knee = KneeLocator(
            x,
            normalized,
            curve="convex",
            direction="decreasing"
        )

        if knee.knee is None:
            return dense_results

        return dense_results[:knee.knee + 1]