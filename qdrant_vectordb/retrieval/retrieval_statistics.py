from statistics import mean, pstdev


class RetrievalStatistics:

    def __init__(self):

        self.history = []

    ##################################################

    def update(self, report):

        self.history.append({

            "average_score": report["average_score"],

            "query_coverage": report["query_coverage"],

            "metadata_match": report["metadata_match"],

            "retriever_consensus": report["retriever_consensus"]

        })

    ##################################################

    def compute(self):

        if len(self.history) < 2:

            return None

        stats = {}

        for metric in self.history[0]:

            values = [

                item[metric]

                for item in self.history

            ]

            stats[metric] = {

                "mean": mean(values),

                "std": pstdev(values),

                "min": min(values),

                "max": max(values)

            }

        return stats