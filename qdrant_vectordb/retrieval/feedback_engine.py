class FeedbackEngine:

    def decide(

        self,

        report,

        statistics

    ):

        ##################################################

        if statistics is None:

            return {

                "confidence": None,

                "action": "Collect More Data",

                "reason": [

                    "Need more benchmark queries."

                ]

            }

        ##################################################

        confidence = {}

        ##################################################

        for metric in [

            "average_score",

            "query_coverage",

            "metadata_match",

            "retriever_consensus"

        ]:

            mean = statistics[metric]["mean"]

            std = statistics[metric]["std"]

            value = report[metric]

            ##################################################

            if std == 0:

                confidence[metric] = 0

            else:

                confidence[metric] = (

                    value -

                    mean

                ) / std

        ##################################################
        # Lowest performing metric
        ##################################################

        weakest = min(

            confidence,

            key=confidence.get

        )

        ##################################################

        mapping = {

            "average_score":

                "Increase Top-K",

            "query_coverage":

                "Expand Query",

            "metadata_match":

                "Use Property Graph",

            "retriever_consensus":

                "Retrieve Again"

        }

        return {

            "confidence": confidence,

            "weakest_metric": weakest,

            "recommended_action":

                mapping[weakest]

        }