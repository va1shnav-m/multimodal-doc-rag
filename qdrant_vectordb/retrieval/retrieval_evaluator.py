from statistics import mean
from utils.logger import logger
import re
from statistics import mean, pstdev
class RetrievalEvaluator:
    

    def evaluate(

        self,

        query,

        reranked_results,

        dense_results,

        bm25_results

    ):

        report = {}

        ##############################################
        # Empty retrieval
        ##############################################

        if not reranked_results:

            return {

                "overall_confidence": 0,

                "signals": {}

            }

        ##############################################
        # 1. Reranker Scores
        ##############################################

        scores = [

            score

            for _, score in reranked_results

        ]

        report["best_score"] = max(scores)
        report["average_score"] = mean(scores)
        report["score_std"] = round(

            pstdev(scores),

            3

        ) if len(scores) > 1 else 0
        report["top_scores"] = [

            round(score, 3)

            for score in scores[:5]

        ]
        
        ##############################################
        # 2. Margin
        ##############################################

        if len(scores) > 1:

            report["margin"] = (

                scores[0]

                -

                scores[1]

            )

        else:

            report["margin"] = scores[0]

        ##############################################
        # 3. Document Diversity
        ##############################################

        documents = {

            doc["metadata"].get(

                "filename",

                "Unknown"

            )

            for doc, _ in reranked_results

        }

        report["document_diversity"] = len(documents)
        report["chunk_count"] = len(

                reranked_results

            )

        ##############################################
        # 4. Retriever Consensus
        ##############################################

        dense_ids = {

            node.node.node_id

            for node in dense_results

        }

        bm25_ids = {

            doc.get(

                "node_id"

            )

            for doc in bm25_results

        }

        overlap = dense_ids.intersection(

            bm25_ids

        )

        report["retriever_consensus"] = len(

            overlap

        )
        coverage, matched = self.calculate_query_coverage(

            query,

            reranked_results

        )

        report["query_coverage"] = round(

            coverage,

            3

        )

        report["matched_terms"] = matched
        ##################################################
        # Metadata Match
        ##################################################

        metadata_score, metadata_matches = (

            self.calculate_metadata_match(

                query,

                reranked_results

            )

        )

        report["metadata_match"] = round(

            metadata_score,

            3

        )

        report["metadata_matches"] = metadata_matches
        ##################################################
        # Metadata Statistics
        ##################################################

        entity_count = 0
        keyword_count = 0
        topic_count = 0

        for doc, _ in reranked_results:

            metadata = doc.get(

                "metadata",

                {}

            )

            entity_count += len(

                metadata.get(

                    "entities",

                    []

                )

            )

            keyword_count += len(

                metadata.get(

                    "keywords",

                    []

                )

            )

            topic_count += len(

                metadata.get(

                    "topics",

                    []

                )

            )

        report["entities_found"] = entity_count

        report["keywords_found"] = keyword_count

        report["topics_found"] = topic_count

        ##############################################
        # Overall Report
        ##############################################

        logger.info("=" * 80)

        logger.info(

            "Retrieval Evaluation"

        )

        logger.info("=" * 80)
        logger.info("RETRIEVAL QUALITY REPORT")
        logger.info("=" * 80)

        for key, value in report.items():

            logger.info(

                f"{key:<25}: {value}"

            )

        logger.info("=" * 80)

        logger.info("=" * 80)

        return report
    def calculate_query_coverage(

    self,

    query,

    reranked_results

):

    ##################################################
    # Extract words from query
    ##################################################

        query_terms = set(

            re.findall(

                r"\w+",

                query.lower()

            )

        )

        ##################################################
        # Build retrieved text
        ##################################################

        retrieved_text = ""

        for doc, _ in reranked_results:

            retrieved_text += " "

            retrieved_text += doc["text"].lower()

        ##################################################
        # Count matches
        ##################################################

        matched = [

            term

            for term in query_terms

            if term in retrieved_text

        ]

        coverage = len(matched) / max(

            len(query_terms),

            1

        )

        return coverage, matched
    def calculate_metadata_match(

    self,

    query,

    reranked_results

):

    ##################################################
    # Extract words from query
    ##################################################

        query_terms = set(

            re.findall(

                r"\w+",

                query.lower()

            )

        )

        ##################################################
        # Collect metadata terms
        ##################################################

        metadata_terms = set()

        for doc, _ in reranked_results:

            metadata = doc.get(

                "metadata",

                {}

            )

            ##################################################
            # Entities
            ##################################################

            for entity in metadata.get(

                "entities",

                []

            ):

                metadata_terms.update(

                    re.findall(

                        r"\w+",

                        str(entity).lower()

                    )

                )

            ##################################################
            # Keywords
            ##################################################

            for keyword in metadata.get(

                "keywords",

                []

            ):

                metadata_terms.update(

                    re.findall(

                        r"\w+",

                        str(keyword).lower()

                    )

                )

            ##################################################
            # Topics
            ##################################################

            for topic in metadata.get(

                "topics",

                []

            ):

                metadata_terms.update(

                    re.findall(

                        r"\w+",

                        str(topic).lower()

                    )

                )

        ##################################################
        # Find matching metadata terms
        ##################################################

        matched = sorted(

            query_terms.intersection(

                metadata_terms

            )

        )

        coverage = len(

            matched

        ) / max(

            len(query_terms),

            1

        )

        return coverage, matched