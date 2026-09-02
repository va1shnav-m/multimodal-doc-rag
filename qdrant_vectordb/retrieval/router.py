from retrieval.simple_search import SimpleSearch
from retrieval.hybrid_search import HybridSearch
from retrieval.route_selector import RouteSelector
from retrieval.router_schema import Pipeline
from utils.logger import logger


class AdaptiveRouter:

    def __init__(self, retriever, llm_choice: str = None):

        self.simple = SimpleSearch(retriever)
        self.hybrid = HybridSearch(retriever)

        self.selector = RouteSelector(llm_choice=llm_choice)

    def retrieve(self, query):

        decision = self.selector.select(query)

        logger.info("=" * 60)
        logger.info("Adaptive Router")
        logger.info(f"Pipeline : {decision.pipeline}")
        logger.info(f"Reason   : {decision.reason}")
        logger.info("=" * 60)

        if decision.pipeline == Pipeline.DIRECT:
            logger.info("Skipping document retrieval -> Direct LLM generation")
            return {
                "context": "",
                "supporting_evidence": [],
                "documents_used": [],
                "analytics": {
                    "pipeline": "direct",
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                },
                "retrieval_report": f"DIRECT Response (No retrieval needed): {decision.reason}",
            }

        if decision.pipeline == Pipeline.FAST:
            return self.simple.retrieve(query)

        return self.hybrid.retrieve(query)