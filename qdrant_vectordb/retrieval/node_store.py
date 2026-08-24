import os
import pickle
from retrieval.settings import STORAGE_ROOT
from utils.logger import logger

NODES_PATH = str(STORAGE_ROOT / "nodes.pkl")


class NodeStore:

    @staticmethod
    def load():

        if not os.path.exists(NODES_PATH):
            return []

        with open(NODES_PATH, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save(nodes):

        with open(NODES_PATH, "wb") as f:
            pickle.dump(nodes, f)

    @staticmethod
    def append(new_nodes):

        nodes = NodeStore.load()

        nodes.extend(new_nodes)

        NodeStore.save(nodes)

        logger.info(f"Total stored nodes: {len(nodes)}")

    @staticmethod
    def replace_document(filename, new_nodes):

        nodes = NodeStore.load()

        nodes = [
            node
            for node in nodes
            if node.metadata.get("filename") != filename
        ]

        nodes.extend(new_nodes)

        NodeStore.save(nodes)

        logger.info(
            f"Updated node hierarchy for {filename}"
        )

    @staticmethod
    def clear():

        if os.path.exists(NODES_PATH):
            os.remove(NODES_PATH)