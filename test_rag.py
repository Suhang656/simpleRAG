import unittest
import numpy as np
from simpleRAG import RAGConfig, RAGSystem

class FakeEmbedder:
    def embed(self, texts):
        return np.asarray([[t.count("地球")+t.count("太阳"), t.count("大雄")+t.count("哆啦")] for t in texts], dtype="float32") + 1e-6

class Tests(unittest.TestCase):
    def setUp(self):
        self.rag = RAGSystem(config=RAGConfig(chunk_size=20, chunk_overlap=2, min_relevance=.01), embedder=FakeEmbedder())
        self.rag.load_and_index("地球围绕太阳运行。\n\n大雄和哆啦A梦是朋友。", "text", "test")
    def test_hybrid_retrieval(self):
        self.assertIn("哆啦A梦", self.rag.retrieve("大雄的朋友是谁？")[0].chunk.text)
    def test_offline_citation(self):
        answer = self.rag.query("地球围绕什么运行？", mode="fast")
        self.assertIn("[1]", answer); self.assertIn("太阳", answer)

if __name__ == "__main__": unittest.main()
