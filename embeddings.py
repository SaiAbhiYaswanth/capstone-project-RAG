from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class EmbeddingModel:

    _model = None

    @classmethod
    def load_model(cls):

        if cls._model is None:
            cls._model = SentenceTransformer(EMBEDDING_MODEL)

        return cls._model

    @classmethod
    def embed(cls, texts):

        model = cls.load_model()

        return model.encode(texts).tolist()