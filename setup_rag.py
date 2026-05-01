# --- 雲端 SQLite 版本過舊的修正補丁 ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -------------------------------------

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# --- SOP 知識庫文件 ---
SOP_DOCUMENTS = [
    (
        "SOP-001",
        "SOP-001 [細胞壅塞處理]：當 PRB 利用率持續高於 85% 或出現 Cell Congestion Alarm 時，"
        "系統應自動啟動 Load Balancing 機制將邊緣用戶移轉至相鄰基站。"
        "若壅塞未解，建議遠端調降天線 Electrical Tilt (電氣下傾角) 2-3 度以縮小實體涵蓋範圍，減少干擾。",
    ),
    (
        "SOP-002",
        "SOP-002 [連線建立失效處理]：當 RRC 建立成功率跌破 80% 或出現大量 Handover Failure 時，"
        "為防止基地台核心網路癱瘓，應立即啟動 Admission Control (接納控制)。"
        "同時調整 QoS，降低 P2P 與一般影音串流的 QCI 優先級，強制保留資源給 VoLTE 語音通話與緊急簡訊。",
    ),
    (
        "SOP-003",
        "SOP-003 [硬體模組故障排除]：若系統持續回報 Radio Link Failure 或 Hardware Fault，"
        "且各項軟體參數調整皆無效，初步判定為 RRU (射頻拉遠單元) 或光纖線路實體毀損。"
        "需立即生成實體派工單，指派距離最近之外勤工程師攜帶備用射頻模組與融接機前往站台搶修。",
    ),
]

COLLECTION_NAME = "cht_sop_knowledge_base"
DB_PATH = "./chroma_db"

# --- 自訂 Embedding Function（讓 ChromaDB 使用 sentence-transformers）---
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"載入 Embedding 模型：{model_name} ...")
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()


def build_vector_db() -> chromadb.Collection:
    """初始化 ChromaDB，寫入 SOP 文件並回傳 Collection。"""
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = LocalEmbeddingFunction()

    # 若已存在則先刪除，確保重跑腳本時數據是最新的
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"已清除舊版 Collection：{COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},   # 使用 Cosine 相似度
    )

    ids, documents, metadatas = [], [], []
    for sop_id, sop_text in SOP_DOCUMENTS:
        ids.append(sop_id)
        documents.append(sop_text)
        metadatas.append({"source": sop_id})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"已寫入 {len(ids)} 筆 SOP 文件至 ChromaDB（路徑：{DB_PATH}）")
    return collection


def get_relevant_sop(query_text: str, n_results: int = 1) -> list[dict]:
    """
    對 ChromaDB 進行語意相似度搜尋，回傳最相關的 SOP。

    Returns:
        List of dict，每筆包含 'id', 'document', 'distance'
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = LocalEmbeddingFunction()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 — RAG 知識庫建置")
    print("=" * 60)

    # Step 1：建立向量資料庫並寫入 SOP
    build_vector_db()

    # Step 2：測試檢索
    test_queries = [
        "目前 RRC 成功率掉到 60% 以下，用戶連不上線",
        "PRB 利用率超過 90%，基站出現壅塞告警",
        "Radio Link Failure 一直跳，懷疑是硬體問題",
    ]

    print("\n" + "=" * 60)
    print("RAG 檢索測試")
    print("=" * 60)

    for query in test_queries:
        print(f"\n【查詢】{query}")
        hits = get_relevant_sop(query, n_results=1)
        for hit in hits:
            print(f"  >> 命中：{hit['id']}  (Cosine Distance: {hit['distance']:.4f})")
            print(f"  {hit['document']}")
