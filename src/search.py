def search(self, query, collection_names=None, top_k=2):
    if not self.collections:
        return [
            {
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "로드된 컬렉션이 없습니다."},
            }
        ]

    use_collections = [
        c
        for c in self.collections
        if not collection_names or c["name"] in collection_names
    ]
    if not use_collections:
        return [
            {
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "지정된 컬렉션을 찾을 수 없습니다."},
            }
        ]

    print(f"\n🔍 쿼리: {query}")
    print(f"🎯 검색 대상: {[c['name'] for c in use_collections]}")
    print(f"📌 Top K: {top_k}")

    all_results = []
    for coll in use_collections:
        name = coll["name"]
        vectordb = coll["vectordb"]
        try:
            docs_and_scores = vectordb.similarity_search_with_score(query, k=top_k)
            print(f"✅ '{name}' → {len(docs_and_scores)}개 결과")

            for doc, score in docs_and_scores:
                all_results.append(
                    {
                        "collection": name,
                        "id": doc.metadata.get("id", "N/A"),
                        "score": 1 - score,  # 낮을수록 유사한 거리 ➜ 반전
                        "metadata": doc.metadata,
                    }
                )

        except Exception as e:
            print(f"❌ '{name}' 컬렉션 검색 중 오류: {e}")
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)

    return (
        all_results
        if all_results
        else [
            {
                "collection": "default",
                "id": "0",
                "score": 1.0,
                "metadata": {"text": "검색 결과를 찾을 수 없습니다."},
            }
        ]
    )
