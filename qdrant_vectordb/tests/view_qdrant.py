from qdrant_client import QdrantClient

COLLECTION_NAME = "production_rag"

client = QdrantClient(
    path="storage/qdrant"
)

print("=" * 80)
print("COLLECTION INFORMATION")
print("=" * 80)

collection = client.get_collection(
    COLLECTION_NAME
)

print(collection)

print("\n")
print("=" * 80)
print("FIRST 5 STORED POINTS")
print("=" * 80)

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_payload=True,
    with_vectors=True
)

for i, point in enumerate(points, start=1):

    print(f"\nPoint {i}")
    print("-" * 60)

    print(f"ID: {point.id}")

    print("\nPayload (Metadata):")
    print(point.payload)

    print("\nVector Dimension:")
    print(len(point.vector))

    print("\nFirst 10 Vector Values:")
    print(point.vector[:10])

    print("\n")