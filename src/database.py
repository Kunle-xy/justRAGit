import weaviate
import dotenv
import os

dotenv.load_dotenv()

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")


def create_client():
    return weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)


if __name__ == "__main__":
    client = create_client()
    print(client)
