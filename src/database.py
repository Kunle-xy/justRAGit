import weaviate
import dotenv
import os

dotenv.load_dotenv()

APIKEY = os.getenv("WEAVIATE_API_KEY")
URL = os.getenv("WEAVIATE_URL")
OPENAI_APIKEY = os.getenv("OPENAI_API_KEY")


def create_client():
    return weaviate.connect_to_wcs(
            cluster_url=URL,  # Replace with your WCS URL
            auth_credentials=weaviate.auth.AuthApiKey(APIKEY),
            headers={
            "X-OpenAI-Api-Key": OPENAI_APIKEY # Replace with your inference API key
                    } # Replace with your WCS key
        )



if __name__ == "__main__":
    client =  create_client()
    print(client.is_ready())
