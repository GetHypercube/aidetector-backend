import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

mongodb_url = os.getenv("MONGODB_URL")

try:
    with MongoClient(mongodb_url) as client:
        db = client['test']
        collection = db['aidetectorresults']

        document = {
            "image_path": '/image.png',
            "inference_results": {
                "inferenceResults": {
                    "executionTime": 1.7613134384155273,
                    "fusedLogit": -15.15683889389038,
                    "fusedProbability": 0.00027088913673041477,
                    "isDiffusionImage": 'false',
                    "logits": {
                        "Grag2021_latent": -7.520112037658691,
                        "Grag2021_progan": -22.79356575012207
                    },
                    "probabilities": {
                        "Grag2021_latent": 0.000541778147312887,
                        "Grag2021_progan": 1.261479425851178e-10
                    }
                }
            }
        }

        result = collection.insert_one(document)

        if result.inserted_id:
            print("Document successfully inserted", result.inserted_id)
        else:
            print("Document insertion was unsuccessful")

except Exception as e:
    print("Error:", str(e))
