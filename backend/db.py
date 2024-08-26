# mongodb driver
from dotenv import load_dotenv
import pymongo
from bson.json_util import dumps
import os 
from typing import List, Dict, Tuple, Set
from bson.objectid import ObjectId   


load_dotenv()
# uri = fromenv
uri = os.getenv("DB_URI")


class Mongo:
  def __init__(self) -> None:
      load_dotenv()
      uri = os.getenv("DB_URI")
      self.client = pymongo.MongoClient(uri)
      #self.db = self.client.ai
      self.client['zuum_api_prod']
  
  def get_shipment_status(self, _id) -> str:
      status = self.db.shipments.find(
          {"_id": _id}
      )
      print(status.retrieved)

      if status.retrieved < 1:
         raise Exception("gg")
      return status[0]
  
if __name__ == "__main__":
  mongoDB = Mongo()
  status = mongoDB.get_shipment_status(ObjectId("5a6f4d14e8083c5637de984d"))
  print(dumps(status, indent=2))
