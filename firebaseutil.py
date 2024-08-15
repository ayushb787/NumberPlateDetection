import firebase_admin
from firebase_admin import credentials
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime

cred = credentials.Certificate("deeplearningproject-a8099-firebase-adminsdk-81u28-c85bb2b075.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def findNoPlate(vehNo):
    query = (db.collection('vehicledata')
             .where(filter=FieldFilter("vehNo", "==", vehNo))
             ).limit(1).stream()
    existing_entry = None
    for doc in query:
        existing_entry = doc.reference
        break
    if existing_entry:
        existing_entry.update({'last_updated': datetime.now()})
        print("Entry updated with current date and time - " + vehNo)
    else:
        print("No existing entry found for the provided vehicle number - " + vehNo)


firebase_admin.delete_app(firebase_admin.get_app())
