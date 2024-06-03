from flask import Flask, request, render_template
import pandas as pd
import pickle
from pymongo import MongoClient
from main import hybrid_recommendation
import os
app = Flask(__name__)

# Load models and data
with open('models/similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)
    
with open('models/candidate_similarity_matrix.pkl', 'rb') as f:
    candidate_similarity_matrix = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/jobs.pkl', 'rb') as f:
    jobs = pickle.load(f)

with open('models/companies.pkl', 'rb') as f:
    companies = pickle.load(f)

def load_candidates():
    client = MongoClient(os.getenv('STRING_KEY'))
    db = client['job_recommendation']
    candidates_collection = db['candidates']
    candidates = pd.DataFrame(list(candidates_collection.find()))
    candidates['_id'] = candidates['_id'].apply(str)
    candidates = candidates.rename(columns={
        '_id': 'CandidateID',
        'designation': 'Designation',
        'qualification': 'Education',
        'city': 'Location',
        'description': 'Description',
        'experience': 'Experience',
        'jobType': 'JobType',
        'skills': 'Skills',
    })
    
    return candidates

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    candidate_id = request.form['candidate_id']
    candidates = load_candidates()
    
    content_based, recommended_companies, candidates_for_companies = hybrid_recommendation(
        candidate_id, candidates, jobs, companies, similarity_matrix, candidate_similarity_matrix
    )
    
    return render_template('recommendations.html', jobs=content_based, companies=recommended_companies, candidates=candidates_for_companies)

if __name__ == '__main__':
    app.run(debug=True)
