import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle
import logging

# MongoDB setup
client = MongoClient(os.getenv('STRING_KEY'))
db = client['job_recommendation']

candidates_collection = db['candidates']
jobs_collection = db['jobs']
companies_collection = db['companies']

def load_data(collection):
    data = pd.DataFrame(list(collection.find()))
    if '_id' in data.columns:
        data['_id'] = data['_id'].apply(str)
    return data

# Load data
companies = load_data(companies_collection)
candidates = load_data(candidates_collection)
jobs = load_data(jobs_collection)

# Process jobs data
jobs['companyProfile'] = jobs['creator'].apply(lambda x: x.get('companyProfile'))
jobs['user'] = jobs['creator'].apply(lambda x: x.get('user'))
jobs.drop(columns=['creator'], inplace=True)

# Rename columns
jobs = jobs.rename(columns={
    '_id': 'JobID',
    'jobTitle': 'JobTitle',
    'experience': 'Experience',
    'education': 'Education',
    'skills': 'Skills',
    'location': 'Location',
    'jobDescription': 'Description',
    'companyProfile': 'CompanyID',
    'jobType': 'JobType'
})

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

companies = companies.rename(columns={
    '_id': 'CompanyID',
    'location': 'Location',
    'description': 'Description'
})

# Ensure 'Skills' columns are lists
jobs['Skills'] = jobs['Skills'].apply(lambda x: x if isinstance(x, list) else [x])
candidates['Skills'] = candidates['Skills'].apply(lambda x: x if isinstance(x, list) else [x])

# Fill NaN values in 'Description' and 'JobType' columns with empty strings
jobs['Description'] = jobs['Description'].fillna('')
candidates['Description'] = candidates['Description'].fillna('')
jobs['JobType'] = jobs['JobType'].fillna('')
candidates['JobType'] = candidates['JobType'].fillna('')

# Convert 'JobType' to strings
jobs['JobType'] = jobs['JobType'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
candidates['JobType'] = candidates['JobType'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Convert 'Designation' to strings and fill NaN values
jobs['Designation'] = jobs['JobTitle']
candidates['Designation'] = candidates['Designation'].fillna('')

# Fit TF-IDF Vectorizer on job and candidate descriptions
tfidf_vectorizer = TfidfVectorizer()
job_descriptions_tfidf = tfidf_vectorizer.fit_transform(jobs['Description'])
candidate_descriptions_tfidf = tfidf_vectorizer.transform(candidates['Description'])

# Encode skills
skills_set = list(set([skill for sublist in jobs['Skills'].tolist() + candidates['Skills'].tolist() for skill in sublist]))

def encode_features(df, feature_set, column_name):
    encoded_features = []
    for features in df[column_name]:
        encoded = [1 if feature in features else 0 for feature in feature_set]
        encoded_features.append(encoded)
    return np.array(encoded_features)

jobs_skills_encoded = encode_features(jobs, skills_set, 'Skills')
candidates_skills_encoded = encode_features(candidates, skills_set, 'Skills')

# Combine job and candidate data to fit label encoders
combined_designations = pd.concat([candidates['Designation'], jobs['Designation']])
combined_job_types = pd.concat([candidates['JobType'], jobs['JobType']])

designation_encoder = LabelEncoder()
designation_encoder.fit(combined_designations)

job_type_encoder = LabelEncoder()
job_type_encoder.fit(combined_job_types)

label_encoders = {
    'Designation': designation_encoder,
    'JobType': job_type_encoder
}

for col in ['Experience', 'Education', 'Location', 'Designation', 'JobType']:
    label_encoders[col] = LabelEncoder()
    combined_col_data = pd.concat([candidates[col].astype(str), jobs[col].astype(str)])
    label_encoders[col].fit(combined_col_data)

for col in ['Experience', 'Education', 'Location', 'Designation', 'JobType']:
    candidates[col] = label_encoders[col].transform(candidates[col].astype(str))
    jobs[col] = label_encoders[col].transform(jobs[col].astype(str))



# jobs_skills_encoded = np.tile(jobs_skills_encoded, (10, 1))
# job_descriptions_tfidf = job_descriptions_tfidf.toarray()


# Combine all features into the final array for similarity computation
jobs_combined = np.hstack((
    jobs[['Experience', 'Education', 'Location', 'Designation', 'JobType']],
    jobs_skills_encoded,
    job_descriptions_tfidf.toarray()
))

# candidates_skills_encoded = np.tile(candidates_skills_encoded, (93, 1)) 
# candidate_descriptions_tfidf = candidate_descriptions_tfidf.toarray()

candidates_combined = np.hstack((
    candidates[['Experience', 'Education', 'Location', 'Designation', 'JobType']],
    candidates_skills_encoded,
    candidate_descriptions_tfidf.toarray()
))



# extra_features = candidates_combined[1] - jobs_combined[1]
# if extra_features > 0:
#     jobs_combined = pd.hstack([jobs_combined, np.zeros((jobs_combined[0], encode_features))])
# else:
#     jobs_combined = jobs_combined
    
# Compute similarity matrices   
similarity_matrix = cosine_similarity(jobs_combined, candidates_combined)
candidate_similarity_matrix = similarity_matrix.T

# true_labels = np.random.randint(0, 2, size=(10 * 93))


# cosin_similarity = candidate_similarity_matrix.flatten()
# threshold = 0.5

# predicted_label = (cosin_similarity >= threshold).astype(int)

# Accuracy_score = accuracy_score(true_labels , predicted_label)
# F1_score = f1_score(true_labels, predicted_label)
# Precision_score = precision_score(true_labels, predicted_label)
# Recall_score = recall_score(true_labels, predicted_label)




# print(f"\n{Accuracy_score * 100}%: Percentage Accuracy")
# print(f"\n{F1_score * 100}%: f1 score Accuracy")
# print(f"\n{Precision_score * 100}%: Precision score Accuracy")
# print(f"\n{Recall_score * 100}: recall score Accuracy")

# Save similarity matrix and label encoders to pickle files 
with open('models/similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

with open('models/candidate_similarity_matrix.pkl', 'wb') as f:
    pickle.dump(candidate_similarity_matrix, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save jobs and companies data to pickle files for use in app.py
with open('models/jobs.pkl', 'wb') as f:
    pickle.dump(jobs, f)

with open('models/companies.pkl', 'wb') as f:
    pickle.dump(companies, f)

def hybrid_recommendation(candidate_id, candidates, jobs, companies, similarity_matrix, candidate_similarity_matrix):
    try:
        candidate_idx = candidates[candidates['CandidateID'] == candidate_id].index[0]

        # Top 3 job recommendations for the candidate
        job_indices = np.argsort(-similarity_matrix[:, candidate_idx])[:3]
        content_based = jobs.iloc[job_indices].copy()
        content_based['Similarity'] = similarity_matrix[job_indices, candidate_idx]

        # Recommend companies related to the top jobs
        recommended_companies = companies[companies['CompanyID'].isin(content_based['CompanyID'])]
        if recommended_companies.empty:
            recommended_companies = companies[
                (companies['Location'] == candidates.at[candidate_idx, 'Location']) |
                (companies['CompanyID'].isin(content_based['CompanyID']))
            ]

        # Recommend similar candidates for the top job
        job_id = jobs.iloc[job_indices[0]]['JobID']
        job_idx = jobs[jobs['JobID'] == job_id].index[0]
        candidate_indices = np.argsort(-candidate_similarity_matrix[:, job_idx])[:3]
        candidates_for_companies = candidates.iloc[candidate_indices].copy()
        candidates_for_companies['Similarity'] = candidate_similarity_matrix[candidate_indices, job_idx]

        return content_based.sort_values(by='Similarity', ascending=False).reset_index(drop=True), recommended_companies, candidates_for_companies.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    except IndexError:
        logging.error(f"No candidate found with ID {candidate_id}.")
        return pd.DataFrame(columns=['JobID', 'JobTitle', 'Similarity']), pd.DataFrame(columns=['CompanyID', 'Location']), pd.DataFrame(columns=['CandidateID', 'Similarity'])
