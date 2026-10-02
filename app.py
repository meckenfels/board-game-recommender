import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

data = {
    'Title': ['Heat: Pedal to the Metal', 'Nemesis', 'Roll Player', 'ISS Vanguard', 'Formula D'],
    'Description': [
        'Racing Motorsport. Mechanics: hand management, deck building.',
        'Sci-Fi Survival. Mechanics: semi-cooperative, hidden traitor.',
        'Fantasy. Mechanics: dice manipulation, character building.',
        'Sci-Fi Exploration. Mechanics: cooperative, campaign, narrative.',
        'Racing Motorsport. Mechanics: dice rolling, push your luck.'
    ]
}
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title):
    idx = df.index[df['Title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3] 
    game_indices = [i[0] for i in sim_scores]
    recommendations = df['Title'].iloc[game_indices].tolist()
    return f"1. {recommendations[0]}\n2. {recommendations[1]}"

game_list = df['Title'].tolist()

demo = gr.Interface(
    fn=recommend,
    inputs=gr.Dropdown(choices=game_list, label="Select a Board Game you enjoy:"),
    outputs=gr.Textbox(label="The AI Recommends:"),
    title="Board Game Recommender",
    description="A content-based recommendation system using TF-IDF and Cosine Similarity."
)

if __name__ == "__main__":
    demo.launch()
