from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import process
import torch

app = Flask(__name__)

# ========== تحميل البيانات ==========
df = pd.read_csv("movie.csv")
df = df.dropna(subset=['title', 'description', 'listed_in', 'rating'])

# ========== تجهيز النصوص ==========
df['content'] = (
    "Title: " + df['title'] + ". " +
    "Genres: " + ((df['listed_in'] + ". ") * 2) +
    "Rating: " + df['rating'] + ". " +
    "Description: " + ((df['description'] + ". ") * 3)
)

# ========== تحميل النموذج ==========
model = SentenceTransformer('all-MiniLM-L6-v2')

# ========== إنشاء التضمينات ==========
print("⚙️ يتم إنشاء التضمينات...")
embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True, show_progress_bar=True)
print("✅ تم إنشاء التضمينات.")

# ========== دالة تصحيح اسم الفيلم ==========
def correct_title(user_input, titles):
    best_match, score, _ = process.extractOne(user_input, titles)
    if score >= 90:
        return best_match
    return None

# ========== دالة التوصية ==========
def recommend(user_input):
    titles = df['title'].tolist()
    corrected = correct_title(user_input, titles)

    if corrected:
        # إذا وجدنا تطابق قوي
        idx = df[df['title'] == corrected].index[0]
        input_content = df.at[idx, 'content']
        query_embedding = model.encode([input_content], convert_to_tensor=True)
    else:
        # لم نجد تطابق جيد -> نستخدم العنوان نفسه
        query_text = "Title: " + user_input
        query_embedding = model.encode([query_text], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cosine_scores, k=10)

    recommendations = []
    for i in top_results.indices:
        i = int(i)
        candidate_title = df.iloc[i]['title']
        if corrected and candidate_title == corrected:
            continue
        if candidate_title not in recommendations:
            recommendations.append(candidate_title)
        if len(recommendations) == 5:
            break

    if not recommendations:
        recommendations = ["❌ لم أتمكن من العثور على توصيات مشابهة."]

    return recommendations, None

# ========== صفحة الويب ==========
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    error_message = None
    if request.method == "POST":
        user_input = request.form["movie"]
        recommendations, error_message = recommend(user_input)

    return render_template("index.html", recommendations=recommendations, error_message=error_message)

# ========== تشغيل التطبيق ==========
if __name__ == "__main__":
    app.run(debug=True)

